import os
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F
import yaml

from sam2.build_sam import build_sam2_video_predictor
from sam2.utils.transforms import SAM2Transforms
from utils.utils import keep_largest_component, determine_tracker
from vot.region import RegionType
from vot.region.raster import calculate_overlaps
from vot.region.shapes import Mask

# 加载配置文件和设置随机种子
config_path = Path(__file__).parent / "dam4sam_config.yaml"
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

seed = config["seed"]
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class DAM4SAMTracker():
    def __init__(self, tracker_name="sam21pp-L"):
        """
        DAM4SAM (2.1) 跟踪器的构造函数。
        """
        self.checkpoint, self.model_cfg = determine_tracker(tracker_name)

        # 图像预处理参数
        self.input_image_size = 1024
        self.img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        self.img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]

        self.predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint, device="cuda:0")
        self.tracking_times = []

    def _prepare_image(self, img_pil):
        """将 PIL 图像转换为模型所需的张量。"""
        img = torch.from_numpy(np.array(img_pil)).to(self.inference_state["device"])
        img = img.permute(2, 0, 1).float() / 255.0
        img = F.resize(img, (self.input_image_size, self.input_image_size))
        img = (img - self.img_mean) / self.img_std
        return img

    @torch.inference_mode()
    def init_state_tw(self):
        """初始化一个推理状态字典。"""
        compute_device = torch.device("cuda")
        inference_state = {}
        inference_state["images"] = None
        inference_state["num_frames"] = 0
        inference_state["offload_video_to_cpu"] = False
        inference_state["offload_state_to_cpu"] = False
        inference_state["video_height"] = None
        inference_state["video_width"] = None
        inference_state["device"] = compute_device
        inference_state["storage_device"] = compute_device
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        inference_state["adds_in_drm_per_obj"] = {}
        inference_state["cached_features"] = {}
        inference_state["constants"] = {}
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        inference_state["output_dict_per_obj"] = {}
        inference_state["temp_output_dict_per_obj"] = {}
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),
            "non_cond_frame_outputs": set(),
        }
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        inference_state["frames_tracked_per_obj"] = {}

        self.img_mean = self.img_mean.to(compute_device)
        self.img_std = self.img_std.to(compute_device)

        return inference_state

    @torch.inference_mode()
    def initialize(self, image, init_mask=None, init_prompts=None):
        """
        使用第一帧和一组混合提示（点和/或框）或一个初始掩码来初始化跟踪器。
        """
        self.frame_index = 0
        self.object_sizes = []
        self.last_added = -1

        self.img_width = image.width
        self.img_height = image.height
        self.inference_state = self.init_state_tw()
        self.inference_state["video_height"] = self.img_height
        self.inference_state["video_width"] = self.img_width

        prepared_img = self._prepare_image(image)
        self.inference_state["images"] = {0: prepared_img}
        self.inference_state["num_frames"] = 1

        self.predictor.reset_state(self.inference_state)
        self.predictor._get_image_feature(self.inference_state, frame_idx=0, batch_size=1)

        if init_mask is None:
            if init_prompts and (
                    init_prompts.get("pos_points") or init_prompts.get("neg_points") or init_prompts.get("box")):
                init_mask = self.estimate_mask_from_prompts(init_prompts)
            else:
                raise ValueError("错误：必须提供 init_mask 或有效的 init_prompts。")

        if init_mask is None:
            raise ValueError("错误：从提供的提示中未能成功生成掩码。")

        _, _, out_mask_logits = self.predictor.add_new_mask(
            inference_state=self.inference_state,
            frame_idx=0,
            obj_id=0,
            mask=init_mask,
        )

        m = (out_mask_logits[0, 0] > 0).float().cpu().numpy().astype(np.uint8)
        self.inference_state["images"].pop(self.frame_index, None)

        return {'pred_mask': m}

    @torch.inference_mode()
    def track(self, image, init=False):
        """
        跟踪下一帧中的对象。
        """
        torch.cuda.empty_cache()
        prepared_img = self._prepare_image(image).unsqueeze(0)
        if not init:
            self.frame_index += 1
            self.inference_state["num_frames"] += 1
        self.inference_state["images"][self.frame_index] = prepared_img

        for out in self.predictor.propagate_in_video(self.inference_state, start_frame_idx=self.frame_index,
                                                     max_frame_num_to_track=0, return_all_masks=True):
            if len(out) == 3:
                out_frame_idx, _, out_mask_logits = out
                m = (out_mask_logits[0][0] > 0.0).float().cpu().numpy().astype(np.uint8)
            else:
                out_frame_idx, _, out_mask_logits, alternative_masks_ious = out
                m = (out_mask_logits[0][0] > 0.0).float().cpu().numpy().astype(np.uint8)

                alternative_masks, out_all_ious = alternative_masks_ious
                m_idx = np.argmax(out_all_ious)
                m_iou = out_all_ious[m_idx]
                alternative_masks = [mask for i, mask in enumerate(alternative_masks) if i != m_idx]

                n_pixels = (m == 1).sum()
                self.object_sizes.append(n_pixels)
                if len(self.object_sizes) > 1 and n_pixels >= 1:
                    obj_sizes_ratio = n_pixels / np.median(
                        [size for size in self.object_sizes[-300:] if size >= 1][-10:])
                else:
                    obj_sizes_ratio = -1

                if m_iou > 0.8 and obj_sizes_ratio >= 0.8 and obj_sizes_ratio <= 1.2 and n_pixels >= 1 and (
                        self.frame_index - self.last_added > 5 or self.last_added == -1):
                    alternative_masks = [Mask((m_[0][0] > 0.0).cpu().numpy()).rasterize(
                        (0, 0, self.img_width - 1, self.img_height - 1)).astype(np.uint8)
                                         for m_ in alternative_masks]
                    chosen_mask_np = m.copy()
                    chosen_bbox = Mask(chosen_mask_np).convert(RegionType.RECTANGLE)
                    alternative_masks = [np.logical_and(m_, np.logical_not(chosen_mask_np)).astype(np.uint8) for m_ in
                                         alternative_masks]
                    alternative_masks = [keep_largest_component(m_) for m_ in alternative_masks if np.sum(m_) >= 1]
                    if len(alternative_masks) > 0:
                        alternative_masks = [np.logical_or(m_, chosen_mask_np).astype(np.uint8) for m_ in
                                             alternative_masks]
                        alternative_bboxes = [Mask(m_).convert(RegionType.RECTANGLE) for m_ in alternative_masks]
                        ious = [calculate_overlaps([chosen_bbox], [bbox])[0] for bbox in alternative_bboxes]
                        if np.min(np.array(ious)) <= 0.7:
                            self.last_added = self.frame_index
                            self.predictor.add_to_drm(
                                inference_state=self.inference_state,
                                frame_idx=out_frame_idx,
                                obj_id=0,
                            )

            out_dict = {'pred_mask': m}
            self.inference_state["images"].pop(self.frame_index, None)
            return out_dict

    def estimate_mask_from_prompts(self, prompts):
        """
        从一组混合提示（前景/背景点和/或框）中估计初始掩码。
        """
        # 步骤 1: 获取已缓存的图像特征
        (
            _, _, current_vision_feats, _, feat_sizes
        ) = self.predictor._get_image_feature(self.inference_state, 0, 1)
        device = current_vision_feats[0].device

        _transforms = SAM2Transforms(
            resolution=self.predictor.image_size,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0
        )

        points_for_encoder = None
        box_for_encoder = None

        # 步骤 2: 准备点提示 (已修复坐标变换)
        pos_points = prompts.get("pos_points", [])
        neg_points = prompts.get("neg_points", [])

        if pos_points or neg_points:
            all_coords_np = np.array(pos_points + neg_points, dtype=np.float32)
            all_labels_np = np.array([1] * len(pos_points) + [0] * len(neg_points), dtype=np.int32)

            point_coords_orig = torch.as_tensor(all_coords_np, dtype=torch.float, device=device)
            point_labels = torch.as_tensor(all_labels_np, dtype=torch.int, device=device).unsqueeze(0)

            # --- 对点坐标进行和BOX一样的变换，使其匹配模型输入尺寸(1024x1024) ---
            transformed_point_coords = _transforms.transform_coords(
                point_coords_orig, normalize=True, orig_hw=(self.img_height, self.img_width)
            ).unsqueeze(0)

            points_for_encoder = (transformed_point_coords, point_labels)

        # 步骤 3: 准备框提示
        bbox = prompts.get("box")
        if bbox:
            box_coords = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])[None, :]
            box_tensor = torch.as_tensor(box_coords, dtype=torch.float, device=device)

            box_for_encoder = _transforms.transform_boxes(
                box_tensor, normalize=True, orig_hw=(self.img_height, self.img_width)
            )

        # 步骤 4: 调用 Prompt Encoder
        sparse_embeddings, dense_embeddings = self.predictor.sam_prompt_encoder(
            points=points_for_encoder,
            boxes=box_for_encoder,
            masks=None
        )

        # 步骤 5: 调用 Mask Decoder
        high_res_features = []
        for i in range(2):
            _, b_, c_ = current_vision_feats[i].shape
            high_res_features.append(
                current_vision_feats[i].permute(1, 2, 0).view(b_, c_, feat_sizes[i][0], feat_sizes[i][1]))

        if self.predictor.directly_add_no_mem_embed:
            img_embed = current_vision_feats[2] + self.predictor.no_mem_embed
        else:
            img_embed = current_vision_feats[2]
        _, b_, c_ = current_vision_feats[2].shape
        img_embed = img_embed.permute(1, 2, 0).view(b_, c_, feat_sizes[2][0], feat_sizes[2][1])

        low_res_masks, iou_predictions, _, _ = self.predictor.sam_mask_decoder(
            image_embeddings=img_embed,
            image_pe=self.predictor.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        # 步骤 6: 后处理并选出最佳掩码
        masks = _transforms.postprocess_masks(low_res_masks, (self.img_height, self.img_width)) > 0

        masks_np = masks.squeeze(0).float().detach().cpu().numpy()
        iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()

        best_mask_idx = np.argmax(iou_predictions_np)
        init_mask = masks_np[best_mask_idx]


        return init_mask