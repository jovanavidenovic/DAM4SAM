import numpy as np
import yaml
import torch
from vot.region.raster import calculate_overlaps
from vot.region.shapes import Mask
from vot.region import RegionType
from sam2.build_sam import build_sam2_video_predictor
from collections import OrderedDict
import random
import os
from utils.utils import keep_largest_component, determine_tracker

with open("./dam4sam_config.yaml") as f:
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
        Constructor for the SAM 2.1++ tracking wrapper.

        Args:
        - tracker_name (str): Name of the tracker to use. Options are:
            - "sam21pp-L": SAM 2.1++ Hiera Large
            - "sam21pp-B": SAM 2.1++ Hiera Base+
            - "sam21pp-S": SAM 2.1++ Hiera Small
            - "sam21pp-T": SAM 2.1++ Hiera Tiny
            - "sam2pp-L": SAM 2++ Hiera Large
            - "sam2pp-B": SAM 2++ Hiera Base+
            - "sam2pp-S": SAM 2++ Hiera Small
            - "sam2pp-T": SAM 2++ Hier Tiny
        """
        self.checkpoint, self.model_cfg = determine_tracker(tracker_name)

        # Image preprocessing parameters
        self.input_image_size = 1024       
        self.img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        self.img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]

    def _prepare_image(self, img_pil):
        """
        Function to prepare an image for input to the model.

        Args:
        - img_pil (PIL Image): Input image.

        Returns:
        - img (torch tensor): Preprocessed image tensor.
        """
        img_np = np.array(img_pil.convert("RGB").resize((self.input_image_size, self.input_image_size)))
        img = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        img -= self.img_mean
        img /= self.img_std
        return img

    @torch.inference_mode()
    def init_state_tw(
        self,
    ):
        """Initialize an inference state."""
        compute_device = "cuda:0" 
        inference_state = {}
        inference_state["images"] = None #Images are added later, frame-by-frame
        inference_state["num_frames"] = 0 #Number of frames is updated later, frame-by-frame
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        # this is useful when the video is very long and the GPU memory is limited
        inference_state["offload_video_to_cpu"] = True
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = True
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = None #Adding this later, in tracker.initialize
        inference_state["video_width"] =  None #Adding this later, in tracker.initialize
        inference_state["device"] = compute_device
        inference_state["storage_device"] = torch.device("cpu")
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        inference_state["adds_in_drm_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # A storage to hold the model's tracking results and states on each frame
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        
        return inference_state

    @torch.inference_mode()
    def initialize(self, image, init_mask):
        """
        Initialize the tracker with the first frame and mask.
        Function builds the SAM 2.1++ tracker and initializes it with the first frame and mask.

        Args:
        - image (PIL Image): First frame of the video.
        - init_mask (numpy array): Binary mask for the initialization
        
        Returns:
        - out_dict (dict): Dictionary containing the mask for the initialization frame.
        """
        if type(init_mask) is list:
            init_mask = init_mask[0]
        self.predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint, device="cuda:0")
        self.frame_index = 0 # Current frame index, updated frame-by-frame
        self.object_sizes = [] # List to store object sizes (number of pixels) 
        self.last_added = -1 # Frame index of the last added frame into DRM memory
        
        self.img_width = image.width # Original image width
        self.img_height = image.height # Original image height
        self.inference_state = self.init_state_tw()
        self.inference_state["images"] = image
        video_width, video_height = image.size 
        self.inference_state["video_height"] = video_height
        self.inference_state["video_width"] =  video_width
        prepared_img = self._prepare_image(image).cpu()
        self.inference_state["images"] = {0 : prepared_img}
        self.inference_state["num_frames"] = 1
        self.predictor.reset_state(self.inference_state)

        # Initialize the tracker with the first frame and mask
        _, _, out_mask_logits = self.predictor.add_new_mask(
            inference_state=self.inference_state,
            frame_idx=0,
            obj_id=0,
            mask=init_mask,
        )   
        m = (out_mask_logits[0, 0] > 0).float().cpu().numpy().astype(np.uint8)

        out_dict = {'pred_mask': m}
        return out_dict

    @torch.inference_mode()
    def track(self, image, init=False):
        """
        Function to track the object in the next frame.

        Args:
        - image (PIL Image): Next frame of the video.
        - init (bool): Whether the current frame is the initialization frame.

        Returns:
        - out_dict (dict): Dictionary containing the predicted mask for the current frame.
        """
        # Prepare the image for input to the model
        prepared_img = self._prepare_image(image).unsqueeze(0).cpu()
        if not init:
            self.frame_index += 1
            self.inference_state["num_frames"] += 1
        self.inference_state["images"][self.frame_index] = prepared_img

        # Propagate the tracking to the next frame
        # return_all_masks=True returns all predicted (chosen and alternative) masks and corresponding IoUs
        for out in self.predictor.propagate_in_video(self.inference_state, start_frame_idx=self.frame_index, max_frame_num_to_track=0, return_all_masks=True):
            if len(out) == 3:
                # There are 3 outputs when the tracking is done on the initialization frame
                out_frame_idx, _, out_mask_logits = out
                m = (out_mask_logits[0][0] > 0.0).float().cpu().numpy().astype(np.uint8)
            else:
                # There are 4 outputs when the tracking is done on a non-initialization frame
                # alternative_masks_ious is a tuple containing chosen and alternative masks and corresponding predicted IoUs
                out_frame_idx, _, out_mask_logits, alternative_masks_ious = out
                m = (out_mask_logits[0][0] > 0.0).float().cpu().numpy().astype(np.uint8)

                alternative_masks, out_all_ious = alternative_masks_ious # Extract all predicted masks (chosen and alternatives) and IoUs
                m_idx = np.argmax(out_all_ious) # Index of the chosen predicted mask
                m_iou = out_all_ious[m_idx] # Predicted IoU of the chosen predicted mask
                # Delete the chosen predicted mask from the list of all predicted masks, leading to only alternative masks
                alternative_masks = [mask for i, mask in enumerate(alternative_masks) if i != m_idx]

                # Determine if the object ratio between the current frame and the previous frames is within a certain range
                n_pixels = (m == 1).sum() 
                self.object_sizes.append(n_pixels)
                if len(self.object_sizes) > 1 and n_pixels >= 1:
                    obj_sizes_ratio = n_pixels / np.median([
                        size for size in self.object_sizes[-300:] if size >= 1
                    ][-10:])
                else:
                    obj_sizes_ratio = -1

                # The first condition checks if:
                #  - the chosen predicted mask has a high predicted IoU, 
                #  - the object size ratio is within a +- 20% range compared to the previous frames, 
                #  - the target is present in the current frame,
                #  - the last added frame to DRM is more than 5 frames ago or no frame has been added yet
                if m_iou > 0.8 and obj_sizes_ratio >= 0.8 and obj_sizes_ratio <= 1.2 and n_pixels >= 1 and (self.frame_index - self.last_added > 5 or self.last_added == -1):
                    alternative_masks = [Mask((m_[0][0] > 0.0).cpu().numpy()).rasterize((0, 0, self.img_width - 1, self.img_height - 1)).astype(np.uint8) 
                                     for m_ in alternative_masks]

                    # Numpy array of the chosen mask and corresponding bounding box
                    chosen_mask_np = m.copy()
                    chosen_bbox = Mask(chosen_mask_np).convert(RegionType.RECTANGLE)

                    # Delete the parts of the alternative masks that overlap with the chosen mask
                    alternative_masks = [np.logical_and(m_, np.logical_not(chosen_mask_np)).astype(np.uint8) for m_ in alternative_masks]
                    # Keep only the largest connected component of the processed alternative masks
                    alternative_masks = [keep_largest_component(m_) for m_ in alternative_masks if np.sum(m_) >= 1]
                    if len(alternative_masks) > 0:
                        # Make the union of the chosen mask and the processed alternative masks (corresponding to the largest connected component)
                        alternative_masks = [np.logical_or(m_, chosen_mask_np).astype(np.uint8) for m_ in alternative_masks]
                        # Convert the processed alternative masks to bounding boxes to calculate the IoUs bounding box-wise
                        alternative_bboxes = [Mask(m_).convert(RegionType.RECTANGLE) for m_ in alternative_masks]
                        # Calculate the IoUs between the chosen bounding box and the processed alternative bounding boxes
                        ious = [calculate_overlaps([chosen_bbox], [bbox])[0] for bbox in alternative_bboxes]
                        
                        # The second condition checks if within the calculated IoUs, there is at least one IoU that is less than 0.7
                        # That would mean that there are significant differences between the chosen mask and the processed alternative masks, 
                        # leading to possible detections of distractors within alternative masks.
                        if np.min(np.array(ious)) <= 0.7:
                            self.last_added = self.frame_index # Update the last added frame index

                            self.predictor.add_to_drm(
                                inference_state=self.inference_state,
                                frame_idx=out_frame_idx,
                                obj_id=0,
                            )

            # Return the predicted mask for the current frame
            out_dict = {'pred_mask': m}
            return out_dict