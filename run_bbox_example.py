import os
import glob
import argparse
import shutil

import numpy as np
import cv2
from PIL import Image

from dam4sam_tracker import DAM4SAMTracker
from utils.visualization_utils import overlay_mask, overlay_rectangle


# =================================================================================
# 全新重构：一个支持混合输入的标注器
# =================================================================================
class InputSelector:
    """
    一个统一的输入选择器，支持混合标注：
    - 左键单击: 添加前景点 (绿色)
    - 右键单击: 添加背景点 (红色)
    - 左键拖拽: 绘制边界框 (蓝色)
    - 'c'键: 清除所有当前标注
    - 'Enter'键: 确认并继续
    """

    def __init__(self, window_name, image):
        self.window_name = window_name
        self.image = image.copy()
        self.display_image = image.copy()

        # 存储所有类型的提示
        self.prompts = {
            "pos_points": [],
            "neg_points": [],
            "box": None
        }

        self.start_point = None
        self.is_drawing = False

    def _mouse_callback(self, event, x, y, flags, param):
        # 左键按下，开始绘制或标记点
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.is_drawing = True

        # 鼠标移动，如果正在绘制，则实时显示边界框
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing:
                # 创建一个临时图像以显示拖拽过程，避免在原图上留下痕迹
                temp_image = self.display_image.copy()
                cv2.rectangle(temp_image, self.start_point, (x, y), (255, 0, 0), 2)
                cv2.imshow(self.window_name, temp_image)

        # 左键抬起，结束绘制或标记点
        elif event == cv2.EVENT_LBUTTONUP:
            end_point = (x, y)
            self.is_drawing = False

            # 如果起始点和结束点非常接近，视为一次点击
            if np.linalg.norm(np.array(self.start_point) - np.array(end_point)) < 5:
                self.prompts["pos_points"].append(end_point)
                print(f"添加前景点: {end_point}")
                cv2.circle(self.display_image, end_point, 5, (0, 255, 0), -1)
            # 否则，视为一次拖拽，形成边界框
            else:
                x1, y1 = self.start_point
                x2, y2 = end_point
                # 如果已有框，新的框会覆盖旧的
                self.prompts["box"] = (min(x1, x2), min(y1, y2), abs(x1 - x2), abs(y1 - y2))
                print(f"设置边界框: {self.prompts['box']}")
                # 清除旧框的痕迹，重新绘制所有标注
                self._redraw_prompts()

            cv2.imshow(self.window_name, self.display_image)

        # 右键单击，添加背景点
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.prompts["neg_points"].append((x, y))
            print(f"添加背景点: {(x, y)}")
            cv2.circle(self.display_image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(self.window_name, self.display_image)

    def _redraw_prompts(self):
        """重新绘制所有当前的提示（点和框），用于更新或清除"""
        self.display_image = self.image.copy()
        # 画框
        if self.prompts["box"]:
            x, y, w, h = self.prompts["box"]
            cv2.rectangle(self.display_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # 画前景点
        for pt in self.prompts["pos_points"]:
            cv2.circle(self.display_image, pt, 5, (0, 255, 0), -1)
        # 画背景点
        for pt in self.prompts["neg_points"]:
            cv2.circle(self.display_image, pt, 5, (0, 0, 255), -1)

    def select_input(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        print("\n--- 混合标注模式 ---")
        print("左键单击: 添加前景点 (绿色)")
        print("右键单击: 添加背景点 (红色)")
        print("左键拖拽: 绘制/替换边界框 (蓝色)")
        print("按 'c' 清除所有标注")
        print("按 'Enter' 确认并开始跟踪")
        print("--------------------\n")

        while True:
            cv2.imshow(self.window_name, self.display_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                print("清除所有标注。")
                self._reset_selection()
                self._redraw_prompts()

            elif key == 13:  # Enter 键
                break

        cv2.destroyWindow(self.window_name)

        # 检查是否至少有一个有效的提示
        if not self.prompts["pos_points"] and not self.prompts["neg_points"] and not self.prompts["box"]:
            return None

        return self.prompts

    def _reset_selection(self):
        self.prompts = {
            "pos_points": [],
            "neg_points": [],
            "box": None
        }
        self.start_point = None
        self.is_drawing = False


def run_sequence(dir_path, file_extension, output_dir):
    frames_dir = sorted(glob.glob(os.path.join(dir_path, '*.%s' % file_extension)))
    if not frames_dir:
        print('Error: 指定目录下没有图像文件。')
        return

    img0_bgr = cv2.imread(frames_dir[0])

    # 使用新的 InputSelector
    input_selector = InputSelector('混合标注界面', img0_bgr)
    init_prompts = input_selector.select_input()

    if not init_prompts:
        print('错误: 未提供任何有效的初始化标注 (点或框)。')
        return

    tracker = DAM4SAMTracker('sam21pp-L')

    if output_dir:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    print('开始分割序列帧...')
    for i, frame_path in enumerate(frames_dir):
        img = Image.open(frame_path).convert('RGB')
        img_vis = np.array(img)

        if i == 0:
            # 使用包含多种提示的字典来初始化 tracker
            outputs = tracker.initialize(img, init_prompts=init_prompts)

            # 可视化所有初始提示
            if not output_dir:
                if init_prompts.get("box"):
                    overlay_rectangle(img_vis, init_prompts["box"], color=(255, 0, 0), line_width=2)
                for point in init_prompts.get("pos_points", []):
                    cv2.circle(img_vis, point, 5, (0, 255, 0), -1)
                for point in init_prompts.get("neg_points", []):
                    cv2.circle(img_vis, point, 5, (0, 0, 255), -1)

            if not output_dir:
                window_name = 'win'
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                wait_ = 0
        else:
            outputs = tracker.track(img)

        pred_mask = outputs['pred_mask']

        if output_dir:
            frame_name = os.path.splitext(os.path.basename(frame_path))[0]
            output_path = os.path.join(output_dir, f'{frame_name}.png')
            cv2.imwrite(output_path, pred_mask * 255)
        else:
            overlay_mask(img_vis, pred_mask, (255, 255, 0), line_width=1, alpha=0.55)
            cv2.imshow(window_name, cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
            key_ = cv2.waitKey(wait_)

            if key_ == 27:
                exit(0)
            elif key_ == 32:
                wait_ = 0 if wait_ else 1

    print('分割完成。')


# main 函数保持不变
def main():
    parser = argparse.ArgumentParser(description='Run on a sequence of frames-dir.')
    parser.add_argument('--dir', type=str, required=True, help='Path to directory with frames-dir.')
    parser.add_argument('--ext', type=str, default='jpg', help='Image file extension.')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to the output directory.')

    args = parser.parse_args()
    run_sequence(args.dir, args.ext, args.output_dir)


if __name__ == "__main__":
    main()