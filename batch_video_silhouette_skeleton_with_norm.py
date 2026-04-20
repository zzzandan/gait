# -*- coding: utf-8 -*-
"""
批量处理一个目录中的视频，按“每个视频单独一个文件夹”输出：
1. 提取视频帧
2. 使用 YOLO-seg 生成人体剪影图
3. 对裁剪后的剪影图做标准化（默认 64x44，底部对齐，水平居中）
4. 使用 YOLO 检测 + YOLO Pose 生成骨架图 / 骨架可视化 / 关键点 JSON

输出目录示例：
output_root/
└── walk1/
    ├── frames/
    │   ├── frame_000000.jpg
    │   └── ...
    ├── silhouette/
    │   ├── full_mask/
    │   ├── crop_mask/
    │   ├── crop_mask_norm/
    │   ├── crop_rgb/
    │   └── debug/
    └── skeleton/
        ├── skeleton_npy/
        ├── skeleton_vis/
        ├── crop_rgb/
        ├── keypoints_json/
        └── debug/

依赖：
pip install opencv-python numpy ultralytics
"""

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError("缺少 ultralytics，请先安装：pip install ultralytics") from e


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".mpeg", ".mpg", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================================================
# 一、通用工具
# =========================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def natural_sort_key(path: Path):
    return path.name.lower()


def find_videos(input_dir: Path) -> List[Path]:
    return sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS],
        key=natural_sort_key
    )


def box_center(box: Tuple[int, int, int, int, float]) -> Tuple[float, float]:
    x1, y1, x2, y2, _ = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def pad_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
    pad_ratio_x: float,
    pad_ratio_y: float
) -> Tuple[int, int, int, int]:
    bw = x2 - x1
    bh = y2 - y1

    pad_x = bw * pad_ratio_x
    pad_y = bh * pad_ratio_y

    nx1 = max(0, int(round(x1 - pad_x)))
    ny1 = max(0, int(round(y1 - pad_y)))
    nx2 = min(img_w, int(round(x2 + pad_x)))
    ny2 = min(img_h, int(round(y2 + pad_y)))

    return nx1, ny1, nx2, ny2


def extract_frames_from_video(
    video_path: Path,
    frames_dir: Path,
    frame_step: int = 1,
    jpg_quality: int = 95,
) -> int:
    ensure_dir(frames_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[提帧] 错误：无法打开视频 {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[提帧] {video_path.name} -> 帧率={fps:.2f}，总帧数={total_frames}")

    frame_idx = 0
    saved_count = 0
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            save_path = frames_dir / f"frame_{saved_count:06d}.jpg"
            cv2.imwrite(str(save_path), frame, encode_params)
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"[提帧] 完成：{video_path.name}，保存 {saved_count} 张到 {frames_dir}")
    return saved_count


# =========================================================
# 二、剪影生成与标准化
# =========================================================

def keep_largest_component(mask: np.ndarray, min_area: int = 0) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask)

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_area = stats[largest_label, cv2.CC_STAT_AREA]
    if largest_area < min_area:
        return np.zeros_like(mask)

    out = np.zeros_like(mask)
    out[labels == largest_label] = 255
    return out


def resize_mask_if_needed(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = mask.shape[:2]
    if h == target_h and w == target_w:
        return mask
    return cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def find_foreground_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = xs.min()
    x2 = xs.max() + 1
    y1 = ys.min()
    y2 = ys.max() + 1
    return x1, y1, x2, y2


def normalize_one_silhouette(
    mask: np.ndarray,
    out_h: int = 64,
    out_w: int = 44,
    pad: int = 2
) -> np.ndarray:
    bbox = find_foreground_bbox(mask)
    canvas = np.zeros((out_h, out_w), dtype=np.uint8)

    if bbox is None:
        return canvas

    x1, y1, x2, y2 = bbox
    person = mask[y1:y2, x1:x2]

    ph, pw = person.shape[:2]
    if ph == 0 or pw == 0:
        return canvas

    avail_h = out_h - 2 * pad
    avail_w = out_w - 2 * pad
    if avail_h <= 0 or avail_w <= 0:
        raise ValueError("输出尺寸太小或 pad 太大。")

    scale_h = avail_h / ph
    scale_w = avail_w / pw
    scale = min(scale_h, scale_w)

    new_h = max(1, int(round(ph * scale)))
    new_w = max(1, int(round(pw * scale)))

    resized = cv2.resize(person, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    start_x = (out_w - new_w) // 2
    start_y = out_h - pad - new_h

    start_x = max(0, start_x)
    start_y = max(0, start_y)

    end_x = min(out_w, start_x + new_w)
    end_y = min(out_h, start_y + new_h)

    canvas[start_y:end_y, start_x:end_x] = resized[:end_y - start_y, :end_x - start_x]
    canvas = (canvas > 0).astype(np.uint8) * 255
    return canvas


class SilhouetteGenerator:
    def __init__(
        self,
        model_path: str,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        person_class_id: int = 0,
        min_component_area: int = 100,
        open_kernel_size: int = 3,
        close_kernel_size: int = 5,
        pad_ratio_x: float = 0.05,
        pad_ratio_y: float = 0.08,
        use_temporal_tracking: bool = False,
        save_crop_mask: bool = True,
        save_crop_rgb: bool = True,
        save_debug: bool = True,
        norm_h: int = 64,
        norm_w: int = 44,
        norm_pad: int = 2,
    ):
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.person_class_id = person_class_id
        self.min_component_area = min_component_area
        self.open_kernel_size = open_kernel_size
        self.close_kernel_size = close_kernel_size
        self.pad_ratio_x = pad_ratio_x
        self.pad_ratio_y = pad_ratio_y
        self.use_temporal_tracking = use_temporal_tracking
        self.save_crop_mask = save_crop_mask
        self.save_crop_rgb = save_crop_rgb
        self.save_debug = save_debug
        self.norm_h = norm_h
        self.norm_w = norm_w
        self.norm_pad = norm_pad

    def postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        kernel_open = np.ones((self.open_kernel_size, self.open_kernel_size), np.uint8)
        kernel_close = np.ones((self.close_kernel_size, self.close_kernel_size), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = keep_largest_component(mask, min_area=self.min_component_area)
        return mask

    def choose_main_person(self, result, img_h: int, img_w: int, prev_center=None):
        if result.boxes is None or result.masks is None:
            return None

        boxes = result.boxes
        masks = result.masks
        if len(boxes) == 0 or masks is None or masks.data is None:
            return None

        candidates = []
        mask_data = masks.data.cpu().numpy()

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())

            if cls_id != self.person_class_id or conf < self.conf_thres:
                continue

            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
            x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])

            mask = (mask_data[i] > 0.5).astype(np.uint8) * 255
            mask = resize_mask_if_needed(mask, img_h, img_w)
            mask = self.postprocess_mask(mask)

            area = int((mask > 0).sum())
            if area < self.min_component_area:
                continue

            px1, py1, px2, py2 = pad_box(
                x1, y1, x2, y2, img_w, img_h, self.pad_ratio_x, self.pad_ratio_y
            )

            candidates.append({
                "box": (px1, py1, px2, py2, conf),
                "mask": mask,
                "area": area,
            })

        if not candidates:
            return None

        if (not self.use_temporal_tracking) or (prev_center is None):
            return max(candidates, key=lambda x: x["area"])

        prev_cx, prev_cy = prev_center

        def score_fn(item):
            x1, y1, x2, y2, _ = item["box"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            return (cx - prev_cx) ** 2 + (cy - prev_cy) ** 2

        return min(candidates, key=score_fn)

    @staticmethod
    def save_debug_image(original_bgr: np.ndarray, full_mask: np.ndarray, box, save_path: Path) -> None:
        x1, y1, x2, y2, conf = box

        debug = original_bgr.copy()
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            debug,
            f"person {conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        overlay = debug.copy()
        overlay[full_mask > 0] = (0, 0, 255)
        vis = cv2.addWeighted(debug, 0.65, overlay, 0.35, 0)
        cv2.imwrite(str(save_path), vis)

    def process_frames(self, frames_dir: Path, out_dir: Path) -> None:
        ensure_dir(out_dir)
        full_mask_dir = out_dir / "full_mask"
        crop_mask_dir = out_dir / "crop_mask"
        crop_mask_norm_dir = out_dir / "crop_mask_norm"
        crop_rgb_dir = out_dir / "crop_rgb"
        debug_dir = out_dir / "debug"

        ensure_dir(full_mask_dir)
        if self.save_crop_mask:
            ensure_dir(crop_mask_dir)
            ensure_dir(crop_mask_norm_dir)
        if self.save_crop_rgb:
            ensure_dir(crop_rgb_dir)
        if self.save_debug:
            ensure_dir(debug_dir)

        image_files = sorted(
            [p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS],
            key=natural_sort_key
        )

        if not image_files:
            print(f"[剪影] 没有找到图片：{frames_dir}")
            return

        print(f"[剪影] 开始处理：{frames_dir.name}，共 {len(image_files)} 张")
        prev_center = None

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[剪影] 跳过，无法读取：{img_path}")
                continue

            img_h, img_w = img.shape[:2]
            results = self.model.predict(
                source=img,
                conf=self.conf_thres,
                iou=self.iou_thres,
                verbose=False,
            )

            if not results:
                print(f"[剪影] 无推理结果：{img_path.name}")
                continue

            selected = self.choose_main_person(
                result=results[0],
                img_h=img_h,
                img_w=img_w,
                prev_center=prev_center,
            )

            if selected is None:
                print(f"[剪影] 未找到合格 person：{img_path.name}")
                continue

            box = selected["box"]
            full_mask = selected["mask"]
            prev_center = box_center(box)

            x1, y1, x2, y2, _ = box
            name = img_path.stem

            cv2.imwrite(str(full_mask_dir / f"{name}.png"), full_mask)

            crop_rgb = img[y1:y2, x1:x2]
            crop_mask = full_mask[y1:y2, x1:x2]

            if self.save_crop_rgb and crop_rgb.size > 0:
                cv2.imwrite(str(crop_rgb_dir / f"{name}.png"), crop_rgb)

            if self.save_crop_mask and crop_mask.size > 0:
                crop_mask = (crop_mask > 0).astype(np.uint8) * 255
                cv2.imwrite(str(crop_mask_dir / f"{name}.png"), crop_mask)

                norm_mask = normalize_one_silhouette(
                    mask=crop_mask,
                    out_h=self.norm_h,
                    out_w=self.norm_w,
                    pad=self.norm_pad
                )
                cv2.imwrite(str(crop_mask_norm_dir / f"{name}.png"), norm_mask)

            if self.save_debug:
                self.save_debug_image(img, full_mask, box, debug_dir / f"{name}_debug.jpg")

        print(f"[剪影] 完成：{frames_dir.name}")


# =========================================================
# 三、骨架图生成（YOLO 检测 + YOLO Pose）
# =========================================================

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

COCO_SKELETON_EDGES = [
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (0, 5), (0, 6)
]


def gaussian_heatmap(height, width, cx, cy, sigma=1.5):
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    heat = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
    return heat


def normalize_points_to_canvas(kpts_xy, crop_w, crop_h, out_w, out_h):
    mapped = np.zeros_like(kpts_xy, dtype=np.float32)

    if crop_w <= 1 or crop_h <= 1:
        return mapped

    mapped[:, 0] = kpts_xy[:, 0] * (out_w - 1) / (crop_w - 1)
    mapped[:, 1] = kpts_xy[:, 1] * (out_h - 1) / (crop_h - 1)
    return mapped


def render_skeleton_map(kpts_xy, kpts_conf, out_h=64, out_w=44,
                        sigma=1.5, limb_thickness=2, conf_thres=0.2):
    joint_map = np.zeros((out_h, out_w), dtype=np.float32)
    limb_map = np.zeros((out_h, out_w), dtype=np.float32)

    for i in range(len(kpts_xy)):
        if float(kpts_conf[i]) < conf_thres:
            continue
        x, y = kpts_xy[i]
        if not (0 <= x < out_w and 0 <= y < out_h):
            continue
        heat = gaussian_heatmap(out_h, out_w, x, y, sigma=sigma)
        joint_map = np.maximum(joint_map, heat)

    line_img = np.zeros((out_h, out_w), dtype=np.uint8)
    for a, b in COCO_SKELETON_EDGES:
        if float(kpts_conf[a]) < conf_thres or float(kpts_conf[b]) < conf_thres:
            continue
        xa, ya = kpts_xy[a]
        xb, yb = kpts_xy[b]
        if not (0 <= xa < out_w and 0 <= ya < out_h and 0 <= xb < out_w and 0 <= yb < out_h):
            continue

        cv2.line(
            line_img,
            (int(round(xa)), int(round(ya))),
            (int(round(xb)), int(round(yb))),
            color=255,
            thickness=limb_thickness
        )

    limb_map = line_img.astype(np.float32) / 255.0
    skeleton_map = np.stack([joint_map, limb_map], axis=0)

    vis = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    vis[:, :, 1] = np.clip(limb_map * 255, 0, 255).astype(np.uint8)

    for i in range(len(kpts_xy)):
        if float(kpts_conf[i]) < conf_thres:
            continue
        x, y = kpts_xy[i]
        cv2.circle(vis, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)

    return skeleton_map, vis


class SkeletonGenerator:
    def __init__(
        self,
        det_model_path: str = "yolov8n.pt",
        pose_model_path: str = "yolov8n-pose.pt",
        person_class_id: int = 0,
        det_conf: float = 0.25,
        det_iou: float = 0.45,
        pose_conf: float = 0.25,
        pose_iou: float = 0.45,
        out_h: int = 64,
        out_w: int = 44,
        pad_ratio_x: float = 0.08,
        pad_ratio_y: float = 0.10,
        kpt_conf_thres: float = 0.20,
        min_valid_kpts: int = 5,
        gaussian_sigma: float = 1.5,
        limb_thickness: int = 2,
        use_temporal_tracking: bool = False,
        save_crop_rgb: bool = True,
        save_keypoints_json: bool = True,
        save_vis: bool = True,
        save_debug: bool = True,
    ):
        self.det_model = YOLO(det_model_path)
        self.pose_model = YOLO(pose_model_path)
        self.person_class_id = person_class_id
        self.det_conf = det_conf
        self.det_iou = det_iou
        self.pose_conf = pose_conf
        self.pose_iou = pose_iou
        self.out_h = out_h
        self.out_w = out_w
        self.pad_ratio_x = pad_ratio_x
        self.pad_ratio_y = pad_ratio_y
        self.kpt_conf_thres = kpt_conf_thres
        self.min_valid_kpts = min_valid_kpts
        self.gaussian_sigma = gaussian_sigma
        self.limb_thickness = limb_thickness
        self.use_temporal_tracking = use_temporal_tracking
        self.save_crop_rgb = save_crop_rgb
        self.save_keypoints_json = save_keypoints_json
        self.save_vis = save_vis
        self.save_debug = save_debug

    @staticmethod
    def save_debug_image(img, box, save_path):
        x1, y1, x2, y2, conf = box
        debug = img.copy()
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            debug,
            f"person {conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        cv2.imwrite(str(save_path), debug)

    def choose_main_person_box(self, det_result, img_w, img_h, prev_center=None):
        if det_result.boxes is None or len(det_result.boxes) == 0:
            return None

        candidates = []

        for i in range(len(det_result.boxes)):
            cls_id = int(det_result.boxes.cls[i].item())
            conf = float(det_result.boxes.conf[i].item())

            if cls_id != self.person_class_id or conf < self.det_conf:
                continue

            x1, y1, x2, y2 = det_result.boxes.xyxy[i].cpu().numpy().tolist()
            x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])

            x1, y1, x2, y2 = pad_box(
                x1, y1, x2, y2, img_w, img_h, self.pad_ratio_x, self.pad_ratio_y
            )
            area = max(0, x2 - x1) * max(0, y2 - y1)

            candidates.append((x1, y1, x2, y2, conf, area))

        if len(candidates) == 0:
            return None

        if (not self.use_temporal_tracking) or (prev_center is None):
            best = max(candidates, key=lambda x: x[5])
            return best[:5]

        prev_cx, prev_cy = prev_center

        def score_fn(item):
            x1, y1, x2, y2, conf, area = item
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            return (cx - prev_cx) ** 2 + (cy - prev_cy) ** 2

        best = min(candidates, key=score_fn)
        return best[:5]

    @staticmethod
    def get_best_pose_keypoints(pose_result):
        if pose_result.keypoints is None or pose_result.boxes is None:
            return None, None

        if len(pose_result.boxes) == 0:
            return None, None

        confs = pose_result.boxes.conf.cpu().numpy()
        idx = int(np.argmax(confs))

        kpts_xy = pose_result.keypoints.xy[idx].cpu().numpy()

        if hasattr(pose_result.keypoints, "conf") and pose_result.keypoints.conf is not None:
            kpts_conf = pose_result.keypoints.conf[idx].cpu().numpy()
        else:
            data = pose_result.keypoints.data[idx].cpu().numpy()
            if data.shape[1] >= 3:
                kpts_conf = data[:, 2]
            else:
                kpts_conf = np.ones((kpts_xy.shape[0],), dtype=np.float32)

        return kpts_xy, kpts_conf

    def process_frames(self, frames_dir: Path, out_dir: Path) -> None:
        ensure_dir(out_dir)

        skeleton_npy_dir = out_dir / "skeleton_npy"
        skeleton_vis_dir = out_dir / "skeleton_vis"
        crop_rgb_dir = out_dir / "crop_rgb"
        keypoints_json_dir = out_dir / "keypoints_json"
        debug_dir = out_dir / "debug"

        ensure_dir(skeleton_npy_dir)
        if self.save_vis:
            ensure_dir(skeleton_vis_dir)
        if self.save_crop_rgb:
            ensure_dir(crop_rgb_dir)
        if self.save_keypoints_json:
            ensure_dir(keypoints_json_dir)
        if self.save_debug:
            ensure_dir(debug_dir)

        image_files = sorted(
            [p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS],
            key=natural_sort_key
        )

        if not image_files:
            print(f"[骨架] 没有找到图片：{frames_dir}")
            return

        print(f"[骨架] 开始处理：{frames_dir.name}，共 {len(image_files)} 张")
        prev_center = None

        for img_path in image_files:
            name = img_path.stem

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[骨架] 跳过，无法读取：{img_path}")
                continue

            img_h, img_w = img.shape[:2]

            det_results = self.det_model.predict(
                source=img,
                conf=self.det_conf,
                iou=self.det_iou,
                verbose=False
            )

            if len(det_results) == 0:
                print(f"[骨架] 无检测结果：{img_path.name}")
                continue

            det_result = det_results[0]
            box = self.choose_main_person_box(det_result, img_w=img_w, img_h=img_h, prev_center=prev_center)

            if box is None:
                print(f"[骨架] 未检测到合格人体框：{img_path.name}")
                continue

            x1, y1, x2, y2, conf = box
            prev_center = box_center(box)

            if self.save_debug:
                self.save_debug_image(img, box, debug_dir / f"{name}_det.jpg")

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                print(f"[骨架] 裁剪区域为空：{img_path.name}")
                continue

            if self.save_crop_rgb:
                cv2.imwrite(str(crop_rgb_dir / f"{name}.png"), crop)

            crop_h, crop_w = crop.shape[:2]

            pose_results = self.pose_model.predict(
                source=crop,
                conf=self.pose_conf,
                iou=self.pose_iou,
                verbose=False
            )

            if len(pose_results) == 0:
                skeleton_map = np.zeros((2, self.out_h, self.out_w), dtype=np.float32)
                np.save(str(skeleton_npy_dir / f"{name}.npy"), skeleton_map)
                continue

            pose_result = pose_results[0]
            kpts_xy, kpts_conf = self.get_best_pose_keypoints(pose_result)

            if kpts_xy is None or kpts_conf is None:
                skeleton_map = np.zeros((2, self.out_h, self.out_w), dtype=np.float32)
                np.save(str(skeleton_npy_dir / f"{name}.npy"), skeleton_map)
                continue

            valid_count = int((kpts_conf >= self.kpt_conf_thres).sum())

            if valid_count < self.min_valid_kpts:
                skeleton_map = np.zeros((2, self.out_h, self.out_w), dtype=np.float32)
                np.save(str(skeleton_npy_dir / f"{name}.npy"), skeleton_map)

                if self.save_vis:
                    empty_vis = np.zeros((self.out_h, self.out_w, 3), dtype=np.uint8)
                    cv2.imwrite(str(skeleton_vis_dir / f"{name}.png"), empty_vis)
                continue

            mapped_xy = normalize_points_to_canvas(
                kpts_xy=kpts_xy,
                crop_w=crop_w,
                crop_h=crop_h,
                out_w=self.out_w,
                out_h=self.out_h
            )

            skeleton_map, vis_img = render_skeleton_map(
                kpts_xy=mapped_xy,
                kpts_conf=kpts_conf,
                out_h=self.out_h,
                out_w=self.out_w,
                sigma=self.gaussian_sigma,
                limb_thickness=self.limb_thickness,
                conf_thres=self.kpt_conf_thres
            )

            np.save(str(skeleton_npy_dir / f"{name}.npy"), skeleton_map)

            if self.save_vis:
                cv2.imwrite(str(skeleton_vis_dir / f"{name}.png"), vis_img)

            if self.save_keypoints_json:
                info = {
                    "image_name": img_path.name,
                    "det_box_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                    "crop_hw": [int(crop_h), int(crop_w)],
                    "valid_keypoint_count": valid_count,
                    "keypoint_names": COCO_KEYPOINT_NAMES,
                    "keypoints_xy_crop": kpts_xy.tolist(),
                    "keypoints_conf": kpts_conf.tolist(),
                    "keypoints_xy_canvas": mapped_xy.tolist(),
                }
                with open(keypoints_json_dir / f"{name}.json", "w", encoding="utf-8") as f:
                    json.dump(info, f, ensure_ascii=False, indent=2)

        print(f"[骨架] 完成：{frames_dir.name}")


# =========================================================
# 四、批量主流程
# =========================================================

def process_all_videos(
    input_dir: Path,
    output_root: Path,
    seg_model_path: str,
    det_model_path: str,
    pose_model_path: str,
    frame_step: int,
    use_temporal_tracking: bool,
    seg_conf_thres: float,
    seg_iou_thres: float,
) -> None:
    ensure_dir(output_root)

    videos = find_videos(input_dir)
    if not videos:
        print(f"[退出] 输入目录中没有找到视频文件：{input_dir}")
        return

    silhouette_generator = SilhouetteGenerator(
        model_path=seg_model_path,
        conf_thres=seg_conf_thres,
        iou_thres=seg_iou_thres,
        use_temporal_tracking=use_temporal_tracking,
    )

    skeleton_generator = SkeletonGenerator(
        det_model_path=det_model_path,
        pose_model_path=pose_model_path,
        use_temporal_tracking=use_temporal_tracking,
    )

    for idx, video_path in enumerate(videos, start=1):
        video_name = video_path.stem
        video_out_dir = output_root / video_name

        frames_dir = video_out_dir / "frames"
        silhouette_dir = video_out_dir / "silhouette"
        skeleton_dir = video_out_dir / "skeleton"

        print("=" * 90)
        print(f"[{idx}/{len(videos)}] 开始处理视频：{video_path.name}")

        extract_frames_from_video(
            video_path=video_path,
            frames_dir=frames_dir,
            frame_step=frame_step
        )

        silhouette_generator.process_frames(frames_dir, silhouette_dir)
        skeleton_generator.process_frames(frames_dir, skeleton_dir)

        print(f"[完成] 视频处理结束：{video_path.name}")
        print("=" * 90)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量将视频转为帧图、剪影图、标准化剪影图、骨架图")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/zzzandan/desk/gait/gait/datasets",
        help="存放多个视频文件的输入目录"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/home/zzzandan/desk/gait/gait/gait/data/output",
        help="总输出目录，每个视频会单独建立一个子文件夹"
    )
    parser.add_argument(
        "--seg_model_path",
        type=str,
        default="yolov8s-seg.pt",
        help="YOLO-seg 模型路径，例如 yolov8s-seg.pt"
    )
    parser.add_argument(
        "--det_model_path",
        type=str,
        default="yolov8n.pt",
        help="人体检测模型路径，例如 yolov8n.pt"
    )
    parser.add_argument(
        "--pose_model_path",
        type=str,
        default="yolov8n-pose.pt",
        help="姿态模型路径，例如 yolov8n-pose.pt"
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=1,
        help="提帧间隔。1=每帧都保存，2=每隔1帧保存1帧"
    )
    parser.add_argument(
        "--seg_conf_thres",
        type=float,
        default=0.25,
        help="剪影分割置信度阈值"
    )
    parser.add_argument(
        "--seg_iou_thres",
        type=float,
        default=0.45,
        help="剪影分割 NMS IoU 阈值"
    )
    parser.add_argument(
        "--use_temporal_tracking",
        action="store_true",
        help="启用上一帧位置连续性约束"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")

    process_all_videos(
        input_dir=input_dir,
        output_root=output_root,
        seg_model_path=args.seg_model_path,
        det_model_path=args.det_model_path,
        pose_model_path=args.pose_model_path,
        frame_step=args.frame_step,
        use_temporal_tracking=args.use_temporal_tracking,
        seg_conf_thres=args.seg_conf_thres,
        seg_iou_thres=args.seg_iou_thres,
    )


if __name__ == "__main__":
    main()
