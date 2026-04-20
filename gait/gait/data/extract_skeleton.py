# -*- coding: utf-8 -*-
"""

流程：
1. 读取输入文件夹中的所有图片
2. 用 YOLO 检测 person
3. 选择主人体（默认选面积最大的，或用上一帧位置约束）
4. 裁剪人体区域
5. 在裁剪图上用 YOLO Pose 提取关键点
6. 将关键点渲染成 2 通道 Skeleton Map
7. 保存：
   - skeleton_npy/   : (2, H, W) 的骨架图
   - skeleton_vis/   : 可视化图
   - crop_rgb/       : 裁剪后人体图
   - keypoints_json/ : 关键点信息
   - debug/          : 原图上的检测框可视化

说明：
- Skeleton Map 采用 2 通道：
  channel 0: 关节点热图
  channel 1: 骨骼连线图
- 默认使用 COCO 17 点骨架拓扑
"""

import os
import json
import cv2
import numpy as np
from ultralytics import YOLO


# =========================================================
# 一、路径与参数配置
# =========================================================

# 输入图片文件夹
INPUT_DIR = "/home/zzzandan/desk/gait/gait/gait/data/output/frames"

# 输出总文件夹
OUTPUT_DIR = "/home/zzzandan/desk/gait/gait/gait/data/output/skeleton_maps"

# YOLO 检测模型
DET_MODEL_PATH = "yolov8n.pt"

# YOLO 姿态模型
POSE_MODEL_PATH = "yolov8n-pose.pt"

# person 类别 id（COCO 中 person 通常是 0）
PERSON_CLASS_ID = 0

# 检测阈值
DET_CONF = 0.25
DET_IOU = 0.45

# pose 阈值
POSE_CONF = 0.25
POSE_IOU = 0.45

# 输出 Skeleton Map 尺寸
OUT_H = 64
OUT_W = 44

# 对检测框加 padding，防止裁剪时切掉头和脚
PAD_RATIO_X = 0.08
PAD_RATIO_Y = 0.10

# 关键点置信度阈值
KPT_CONF_THRES = 0.20

# 至少保留多少个有效关键点，否则该帧记为空骨架图
MIN_VALID_KPTS = 5

# 渲染参数
GAUSSIAN_SIGMA = 1.5
LIMB_THICKNESS = 2

# 是否启用“上一帧位置连续性约束”
# False：每帧选面积最大的 person
# True：第一帧选最大框，后续优先选离上一帧最近的框
USE_TEMPORAL_TRACKING = False

# 是否保存中间结果
SAVE_CROP_RGB = True
SAVE_KEYPOINTS_JSON = True
SAVE_VIS = True
SAVE_DEBUG = True


# =========================================================
# 二、COCO 17 点定义
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


# =========================================================
# 三、基础工具函数
# =========================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def box_center(box):
    """
    输入 box=(x1, y1, x2, y2, conf)
    返回中心点 (cx, cy)
    """
    x1, y1, x2, y2, _ = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def pad_box(x1, y1, x2, y2, img_w, img_h):
    """
    对检测框做 padding，避免裁剪过紧。
    """
    bw = x2 - x1
    bh = y2 - y1

    pad_x = bw * PAD_RATIO_X
    pad_y = bh * PAD_RATIO_Y

    nx1 = max(0, int(round(x1 - pad_x)))
    ny1 = max(0, int(round(y1 - pad_y)))
    nx2 = min(img_w, int(round(x2 + pad_x)))
    ny2 = min(img_h, int(round(y2 + pad_y)))

    return nx1, ny1, nx2, ny2


def gaussian_heatmap(height, width, cx, cy, sigma=1.5):
    """
    在固定尺寸画布上生成单个关键点的高斯热图。
    """
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    heat = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
    return heat


def normalize_points_to_canvas(kpts_xy, crop_w, crop_h, out_w, out_h):
    """
    将裁剪图内的关键点坐标映射到 Skeleton Map 画布坐标。
    """
    mapped = np.zeros_like(kpts_xy, dtype=np.float32)

    if crop_w <= 1 or crop_h <= 1:
        return mapped

    mapped[:, 0] = kpts_xy[:, 0] * (out_w - 1) / (crop_w - 1)
    mapped[:, 1] = kpts_xy[:, 1] * (out_h - 1) / (crop_h - 1)

    return mapped


def render_skeleton_map(kpts_xy, kpts_conf, out_h=64, out_w=44,
                        sigma=1.5, limb_thickness=2, conf_thres=0.2):
    """
    将关键点和连线渲染成 2 通道 Skeleton Map。

    返回：
    - skeleton_map: shape (2, H, W), float32, 值域 0~1
    - vis_img:      shape (H, W, 3), uint8, 便于检查
    """
    joint_map = np.zeros((out_h, out_w), dtype=np.float32)
    limb_map = np.zeros((out_h, out_w), dtype=np.float32)

    # 关节点热图
    for i in range(len(kpts_xy)):
        if float(kpts_conf[i]) < conf_thres:
            continue
        x, y = kpts_xy[i]
        if not (0 <= x < out_w and 0 <= y < out_h):
            continue
        heat = gaussian_heatmap(out_h, out_w, x, y, sigma=sigma)
        joint_map = np.maximum(joint_map, heat)

    # 骨骼连线图
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

    # 可视化图
    vis = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    vis[:, :, 1] = np.clip(limb_map * 255, 0, 255).astype(np.uint8)

    for i in range(len(kpts_xy)):
        if float(kpts_conf[i]) < conf_thres:
            continue
        x, y = kpts_xy[i]
        cv2.circle(vis, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)

    return skeleton_map, vis


def save_debug_image(img, box, save_path):
    """
    保存原图上的检测框调试图。
    """
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
    cv2.imwrite(save_path, debug)


def choose_main_person_box(det_result, img_w, img_h, prev_center=None):
    """
    从检测结果里选择一个主人体框。

    规则：
    - 只保留 person 类
    - 不启用时序时：选面积最大的
    - 启用时序时：如果有上一帧中心，则优先选最近的
    """
    if det_result.boxes is None or len(det_result.boxes) == 0:
        return None

    candidates = []

    for i in range(len(det_result.boxes)):
        cls_id = int(det_result.boxes.cls[i].item())
        conf = float(det_result.boxes.conf[i].item())

        if cls_id != PERSON_CLASS_ID or conf < DET_CONF:
            continue

        x1, y1, x2, y2 = det_result.boxes.xyxy[i].cpu().numpy().tolist()
        x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])

        x1, y1, x2, y2 = pad_box(x1, y1, x2, y2, img_w, img_h)
        area = max(0, x2 - x1) * max(0, y2 - y1)

        candidates.append((x1, y1, x2, y2, conf, area))

    if len(candidates) == 0:
        return None

    # 不启用时序
    if (not USE_TEMPORAL_TRACKING) or (prev_center is None):
        best = max(candidates, key=lambda x: x[5])
        return best[:5]

    # 启用时序：选离上一帧中心最近的
    prev_cx, prev_cy = prev_center

    def score_fn(item):
        x1, y1, x2, y2, conf, area = item
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dist = (cx - prev_cx) ** 2 + (cy - prev_cy) ** 2
        return dist

    best = min(candidates, key=score_fn)
    return best[:5]


def get_best_pose_keypoints(pose_result):
    """
    从 pose 结果中取一个最靠谱的人体关键点结果。
    因为输入已经是裁剪后的人体区域，通常只会有一个人。

    返回：
    - kpts_xy:   [17, 2]
    - kpts_conf: [17]
    如果没有结果，返回 (None, None)
    """
    if pose_result.keypoints is None or pose_result.boxes is None:
        return None, None

    if len(pose_result.boxes) == 0:
        return None, None

    # 选框置信度最高的一个
    confs = pose_result.boxes.conf.cpu().numpy()
    idx = int(np.argmax(confs))

    # 坐标
    kpts_xy = pose_result.keypoints.xy[idx].cpu().numpy()  # [17,2]

    # 关键点置信度
    kpts_conf = None
    if hasattr(pose_result.keypoints, "conf") and pose_result.keypoints.conf is not None:
        kpts_conf = pose_result.keypoints.conf[idx].cpu().numpy()
    else:
        # 兼容写法：如果没有 conf 属性，就尝试从 data 里取
        data = pose_result.keypoints.data[idx].cpu().numpy()
        if data.shape[1] >= 3:
            kpts_conf = data[:, 2]
        else:
            kpts_conf = np.ones((kpts_xy.shape[0],), dtype=np.float32)

    return kpts_xy, kpts_conf


# =========================================================
# 四、主流程
# =========================================================

def process_images():
    ensure_dir(OUTPUT_DIR)

    skeleton_npy_dir = os.path.join(OUTPUT_DIR, "skeleton_npy")
    skeleton_vis_dir = os.path.join(OUTPUT_DIR, "skeleton_vis")
    crop_rgb_dir = os.path.join(OUTPUT_DIR, "crop_rgb")
    keypoints_json_dir = os.path.join(OUTPUT_DIR, "keypoints_json")
    debug_dir = os.path.join(OUTPUT_DIR, "debug")

    ensure_dir(skeleton_npy_dir)
    if SAVE_VIS:
        ensure_dir(skeleton_vis_dir)
    if SAVE_CROP_RGB:
        ensure_dir(crop_rgb_dir)
    if SAVE_KEYPOINTS_JSON:
        ensure_dir(keypoints_json_dir)
    if SAVE_DEBUG:
        ensure_dir(debug_dir)

    # 加载模型
    det_model = YOLO(DET_MODEL_PATH)
    pose_model = YOLO(POSE_MODEL_PATH)

    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_files = sorted([
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(valid_exts)
    ])

    if len(image_files) == 0:
        print(f"[退出] 输入目录没有图片: {INPUT_DIR}")
        return

    print(f"共找到 {len(image_files)} 张图片，开始处理...")

    prev_center = None

    for img_path in image_files:
        name = os.path.splitext(os.path.basename(img_path))[0]

        img = cv2.imread(img_path)
        if img is None:
            print(f"[跳过] 无法读取图片: {img_path}")
            continue

        img_h, img_w = img.shape[:2]

        # ---------- 第一步：检测人体 ----------
        det_results = det_model.predict(
            source=img,
            conf=DET_CONF,
            iou=DET_IOU,
            verbose=False
        )

        if len(det_results) == 0:
            print(f"[跳过] 无检测结果: {img_path}")
            continue

        det_result = det_results[0]
        box = choose_main_person_box(det_result, img_w=img_w, img_h=img_h, prev_center=prev_center)

        if box is None:
            print(f"[跳过] 未检测到合格的人体框: {img_path}")
            continue

        x1, y1, x2, y2, conf = box
        prev_center = box_center(box)

        # 保存原图上的检测框可视化
        if SAVE_DEBUG:
            save_debug_image(img, box, os.path.join(debug_dir, f"{name}_det.jpg"))

        # ---------- 第二步：裁剪人体区域 ----------
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            print(f"[跳过] 裁剪区域为空: {img_path}")
            continue

        if SAVE_CROP_RGB:
            cv2.imwrite(os.path.join(crop_rgb_dir, f"{name}.png"), crop)

        crop_h, crop_w = crop.shape[:2]

        # ---------- 第三步：在裁剪图上做 pose ----------
        pose_results = pose_model.predict(
            source=crop,
            conf=POSE_CONF,
            iou=POSE_IOU,
            verbose=False
        )

        if len(pose_results) == 0:
            print(f"[弱结果] pose 无结果: {img_path}")
            skeleton_map = np.zeros((2, OUT_H, OUT_W), dtype=np.float32)
            np.save(os.path.join(skeleton_npy_dir, f"{name}.npy"), skeleton_map)
            continue

        pose_result = pose_results[0]
        kpts_xy, kpts_conf = get_best_pose_keypoints(pose_result)

        if kpts_xy is None or kpts_conf is None:
            print(f"[弱结果] pose 未返回关键点: {img_path}")
            skeleton_map = np.zeros((2, OUT_H, OUT_W), dtype=np.float32)
            np.save(os.path.join(skeleton_npy_dir, f"{name}.npy"), skeleton_map)
            continue

        valid_count = int((kpts_conf >= KPT_CONF_THRES).sum())

        if valid_count < MIN_VALID_KPTS:
            print(f"[弱结果] 有效关键点过少 ({valid_count}): {img_path}")
            skeleton_map = np.zeros((2, OUT_H, OUT_W), dtype=np.float32)
            np.save(os.path.join(skeleton_npy_dir, f"{name}.npy"), skeleton_map)

            if SAVE_VIS:
                empty_vis = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
                cv2.imwrite(os.path.join(skeleton_vis_dir, f"{name}.png"), empty_vis)
            continue

        # ---------- 第四步：坐标归一化到固定画布 ----------
        mapped_xy = normalize_points_to_canvas(
            kpts_xy=kpts_xy,
            crop_w=crop_w,
            crop_h=crop_h,
            out_w=OUT_W,
            out_h=OUT_H
        )

        # ---------- 第五步：渲染 Skeleton Map ----------
        skeleton_map, vis_img = render_skeleton_map(
            kpts_xy=mapped_xy,
            kpts_conf=kpts_conf,
            out_h=OUT_H,
            out_w=OUT_W,
            sigma=GAUSSIAN_SIGMA,
            limb_thickness=LIMB_THICKNESS,
            conf_thres=KPT_CONF_THRES
        )

        # 保存骨架图
        np.save(os.path.join(skeleton_npy_dir, f"{name}.npy"), skeleton_map)

        # 保存可视化图
        if SAVE_VIS:
            cv2.imwrite(os.path.join(skeleton_vis_dir, f"{name}.png"), vis_img)

        # 保存关键点信息
        if SAVE_KEYPOINTS_JSON:
            info = {
                "image_name": os.path.basename(img_path),
                "det_box_xyxy": [x1, y1, x2, y2],
                "crop_hw": [crop_h, crop_w],
                "valid_keypoint_count": valid_count,
                "keypoint_names": COCO_KEYPOINT_NAMES,
                "keypoints_xy_crop": kpts_xy.tolist(),
                "keypoints_conf": kpts_conf.tolist(),
                "keypoints_xy_canvas": mapped_xy.tolist(),
            }
            with open(os.path.join(keypoints_json_dir, f"{name}.json"), "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)

        print(f"[完成] {name} -> Skeleton Map 已保存")

    print("全部处理完成。")


# =========================================================
# 五、程序入口
# =========================================================

if __name__ == "__main__":
    process_images()