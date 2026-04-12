# -*- coding: utf-8 -*-
"""
YOLO-seg 直接提取 person mask，生成二值剪影图

流程：
1. 读取输入文件夹中的所有图片
2. 用 YOLO-seg 做实例分割
3. 只保留 person 类
4. 默认选择“主人体”（面积最大，或结合上一帧位置连续性）
5. 输出：
   - full_mask: 原图坐标系下的整幅二值剪影图
   - crop_mask: 裁剪后的人体局部二值剪影图
   - debug: 调试可视化图（检测框 + mask 叠加）

适用场景：
- 视频逐帧图片
- 想提取步态识别所需的二值人体轮廓
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO


# =========================================================
# 一、路径与参数配置
# =========================================================

# 输入帧文件夹
INPUT_DIR = "/home/zzzandan/desk/gait/gait/gait/data/output/frames"

# 输出总文件夹
OUTPUT_DIR = "/home/zzzandan/desk/gait/gait/gait/data/output/silhouettes_yoloseg"

# YOLO-seg 模型
# 可选：
#   yolov8n-seg.pt  速度快
#   yolov8s-seg.pt  更准一点
MODEL_PATH = "yolov8n-seg.pt"

# person 类别 id（COCO 数据集里 person 通常是 0）
PERSON_CLASS_ID = 0

# 检测置信度阈值
CONF_THRES = 0.25

# NMS IoU 阈值
IOU_THRES = 0.45

# 是否启用上一帧位置连续性约束
# False：每帧选面积最大的 person
# True：第一帧选最大 person，后续优先选离上一帧最近的 person
USE_TEMPORAL_TRACKING = False

# 如果启用位置连续性约束，距离惩罚系数越小越偏向“离上一帧近”
# 当前简单实现不需要显式用这个参数，保留是为了后续扩展
TRACKING_DISTANCE_WEIGHT = 1.0

# 是否保存调试图
SAVE_DEBUG = True

# 是否保存裁剪后的小剪影图
SAVE_CROP_MASK = True

# 是否保存裁剪后的人体 RGB 图
SAVE_CROP_RGB = True

# 后处理：开运算核大小
OPEN_KERNEL_SIZE = 3

# 后处理：闭运算核大小
CLOSE_KERNEL_SIZE = 5

# 最小连通域面积，小于这个阈值的最大连通域会被视为无效
MIN_COMPONENT_AREA = 100

# 检测框 padding，防止裁剪时切掉头脚
PAD_RATIO_X = 0.05
PAD_RATIO_Y = 0.08


# =========================================================
# 二、工具函数
# =========================================================

def ensure_dir(path: str):
    """如果目录不存在，则创建。"""
    os.makedirs(path, exist_ok=True)


def keep_largest_component(mask: np.ndarray, min_area: int = 0) -> np.ndarray:
    """
    只保留最大连通域。

    为什么要这样做：
    - YOLO-seg 输出的 mask 有时会有零碎小块
    - 步态识别更希望得到“单个人体”的完整轮廓
    """
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


def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    """
    对 mask 做后处理：
    1. 开运算去小噪点
    2. 闭运算补小孔洞
    3. 保留最大连通域
    """
    kernel_open = np.ones((OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE), np.uint8)
    kernel_close = np.ones((CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = keep_largest_component(mask, min_area=MIN_COMPONENT_AREA)
    return mask


def box_center(box):
    """
    输入 box=(x1, y1, x2, y2, conf)，返回中心点(cx, cy)
    """
    x1, y1, x2, y2, _ = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def pad_box(x1, y1, x2, y2, img_w, img_h):
    """
    对检测框加一点 padding，避免裁剪太紧。
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


def resize_mask_if_needed(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    如果 YOLO 输出的 mask 尺寸和原图不同，则 resize 到原图大小。
    一定要用最近邻插值，避免二值边界被模糊。
    """
    h, w = mask.shape[:2]
    if h == target_h and w == target_w:
        return mask
    return cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def choose_main_person(result, img_h: int, img_w: int, prev_center=None):
    """
    从 YOLO-seg 结果中选择一个“主人体”。

    规则：
    - 只考虑 person 类，且置信度 >= CONF_THRES
    - 如果不启用时序约束，直接选面积最大的那个
    - 如果启用时序约束，第一帧选最大，后续选离上一帧最近的

    返回：
    {
        "box": (x1, y1, x2, y2, conf),
        "mask": full_mask_uint8  # 原图坐标系下，0/255
    }
    如果没有找到，返回 None
    """
    if result.boxes is None or result.masks is None:
        return None

    boxes = result.boxes
    masks = result.masks

    if len(boxes) == 0 or masks is None or masks.data is None:
        return None

    candidates = []

    # masks.data 与 boxes 顺序一一对应
    mask_data = masks.data.cpu().numpy()  # [N, H, W], 值一般为0/1
    orig_h, orig_w = img_h, img_w

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())

        if cls_id != PERSON_CLASS_ID or conf < CONF_THRES:
            continue

        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
        x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])

        # 取对应 mask
        mask = mask_data[i]
        mask = (mask > 0.5).astype(np.uint8) * 255
        mask = resize_mask_if_needed(mask, orig_h, orig_w)

        # 后处理
        mask = postprocess_mask(mask)

        # 面积
        area = int((mask > 0).sum())
        if area < MIN_COMPONENT_AREA:
            continue

        # 加 padding 后的框
        px1, py1, px2, py2 = pad_box(x1, y1, x2, y2, img_w=img_w, img_h=img_h)

        candidates.append({
            "box": (px1, py1, px2, py2, conf),
            "mask": mask,
            "area": area
        })

    if len(candidates) == 0:
        return None

    # 不启用时序约束：直接选面积最大的
    if (not USE_TEMPORAL_TRACKING) or (prev_center is None):
        best = max(candidates, key=lambda x: x["area"])
        return best

    # 启用时序约束：选距离上一帧中心最近的
    prev_cx, prev_cy = prev_center

    def score_fn(item):
        x1, y1, x2, y2, _ = item["box"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dist = (cx - prev_cx) ** 2 + (cy - prev_cy) ** 2
        return dist

    best = min(candidates, key=score_fn)
    return best


def save_debug_image(original_bgr: np.ndarray, full_mask: np.ndarray, box, save_path: str):
    """
    保存调试图：
    - 画 person 框
    - 用红色叠加最终 mask
    """
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
        2
    )

    overlay = debug.copy()
    overlay[full_mask > 0] = (0, 0, 255)

    vis = cv2.addWeighted(debug, 0.65, overlay, 0.35, 0)
    cv2.imwrite(save_path, vis)


# =========================================================
# 三、主流程
# =========================================================

def process_images():
    """
    主流程：
    1. 加载 YOLO-seg 模型
    2. 遍历所有图片
    3. 对每张图：
       - 做实例分割
       - 选主人体
       - 保存 full mask / crop mask / crop rgb / debug
    """
    ensure_dir(OUTPUT_DIR)

    full_mask_dir = os.path.join(OUTPUT_DIR, "full_mask")
    crop_mask_dir = os.path.join(OUTPUT_DIR, "crop_mask")
    crop_rgb_dir = os.path.join(OUTPUT_DIR, "crop_rgb")
    debug_dir = os.path.join(OUTPUT_DIR, "debug")

    ensure_dir(full_mask_dir)
    if SAVE_CROP_MASK:
        ensure_dir(crop_mask_dir)
    if SAVE_CROP_RGB:
        ensure_dir(crop_rgb_dir)
    if SAVE_DEBUG:
        ensure_dir(debug_dir)

    # 加载模型
    model = YOLO(MODEL_PATH)

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

        # 做实例分割
        results = model.predict(
            source=img,
            conf=CONF_THRES,
            iou=IOU_THRES,
            verbose=False
        )

        if len(results) == 0:
            print(f"[跳过] 无推理结果: {img_path}")
            continue

        selected = choose_main_person(
            results[0],
            img_h=img_h,
            img_w=img_w,
            prev_center=prev_center
        )

        if selected is None:
            print(f"[跳过] 未找到合格的 person: {img_path}")
            continue

        box = selected["box"]
        full_mask = selected["mask"]

        x1, y1, x2, y2, conf = box
        prev_center = box_center(box)

        # 保存整图坐标系下的二值 mask
        full_mask_path = os.path.join(full_mask_dir, f"{name}.png")
        cv2.imwrite(full_mask_path, full_mask)

        # 裁剪局部 RGB 和局部 mask
        crop_rgb = img[y1:y2, x1:x2]
        crop_mask = full_mask[y1:y2, x1:x2]

        if SAVE_CROP_RGB and crop_rgb.size > 0:
            crop_rgb_path = os.path.join(crop_rgb_dir, f"{name}.png")
            cv2.imwrite(crop_rgb_path, crop_rgb)

        if SAVE_CROP_MASK and crop_mask.size > 0:
            crop_mask_path = os.path.join(crop_mask_dir, f"{name}.png")
            cv2.imwrite(crop_mask_path, crop_mask)

        # 保存 debug 图
        if SAVE_DEBUG:
            debug_path = os.path.join(debug_dir, f"{name}_debug.jpg")
            save_debug_image(img, full_mask, box, debug_path)

        print(f"[完成] {os.path.basename(img_path)} -> {full_mask_path}")

    print("全部处理完成。")


# =========================================================
# 四、程序入口
# =========================================================

if __name__ == "__main__":
    process_images()