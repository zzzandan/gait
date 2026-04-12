# -*- coding: utf-8 -*-
"""
YOLO 检测人 → 裁剪人体区域 → rembg 做人体分割 → 生成二值剪影图

适用场景：
- 原始输入是一系列视频帧图片
- 希望从复杂背景中先检测出人，再对“人体局部区域”做分割
- 用于步态识别前处理，得到人体二值剪影图

输出内容：
1. full_mask/      -> 放回原图坐标系的整幅二值剪影图
2. crop_mask/      -> 裁剪后人体区域内的二值剪影图
3. debug/          -> 调试图（检测框 + 分割结果可视化）
4. crop_rgb/       -> 裁剪后的人体RGB图（可选，方便检查）
"""

import os
import io
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from rembg import remove, new_session


# =========================================================
# 一、路径与参数配置区
# =========================================================

# 输入帧文件夹
INPUT_DIR = "/home/zzzandan/desk/gait/gait/gait/data/output/frames"

# 输出总文件夹
OUTPUT_DIR = "/home/zzzandan/desk/gait/gait/gait/data/output/silhouettes_yolo"

# YOLO 检测模型
# 可选: yolov8n.pt / yolov8s.pt / yolov8m.pt
# n 最快，m 更准一些，但更慢
YOLO_MODEL_PATH = "yolov8n.pt"

# YOLO person 类别 id（COCO 数据集里 person 通常是 0）
PERSON_CLASS_ID = 0

# 检测置信度阈值
# 低了会引入误检，高了可能漏检
YOLO_CONF_THRES = 0.25

# NMS IoU 阈值（通常默认就可以）
YOLO_IOU_THRES = 0.45

# 对检测框做 padding，防止裁剪时切掉手脚
# 建议 0.05 ~ 0.12 之间调
PAD_RATIO_X = 0.08
PAD_RATIO_Y = 0.10

# rembg 使用的人体分割模型
# u2net_human_seg 比默认模型更适合人体
REMBG_MODEL_NAME = "u2net_human_seg"

# rembg 输出 alpha 通道阈值
# 越大越严格，前景更干净，但也更容易丢细节
# 建议尝试 160 / 180 / 200
ALPHA_THRESH = 180

# 形态学参数
OPEN_KERNEL_SIZE = 3
CLOSE_KERNEL_SIZE = 5

# 连通域面积阈值
# 用来过滤很小的噪点
MIN_COMPONENT_AREA = 100

# 是否保存调试图
SAVE_DEBUG = True

# 是否保存裁剪后 RGB 小图
SAVE_CROP_RGB = True

# 是否启用“上一帧位置连续性约束”
# False：每帧选面积最大的 person
# True：第一帧选最大 person，后续优先选离上一帧最近的人
USE_TEMPORAL_TRACKING = False


# =========================================================
# 二、工具函数
# =========================================================

def ensure_dir(path: str):
    """如果文件夹不存在，则创建。"""
    os.makedirs(path, exist_ok=True)


def keep_largest_component(mask: np.ndarray, min_area: int = 0) -> np.ndarray:
    """
    只保留最大连通域。

    为什么要这么做：
    - rembg 在人体局部区域里仍可能分出多个碎块
    - 例如衣服边缘噪点、地面亮块、旁边物体等
    - 步态识别更需要“完整人体轮廓”，所以通常保留最大那一块更合理

    参数：
    - mask: 二值图，前景为255，背景为0
    - min_area: 最大连通域面积如果小于这个阈值，则认为无效，返回全黑图
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # 没有前景
    if num_labels <= 1:
        return np.zeros_like(mask)

    # 跳过背景（第0类），找到面积最大的前景连通域
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_area = stats[largest_label, cv2.CC_STAT_AREA]

    # 面积太小，直接判为无效
    if largest_area < min_area:
        return np.zeros_like(mask)

    out = np.zeros_like(mask)
    out[labels == largest_label] = 255
    return out


def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    """
    对二值 mask 做后处理：
    1. 开运算去小噪点
    2. 闭运算补小孔洞
    3. 只保留最大连通域

    这样处理后的人体轮廓通常会更干净。
    """
    kernel_open = np.ones((OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE), np.uint8)
    kernel_close = np.ones((CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE), np.uint8)

    # 去掉小的孤立噪点
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    # 填补人体内部的小空洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # 只保留最大人体区域
    mask = keep_largest_component(mask, min_area=MIN_COMPONENT_AREA)

    return mask


def box_center(box):
    """
    输入 box=(x1,y1,x2,y2,conf)，返回中心点(cx, cy)
    """
    x1, y1, x2, y2, _ = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return cx, cy


def pad_box(x1, y1, x2, y2, img_w, img_h, pad_ratio_x=0.08, pad_ratio_y=0.10):
    """
    对检测框加 padding，避免切掉人体边缘。
    """
    bw = x2 - x1
    bh = y2 - y1

    pad_x = bw * pad_ratio_x
    pad_y = bh * pad_ratio_y

    nx1 = max(0, int(round(x1 - pad_x)))
    ny1 = max(0, int(round(y1 - pad_y)))
    nx2 = min(img_w, int(round(x2 + pad_x)))
    ny2 = min(img_h, int(round(y2 + pad_y)))

    return nx1, ny1, nx2, ny2


def choose_person_box(result, img_w, img_h, prev_center=None):
    """
    从 YOLO 检测结果中选一个“主人体框”。

    两种策略：
    1. 不启用时序跟踪：
       - 直接从所有 person 框中选择面积最大的
    2. 启用时序跟踪：
       - 如果没有上一帧中心，第一帧仍然选最大框
       - 如果有上一帧中心，则优先选“与上一帧中心距离最近”的框

    为什么要这样：
    - 多人场景下，只按面积最大可能会跳人
    - 加一个最简单的时序连续性约束，会稳很多

    返回：
    - best_box = (x1, y1, x2, y2, conf)
    - 如果没有 person，返回 None
    """
    if result.boxes is None or len(result.boxes) == 0:
        return None

    person_boxes = []

    for box in result.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        if cls_id != PERSON_CLASS_ID or conf < YOLO_CONF_THRES:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])

        # 加 padding
        px1, py1, px2, py2 = pad_box(
            x1, y1, x2, y2,
            img_w=img_w,
            img_h=img_h,
            pad_ratio_x=PAD_RATIO_X,
            pad_ratio_y=PAD_RATIO_Y
        )

        area = max(0, px2 - px1) * max(0, py2 - py1)
        person_boxes.append((px1, py1, px2, py2, conf, area))

    if len(person_boxes) == 0:
        return None

    # 不启用连续性约束：选面积最大的
    if (not USE_TEMPORAL_TRACKING) or (prev_center is None):
        best = max(person_boxes, key=lambda x: x[5])
        return best[:5]

    # 启用连续性约束：选距离上一帧中心最近的
    prev_cx, prev_cy = prev_center

    def score_fn(b):
        x1, y1, x2, y2, conf, area = b
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dist = (cx - prev_cx) ** 2 + (cy - prev_cy) ** 2
        return dist

    best = min(person_boxes, key=score_fn)
    return best[:5]


def rembg_segment_human(crop_bgr: np.ndarray, session) -> np.ndarray:
    """
    对裁剪后的人体 RGB 图做 rembg 分割，返回二值剪影图。

    核心注意点：
    - rembg 输出的是 RGBA 图，不要直接转灰度阈值
    - 一定要取 alpha 通道
    - 再对 alpha 通道做阈值化

    返回：
    - mask: uint8 二值图，前景=255，背景=0
    """
    # OpenCV 读入的是 BGR，先转 RGB
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    # 转成 PIL 图，交给 rembg
    pil_img = Image.fromarray(crop_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    input_bytes = buf.getvalue()

    # rembg 去背景
    output_bytes = remove(input_bytes, session=session)

    # 读取结果，注意这里要保留 alpha 通道
    rgba = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

    # 只取 alpha 通道，alpha 越大说明越像前景
    alpha = np.array(rgba.getchannel("A"), dtype=np.uint8)

    # 二值化
    _, mask = cv2.threshold(alpha, ALPHA_THRESH, 255, cv2.THRESH_BINARY)

    # 后处理
    mask = postprocess_mask(mask)

    return mask


def save_debug_image(
    original_bgr: np.ndarray,
    full_mask: np.ndarray,
    box,
    save_path: str
):
    """
    保存调试图：
    - 原图上画出检测框
    - 用红色叠加显示分割结果
    """
    debug = original_bgr.copy()

    x1, y1, x2, y2, conf = box
    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        debug,
        f"person {conf:.2f}",
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    overlay = debug.copy()
    overlay[full_mask > 0] = (0, 0, 255)

    vis = cv2.addWeighted(debug, 0.65, overlay, 0.35, 0)
    cv2.imwrite(save_path, vis)


# =========================================================
# 三、主处理函数
# =========================================================

def process_images():
    """
    主流程：
    1. 加载 YOLO 检测模型
    2. 加载 rembg 人体分割模型
    3. 遍历输入文件夹中的图片
    4. 对每张图：
       - 检测 person
       - 选主人体框
       - 裁剪
       - rembg 分割
       - 保存 full mask / crop mask / debug 图
    """
    ensure_dir(OUTPUT_DIR)

    full_mask_dir = os.path.join(OUTPUT_DIR, "full_mask")
    crop_mask_dir = os.path.join(OUTPUT_DIR, "crop_mask")
    debug_dir = os.path.join(OUTPUT_DIR, "debug")
    crop_rgb_dir = os.path.join(OUTPUT_DIR, "crop_rgb")

    ensure_dir(full_mask_dir)
    ensure_dir(crop_mask_dir)
    if SAVE_DEBUG:
        ensure_dir(debug_dir)
    if SAVE_CROP_RGB:
        ensure_dir(crop_rgb_dir)

    # 加载 YOLO 模型
    detector = YOLO(YOLO_MODEL_PATH)

    # 加载 rembg session，指定人体分割模型
    session = new_session(REMBG_MODEL_NAME)

    # 支持的输入图片格式
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    image_files = sorted([
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(valid_exts)
    ])

    if len(image_files) == 0:
        print(f"[退出] 输入目录中没有图片: {INPUT_DIR}")
        return

    print(f"共找到 {len(image_files)} 张图片，开始处理...")

    # 上一帧目标中心，用于简单时序连续性约束
    prev_center = None

    for img_path in image_files:
        name = os.path.splitext(os.path.basename(img_path))[0]

        # 读取原图
        img = cv2.imread(img_path)
        if img is None:
            print(f"[跳过] 无法读取图片: {img_path}")
            continue

        h, w = img.shape[:2]

        # YOLO 检测
        results = detector.predict(
            source=img,
            conf=YOLO_CONF_THRES,
            iou=YOLO_IOU_THRES,
            verbose=False
        )

        if len(results) == 0:
            print(f"[跳过] YOLO 无结果: {img_path}")
            continue

        # 选择主人体框
        box = choose_person_box(results[0], img_w=w, img_h=h, prev_center=prev_center)
        if box is None:
            print(f"[跳过] 未检测到合格的人体框: {img_path}")
            continue

        x1, y1, x2, y2, conf = box

        # 更新上一帧中心（如果启用了时序约束，这会用于下一帧）
        prev_center = box_center(box)

        # 裁剪人体区域
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            print(f"[跳过] 裁剪区域为空: {img_path}")
            continue

        # 保存裁剪后的 RGB 图，方便检查
        if SAVE_CROP_RGB:
            cv2.imwrite(os.path.join(crop_rgb_dir, f"{name}.png"), crop)

        # rembg 分割
        try:
            crop_mask = rembg_segment_human(crop, session)
        except Exception as e:
            print(f"[失败] rembg 处理异常: {img_path} | {e}")
            continue

        # 将裁剪区域的 mask 放回原图坐标系
        full_mask = np.zeros((h, w), dtype=np.uint8)
        mh, mw = crop_mask.shape[:2]
        full_mask[y1:y1 + mh, x1:x1 + mw] = crop_mask

        # 保存整图坐标系下的 mask
        full_mask_path = os.path.join(full_mask_dir, f"{name}.png")
        cv2.imwrite(full_mask_path, full_mask)

        # 保存局部裁剪区域的 mask
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