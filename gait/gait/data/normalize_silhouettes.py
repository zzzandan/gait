import os
import cv2
import numpy as np


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def find_foreground_bbox(mask: np.ndarray):
    """
    找到前景外接框。
    输入:
        mask: 二值图，前景为255，背景为0
    返回:
        (x1, y1, x2, y2)
        如果没有前景，返回 None
    """
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
):
    """
    将单张二值剪影图标准化为固定大小。

    处理流程：
    1. 找前景外接框
    2. 裁剪前景
    3. 按比例缩放
    4. 放到固定大小画布上
       - 水平居中
       - 底部对齐

    参数：
    - out_h, out_w: 输出尺寸
    - pad: 给人物四周预留一点边界，避免贴边太死

    返回：
    - norm_mask: shape [out_h, out_w] 的 uint8 二值图
    """
    bbox = find_foreground_bbox(mask)
    canvas = np.zeros((out_h, out_w), dtype=np.uint8)

    # 如果整张图没有前景，直接返回全黑图
    if bbox is None:
        return canvas

    x1, y1, x2, y2 = bbox
    person = mask[y1:y2, x1:x2]

    ph, pw = person.shape[:2]
    if ph == 0 or pw == 0:
        return canvas

    # -------------------------------------------------
    # 目标可用区域（留一点边界）
    # 例如 out_h=64, pad=2，则可用高度为 60
    # -------------------------------------------------
    avail_h = out_h - 2 * pad
    avail_w = out_w - 2 * pad

    if avail_h <= 0 or avail_w <= 0:
        raise ValueError("输出尺寸太小或 pad 太大。")

    # -------------------------------------------------
    # 保持宽高比缩放
    # 优先按高度缩放，如果宽度超出，则再按宽度限制
    # -------------------------------------------------
    scale_h = avail_h / ph
    scale_w = avail_w / pw
    scale = min(scale_h, scale_w)

    new_h = max(1, int(round(ph * scale)))
    new_w = max(1, int(round(pw * scale)))

    # 二值图缩放必须用最近邻插值
    resized = cv2.resize(person, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # -------------------------------------------------
    # 放到固定画布上：
    # - 水平居中
    # - 底部对齐
    # -------------------------------------------------
    start_x = (out_w - new_w) // 2
    start_y = out_h - pad - new_h   # 底部对齐

    # 安全裁剪，防止越界
    start_x = max(0, start_x)
    start_y = max(0, start_y)

    end_x = min(out_w, start_x + new_w)
    end_y = min(out_h, start_y + new_h)

    canvas[start_y:end_y, start_x:end_x] = resized[:end_y-start_y, :end_x-start_x]

    # 强制二值化，防止缩放/写入时出现非 0/255
    canvas = (canvas > 0).astype(np.uint8) * 255
    return canvas


def batch_normalize_silhouettes(
    input_dir: str,
    output_dir: str,
    out_h: int = 64,
    out_w: int = 44,
    pad: int = 2
):
    """
    批量标准化一个文件夹里的剪影图。
    """
    ensure_dir(output_dir)

    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(valid_exts)
    ])

    if len(image_files) == 0:
        print(f"[退出] 输入目录没有图片: {input_dir}")
        return

    print(f"共找到 {len(image_files)} 张剪影图，开始标准化...")

    for fname in image_files:
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, os.path.splitext(fname)[0] + ".png")

        mask = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[跳过] 无法读取: {in_path}")
            continue

        # 防止输入不是严格二值图
        mask = (mask > 127).astype(np.uint8) * 255

        norm_mask = normalize_one_silhouette(
            mask=mask,
            out_h=out_h,
            out_w=out_w,
            pad=pad
        )

        cv2.imwrite(out_path, norm_mask)
        print(f"[完成] {fname} -> {out_path}")

    print("全部标准化完成。")


if __name__ == "__main__":
    input_folder = "/home/zzzandan/desk/gait/gait/gait/data/output/silhouettes_yoloseg/crop_mask"
    output_folder = "/home/zzzandan/desk/gait/gait/gait/data/output/silhouettes_yoloseg/crop_mask_norm"

    batch_normalize_silhouettes(
        input_dir=input_folder,
        output_dir=output_folder,
        out_h=64,
        out_w=44,
        pad=2
    )