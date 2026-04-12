import os
import cv2
import numpy as np


def build_background(frame_paths, sample_step=10, roi=None):
    """
    构建背景图。

    思路：
    1. 从视频帧中每隔 sample_step 抽取一张图；
    2. 转成灰度图并做高斯模糊，降低细碎噪声和纹理干扰；
    3. 对所有抽样帧按像素取中值，得到背景图。

    为什么这样做：
    - 对固定机位视频来说，如果人一直在移动，那么在大量帧中同一位置大部分时间是背景。
    - 中值背景建模比简单平均更稳，不容易把移动目标“平均”进去。

    注意事项：
    - 如果视频太短、抽样帧太少，背景图可能不稳。
    - 如果人长时间停留在一个位置，中值背景里可能残留人影。
    - 如果背景本身会动（树叶、屏幕闪烁、光照变化），背景图质量会下降。
    - 建议后面一定要把背景图保存出来肉眼检查：
      看看 background.png 是否“像真正的空场景”。

    参数建议：
    - sample_step 越大，背景样本越稀疏，速度更快，但可能不够稳定。
    - 一般可以尝试 5 / 10 / 20。
    - 如果视频比较长，sample_step=10 往往够用。

    ROI 说明：
    - roi 形如 (x1, y1, x2, y2)
    - 如果你知道人只会出现在画面某一块区域，建议只对这块区域建背景。
    - 这样可以显著减少广告牌、远处反光、道路高亮等干扰。
    """
    samples = []
    for i, p in enumerate(frame_paths):
        if i % sample_step != 0:
            continue

        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        if roi is not None:
            x1, y1, x2, y2 = roi
            img = img[y1:y2, x1:x2]

        # 高斯模糊：
        # 用来降低高频噪声和细碎纹理，能让后面的背景差分更平稳。
        img = cv2.GaussianBlur(img, (5, 5), 0)
        samples.append(img)

    if len(samples) == 0:
        raise ValueError("没有成功读取到用于建模背景的样本帧。")

    bg = np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)
    return bg


def select_human_component(mask, min_area=500):
    """
    从连通域中选一个“更像人”的区域。

    当前策略：
    1. 连通域面积不能太小；
    2. 连通域长宽比要满足“竖着站的人”的大致形状；
    3. 在所有候选中选面积最大的那个。

    为什么不能直接“保留最大连通域”：
    - 最大连通域不一定是人；
    - 可能是地面大片反光；
    - 可能是广告牌亮区；
    - 可能是阴影或背景中的大块变化区域。

    min_area 的作用：
    - 用来过滤掉小噪点、小碎块。
    - 如果人很小，min_area 不能设太大；
    - 如果噪声很多，min_area 不能设太小。
    - 建议尝试：200 / 500 / 1000。

    aspect_ratio 的作用：
    - 人体一般“高 > 宽”，所以高宽比通常不会太小。
    - 这里设置成 1.0 ~ 6.0 是一个比较宽松的经验范围。
    - 如果你的行人比较远、比较瘦，可以适当放宽上限；
    - 如果误选到很多横向噪声块，可以适当提高下限。

    当前方法的局限：
    - 仍然是单帧独立判断；
    - 没有利用“上一帧的人在哪里”这个时序信息；
    - 所以后面如果你还想继续提升稳定性，最推荐的方向是：
      在此基础上加入“上一帧中心位置约束”，优先选择位置连续的目标。
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    best_mask = np.zeros_like(mask)
    best_area = 0

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area < min_area:
            continue

        aspect_ratio = h / max(w, 1)

        # 人一般比宽更高，这里给一个比较宽松的范围
        if aspect_ratio < 1.0 or aspect_ratio > 6.0:
            continue

        if area > best_area:
            best_area = area
            best_mask[:] = 0
            best_mask[labels == i] = 255

    return best_mask


def extract_fg_with_bg_subtraction(
    input_dir,
    output_dir,
    thresh=30,
    sample_step=10,
    min_area=500,
    roi=None
):
    """
    使用背景减除法批量提取前景剪影。

    总体流程：
    1. 读取输入文件夹中的所有帧；
    2. 构建背景图；
    3. 当前帧与背景图做差分；
    4. 阈值化得到粗前景；
    5. 形态学操作去噪、补洞；
    6. 保留更像人的连通域；
    7. 保存二值剪影图。

    参数说明：
    - thresh:
        差分阈值。越小，越容易把轻微变化也当作前景；
        越大，越严格，只保留变化明显的区域。
        建议尝试：20 / 30 / 40 / 50。
        如果提取出来前景太碎、太多，增大 thresh；
        如果人经常被断开或提不出来，减小 thresh。

    - sample_step:
        背景建模抽样步长。
        一般视频足够长时，10 是一个比较稳妥的初值。

    - min_area:
        连通域最小面积阈值。
        过滤噪声用。

    - roi:
        感兴趣区域。
        如果已知人只在画面某一块区域活动，强烈建议加 ROI。
        这是提升效果最直接的方法之一。

    为什么这个方法适合步态视频：
    - 固定机位下，背景减除通常比通用抠图模型更符合步态任务；
    - 因为步态识别本来就更关心“运动前景”，而不是通用显著目标。

    这个方法的不足：
    - 对光照变化敏感；
    - 对阴影敏感；
    - 对固定背景中的动态物体敏感；
    - 对相机抖动不鲁棒。
    如果这些问题明显，下一步就要考虑：
    1. 加 ROI；
    2. 加时序跟踪约束；
    3. 或改成“人体检测 + 局部分割”的方案。
    """
    os.makedirs(output_dir, exist_ok=True)

    frame_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if len(frame_files) == 0:
        raise ValueError("输入文件夹中没有找到图片。")

    # =========================
    # 第一步：建背景
    # =========================
    bg = build_background(frame_files, sample_step=sample_step, roi=roi)

    # 强烈建议把背景图保存出来检查。
    # 很多时候前景提取效果差，不是因为差分错了，而是背景图本身建错了。
    cv2.imwrite(os.path.join(output_dir, "background.png"), bg)

    for path in frame_files:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        if roi is not None:
            x1, y1, x2, y2 = roi
            img_roi = img[y1:y2, x1:x2]
        else:
            img_roi = img

        # =========================
        # 第二步：模糊 + 背景差分
        # =========================
        # 先模糊再差分，能够减少噪点和细小纹理造成的误检。
        img_blur = cv2.GaussianBlur(img_roi, (5, 5), 0)
        diff = cv2.absdiff(img_blur, bg)

        # =========================
        # 第三步：阈值化
        # =========================
        # 将背景差分图变成二值图。
        # diff > thresh 的区域视为前景。
        _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)

        # =========================
        # 第四步：形态学处理
        # =========================
        # 开运算：
        # 去掉小噪点。
        kernel_open = np.ones((3, 3), np.uint8)

        # 闭运算：
        # 填补前景内部小孔洞，让人体区域更完整。
        kernel_close = np.ones((7, 7), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        # =========================
        # 第五步：连通域筛选
        # =========================
        # 不直接保留最大块，而是保留“更像人”的块。
        mask = select_human_component(mask, min_area=min_area)

        # =========================
        # 第六步：如果用了 ROI，放回原图
        # =========================
        if roi is not None:
            full_mask = np.zeros_like(img)
            full_mask[y1:y2, x1:x2] = mask
            mask = full_mask

        out_path = os.path.join(output_dir, os.path.basename(path))
        cv2.imwrite(out_path, mask)
        print("处理完成:", out_path)


if __name__ == "__main__":
    input_folder = "/home/zzzandan/desk/gait/gait/gait/data/output/frames"
    output_folder = "/home/zzzandan/desk/gait/gait/gait/data/output/silhouettes_bg"

    # ROI 建议：
    # 如果你知道人主要在哪个区域活动，尽量设置 ROI。
    # 例如：
    # roi = (100, 50, 1000, 700)
    # 这样能显著减少背景干扰。
    # 如果暂时不确定，就先设为 None，先看 background.png 和输出结果。
    roi = None

    extract_fg_with_bg_subtraction(
        input_dir=input_folder,
        output_dir=output_folder,

        # 差分阈值：
        # 如果输出里噪声太多，就增大；
        # 如果人体断裂严重或提不出来，就减小。
        thresh=30,

        # 背景建模抽样步长：
        # 越小背景越稳，但建模更慢；
        # 越大背景更快，但可能不够稳定。
        sample_step=10,

        # 最小前景面积：
        # 用于过滤小噪点；
        # 人比较小就调低；
        # 噪声比较大就调高。
        min_area=500,

        roi=roi
    )