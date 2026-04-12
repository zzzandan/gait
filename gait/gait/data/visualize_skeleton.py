import os
import json
import cv2
import numpy as np

# MediaPipe 定义的关键点连接关系（骨骼）
# 来源：mp.solutions.pose.POSE_CONNECTIONS
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]

def draw_skeleton(image, keypoints, confidence_threshold=0.5):
    """
    在图像上绘制骨架关键点和连线。
    
    Args:
        image: numpy array (BGR格式)
        keypoints: list of dict, 每个dict包含 'x_pixel', 'y_pixel', 'visibility'
        confidence_threshold: 低于此阈值的关键点不绘制
    Returns:
        绘制后的图像
    """
    img_copy = image.copy()
    h, w = img_copy.shape[:2]
    
    # 过滤出可见关键点（像素坐标）
    points = []
    for kp in keypoints:
        vis = kp.get('visibility', 0)
        if vis >= confidence_threshold:
            x = kp['x_pixel']
            y = kp['y_pixel']
            points.append((x, y))
        else:
            points.append(None)
    
    # 绘制连线
    for connection in POSE_CONNECTIONS:
        idx1, idx2 = connection
        if idx1 < len(points) and idx2 < len(points):
            pt1 = points[idx1]
            pt2 = points[idx2]
            if pt1 is not None and pt2 is not None:
                cv2.line(img_copy, pt1, pt2, (0, 255, 0), 2)  # 绿色连线
    
    # 绘制关键点（圆形）
    for i, pt in enumerate(points):
        if pt is not None:
            cv2.circle(img_copy, pt, 4, (0, 0, 255), -1)  # 红色圆点
    
    return img_copy

def batch_visualize(frames_dir, skeletons_dir, output_dir, confidence_threshold=0.5):
    """
    批量生成骨架可视化图像。
    
    Args:
        frames_dir: 原始帧图片文件夹（与提取时使用的相同）
        skeletons_dir: 骨架 JSON 文件夹
        output_dir: 输出可视化图像的文件夹
        confidence_threshold: 关键点可见性阈值
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有 JSON 文件
    json_files = [f for f in os.listdir(skeletons_dir) if f.endswith('.json')]
    
    for json_file in json_files:
        base_name = os.path.splitext(json_file)[0]
        # 找到对应的原始帧图片（支持 jpg, jpeg, png）
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            candidate = os.path.join(frames_dir, base_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            print(f"警告：找不到 {base_name} 对应的原始图片")
            continue
        
        # 读取原始图片
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告：无法读取图片 {img_path}")
            continue
        
        # 读取 JSON
        json_path = os.path.join(skeletons_dir, json_file)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        keypoints = data.get('keypoints', [])
        if not keypoints:
            print(f"✗ {json_file} 中没有关键点数据，跳过")
            continue
        
        # 绘制骨架
        vis_image = draw_skeleton(image, keypoints, confidence_threshold)
        
        # 保存可视化图像
        out_path = os.path.join(output_dir, base_name + '_skeleton.jpg')
        cv2.imwrite(out_path, vis_image)
        print(f"✓ 已保存可视化: {out_path}")
    
    print("所有可视化图像生成完毕！")

if __name__ == "__main__":
    # 请根据你的实际路径修改
    frames_dir = "/home/zzzandan/desk/gait/gait/gait/data/output/frames"        # 原始帧文件夹
    skeletons_dir = "/home/zzzandan/desk/gait/gait/gait/data/output/skeletons_from_frames"  # 骨架 JSON 文件夹
    output_dir = "/home/zzzandan/desk/gait/gait/gait/data/output/skeleton_from_frames_viz"  # 可视化输出文件夹
    
    batch_visualize(frames_dir, skeletons_dir, output_dir, confidence_threshold=0.5)