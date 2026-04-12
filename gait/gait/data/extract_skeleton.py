import os
import json
import cv2
import mediapipe as mp  # 标准导入方式

def extract_skeletons(input_dir, output_json_dir, confidence_threshold=0.5):
    """
    批量提取文件夹内所有图片的人体骨架关键点。
    
    Args:
        input_dir (str): 输入图片文件夹路径 (建议使用原始帧文件夹，而非剪影)
        output_json_dir (str): 输出骨架 JSON 文件的文件夹路径
        confidence_threshold (float): 关键点置信度阈值，低于此值标记为不可见
    """
    os.makedirs(output_json_dir, exist_ok=True)

    # 使用标准方式初始化 Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,         # 处理静态图片，精度更高
        model_complexity=1,             # 模型复杂度 (0-2)，1为平衡选择
        min_detection_confidence=0.5
    )

    # 支持的图片扩展名
    extensions = ('.jpg', '.jpeg', '.png')

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(extensions):
            continue

        in_path = os.path.join(input_dir, fname)
        base_name = os.path.splitext(fname)[0]

        # 读取并转换图像 (BGR -> RGB)
        image = cv2.imread(in_path)
        if image is None:
            print(f"警告：无法读取图片 {in_path}，已跳过")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # 进行姿态估计
        results = pose.process(image_rgb)

        # 提取关键点数据
        keypoints = []
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                px = int(landmark.x * w)
                py = int(landmark.y * h)
                visibility = landmark.visibility

                keypoints.append({
                    'id': idx,
                    'x_norm': landmark.x,
                    'y_norm': landmark.y,
                    'x_pixel': px,
                    'y_pixel': py,
                    'z': landmark.z,
                    'visibility': visibility,
                    'is_visible': visibility >= confidence_threshold
                })

            skeleton_data = {
                'image': in_path,
                'image_size': {'width': w, 'height': h},
                'num_keypoints': len(keypoints),
                'keypoints': keypoints
            }

            json_path = os.path.join(output_json_dir, f"{base_name}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(skeleton_data, f, indent=2, ensure_ascii=False)
            print(f"✓ 骨架提取完成: {fname} -> {json_path}")
        else:
            print(f"✗ 未检测到人体姿态: {fname}，已跳过")
            empty_data = {
                'image': in_path,
                'error': 'No pose detected',
                'keypoints': []
            }
            json_path = os.path.join(output_json_dir, f"{base_name}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(empty_data, f, indent=2, ensure_ascii=False)

    pose.close()
    print("\n所有图片处理完毕！")

if __name__ == "__main__":
    # 使用原始RGB帧路径
    input_folder = "/home/zzzandan/desk/gait/gait/gait/data/output/frames"
    output_json_folder = "/home/zzzandan/desk/gait/gait/gait/data/output/skeletons"
    extract_skeletons(input_folder, output_json_folder)