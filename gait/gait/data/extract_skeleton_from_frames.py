import os
import json
import cv2
import mediapipe as mp

def extract_skeletons_from_frames(input_dir, output_json_dir, confidence_threshold=0.5):
    """
    从原始RGB帧提取人体骨架关键点。
    
    Args:
        input_dir: 原始帧文件夹（如 frames/）
        output_json_dir: 输出骨架 JSON 的文件夹
        confidence_threshold: 关键点可见性阈值
    """
    os.makedirs(output_json_dir, exist_ok=True)

    # 初始化 MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    )

    extensions = ('.jpg', '.jpeg', '.png')

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(extensions):
            continue

        in_path = os.path.join(input_dir, fname)
        base_name = os.path.splitext(fname)[0]

        # 读取原始图像（BGR）
        image = cv2.imread(in_path)
        if image is None:
            print(f"警告：无法读取 {in_path}")
            continue

        # 转换为 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # 姿态估计
        results = pose.process(image_rgb)

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
            print(f"✓ {fname} -> {json_path}")
        else:
            print(f"✗ 未检测到姿态: {fname}")
            empty_data = {'image': in_path, 'error': 'No pose detected', 'keypoints': []}
            json_path = os.path.join(output_json_dir, f"{base_name}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(empty_data, f, indent=2, ensure_ascii=False)

    pose.close()
    print("所有骨架提取完成！")

if __name__ == "__main__":
    # 请修改为你的原始帧文件夹路径
    frames_dir = "/home/zzzandan/desk/gait/gait/gait/data/output/frames"
    skeletons_dir = "/home/zzzandan/desk/gait/gait/gait/data/output/skeletons_from_frames"
    extract_skeletons_from_frames(frames_dir, skeletons_dir)