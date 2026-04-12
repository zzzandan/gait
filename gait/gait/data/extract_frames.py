import cv2
import os

def extract_frames(video_path, output_dir):
    """
    从视频文件中提取所有帧并保存为图片。
    
    参数:
        video_path (str): 视频文件的路径。
        output_dir (str): 保存图片的目录路径。
    """
    # 1. 创建VideoCapture对象，打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 2. 错误检查：确认视频是否成功打开
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 '{video_path}'。请检查路径和文件完整性。")
        return
    
    # 3. 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 4. 获取视频属性（可选，用于打印信息）
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频信息：帧率={fps:.2f} FPS，总帧数={total_frames}")

    # 5. 初始化帧计数器
    frame_count = 0
    
    # 6. 循环读取视频帧
    while True:
        # 读取一帧：ret为布尔值表示是否成功，frame为读取到的图像数据
        ret, frame = cap.read()
        
        # 如果读取失败（例如到达视频末尾），则跳出循环
        if not ret:
            print("视频读取完毕或发生错误。")
            break
        
        # 7. 保存当前帧为图片文件
        # 生成文件名，如 frame_000001.jpg，便于排序
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        # 可选：打印进度
        if frame_count % 100 == 0:
            print(f"已处理 {frame_count} 帧...")
        
        frame_count += 1

    # 8. 释放VideoCapture对象，释放资源
    cap.release()
    print(f"完成！共提取 {frame_count} 帧，保存至: {output_dir}")

# 使用示例
if __name__ == "__main__":
    # 请将这里的路径替换为你的视频文件路径和输出目录
    video_file = "/home/zzzandan/desk/gait/gait/datasets/walk1.mp4"
    output_folder = "/home/zzzandan/desk/gait/gait/gait/data/output/frames"
    extract_frames(video_file, output_folder)