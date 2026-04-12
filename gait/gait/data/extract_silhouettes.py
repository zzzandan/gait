"""
批量提取人体剪影（二值图）
输入：包含原始帧图片的文件夹（支持 .jpg, .jpeg, .png）
输出：对应的二值剪影图，保存在输出文件夹中
"""

import os
import io
from rembg import remove
from PIL import Image

def extract_silhouettes(input_dir, output_dir):
    """
    将文件夹内的所有图片转换为二值剪影图。
    
    Args:
        input_dir (str): 输入图片文件夹路径
        output_dir (str): 输出剪影文件夹路径
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的图片扩展名
    extensions = ('.jpg', '.jpeg', '.png')
    
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(extensions):
            continue
        
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)
        
        # 读取原始图片
        with open(in_path, 'rb') as f:
            data = f.read()
        
        # 移除背景，得到透明背景的PNG数据
        out_data = remove(data)
        
        # 转换为PIL图像并灰度化
        img = Image.open(io.BytesIO(out_data)).convert('L')
        
        # 二值化：前景白（255），背景黑（0）
        img = img.point(lambda x: 255 if x > 0 else 0, '1')
        
        # 保存
        img.save(out_path)
        print(f"处理完成: {fname} -> {out_path}")

if __name__ == "__main__":
    # 直接运行脚本时执行的代码
    input_folder = "/home/zzzandan/desk/gait/gait/gait/data/output/frames"   # 请根据实际情况修改
    output_folder = "/home/zzzandan/desk/gait/gait/gait/data/output/silhouettes"
    extract_silhouettes(input_folder, output_folder)
    print("所有图片处理完毕！")