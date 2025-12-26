import torch
import os
import shutil # 用于清理临时目录 (如果需要)
import time
from PIL import Image, ImageOps # Pillow 用于图像处理
import numpy as np
import folder_paths # 用于获取输出目录
from comfy.utils import ProgressBar
from tqdm import tqdm
import glob # 用于查找文件

# 可能需要 common_upscale 类似的缩放，但我们会用 Pillow 实现
# from comfy.model_management import common_upscale # 如果你打算复用它，但对于磁盘文件，Pillow更直接

class AIIA_Utils_Image_Concanate:
    NODE_NAME = "AIIA Utils Image Concatenate (Disk)" # 加上 (Disk) 以示区别
    CATEGORY = "AIIA/Utils" # 或者 AIIA/Image Utils
    FUNCTION = "concatenate_images_from_disk"
    RETURN_TYPES = ("STRING", "INT") # output_directory, frame_count
    RETURN_NAMES = ("output_directory", "concatenated_frame_count")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_1": ("STRING", {"default": "path/to/frames1", "multiline": False}),
                "directory_2": ("STRING", {"default": "path/to/frames2", "multiline": False}),
                "direction": (['right', 'down', 'left', 'up'], {"default": 'right'}),
                "match_image_size": ("BOOLEAN", {"default": True}),
                # 如果 match_image_size 为 True，以哪个目录的图像尺寸为基准
                "base_size_from": (['directory_1', 'directory_2'], {"default": 'directory_1'}), 
                "output_subdir_name": ("STRING", {"default": "concatenated_frames_AIIA"}),
            },
            "optional": {
                # 可以添加填充颜色等
                "background_color": ("STRING", {"default": "white", "tooltip": "背景颜色 (例如 'white', 'black', '#RRGGBB')"}),
            }
        }

    def _get_sorted_image_files(self, directory_path):
        """辅助函数：获取目录下排序后的图像文件列表"""
        # 支持常见的图像格式，扩展大小写支持以适应 Linux
        supported_extensions = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.bmp", "*.BMP", "*.webp", "*.WEBP"]
        image_files = []
        for ext in supported_extensions:
            image_files.extend(glob.glob(os.path.join(directory_path, ext)))
        
        # 按文件名排序 (假设文件名包含数字序列如 frame_001.png)
        # 更稳健的排序可能需要解析文件名中的数字
        try:
            # 尝试基于文件名中的数字进行自然排序
            image_files.sort(key=lambda f: int("".join(filter(str.isdigit, os.path.basename(f))) or 0))
        except ValueError:
            # 如果无法解析数字，则按普通字符串排序
            image_files.sort()
            print(f"警告: 目录 {directory_path} 中的文件无法按数字序列智能排序，将使用标准字符串排序。")

        if not image_files:
            print(f"警告: 目录 {directory_path} 中未找到支持的图像文件。")
        return image_files

    def concatenate_images_from_disk(self, directory_1, directory_2, direction, 
                                     match_image_size, base_size_from, output_subdir_name,
                                     background_color="white"):
        node_name_log = f"[{self.__class__.NODE_NAME}]"
        print(f"{node_name_log} 开始拼接图像 (从磁盘)。")
        start_time_process = time.time()

        files1 = self._get_sorted_image_files(directory_1)
        files2 = self._get_sorted_image_files(directory_2)
        
        num_frames1 = len(files1)
        num_frames2 = len(files2)
        print(f"{node_name_log} DEBUG: 目录1 ('{os.path.basename(directory_1)}') 包含 {num_frames1} 个图像文件。")
        print(f"{node_name_log} DEBUG: 目录2 ('{os.path.basename(directory_2)}') 包含 {num_frames2} 个图像文件。")

        if not files1 or not files2:
            error_msg = "输入目录为空或未找到图像文件。"
            if not files1: error_msg += f" 检查目录1: {directory_1}"
            if not files2: error_msg += f" 检查目录2: {directory_2}"
            print(f"错误: {node_name_log} {error_msg}")
            return (f"错误: {error_msg}", 0)

        # --- 处理帧数不匹配 ---
        if num_frames1 != num_frames2:
            # 当前简单处理：按最短的序列处理
            num_frames_to_process = min(num_frames1, num_frames2)
            print(f"警告: {node_name_log} 输入目录帧数不匹配 ({num_frames1} vs {num_frames2})。将处理 {num_frames_to_process} 帧。")
            files1 = files1[:num_frames_to_process]
            files2 = files2[:num_frames_to_process]
        else:
            num_frames_to_process = num_frames1
        
        if num_frames_to_process == 0:
            print(f"错误: {node_name_log} 没有可供处理的匹配帧。")
            return ("错误: 没有可供处理的匹配帧。", 0)

        # --- 创建输出目录 ---
        output_main_dir = folder_paths.get_output_directory()
        timestamp_str = time.strftime("%Y%m%d-%H%M%S")
        unique_folder_name = f"{output_subdir_name}_{timestamp_str}_{int(torch.randint(0,10000,(1,)).item())}"
        final_output_dir = os.path.join(output_main_dir, unique_folder_name)
        try:
            os.makedirs(final_output_dir, exist_ok=True)
        except Exception as e_mkdir:
            print(f"错误: {node_name_log} 无法创建输出目录 {final_output_dir}: {e_mkdir}")
            return (f"错误: 无法创建输出目录: {e_mkdir}", 0)

        concatenated_count = 0
        comfy_pbar = ProgressBar(num_frames_to_process)
        
        print(f"{node_name_log} 将处理 {num_frames_to_process} 帧组。")

        with tqdm(total=num_frames_to_process, desc=f"{node_name_log} Concatenating", unit="frame") as console_pbar:
            for i in range(num_frames_to_process):
                try:
                    img1_pil = Image.open(files1[i])
                    img2_pil = Image.open(files2[i])

                    # 确保图像是 RGB 或 RGBA，如果一个是P模式（调色板），先转换
                    if img1_pil.mode == 'P': img1_pil = img1_pil.convert('RGBA' if 'A' in img2_pil.mode else 'RGB')
                    if img2_pil.mode == 'P': img2_pil = img2_pil.convert('RGBA' if 'A' in img1_pil.mode else 'RGB')
                    
                    # 统一通道（例如，如果一个是RGB，一个是RGBA，都转为RGBA）
                    if 'A' in img1_pil.mode and 'A' not in img2_pil.mode:
                        img2_pil = img2_pil.convert('RGBA')
                    elif 'A' in img2_pil.mode and 'A' not in img1_pil.mode:
                        img1_pil = img1_pil.convert('RGBA')
                    
                    # 保留原始图像副本，以备尺寸匹配时参考
                    img1_orig_pil = img1_pil.copy()
                    img2_orig_pil = img2_pil.copy()


                    # --- 尺寸匹配逻辑 (借鉴并用Pillow实现) ---
                    if match_image_size:
                        if base_size_from == 'directory_1':
                            base_img_pil = img1_orig_pil
                            target_img_pil = img2_pil # 我们要修改 img2_pil
                            target_img_orig_pil = img2_orig_pil # 用于计算宽高比
                        else: # base_size_from == 'directory_2'
                            base_img_pil = img2_orig_pil
                            target_img_pil = img1_pil # 我们要修改 img1_pil
                            target_img_orig_pil = img1_orig_pil # 用于计算宽高比

                        base_w, base_h = base_img_pil.size
                        target_orig_w, target_orig_h = target_img_orig_pil.size
                        
                        if target_orig_h == 0 or target_orig_w == 0: # 避免除以零
                            print(f"警告: {node_name_log} 第 {i} 帧中, 目标图像尺寸为零，跳过尺寸匹配。")
                        else:
                            aspect_ratio = target_orig_w / target_orig_h

                            if direction in ['left', 'right']: # 匹配高度，调整宽度
                                new_h = base_h
                                new_w = int(new_h * aspect_ratio)
                            else: # up, down: 匹配宽度，调整高度
                                new_w = base_w
                                new_h = int(new_w / aspect_ratio)
                            
                            if new_w <=0 or new_h <=0: # 避免无效尺寸
                                print(f"警告: {node_name_log} 第 {i} 帧中, 计算出的新尺寸无效 ({new_w}x{new_h})，跳过此帧的尺寸匹配。")
                            else:
                                # 使用 LANCZOS (高质量) 或 ANTIALIAS (Pillow 9.0.0+ 推荐)
                                resample_filter = Image.Resampling.LANCZOS if hasattr(Image.Resampling, 'LANCZOS') else Image.LANCZOS
                                resized_target_img = target_img_orig_pil.resize((new_w, new_h), resample=resample_filter)
                                
                                # 更新 PIL 对象
                                if base_size_from == 'directory_1':
                                    img2_pil = resized_target_img
                                else:
                                    img1_pil = resized_target_img
                    
                    # --- 拼接逻辑 ---
                    w1, h1 = img1_pil.size
                    w2, h2 = img2_pil.size

                    if direction == 'right':
                        new_width = w1 + w2
                        new_height = max(h1, h2)
                        # 创建新画布，使用用户指定的背景色
                        result_img = Image.new(img1_pil.mode, (new_width, new_height), background_color)
                        result_img.paste(img1_pil, (0, (new_height - h1) // 2)) # 居中粘贴
                        result_img.paste(img2_pil, (w1, (new_height - h2) // 2))
                    elif direction == 'down':
                        new_width = max(w1, w2)
                        new_height = h1 + h2
                        result_img = Image.new(img1_pil.mode, (new_width, new_height), background_color)
                        result_img.paste(img1_pil, ((new_width - w1) // 2, 0))
                        result_img.paste(img2_pil, ((new_width - w2) // 2, h1))
                    elif direction == 'left':
                        new_width = w1 + w2
                        new_height = max(h1, h2)
                        result_img = Image.new(img1_pil.mode, (new_width, new_height), background_color)
                        result_img.paste(img2_pil, (0, (new_height - h2) // 2))
                        result_img.paste(img1_pil, (w2, (new_height - h1) // 2))
                    elif direction == 'up':
                        new_width = max(w1, w2)
                        new_height = h1 + h2
                        result_img = Image.new(img1_pil.mode, (new_width, new_height), background_color)
                        result_img.paste(img2_pil, ((new_width - w2) // 2, 0))
                        result_img.paste(img1_pil, ((new_width - w1) // 2, h2))
                    else: # 理论上不会到这里，因为 direction 是 COMBO
                        print(f"错误: {node_name_log} 未知的拼接方向: {direction}")
                        continue 

                    output_filename = f"frame_{concatenated_count:06d}.png" # 固定输出为png
                    output_filepath = os.path.join(final_output_dir, output_filename)
                    result_img.save(output_filepath)
                    concatenated_count += 1

                except Exception as e_frame:
                    print(f"错误: {node_name_log} 处理帧 {i} (文件: {files1[i]}, {files2[i]}) 时出错: {e_frame}")
                    import traceback
                    traceback.print_exc()
                    # 可以选择跳过此帧或中止
                
                finally:
                    # 关闭打开的图像文件，以释放资源
                    if 'img1_pil' in locals() and hasattr(img1_pil, 'close'): img1_pil.close()
                    if 'img2_pil' in locals() and hasattr(img2_pil, 'close'): img2_pil.close()
                    if 'img1_orig_pil' in locals() and hasattr(img1_orig_pil, 'close'): img1_orig_pil.close()
                    if 'img2_orig_pil' in locals() and hasattr(img2_orig_pil, 'close'): img2_orig_pil.close()
                    if 'base_img_pil' in locals() and hasattr(base_img_pil, 'close'): base_img_pil.close()
                    if 'target_img_pil' in locals() and hasattr(target_img_pil, 'close'): target_img_pil.close()
                    if 'target_img_orig_pil' in locals() and hasattr(target_img_orig_pil, 'close'): target_img_orig_pil.close()
                    if 'resized_target_img' in locals() and hasattr(resized_target_img, 'close'): resized_target_img.close()
                    if 'result_img' in locals() and hasattr(result_img, 'close'): result_img.close()


                comfy_pbar.update(1)
                console_pbar.update(1)
        
        end_time_process = time.time()
        print(f"{node_name_log} 拼接完成。总共 {concatenated_count} 帧已保存到 {final_output_dir}")
        print(f"{node_name_log} 方法总执行耗时: {end_time_process - start_time_process:.2f} 秒。")

        if concatenated_count == 0 and num_frames_to_process > 0:
            return (f"错误: 未成功拼接任何帧，请检查日志。", 0)
            
        return (final_output_dir, concatenated_count)

# --- ComfyUI 节点注册 ---
NODE_CLASS_MAPPINGS = {
    "AIIA_Utils_Image_Concanate": AIIA_Utils_Image_Concanate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_Utils_Image_Concanate": "Image Concatenate (AIIA Utils, Disk)",
}

# 可以在这里添加一个 main 用于独立测试 (如果需要)
# if __name__ == '__main__':
#     # 创建一些假的目录和图像文件进行测试
#     # ...
#     concatenator = AIIA_Utils_Image_Concanate()
#     # result_dir, count = concatenator.concatenate_images_from_disk(...)
#     # print(f"Test finished. Output to {result_dir}, {count} frames.")