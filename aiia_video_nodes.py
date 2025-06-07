# aiia_video_nodes.py
import torch
import os
import sys
import subprocess
import numpy as np
import re
import datetime
from typing import List, Iterable, Union
import json
from PIL import Image, ExifTags
from PIL.PngImagePlugin import PngInfo
from pathlib import Path
from string import Template
import itertools
import functools
import glob
import hashlib
import shutil
import time
import shlex # For safe shell splitting

import folder_paths
# 尝试从 AIIA 包导入自定义日志记录器，如果失败则使用标准日志记录器
try:
    from .logger import logger # 假设你在 AIIA 包根目录有 logger.py
except ImportError:
    import logging
    logger = logging.getLogger("AIIA_VideoNodes")
try:
    import torchaudio
except ImportError:
    logger.error("torchaudio 未安装。如果需要从 AUDIO 张量输入创建视频，请安装 torchaudio。")
    torchaudio = None # 设置为 None 以便后续检查

import tempfile # 用于创建临时目录和文件

from comfy.utils import ProgressBar

# --- AIIA 特定工具函数 ---

DEFAULT_STD_ENCODING = "utf-8"
DEFAULT_STD_ERRORS = "replace"

def get_aiia_ffmpeg_path() -> str:
    env_path = os.environ.get("AIIA_FFMPEG_PATH")
    if env_path and os.path.isfile(env_path): return env_path
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg: return system_ffmpeg
    logger.warning("未能找到 ffmpeg。将默认使用 'ffmpeg'。")
    return "ffmpeg"

def get_aiia_ffprobe_path() -> str:
    env_path = os.environ.get("AIIA_FFPROBE_PATH")
    if env_path and os.path.isfile(env_path): return env_path
    ffmpeg_dir = os.path.dirname(get_aiia_ffmpeg_path())
    if ffmpeg_dir:
        potential_ffprobe = os.path.join(ffmpeg_dir, "ffprobe" + (".exe" if os.name == 'nt' else ""))
        if os.path.isfile(potential_ffprobe): return potential_ffprobe
    system_ffprobe = shutil.which("ffprobe")
    if system_ffprobe: return system_ffprobe
    logger.warning("未能找到 ffprobe。音频码率自动检测将不可用。")
    return "ffprobe"

ffmpeg_path = get_aiia_ffmpeg_path()
ffprobe_path = get_aiia_ffprobe_path()

def get_audio_bitrate_aiia(audio_file_path: str) -> Union[str, None]:
    if not shutil.which(ffprobe_path):
        logger.warning(f"ffprobe 在路径 '{ffprobe_path}' 不可用，无法检测码率。")
        return None
    command = [ffprobe_path, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=bit_rate", "-of", "default=noprint_wrappers=1:nokey=1", audio_file_path]
    try:
        bitrate = subprocess.check_output(command, text=True, stderr=subprocess.PIPE).strip()
        if bitrate and bitrate.isdigit() and bitrate != "N/A":
            kbps = int(bitrate) // 1000
            return f"{kbps}k"
        return None
    except Exception as e:
        logger.warning(f"使用 ffprobe 检测码率失败: {e}")
        return None

def strip_path_aiia(path_str: str) -> str:
    if path_str is None: return ""
    path_str = path_str.strip()
    if path_str.startswith("\"") and path_str.endswith("\"") and len(path_str) > 1: return path_str[1:-1]
    return path_str

def validate_path_aiia(path_str: str, check_is_file: bool = False, check_is_dir: bool = False) -> bool:
    stripped_path = strip_path_aiia(path_str)
    if not stripped_path: return False
    abs_path = os.path.abspath(stripped_path)
    if check_is_file: return os.path.isfile(abs_path)
    elif check_is_dir: return os.path.isdir(abs_path)
    return os.path.exists(abs_path)

# --- 视频格式处理 ---
aiia_builtin_formats_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_formats_aiia")
os.makedirs(aiia_builtin_formats_dir, exist_ok=True)
aiia_user_formats_path_key = "AIIA_video_formats_user"
if aiia_user_formats_path_key not in folder_paths.folder_names_and_paths:
    custom_nodes_root = folder_paths.get_folder_paths("custom_nodes")[0]
    this_file_path_abs = os.path.abspath(__file__)
    node_dir_name = next((item for item in os.listdir(custom_nodes_root) if os.path.isdir(os.path.join(custom_nodes_root, item)) and this_file_path_abs.startswith(os.path.join(custom_nodes_root, item))), None)
    custom_formats_dir = os.path.join(custom_nodes_root, node_dir_name, "video_formats_user") if node_dir_name else os.path.join(custom_nodes_root, "AIIA_Video_User_Formats")
    os.makedirs(custom_formats_dir, exist_ok=True)
    folder_paths.folder_names_and_paths[aiia_user_formats_path_key] = ([custom_formats_dir], {".json"})

@functools.lru_cache(maxsize=1)
def get_aiia_video_formats():
    format_files = {}
    user_format_paths = folder_paths.get_folder_paths(aiia_user_formats_path_key)
    if user_format_paths:
        for user_format_dir_path in user_format_paths:
            if os.path.isdir(user_format_dir_path):
                for f_name in os.listdir(user_format_dir_path):
                    if f_name.endswith(".json"): format_files[os.path.splitext(f_name)[0]] = os.path.join(user_format_dir_path, f_name)
    if os.path.isdir(aiia_builtin_formats_dir):
        for item in os.scandir(aiia_builtin_formats_dir):
            if item.is_file() and item.name.endswith('.json'):
                key = os.path.splitext(item.name)[0]
                if key not in format_files: format_files[key] = item.path
    formats_list_ui, format_widgets_map_ui = [], {}
    if not format_files: return ["video/fallback_h264_crf"], {}
    for name, path in sorted(format_files.items()):
        try:
            with open(path, 'r', encoding='utf-8') as f: content = json.load(f)
            ui_name = "video/" + name
            formats_list_ui.append(ui_name)
            if "extra_widgets" in content and isinstance(content["extra_widgets"], list):
                widgets = [w for w in content["extra_widgets"] if isinstance(w, list) and len(w) >= 2]
                if widgets: format_widgets_map_ui[ui_name] = widgets
        except Exception as e: logger.error(f"处理格式文件 {path} 时出错: {e}")
    if not formats_list_ui: return ["video/fallback_h264_crf"], {}
    return sorted(list(set(formats_list_ui))), format_widgets_map_ui

def aiia_apply_video_format_config(format_ui_name: str, user_inputs: dict) -> dict:
    name_only = format_ui_name.split("/")[-1]
    path = folder_paths.get_full_path(aiia_user_formats_path_key, name_only + ".json")
    if not (path and os.path.exists(path)): path = os.path.join(aiia_builtin_formats_dir, name_only + ".json")
    if not os.path.exists(path):
        if name_only == "fallback_h264_crf":
            return {"extension": "mp4", "main_pass": ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(user_inputs.get('crf', 23)), "-preset", user_inputs.get('codec_preset','medium')]}
        raise FileNotFoundError(f"视频格式 JSON '{name_only}.json' 未找到。")
    with open(path, 'r', encoding='utf-8') as f: template = json.load(f)
    current_vars = dict(user_inputs)
    if "extra_widgets" in template and isinstance(template["extra_widgets"], list):
        for cfg in template["extra_widgets"]:
            w_name = cfg[0]
            if w_name not in current_vars:
                if len(cfg) > 2 and 'default' in cfg[2]: current_vars[w_name] = cfg[2]['default']
                elif isinstance(cfg[1], list) and cfg[1]: current_vars[w_name] = cfg[1][0]
                else: current_vars[w_name] = {"BOOLEAN": False, "INT": 0, "FLOAT": 0.0, "STRING": ""}.get(str(cfg[1]).upper(), "")
    processed = {}
    for key, val in template.items():
        if isinstance(val, list) and (key.endswith("_pass") or key in ["inputs_main_pass", "main_pass", "audio_pass", "ffmpeg_args"]):
            processed[key] = [Template(str(item)).safe_substitute(**current_vars) for item in val]
        elif isinstance(val, str) and "$" in val: processed[key] = Template(val).safe_substitute(**current_vars)
        else: processed[key] = val
    return processed

class AIIA_VideoCombine:
    NODE_NAME = "AIIA 视频合并 (图像或目录)"
    CATEGORY = "AIIA/视频"
    FUNCTION = "combine_video"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_filepath",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        available_formats, format_widgets_map = get_aiia_video_formats()
        dynamic_widgets = {}
        for widgets in format_widgets_map.values():
            for proto in widgets:
                name, type_or_vals, opts = proto[0], proto[1], proto[2] if len(proto) > 2 else {}
                if name not in dynamic_widgets:
                    dynamic_widgets[name] = (type_or_vals, opts) if not isinstance(type_or_vals, list) else (type_or_vals, opts)
        
        audio_codec_options = ["auto", "aac", "libmp3lame", "libopus", "copy"]
        audio_bitrate_options = ["auto", "96k", "128k", "192k", "256k", "320k", "512k"]

        return {
            "required": {
                "frame_rate": ("FLOAT", {"default": 25.0, "min": 0.01, "max": 1000.0, "step": 0.1}),
                "output_filename_prefix": ("STRING", {"default": "AIIA_Video"}),
                "format": (available_formats, {"formats": format_widgets_map}),
                **dynamic_widgets,
                "crf": ("INT", {"default": 23, "min": 0, "max": 51}),
                "codec_preset": (['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow', 'placebo'], {"default": 'medium'}),
                "save_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "images": ("IMAGE",), "frames_directory": ("STRING", {"default": ""}),
                "filename_pattern": ("STRING", {"default": "frame_%06d.png"}),
                "audio_tensor": ("AUDIO",), "audio_file_path": ("STRING", {"default": ""}),
                # 【逻辑修改】将 'auto' 设为默认值
                "audio_codec": (audio_codec_options, {"default": "auto"}),
                "audio_bitrate": (audio_bitrate_options, {"default": "auto"}),
                "custom_ffmpeg_args": ("STRING", {"default": ""}),
            }, "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}
        }

    def combine_video(
        self, frame_rate: float, output_filename_prefix: str, format: str, crf: int, codec_preset: str, save_output: bool,
        images: Union[torch.Tensor, None] = None, frames_directory: str = "", filename_pattern: str = "frame_%06d.png",
        audio_tensor: Union[dict, None] = None, audio_file_path: str = "",
        audio_codec: str = "auto", audio_bitrate: str = "auto", custom_ffmpeg_args: str = "",
        prompt=None, extra_pnginfo=None, **kwargs):

        node_name_log = f"[{self.NODE_NAME}]"
        logger.info(f"{node_name_log} 开始视频合并。")
        
        temp_image_dir_to_delete, temp_audio_file_to_delete = None, None
        
        try:
            effective_frames_dir, effective_filename_pattern = None, filename_pattern
            if images is not None:
                logger.info(f"检测到 {images.shape[0]} 帧的图像张量输入...")
                temp_image_dir_to_delete = tempfile.mkdtemp(prefix="aiia_frames_")
                effective_frames_dir, effective_filename_pattern = temp_image_dir_to_delete, "frame_%08d.png"
                pbar = ProgressBar(images.shape[0])
                for i, frame_tensor in enumerate(images):
                    Image.fromarray((frame_tensor.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(effective_frames_dir, effective_filename_pattern % (i + 1)))
                    pbar.update(1)
            elif frames_directory:
                effective_frames_dir = strip_path_aiia(frames_directory)
                if not validate_path_aiia(effective_frames_dir, check_is_dir=True):
                    raise ValueError(f"帧目录验证失败: {effective_frames_dir}")
            else: raise ValueError("错误: 必须提供 'images' 或 'frames_directory' 输入。")

            output_dir = folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()
            (full_output_folder, filename_no_counter, _, subfolder, _) = folder_paths.get_save_image_path(output_filename_prefix, output_dir)
            Path(full_output_folder).mkdir(parents=True, exist_ok=True)
            counter = len(glob.glob(os.path.join(full_output_folder, f"{filename_no_counter}_*.*"))) + 1
            
            user_inputs = {'crf': str(crf), 'codec_preset': codec_preset, **kwargs}
            video_format_config = aiia_apply_video_format_config(format, user_inputs)
            output_extension = video_format_config.get('extension', 'mp4')
            final_video_filename = f"{filename_no_counter}_{counter:05}.{output_extension}"
            final_video_filepath = os.path.join(full_output_folder, final_video_filename)

            actual_audio_input_path, is_audio_from_tensor = None, False
            if audio_tensor and torchaudio:
                temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=folder_paths.get_temp_directory())
                torchaudio.save(temp_audio_file.name, audio_tensor['waveform'].squeeze(0).cpu(), audio_tensor['sample_rate'])
                actual_audio_input_path, temp_audio_file_to_delete, is_audio_from_tensor = temp_audio_file.name, temp_audio_file.name, True
            elif audio_file_path and validate_path_aiia(strip_path_aiia(audio_file_path), check_is_file=True):
                actual_audio_input_path = os.path.abspath(strip_path_aiia(audio_file_path))

            final_audio_bitrate = "192k"
            if actual_audio_input_path and audio_bitrate.lower() == 'auto':
                if not is_audio_from_tensor:
                    detected = get_audio_bitrate_aiia(actual_audio_input_path)
                    if detected: final_audio_bitrate = detected
                    else: logger.info("自动检测码率失败，使用默认值 192k。")
                else: logger.info("音频来自张量，'auto'码率使用默认值 192k。")
            elif audio_bitrate: final_audio_bitrate = audio_bitrate.strip()

            image_input_pattern = os.path.join(os.path.abspath(effective_frames_dir), effective_filename_pattern)
            ffmpeg_cmd = [ffmpeg_path, "-y", "-nostdin", "-framerate", str(frame_rate)]
            if "inputs_main_pass" in video_format_config: ffmpeg_cmd.extend(video_format_config["inputs_main_pass"])
            ffmpeg_cmd.extend(["-i", image_input_pattern])
            if actual_audio_input_path: ffmpeg_cmd.extend(["-i", actual_audio_input_path])
            if "main_pass" in video_format_config: ffmpeg_cmd.extend(video_format_config["main_pass"])

            if actual_audio_input_path:
                if audio_codec == "auto":
                    logger.info("音频编解码器为 'auto'，尝试使用格式JSON中的 'audio_pass'。")
                    if "audio_pass" in video_format_config and video_format_config["audio_pass"]:
                        ffmpeg_cmd.extend(video_format_config["audio_pass"])
                    else:
                        logger.warning("未在格式JSON中找到 'audio_pass'，回退到默认编码 (aac, 192k)。")
                        ffmpeg_cmd.extend(["-c:a", "aac", "-b:a", "192k"])
                else:
                    logger.info(f"使用节点UI设置的音频参数: Codec='{audio_codec}', Bitrate='{final_audio_bitrate}'")
                    ffmpeg_cmd.extend(["-c:a", audio_codec])
                    if audio_codec != 'copy' and final_audio_bitrate.lower() != 'auto':
                        ffmpeg_cmd.extend(["-b:a", final_audio_bitrate])
                ffmpeg_cmd.append("-shortest")
            
            if custom_ffmpeg_args: ffmpeg_cmd.extend(shlex.split(custom_ffmpeg_args))
            ffmpeg_cmd.extend(["-metadata", f"creation_time={datetime.datetime.now(datetime.timezone.utc).isoformat()}", "-metadata", "encoder=AIIA Video Node (ComfyUI)"])
            ffmpeg_cmd.append(final_video_filepath)
            
            logger.info(f"执行 FFmpeg: {' '.join(map(shlex.quote, ffmpeg_cmd))}")
            process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy())
            _, stderr = process.communicate()
            if process.returncode != 0: raise RuntimeError(f"FFmpeg 错误:\n{stderr.decode(DEFAULT_STD_ENCODING, errors=DEFAULT_STD_ERRORS)}")

            if not os.path.exists(final_video_filepath) or os.path.getsize(final_video_filepath) == 0:
                raise FileNotFoundError("FFmpeg 执行后未生成有效输出文件。")

            preview_item = {"filename": final_video_filename, "subfolder": subfolder, "type": "output" if save_output else "temp"}
            return {"ui": {"images": [preview_item], "animated": (True,)}, "result": (final_video_filepath,)}
            
        except Exception as e:
            logger.error(f"{node_name_log} 出现错误: {e}", exc_info=True)
            return {"ui": {"errors": [str(e)]}, "result": ("",)}
            
        finally:
            if temp_audio_file_to_delete and os.path.exists(temp_audio_file_to_delete):
                try: os.remove(temp_audio_file_to_delete); logger.info(f"已删除临时音频文件: {temp_audio_file_to_delete}")
                except Exception as e_del: logger.warning(f"删除临时音频文件失败: {e_del}")
            if temp_image_dir_to_delete and os.path.exists(temp_image_dir_to_delete):
                try: shutil.rmtree(temp_image_dir_to_delete); logger.info(f"已清理临时图像帧目录: {temp_image_dir_to_delete}")
                except Exception as e_del: logger.error(f"清理临时图像帧目录失败: {e_del}")

NODE_CLASS_MAPPINGS = { "AIIA_VideoCombine": AIIA_VideoCombine }
NODE_DISPLAY_NAME_MAPPINGS = { "AIIA_VideoCombine": "视频合并 (AIIA, 图像或目录)" }

print("--- AIIA 视频节点 (UI优先, 高级音频控制, v1.5) 已加载 ---")