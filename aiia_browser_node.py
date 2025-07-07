# --- START OF FILE aiia_browser_node.py (V19 - 健壮性终极修复) ---

import os
import server
from aiohttp import web
import json
from pathlib import Path
import time
from PIL import Image, ImageOps
import asyncio
import shutil
import traceback
from io import BytesIO
import re
import tempfile
import folder_paths

print("--- [AIIA] Loading Media Browser API Endpoints (V19 - Robustness Fixes) ---")

# --- START OF FIX: Path and Cache Logic Refinement ---
try:
    # Attempt to find the ComfyUI root directory robustly
    output_dir = Path(folder_paths.get_output_directory())
    print(f"--- [AIIA] Successfully located ComfyUI root. Output directory set to: {output_dir}")

except Exception as e:
    # Fallback path if auto-detection fails
    print(f"--- [AIIA] Warning: Could not auto-detect ComfyUI root path ({e}). Falling back to default './ComfyUI/output'.")
    output_dir = Path("./ComfyUI/output")

# Define cache directories based on the final output_dir. This ensures consistency.
cache_main_dir = output_dir / ".aiia_cache"
image_thumb_dir = cache_main_dir / "thumbnails"
video_poster_dir = cache_main_dir / "posters"

# Create all necessary directories
try:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_main_dir.mkdir(exist_ok=True)
    image_thumb_dir.mkdir(exist_ok=True)
    video_poster_dir.mkdir(exist_ok=True)
    print(f"--- [AIIA] Image thumbnail cache: {image_thumb_dir}")
    print(f"--- [AIIA] Video poster cache: {video_poster_dir}")
except Exception as e:
    print(f"--- [AIIA] CRITICAL ERROR: Could not create output or cache directories at {output_dir}. Please check permissions. Error: {e}")
# --- END OF FIX: Path and Cache Logic Refinement ---


CONCURRENT_LIMIT = 16
semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)

def get_safe_path(base_dir, relative_path_str):
    try:
        safe_base = base_dir.resolve()
        
        # If the relative path is empty or just a dot, return the base directory itself.
        if not relative_path_str or relative_path_str == '.':
            return safe_base

        clean_path = os.path.normpath(relative_path_str)
        if '..' in clean_path.split(os.sep):
             raise PermissionError("Path traversal attempt detected.")
        
        target_path = (safe_base / Path(clean_path)).resolve()
        
        # Ensure the resolved target path is still within the base directory.
        target_path.relative_to(safe_base)
        
        return target_path
    except ValueError:
        raise PermissionError("Forbidden path, not within the base directory.")
    except Exception as e:
        print(f"[AIIA Browser] Error in get_safe_path: {e}")
        raise


async def _get_file_metadata(file_path: Path):
    async with semaphore:
        metadata = {}
        try:
            if not file_path.exists() or file_path.stat().st_size == 0:
                return metadata
        except FileNotFoundError:
             return metadata 

        ext = file_path.suffix.lower()
        try:
            if ext in {'.png', '.jpg', '.jpeg', '.gif', '.webp'}:
                with Image.open(file_path) as img:
                    img.load() # Force loading of metadata chunks
                    metadata["width"], metadata["height"] = img.size
                    if 'prompt' in img.info or 'workflow' in img.info:
                        metadata['has_workflow'] = True
            elif ext in {'.mp4', '.mov', '.avi', '.wav', '.mp3', '.ogg'}:
                if not shutil.which("ffprobe"): return metadata
                command = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(file_path)]
                proc = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                stdout, stderr = await proc.communicate()
                if proc.returncode == 0:
                    ffprobe_data = json.loads(stdout)
                    if 'format' in ffprobe_data:
                        if 'duration' in ffprobe_data['format']:
                            metadata['duration'] = float(ffprobe_data['format']['duration'])
                        # Check for workflow in format tags (e.g., 'comment' for MP4/MOV)
                        tags = ffprobe_data['format'].get('tags', {})
                        for key, value in tags.items():
                            # A more strict check for workflow JSON
                         if isinstance(value, str) and value.strip().startswith('{') and '"nodes"' in value and '"links"' in value:
                             metadata['has_workflow'] = True
                             break # Found it, no need to check other tags
                    
                    video_stream = next((s for s in ffprobe_data.get('streams', []) if s.get('codec_type') == 'video'), None)
                    if video_stream:
                        metadata['width'] = video_stream.get('width')
                        metadata['height'] = video_stream.get('height')
        except Exception as e:
            print(f"--- [AIIA] Warning: Could not get metadata for {file_path}. Reason: {e}")
            pass
        return metadata

async def list_items(request):
    relative_path_str = request.query.get("path", "")
    try:
        target_path = get_safe_path(output_dir, relative_path_str)
        if not target_path.is_dir(): 
            return web.Response(status=404, text=f"Directory not found: {target_path}")
        
        directories, files = [], []
        for item in os.scandir(target_path):
            # --- START OF FIX: Precise filtering ---
            # Only filter our specific cache directory, not all dotfiles.
            if item.name == '.aiia_cache':
                continue
            # --- END OF FIX: Precise filtering ---
            try:
                if item.is_dir():
                    try: item_count = len([name for name in os.listdir(item.path) if name != '.aiia_cache']) # Also exclude from count
                    except OSError: item_count = 0
                    directories.append({"name": item.name, "type": "directory", "mtime": item.stat().st_mtime, "item_count": item_count})
                elif item.is_file():
                    ext = Path(item.name).suffix.lower()
                    if ext in {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.mp4', '.mov', '.avi', '.wav', '.mp3', '.ogg'}:
                        stat_info = item.stat()
                        file_data = {"name": item.name, "type": "file", "size": stat_info.st_size, "mtime": stat_info.st_mtime, "extension": ext}
                        files.append(file_data)
            except Exception:
                continue 
        
        def natural_sort_key(s, _re=re.compile(r'([0-9]+)')):
            return [int(text) if text.isdigit() else text.lower() for text in _re.split(s['name'])]

        directories.sort(key=natural_sort_key)
        files.sort(key=natural_sort_key)

        return web.json_response({"directories": directories, "files": files})
    except Exception as e:
        traceback.print_exc()
        return web.Response(status=500, text=str(e))

async def get_thumbnail(request):
    relative_path_str = request.query.get("path", "")
    filename = request.query.get("filename", "")
    if not filename: return web.Response(status=400, text="Filename is required.")
    
    try:
        original_file_path = get_safe_path(output_dir, os.path.join(relative_path_str, filename))
        if not original_file_path.is_file(): return web.Response(status=404, text="File not found")

        cache_sub_dir = get_safe_path(image_thumb_dir, relative_path_str)
        cache_sub_dir.mkdir(parents=True, exist_ok=True)
        cached_thumb_path = cache_sub_dir / f"{Path(filename).stem}.jpg"

        if cached_thumb_path.exists() and cached_thumb_path.stat().st_mtime > original_file_path.stat().st_mtime:
            return web.FileResponse(cached_thumb_path, headers={'Content-Type': 'image/jpeg', 'Cache-Control': 'public, must-revalidate'})

        with Image.open(original_file_path) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode in ("RGBA", "P"): img = img.convert("RGB")
            img.thumbnail((256, 256), Image.Resampling.LANCZOS)
            
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=85, optimize=True)
            buffer.seek(0)
            
            async def write_cache():
                try:
                    with open(cached_thumb_path, 'wb') as f: f.write(buffer.getvalue())
                except Exception as e: print(f"[AIIA] Error writing thumbnail cache for {filename}: {e}")
            
            asyncio.create_task(write_cache())
            
            return web.Response(body=buffer.getvalue(), content_type='image/jpeg', headers={'Cache-Control': 'public, must-revalidate'})

    except Exception as e:
        traceback.print_exc()
        return web.Response(status=500, text=str(e))

async def get_video_poster(request):
    relative_path_str = request.query.get("path", "")
    filename = request.query.get("filename", "")
    if not filename: return web.Response(status=400, text="Filename is required.")

    if not shutil.which("ffmpeg"):
        return web.Response(status=501, text="ffmpeg not found on server.")

    try:
        original_file_path = get_safe_path(output_dir, os.path.join(relative_path_str, filename))
        if not original_file_path.is_file() or original_file_path.stat().st_size == 0:
            return web.Response(status=404, text="File not found or is empty")

        cache_sub_dir = get_safe_path(video_poster_dir, relative_path_str)
        cache_sub_dir.mkdir(parents=True, exist_ok=True)
        cached_poster_path = cache_sub_dir / f"{Path(filename).stem}.jpg"
        
        if cached_poster_path.exists():
            # If the cached file is valid and up-to-date, serve it.
            if cached_poster_path.stat().st_size > 0 and cached_poster_path.stat().st_mtime > original_file_path.stat().st_mtime:
                return web.FileResponse(cached_poster_path, headers={'Content-Type': 'image/jpeg', 'Cache-Control': 'public, must-revalidate'})
            else:
                # Delete invalid (0-byte) or outdated cache files.
                print(f"--- [AIIA Browser Debug] Deleting invalid or outdated cache file: {cached_poster_path}")
                os.remove(cached_poster_path)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
            tmp_poster_path = tmpfile.name

        # Use the 'thumbnail' filter with the 'update' flag, which is robust for both
        # normal videos and special cases like single-frame videos.
        command = [
            "ffmpeg",
            "-i", str(original_file_path),
            "-vf", "thumbnail,scale=256:-1:force_original_aspect_ratio=decrease",
            "-frames:v", "1",
            "-update", "1",
            "-q:v", "3",
            "-y",
            str(tmp_poster_path)
        ]
        
        proc = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, stderr = await proc.communicate()

        if proc.returncode != 0 or not os.path.exists(tmp_poster_path) or os.path.getsize(tmp_poster_path) == 0:
            if os.path.exists(tmp_poster_path): os.remove(tmp_poster_path)
            print(f"[AIIA] FFmpeg failed to extract thumbnail for {filename}. Error: {stderr.decode()}")
            return web.Response(status=500, text="Failed to extract thumbnail from video.")

        shutil.move(tmp_poster_path, cached_poster_path)
        return web.FileResponse(cached_poster_path, headers={'Content-Type': 'image/jpeg', 'Cache-Control': 'public, must-revalidate'})

    except Exception as e:
        if 'tmp_poster_path' in locals() and os.path.exists(tmp_poster_path):
            os.remove(tmp_poster_path)
        traceback.print_exc()
        return web.Response(status=500, text=str(e))

async def get_batch_metadata(request):
    data = await request.json()
    path = data.get("path", "")
    filenames = data.get("filenames", [])
    try:
        safe_base = output_dir.resolve()
        tasks = []
        valid_filenames = []
        for filename in filenames:
            file_path = (safe_base / Path(path) / filename).resolve()
            if safe_base in file_path.parents and file_path.is_file():
                tasks.append(_get_file_metadata(file_path))
                valid_filenames.append(filename)
        results = await asyncio.gather(*tasks)
        response_data = {valid_filenames[i]: results[i] for i in range(len(valid_filenames))}
        return web.json_response(response_data)
    except Exception as e:
        traceback.print_exc()
        return web.Response(status=500, text=str(e))

import math

def replace_nan_with_null(obj):
    """
    Recursively traverses a Python object (dicts, lists) and replaces
    float('nan') with None, which serializes to JSON 'null'.
    """
    if isinstance(obj, dict):
        return {k: replace_nan_with_null(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan_with_null(elem) for elem in obj]
    # The core of the fix: only replace actual float NaN values
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    else:
        return obj

async def get_workflow(request):
    relative_path_str = request.query.get("path", "")
    filename = request.query.get("filename", "")
    if not filename:
        return web.Response(status=400, text="Filename is required.")

    try:
        file_path = get_safe_path(output_dir, os.path.join(relative_path_str, filename))
        if not file_path.is_file():
            return web.Response(status=404, text="File not found")

        ext = file_path.suffix.lower()
        workflow_obj = None

        if ext in {'.png', '.jpg', '.jpeg', '.gif', '.webp'}:
            with Image.open(file_path) as img:
                workflow_data_str = img.info.get('workflow') or img.info.get('prompt')
                if workflow_data_str:
                    workflow_obj = json.loads(workflow_data_str)
        
        elif ext in {'.mp4', '.mov', '.avi', '.webm'}:
            if not shutil.which("ffprobe"):
                return web.Response(status=501, text="ffprobe is not installed on the server.")
            
            command = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(file_path)]
            proc = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                ffprobe_data = json.loads(stdout)
                tags = ffprobe_data.get("format", {}).get("tags", {})
                for key, value in tags.items():
                    if isinstance(value, str) and value.strip().startswith('{') and '"nodes"' in value:
                        workflow_obj = json.loads(value)
                        break
        
        if workflow_obj:
            sanitized_workflow_obj = replace_nan_with_null(workflow_obj)
            return web.json_response(sanitized_workflow_obj)
        else:
            return web.Response(status=404, text="No workflow data found in the file.")

    except Exception as e:
        traceback.print_exc()
        return web.Response(status=500, text=str(e))

server.PromptServer.instance.app.router.add_get('/api/aiia/v1/browser/list_items', list_items)
server.PromptServer.instance.app.router.add_post('/api/aiia/v1/browser/get_batch_metadata', get_batch_metadata)
server.PromptServer.instance.app.router.add_get('/api/aiia/v1/browser/thumbnail', get_thumbnail)
server.PromptServer.instance.app.router.add_get('/api/aiia/v1/browser/poster', get_video_poster)
server.PromptServer.instance.app.router.add_get('/api/aiia/v1/browser/get_workflow', get_workflow)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

print("--- [AIIA] Media Browser API Endpoints (V19) loaded successfully. ---")