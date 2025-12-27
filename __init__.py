# /app/ComfyUI/custom_nodes/ComfyUI_AIIA/__init__.py
print("--- 正在加载 ComfyUI_AIIA 自定义节点包 ---") # 中文注释

# 初始化空的映射字典
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# --- 模块导入和映射合并 ---
# (为了代码简洁，我将重复的导入逻辑封装成一个函数)

def _load_nodes_from_module(module_name_relative, module_alias_for_log):
    """
    尝试从指定的模块加载节点映射。
    module_name_relative: 相对模块名，例如 ".aiia_float_nodes"
    module_alias_for_log: 用于日志记录的模块别名
    """
    global NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS # 确保我们修改的是全局变量
    
    module_object = None
    try:
        # 使用 importlib 进行更动态的导入
        import importlib
        module_object = importlib.import_module(module_name_relative, package=__name__) # __name__ 是当前包名 "ComfyUI_AIIA"
        
        if hasattr(module_object, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(module_object.NODE_CLASS_MAPPINGS)

        if hasattr(module_object, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(module_object.NODE_DISPLAY_NAME_MAPPINGS)
            
    except ImportError as e_import:
        print(f"错误: 导入 {module_alias_for_log} ({module_name_relative}) 失败: {e_import}")
    except Exception as e_generic:
        print(f"错误: 在 {module_alias_for_log} ({module_name_relative}) 导入或处理过程中发生错误: {e_generic}")


# 1. 处理 aiia_float_nodes.py
_load_nodes_from_module(".aiia_float_nodes", "aiia_float_nodes")

# 2. 处理 aiia_generate_segments.py
_load_nodes_from_module(".aiia_generate_segments", "aiia_generate_segments")

# 3. 处理 aiia_e2e_diarizer.py 
_load_nodes_from_module(".aiia_e2e_diarizer", "aiia_e2e_diarizer")

# 4. 处理 aiia_utils_nodes.py
_load_nodes_from_module(".aiia_utils_nodes", "aiia_utils_nodes")

# 5. 处理 aiia_video_nodes.py (新增部分)
_load_nodes_from_module(".aiia_video_nodes", "aiia_video_nodes")

# 6. 处理 aiia_browser_node.py (新增 - 确保HTTP路由被注册)
_load_nodes_from_module(".aiia_browser_node", "aiia_browser_node")

# 7. 处理 aiia_personalive_nodes.py
_load_nodes_from_module(".aiia_personalive_nodes", "aiia_personalive_nodes")

# 8. 处理 aiia_audio_isolator.py
_load_nodes_from_module(".aiia_audio_isolator", "aiia_audio_isolator")

# 9. 处理 aiia_audio_merger.py
_load_nodes_from_module(".aiia_audio_merger", "aiia_audio_merger")

# 10. 处理 aiia_cosyvoice_nodes.py
_load_nodes_from_module(".aiia_cosyvoice_nodes", "aiia_cosyvoice_nodes")

# 11. 处理 aiia_audio_debug.py (新增调试节点)
_load_nodes_from_module(".aiia_audio_debug", "aiia_audio_debug")

# 11. 处理 aiia_audio_processor.py
_load_nodes_from_module(".aiia_audio_processor", "aiia_audio_processor")

# 告诉 ComfyUI 这个节点包有一个包含网页资源的 'js' 目录
WEB_DIRECTORY = "js"

# --- 最终导出 ---
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

if not NODE_CLASS_MAPPINGS:
    print("严重警告: [ComfyUI_AIIA __init__] 未从任何模块加载到节点类映射。AIIA 节点将不可用。")
else:
    loaded_node_count = len(NODE_CLASS_MAPPINGS)
    print(f"--- ComfyUI_AIIA 自定义节点包: 成功处理导入。最终 NODE_CLASS_MAPPINGS 包含 {loaded_node_count} 个条目 ---")
    # 你可以在这里打印加载了哪些类，用于调试
    # print("已加载的节点类:", list(NODE_CLASS_MAPPINGS.keys()))
    # print("已加载的显示名称:", list(NODE_DISPLAY_NAME_MAPPINGS.keys()))