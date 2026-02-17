"""
AIIA JSON Extractor — 从 STRING 中提取 JSON 某个 key 的值

支持嵌套 key 路径 (e.g. "data.items[0].name")，
异常输入时返回 fallback 值而非崩溃。
"""
import json
import re

class AIIA_JSON_Extractor:
    """从 JSON 字符串中按 key 路径提取值，返回 STRING。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_string": ("STRING", {"forceInput": True}),
                "key_path": ("STRING", {
                    "default": "",
                    "tooltip": "提取路径，支持嵌套和数组索引。例: name, data.items[0].text, [2].speaker"
                }),
            },
            "optional": {
                "fallback": ("STRING", {
                    "default": "",
                    "tooltip": "解析失败或 key 不存在时返回的默认值"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT", "BOOLEAN",)
    RETURN_NAMES = ("value", "value_int", "value_float", "found",)
    FUNCTION = "extract"
    CATEGORY = "AIIA/Utils"

    @staticmethod
    def _parse_path(key_path):
        """
        Parse a dot-separated key path with optional array indices.
        Examples:
            "name"              -> ["name"]
            "data.items[0]"     -> ["data", "items", 0]
            "[2].speaker"       -> [2, "speaker"]
            "lines[0].text"     -> ["lines", 0, "text"]
        """
        if not key_path or not key_path.strip():
            return []

        tokens = []
        # Split by '.' first, then handle [N] inside each part
        for part in key_path.strip().split('.'):
            if not part:
                continue
            # Match segments like "items[0]" or "[3]"
            sub_parts = re.findall(r'([^\[\]]+)|\[(\d+)\]', part)
            for name, idx in sub_parts:
                if idx != '':
                    tokens.append(int(idx))
                elif name:
                    tokens.append(name)
        return tokens

    @staticmethod
    def _navigate(data, tokens):
        """Walk into data following tokens. Returns (value, found)."""
        current = data
        for token in tokens:
            try:
                if isinstance(token, int):
                    if isinstance(current, (list, tuple)) and -len(current) <= token < len(current):
                        current = current[token]
                    else:
                        return None, False
                elif isinstance(current, dict):
                    if token in current:
                        current = current[token]
                    else:
                        return None, False
                else:
                    return None, False
            except (KeyError, IndexError, TypeError):
                return None, False
        return current, True

    def extract(self, json_string, key_path, fallback=""):
        log = "[AIIA JSON Extractor]"

        # --- 1. Parse JSON ---
        data = None
        try:
            data = json.loads(json_string)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"{log} JSON 解析失败: {e}")
            # Try to recover: strip leading/trailing whitespace, BOM, etc.
            if isinstance(json_string, str):
                cleaned = json_string.strip().lstrip('\ufeff')
                try:
                    data = json.loads(cleaned)
                    print(f"{log} 清理后解析成功")
                except Exception:
                    pass

        if data is None:
            print(f"{log} 无法解析 JSON，返回 fallback: {fallback[:50]}")
            return (fallback, 0, 0.0, False)

        # --- 2. Navigate key path ---
        tokens = self._parse_path(key_path)

        if not tokens:
            # No key path: return the entire JSON re-serialized
            if isinstance(data, (dict, list)):
                result_str = json.dumps(data, ensure_ascii=False, indent=2)
            else:
                result_str = str(data)
            return (result_str, self._safe_int(data), self._safe_float(data), True)

        value, found = self._navigate(data, tokens)

        if not found:
            print(f"{log} key '{key_path}' 未找到，返回 fallback")
            return (fallback, 0, 0.0, False)

        # --- 3. Convert to output types ---
        if isinstance(value, (dict, list)):
            value_str = json.dumps(value, ensure_ascii=False, indent=2)
        elif value is None:
            value_str = ""
        elif isinstance(value, bool):
            value_str = str(value).lower()
        else:
            value_str = str(value)

        return (value_str, self._safe_int(value), self._safe_float(value), True)

    @staticmethod
    def _safe_int(value):
        try:
            if isinstance(value, bool):
                return 1 if value else 0
            return int(value)
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def _safe_float(value):
        try:
            if isinstance(value, bool):
                return 1.0 if value else 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0


class AIIA_JSON_Builder:
    """将多个 STRING 输入组装为 JSON 对象字符串。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key_1": ("STRING", {"default": "key1"}),
                "value_1": ("STRING", {"default": "", "forceInput": True}),
            },
            "optional": {
                "key_2": ("STRING", {"default": ""}),
                "value_2": ("STRING", {"default": "", "forceInput": True}),
                "key_3": ("STRING", {"default": ""}),
                "value_3": ("STRING", {"default": "", "forceInput": True}),
                "key_4": ("STRING", {"default": ""}),
                "value_4": ("STRING", {"default": "", "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_string",)
    FUNCTION = "build"
    CATEGORY = "AIIA/Utils"

    def build(self, key_1, value_1, **kwargs):
        result = {}
        pairs = [(key_1, value_1)]
        for i in range(2, 5):
            k = kwargs.get(f"key_{i}", "")
            v = kwargs.get(f"value_{i}", "")
            if k and k.strip():
                pairs.append((k.strip(), v))

        for key, val in pairs:
            # Try to parse value as JSON (for nested objects/arrays/numbers/bools)
            try:
                parsed = json.loads(val)
                result[key] = parsed
            except (json.JSONDecodeError, TypeError):
                result[key] = val

        return (json.dumps(result, ensure_ascii=False, indent=2),)


NODE_CLASS_MAPPINGS = {
    "AIIA_JSON_Extractor": AIIA_JSON_Extractor,
    "AIIA_JSON_Builder": AIIA_JSON_Builder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_JSON_Extractor": "AIIA JSON Extractor 🔑",
    "AIIA_JSON_Builder": "AIIA JSON Builder 🏗️",
}
