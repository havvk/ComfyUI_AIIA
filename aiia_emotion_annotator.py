"""
AIIA Emotion Annotator - LLM 驱动的情感标注节点

使用 OpenAI-compatible API（Groq / Ollama / vLLM）自动为对话剧本标注情感。
插入 Script Parser 和 Dialogue TTS 之间，直接修改 dialogue_json 的 emotion 字段。
"""

import json
import os
import urllib.request
import urllib.error
import ssl

# 情感标签列表（与 AIIA_EMOTION_LIST 对齐，但使用英文 key）
EMOTION_TAGS = [
    "neutral", "happy", "sad", "angry", "excited", "gentle",
    "fearful", "surprised", "disappointed", "serious", "calm",
    "romantic", "sarcastic", "proud", "confused", "anxious",
    "disgusted", "nostalgic", "mysterious", "enthusiastic", "lazy",
    "gossip", "innocent", "nervous"
]

# 英文 → 中文显示名映射（用于日志）
EMOTION_DISPLAY = {
    "neutral": "中性", "happy": "开心", "sad": "悲伤", "angry": "愤怒",
    "excited": "兴奋", "gentle": "温柔", "fearful": "恐惧", "surprised": "惊讶",
    "disappointed": "失望", "serious": "严肃", "calm": "平静", "romantic": "浪漫",
    "sarcastic": "讽刺", "proud": "自豪", "confused": "困惑", "anxious": "焦虑",
    "disgusted": "厌恶", "nostalgic": "怀旧", "mysterious": "神秘",
    "enthusiastic": "热情", "lazy": "慵懒", "gossip": "八卦", "innocent": "天真",
    "nervous": "紧张"
}

# 情感标签 → CosyVoice / Qwen3 兼容格式
EMOTION_TO_TAG = {
    "neutral": None,  # neutral 不注入标签
    "happy": "Happy", "sad": "Sad", "angry": "Angry", "excited": "Excited",
    "gentle": "Gentle", "fearful": "Fearful", "surprised": "Surprised",
    "disappointed": "Disappointed", "serious": "Serious", "calm": "Calm",
    "romantic": "Romantic", "sarcastic": "Sarcastic", "proud": "Proud",
    "confused": "Confused", "anxious": "Anxious", "disgusted": "Disgusted",
    "nostalgic": "Nostalgic", "mysterious": "Mysterious",
    "enthusiastic": "Enthusiastic", "lazy": "Lazy tone",
    "gossip": "Gossip tone", "innocent": "Innocent", "nervous": "Nervous"
}

PROMPT_TEMPLATE = """你是一位专业的有声读物情感标注师。分析以下对话剧本，为每句台词标注最合适的情感。

可选标签（只能从中选一个）：
{tags}

规则：
1. 结合上下文语境整体判断，不要只看单句
2. 日常对话、陈述句多为 neutral，不要过度标注
3. 只输出纯 JSON 数组，不要有任何解释文字

剧本：
{lines}

严格按此格式输出（line 从 0 开始）：
[{{"line":0,"emotion":"neutral"}},{{"line":1,"emotion":"happy"}}]"""

MODEL_LIST = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "qwen-qwq-32b",
    "deepseek-r1-distill-llama-70b",
    "gemma2-9b-it",
]


class AIIA_Emotion_Annotator:
    """
    使用 LLM 自动为对话剧本标注情感标签。
    支持 Groq / Ollama / vLLM 等 OpenAI-compatible API。
    """

    NODE_NAME = "AIIA Emotion Annotator"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dialogue_json": ("STRING", {"forceInput": True}),
                "model": (MODEL_LIST, {"default": "llama-3.1-8b-instant"}),
                "override_mode": (["skip_existing", "overwrite_all"], {"default": "skip_existing"}),
            },
            "optional": {
                "api_base_url": ("STRING", {
                    "default": "https://api.groq.com/openai/v1",
                    "tooltip": "OpenAI-compatible API base URL. Examples:\n"
                               "  Groq: https://api.groq.com/openai/v1\n"
                               "  Ollama: http://localhost:11434/v1\n"
                               "  vLLM: http://localhost:8000/v1"
                }),
                "api_key_override": ("STRING", {
                    "default": "",
                    "tooltip": "可选。留空则使用环境变量 GROQ_API_KEY"
                }),
                "custom_model": ("STRING", {
                    "default": "",
                    "tooltip": "自定义模型名（覆盖下拉选择），用于 Ollama/vLLM 本地模型"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("dialogue_json", "annotation_log")
    FUNCTION = "annotate"
    CATEGORY = "AIIA/Podcast"

    def _get_api_key(self, api_key_override=""):
        """获取 API Key：优先使用 override，否则读取环境变量"""
        if api_key_override and api_key_override.strip():
            return api_key_override.strip()
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            # 尝试从 ~/run.sh 读取（兼容服务器环境）
            run_sh = os.path.expanduser("~/run.sh")
            if os.path.exists(run_sh):
                try:
                    with open(run_sh, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("export GROQ_API_KEY="):
                                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                                break
                except Exception:
                    pass
        return key

    def _call_llm(self, api_base_url, api_key, model, prompt):
        """调用 OpenAI-compatible API"""
        log = f"[{self.NODE_NAME}]"
        url = f"{api_base_url.rstrip('/')}/chat/completions"

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 2048,
        }

        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        # 支持自签名证书的本地服务
        ctx = ssl.create_default_context()
        if "localhost" in api_base_url or "127.0.0.1" in api_base_url:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

        try:
            # 读取代理设置（从环境变量）
            proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or os.environ.get("ALL_PROXY")
            if proxy_url and "localhost" not in api_base_url and "127.0.0.1" not in api_base_url:
                proxy_handler = urllib.request.ProxyHandler({
                    "https": proxy_url,
                    "http": proxy_url
                })
                opener = urllib.request.build_opener(
                    proxy_handler,
                    urllib.request.HTTPSHandler(context=ctx)
                )
            else:
                opener = urllib.request.build_opener(
                    urllib.request.HTTPSHandler(context=ctx)
                )

            with opener.open(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"]
                return content, None
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            return None, f"HTTP {e.code}: {err_body[:200]}"
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    def _parse_llm_response(self, raw_text, line_count):
        """从 LLM 响应中提取 JSON 数组"""
        # 尝试直接解析
        text = raw_text.strip()

        # 去除 markdown 代码块包裹
        if text.startswith("```"):
            lines = text.split("\n")
            # 去首尾 ```
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # 提取 JSON 数组部分
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            return None, f"未找到 JSON 数组: {text[:100]}"

        json_str = text[start:end + 1]
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError as e:
            return None, f"JSON 解析失败: {e}\n原始: {json_str[:200]}"

        if not isinstance(result, list):
            return None, f"期望 JSON 数组，得到 {type(result)}"

        # 验证并规范化
        annotations = {}
        for item in result:
            if isinstance(item, dict) and "line" in item and "emotion" in item:
                line_idx = item["line"]
                emo = item["emotion"].lower().strip()
                if emo in EMOTION_TAGS:
                    annotations[line_idx] = emo
                else:
                    # 模糊匹配：如果 LLM 返回了中文或变体
                    for tag in EMOTION_TAGS:
                        if tag in emo or emo in tag:
                            annotations[line_idx] = tag
                            break
                    else:
                        annotations[line_idx] = "neutral"  # 无法识别则 fallback

        return annotations, None

    def annotate(self, dialogue_json, model, override_mode,
                 api_base_url="https://api.groq.com/openai/v1",
                 api_key_override="", custom_model=""):
        log = f"[{self.NODE_NAME}]"
        logs = []

        # 解析 dialogue_json
        try:
            dialogue = json.loads(dialogue_json)
        except json.JSONDecodeError as e:
            error_msg = f"{log} JSON 解析失败: {e}"
            print(error_msg)
            return (dialogue_json, error_msg)

        if not isinstance(dialogue, list):
            error_msg = f"{log} 错误: dialogue_json 不是列表"
            print(error_msg)
            return (dialogue_json, error_msg)

        # 收集需要标注的句子
        speech_items = []
        speech_indices = []  # 在 dialogue 中的原始索引
        for i, item in enumerate(dialogue):
            if item.get("type") == "speech":
                existing_emotion = item.get("emotion")
                if override_mode == "overwrite_all" or not existing_emotion:
                    speech_items.append(item)
                    speech_indices.append(i)

        if not speech_items:
            msg = f"{log} 所有句子已有情感标注，跳过 (mode={override_mode})"
            print(msg)
            return (dialogue_json, msg)

        # 构造 prompt
        dialogue_lines = []
        for idx, item in enumerate(speech_items):
            speaker = item.get("speaker", "?")
            text = item.get("text", "")
            dialogue_lines.append(f"[{idx}] {speaker}: {text}")

        prompt = PROMPT_TEMPLATE.format(
            tags=", ".join(EMOTION_TAGS),
            lines="\n".join(dialogue_lines)
        )

        # 获取 API Key
        api_key = self._get_api_key(api_key_override)
        actual_model = custom_model.strip() if custom_model and custom_model.strip() else model
        actual_base = api_base_url.strip() if api_base_url else "https://api.groq.com/openai/v1"

        # 非 Groq 的本地服务可能不需要 key
        is_local = "localhost" in actual_base or "127.0.0.1" in actual_base
        if not api_key and not is_local:
            error_msg = (f"{log} 未找到 API Key。请设置环境变量 GROQ_API_KEY "
                         f"或在节点参数中填入 api_key_override")
            print(error_msg)
            return (dialogue_json, error_msg)

        logs.append(f"模型: {actual_model}")
        logs.append(f"API: {actual_base}")
        logs.append(f"待标注: {len(speech_items)} 句")
        print(f"{log} 正在调用 LLM ({actual_model}) 标注 {len(speech_items)} 句情感...")

        # 调用 LLM
        raw_response, api_error = self._call_llm(actual_base, api_key, actual_model, prompt)

        if api_error:
            error_msg = f"{log} API 调用失败: {api_error}"
            print(error_msg)
            logs.append(f"❌ API 错误: {api_error}")
            return (dialogue_json, "\n".join(logs))

        # 解析响应
        annotations, parse_error = self._parse_llm_response(raw_response, len(speech_items))

        if parse_error:
            error_msg = f"{log} 响应解析失败: {parse_error}"
            print(error_msg)
            logs.append(f"❌ 解析错误: {parse_error}")
            logs.append(f"原始响应: {raw_response[:300]}")
            return (dialogue_json, "\n".join(logs))

        # 合并情感标注到 dialogue_json
        annotated_count = 0
        for local_idx, emo in annotations.items():
            if local_idx < len(speech_indices):
                global_idx = speech_indices[local_idx]
                tag = EMOTION_TO_TAG.get(emo)
                if tag:  # neutral = None → 不注入
                    dialogue[global_idx]["emotion"] = tag
                    annotated_count += 1
                    display = EMOTION_DISPLAY.get(emo, emo)
                    speaker = dialogue[global_idx].get("speaker", "?")
                    text_preview = dialogue[global_idx].get("text", "")[:20]
                    logs.append(f"  [{tag}] {speaker}: {text_preview}...")
                else:
                    # neutral: 清除已有标签（如果是 overwrite 模式）
                    if override_mode == "overwrite_all":
                        dialogue[global_idx]["emotion"] = None

        logs.insert(0, f"✅ 标注完成: {annotated_count}/{len(speech_items)} 句获得情感标签")
        summary = "\n".join(logs)
        print(f"{log} {logs[0]}")

        result_json = json.dumps(dialogue, ensure_ascii=False, indent=2)
        return (result_json, summary)


# --- ComfyUI 节点注册 ---
NODE_CLASS_MAPPINGS = {
    "AIIA_Emotion_Annotator": AIIA_Emotion_Annotator,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_Emotion_Annotator": "🎭 AIIA Emotion Annotator (LLM)",
}
