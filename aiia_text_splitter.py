"""
AIIA Text Splitter — 单人文本按标点拆分为 dialogue_json
[v1.13.0 New]

将长文本按句号、问号、感叹号等标点拆分为标准 dialogue_json 格式，
可直接接入 Emotion Annotator → TTS 管线。

支持短句合并（避免碎片）和长句拆分（避免 TTS 单句过长）。
"""

import json
import re


class AIIA_Text_Splitter:
    """
    将纯文本按标点拆分为 dialogue_json 格式。

    支持三种拆分模式：
    - auto: 中英文自动，按句末标点拆分 + 短句合并 + 长句拆分
    - by_sentence: 仅按句号/问号/感叹号拆分
    - by_line: 按换行拆分
    """

    NODE_NAME = "AIIA Text Splitter"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "待拆分的文本。支持多段落、多行。"
                }),
                "speaker_name": ("STRING", {
                    "default": "Narrator",
                    "tooltip": "说话人名称，写入 dialogue_json 的 speaker 字段"
                }),
                "split_mode": (["auto", "by_sentence", "by_line"], {
                    "default": "auto",
                    "tooltip": "拆分模式：\n"
                               "  auto: 按句末标点拆分 + 短句合并 + 长句再拆\n"
                               "  by_sentence: 仅按句号/问号/感叹号拆分\n"
                               "  by_line: 按换行拆分"
                }),
            },
            "optional": {
                "min_chars": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 50,
                    "tooltip": "最小字符数。短于此的句子合并到前一句。"
                }),
                "max_chars": ("INT", {
                    "default": 100,
                    "min": 20,
                    "max": 500,
                    "tooltip": "最大字符数。超长句子在逗号/分号处强制拆分。"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("dialogue_json", "sentence_count")
    FUNCTION = "split_text"
    CATEGORY = "AIIA/Podcast"

    def split_text(self, text, speaker_name="Narrator", split_mode="auto",
                   min_chars=4, max_chars=100):
        """拆分文本为 dialogue_json 格式。"""

        if not text or not text.strip():
            empty = json.dumps([], ensure_ascii=False)
            return (empty, 0)

        text = text.strip()

        if split_mode == "by_line":
            raw_sentences = self._split_by_line(text)
        elif split_mode == "by_sentence":
            raw_sentences = self._split_by_sentence(text)
        else:  # auto
            raw_sentences = self._split_auto(text, min_chars, max_chars)

        # 构建 dialogue_json
        dialogue = []
        for sent in raw_sentences:
            sent = sent.strip()
            if not sent:
                continue
            dialogue.append({
                "type": "speech",
                "speaker": speaker_name,
                "text": sent,
                "emotion": None
            })

        result = json.dumps(dialogue, ensure_ascii=False, indent=2)
        return (result, len(dialogue))

    def _split_by_line(self, text):
        """按换行拆分，空行跳过。"""
        return [line.strip() for line in text.split("\n") if line.strip()]

    def _split_by_sentence(self, text):
        """仅按句末标点拆分（句号、问号、感叹号、省略号）。"""
        # 先按换行分段，再段内按标点拆分
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        sentences = []
        for para in paragraphs:
            parts = self._split_at_sentence_end(para)
            sentences.extend(parts)
        return sentences

    def _split_auto(self, text, min_chars, max_chars):
        """
        智能拆分：
        1. 按换行分段
        2. 段内按句末标点拆分
        3. 短句合并
        4. 长句在逗号/分号处再拆
        """
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        all_sentences = []

        for para in paragraphs:
            # Step 1: 按句末标点拆分
            raw = self._split_at_sentence_end(para)

            # Step 2: 短句合并
            merged = self._merge_short(raw, min_chars)

            # Step 3: 长句拆分
            final = []
            for sent in merged:
                if len(sent) > max_chars:
                    final.extend(self._split_long(sent, max_chars))
                else:
                    final.append(sent)

            all_sentences.extend(final)

        return all_sentences

    def _split_at_sentence_end(self, text):
        """
        在句末标点处拆分，保留标点在前一句末尾。
        支持：。！？!?  以及省略号 …… / ...
        """
        # 按句末标点拆分，保留分隔符
        # 匹配：句号/问号/感叹号（中英文），以及省略号
        parts = re.split(r'((?:\.{3}|…{1,2}|[。！？!?]))', text)

        sentences = []
        buffer = ""
        for i, part in enumerate(parts):
            if i % 2 == 0:
                # 正文部分
                buffer += part
            else:
                # 标点部分，附加到 buffer
                buffer += part
                if buffer.strip():
                    sentences.append(buffer.strip())
                buffer = ""

        # 处理末尾没有标点的残余
        if buffer.strip():
            sentences.append(buffer.strip())

        return sentences

    def _merge_short(self, sentences, min_chars):
        """将短于 min_chars 的句子合并到前一句。"""
        if not sentences:
            return sentences

        merged = [sentences[0]]
        for sent in sentences[1:]:
            if len(sent) < min_chars and merged:
                # 合并到前一句
                merged[-1] = merged[-1] + sent
            else:
                merged.append(sent)

        return merged

    def _split_long(self, text, max_chars):
        """
        将超长句子在逗号/分号/顿号处拆分。
        尽量靠近 max_chars 的位置切，避免太碎。
        """
        # 可选拆分点：逗号、分号、顿号（中英文）
        split_points = []
        for i, ch in enumerate(text):
            if ch in '，,；;、':
                split_points.append(i)

        if not split_points:
            # 没有可拆分点，原样返回
            return [text]

        result = []
        start = 0
        for point in split_points:
            segment = text[start:point + 1]
            if len(segment) >= max_chars // 2:
                # 够长了，切一刀
                result.append(segment.strip())
                start = point + 1

        # 追加剩余部分
        remainder = text[start:].strip()
        if remainder:
            if result and len(remainder) < max_chars // 4:
                # 残余太短，合并到最后一段
                result[-1] = result[-1] + remainder
            else:
                result.append(remainder)

        return result if result else [text]


# === ComfyUI Registration ===
NODE_CLASS_MAPPINGS = {
    "AIIA_Text_Splitter": AIIA_Text_Splitter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_Text_Splitter": "💬 AIIA Text Splitter",
}
