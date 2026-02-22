import json


class AIIA_Podcast_Splitter:
    """
    将多角色对话脚本按说话人拆分为独立文本。
    
    输入: Script Parser 输出的 dialogue_json
    输出: 每个说话人的拼接文本（用于独立 TTS）+ 重组映射表
    """

    NODE_NAME = "AIIA Podcast Splitter"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dialogue_json": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("speaker_A_text", "speaker_B_text", "split_map",)
    FUNCTION = "split_dialogue"
    CATEGORY = "AIIA/Podcast"

    def split_dialogue(self, dialogue_json):
        log = f"[{self.NODE_NAME}]"

        # 解析 dialogue_json
        try:
            dialogue = json.loads(dialogue_json)
        except json.JSONDecodeError as e:
            print(f"{log} JSON 解析失败: {e}")
            empty_map = json.dumps([], ensure_ascii=False)
            return ("", "", empty_map)

        if not isinstance(dialogue, list):
            print(f"{log} 错误: dialogue_json 不是列表")
            empty_map = json.dumps([], ensure_ascii=False)
            return ("", "", empty_map)

        # 收集所有说话人
        speakers_seen = []
        for item in dialogue:
            if item.get("type") == "speech":
                spk = item["speaker"]
                if spk not in speakers_seen:
                    speakers_seen.append(spk)

        if len(speakers_seen) == 0:
            print(f"{log} 警告: 没有找到任何说话人")
            empty_map = json.dumps([], ensure_ascii=False)
            return ("", "", empty_map)

        if len(speakers_seen) > 2:
            print(f"{log} 警告: 发现 {len(speakers_seen)} 个说话人 ({speakers_seen})，仅使用前两个")

        # 按名称排序，确保 speaker_A/speaker_B 的分配不受对话顺序影响
        speakers_seen.sort()

        speaker_A = speakers_seen[0] if len(speakers_seen) > 0 else None
        speaker_B = speakers_seen[1] if len(speakers_seen) > 1 else None

        print(f"{log} Speaker A: {speaker_A}, Speaker B: {speaker_B}")

        # 按说话人分组，同时记录顺序映射
        texts_A = []  # Speaker A 的所有台词
        texts_B = []  # Speaker B 的所有台词
        split_map = []  # 原始顺序映射

        for item in dialogue:
            if item.get("type") != "speech":
                # 暂停等非语音条目也记录到 split_map
                if item.get("type") == "pause":
                    split_map.append({
                        "type": "pause",
                        "duration": item.get("duration", 0.3),
                    })
                continue

            text = item["text"]
            speaker = item["speaker"]
            emotion = item.get("emotion")

            # 如果有情感标签（来自 Emotion Annotator），嵌入到文本中
            # 格式: [Happy] 台词... (CosyVoice 原生支持此格式)
            output_text = f"[{emotion}] {text}" if emotion else text

            if speaker == speaker_A:
                split_map.append({
                    "type": "speech",
                    "speaker": "A",
                    "index": len(texts_A),
                    "text": text,
                    "original_speaker": speaker,
                    "emotion": emotion,
                })
                texts_A.append(output_text)
            elif speaker == speaker_B:
                split_map.append({
                    "type": "speech",
                    "speaker": "B",
                    "index": len(texts_B),
                    "text": text,
                    "original_speaker": speaker,
                    "emotion": emotion,
                })
                texts_B.append(output_text)
            else:
                print(f"{log} 跳过第三个说话人 '{speaker}' 的台词: {text[:30]}...")

        # 拼接每个说话人的文本
        # 每句之间用换行分隔（TTS 会在换行处产生自然停顿）
        speaker_A_text = "\n\n".join(texts_A)
        speaker_B_text = "\n\n".join(texts_B)

        split_map_json = json.dumps(split_map, ensure_ascii=False, indent=2)

        print(f"{log} 拆分完成:")
        print(f"  Speaker A ({speaker_A}): {len(texts_A)} 句, {len(speaker_A_text)} 字符")
        print(f"  Speaker B ({speaker_B}): {len(texts_B)} 句, {len(speaker_B_text)} 字符")
        print(f"  split_map: {len(split_map)} 条目")

        return (speaker_A_text, speaker_B_text, split_map_json)


# --- ComfyUI 节点注册 ---
NODE_CLASS_MAPPINGS = {
    "AIIA_Podcast_Splitter": AIIA_Podcast_Splitter,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_Podcast_Splitter": "✂️ AIIA Podcast Splitter",
}
