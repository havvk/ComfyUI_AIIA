# ComfyUI_AIIA TODO

## Podcast / TTS 增强

- [ ] **LLM 驱动的情感标注**：使用轻量 LLM 分析输入对白文本，自动为每句添加情感控制标签（如 `excited`、`calm`、`serious`），用于控制 TTS 模型的情感表达。适用于支持情感标签的模型（CosyVoice v2/v3、F5-TTS 等），配合 Split/Stitch 流程实现逐句情感分配。

## Stitch / 切分精度

- [ ] **Silero VAD 集成**：用极轻量 VAD 模型（~2MB ONNX）替代纯能量检测，作为切分边界的最终裁决者。ASR 负责粗定位，VAD 精确修正语音起止点，彻底解决尾音误切、清辅音误判等问题。`pip install silero-vad` 或 `torch.hub.load('snakers4/silero-vad', 'silero_vad')`。
- [ ] **Forced Alignment（强制对齐）**：用 Wav2Vec2 基础的对齐模型（如 MFA / mms-fa）替代 ASR 做时间对齐。输入音频+讲稿，输出每个字/音素的毫秒级位置，彻底避免 ASR 错字/漏字的匹配漂移。
