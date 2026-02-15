# ComfyUI_AIIA TODO

## Podcast / TTS 增强

- [ ] **LLM 驱动的情感标注**：使用轻量 LLM 分析输入对白文本，自动为每句添加情感控制标签（如 `excited`、`calm`、`serious`），用于控制 TTS 模型的情感表达。适用于支持情感标签的模型（CosyVoice v2/v3、F5-TTS 等），配合 Split/Stitch 流程实现逐句情感分配。

## Stitch / 切分精度

- [x] **Silero VAD 集成**：用极轻量 VAD 模型（~2MB ONNX）替代纯能量检测，作为切分边界的最终裁决者。ASR 负责粗定位，VAD 精确修正语音起止点，彻底解决尾音误切、清辅音误判等问题。`pip install silero-vad` 或 `torch.hub.load('snakers4/silero-vad', 'silero_vad')`。
- [x] **Forced Alignment（强制对齐）**：用 MMS_FA 模型（torchaudio）实现字级强制对齐。混合策略：FA 精确起点 + 能量检测自然收尾，中文文本自动转拼音。支持 FA/VAD/Energy 三方法交叉验证（IoU），带重叠防护和尾音保护。
