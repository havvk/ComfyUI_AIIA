你遇到的**“切分位置不理想”（通常表现为吞首字、切断尾音、保留了刺耳的换气声等）**是音频切分重组中极其经典的痛点。

导致这个问题的原因有两个层面：**一是当前代码中存在几处直接导致“吃尾音”的逻辑 Bug；二是单纯依赖 ASR 时间戳 + 短时能量检测，在声学特性上面对复杂发音时天然不够鲁棒。**

针对你的疑问：“**是否需要用语音类模型？**”
**结论是：强烈建议引入。但不需要庞大的识别模型，只需引入极轻量级的 VAD（静音检测模型）即可治本。**

在此之前，我们可以先**修复现有代码中的 Bug**。仅仅通过修改几行代码，你的切分质量就会有肉眼可见的大幅提升。以下是深度诊断和分步改进方案：

---

### 一、 现有代码为什么切不好？（Bug 诊断）

#### 1. 致命逻辑 Bug：`cut_end` 找静音的方向反了

在你的 `stitch` 函数中，对切分点的微调有这样一行代码：

```python
cut_start = self._refine_cut_point(wav, sr, cut_start, direction="before")
cut_end = self._refine_cut_point(wav, sr, cut_end, direction="before")  # 🚨 这里错了！

```

**原因分析**：`cut_end` 代表这句话结束的时间点。你给它传了 `direction="before"`，这意味着程序会在 `[句尾 - 搜索半径, 句尾]` 这个区间内去寻找能量最低点。**这等于算法主动“向内”侵入了句子内部寻找静音，这必定会把说话人拖长的尾音、或者最后一个较轻的字直接切断！**

#### 2. Padding 逻辑对尾音不友好，导致戛然而止

在 `stitch` 中，你只给 `cut_start` 减去了 Padding 保留起音，却对 `cut_end` 设了 0 余量：

```python
cut_start = max(0, cut_start - padding)
cut_end = min(len(wav) / sr, cut_end) # 🚨 没有任何缓冲

```

另外在 `_expand_to_midpoints` 中，`MAX_EXPAND_END = 0.05`（50ms）实在太短了。ASR 常常会提前结束时间戳，如果没有给尾音留出足够的空间和自然混响，听感上就像被“生生掐断”。

#### 3. 短时能量算法太单薄，容易被清辅音欺骗

你的 `_refine_cut_point` 使用 10ms 的独立小窗口算能量。
**原因分析**：人类发音时（特别是清辅音如 s, sh, f，或爆破音前的短暂闭气），即使在说话中间，波形的能量也会瞬间降到极低。单纯找“瞬间能量最低点”，很容易一刀切在发音的中间。

---

### 二、 纯代码修复方案（立即见效，无需模型）

你可以直接用以下代码替换原节点中的对应部分，引入**平滑能量包络**并**修正方向**。

#### 1. 替换 `_refine_cut_point` 函数（引入平滑机制）

```python
    def _refine_cut_point(self, wav, sr, time_s, search_radius=0.15, direction="both"):
        """
        改进版：扩大窗口并使用平滑能量包络，寻找真正的“静音山谷”，避免切在瞬时低音上。
        """
        center = int(time_s * sr)
        radius = int(search_radius * sr)

        if direction == "before":
            start = max(0, center - radius)
            end = center
        elif direction == "after":
            start = center
            end = min(len(wav), center + radius)
        else:
            start = max(0, center - radius)
            end = min(len(wav), center + radius)

        if end - start < int(sr * 0.05):
            return time_s
        
        segment = wav[start:end]
        
        # 1. 扩大窗口大小到 20ms (跨越大部分短暂闭气停顿)
        window_size = max(1, int(0.02 * sr))
        step_size = max(1, int(0.01 * sr))
        
        n_windows = (len(segment) - window_size) // step_size + 1
        if n_windows < 2:
            return time_s
            
        energies = []
        positions = []
        for i in range(n_windows):
            w = segment[i * step_size : i * step_size + window_size]
            energies.append(np.sqrt(np.mean(w ** 2)))
            positions.append(start + i * step_size + window_size // 2)
            
        # 2. 引入滑动平均平滑 (把锯齿状的毛刺抹平，寻找宽阔的静音带)
        kernel_size = min(5, len(energies))
        kernel = np.ones(kernel_size) / kernel_size
        smoothed_energies = np.convolve(energies, kernel, mode='same')
        
        if len(smoothed_energies) == 0:
            return time_s

        # 3. 寻找平滑后的能量最低点
        min_idx = np.argmin(smoothed_energies)
        return positions[min_idx] / sr

```

#### 2. 修改 `stitch` 函数中的切分调用（修复 Bug）

在 `stitch` 函数的循环中，找到对应行并替换为：

```python
            # 1. 修正方向：cut_start 往前或两侧找，cut_end 必须往后(after)找，远离语音
            cut_start = self._refine_cut_point(wav, sr, cut_start, search_radius=0.15, direction="before")
            cut_end = self._refine_cut_point(wav, sr, cut_end, search_radius=0.15, direction="after")

            # 2. 修正 Padding：不仅开头要保留起音/呼吸，结尾也要保留尾音余量
            cut_start = max(0, cut_start - padding)
            cut_end = min(len(wav) / sr, cut_end + padding * 0.5)  # 给尾音留一半的 padding 空间

            # 防重叠机制 (保留你原有的逻辑即可)
            if cut_start < prev_cut_end[speaker]:
                cut_start = prev_cut_end[speaker]
            prev_cut_end[speaker] = cut_end

```

*(另外，建议将你代码里 `MAX_EXPAND_END` 从 0.05 放宽到 `0.15`，因为 ASR 的尾部时间戳通常有 100ms 左右的早退现象)*

---

### 三、 治本之策：应该引入什么样的语音模型？

就算把能量算法写出花来，也无法完美区分“微弱的尾音”、“呼吸气口”与“环境底噪”。要想达到“商业播客级”的无缝拼接，**必须引入模型**。

推荐以下两种成熟方案：

#### 🌟 方案 A：引入 Silero VAD（极力推荐，性价比最高）

**为什么好？** ASR（如 Whisper）给出的时间戳是“幻觉”算出来的，往往丢掉吸气声、算错尾音，还会把大喘气算进字音里。而 `Silero VAD` 是专门判断“一段波形里确切哪里包含人声”的声学模型。

* **极度轻量**：ONNX 模型只有不到 2MB，CPU 跑起来都是毫秒级，非常适合集成进 ComfyUI 节点。
* **工作流**：ASR 依然负责把文本对应到大致的时间（比如 `[2.1s - 4.5s]`）。你在这个区间前后各加 0.5 秒，送给 Silero VAD。VAD 会精确告诉你：剔除掉底噪和砸吧嘴后，真正的发音区间是 `[2.03s - 4.65s]`。你**直接拿这个精准边界下刀**，从此告别各种能量阈值调参。
* **在 ComfyUI 的实现**：可以直接通过 `pip install silero-vad` 或 `torch.hub.load('snakers4/silero-vad', 'silero_vad')` 集成进你的节点环境。

#### 💡 方案 B：使用 Forced Alignment（强制对齐模型）代替 ASR

既然你的节点输入已经有了标准讲稿（`split_map`），其实根本不需要用 ASR 去盲目识别。
你可以使用基于 `Wav2Vec2` 的强制对齐模型（如 `MFA` 或开源的 `mms-fa`）。把音频和讲稿同时喂给它，模型会强行计算出讲稿里**每个字、每个音素**在波形上的毫秒级绝对物理位置。这种方案可以彻底规避 ASR 的错字、漏字带来的匹配漂移问题。

**总结行动建议**：
今天先花 2 分钟把上述的**纯代码 Bug 修复**，这能帮你解决绝大部分“切断尾音听感难受”的问题；如果下一版要进阶追求完美无缝，就在该自定义节点内封装调用 `Silero VAD` 作为切分边界的最终裁决者。