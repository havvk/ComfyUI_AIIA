![image](https://github.com/user-attachments/assets/7e38d3fd-2084-4d0c-bf86-4b500eba5ced)

# AIIA Nodes for ComfyUI

欢迎来到 AIIA Nodes for ComfyUI 仓库！这是一个旨在为 ComfyUI 提供一系列强大、直观且高度可定制的节点的集合。这些节点专注于简化复杂的工作流，并为创意工作者提供最大的灵活性。

本节点套件致力于解决 ComfyUI 工作流中的核心痛点。

---

## 🚀 全新旗舰节点：AIIA 媒体浏览器 🚀

**还在费力地翻找 `output` 文件夹，或者对着一堆时间戳命名的文件猜内容吗？**

我们隆重推出 **AIIA 媒体浏览器**——一个完全集成在 ComfyUI 内部的、功能完备的媒体文件管理中心。它的诞生，旨在彻底改变你管理和使用生成结果的方式，让整个过程变得高效、直观且充满乐趣。
![image](https://github.com/user-attachments/assets/358b9ca9-59c8-4433-b84c-c150503af04a)

### ✨ 为何选择 AIIA 浏览器？

* **告别猜测：** 无需离开 ComfyUI 即可即时预览图片、视频和音频。
* **为性能而生：** 专为处理包含数千乃至数万个文件的海量输出文件夹而设计，极致流畅，拒绝卡顿。
* **精美且直观：** 拥有现代化的用户界面，与 ComfyUI 原生主题无缝融合，宛若天生。

### 核心亮点

#### 1. 强大的核心功能

* **一键加载工作流**: 直接从浏览器中加载任何图片、视频或工作流文件，瞬间恢复你的创作环境。
* **直接下载文件**: 无需再打开系统文件夹，一键下载你需要的任何文件。
* **内容丰富的工具提示 (Tooltip)**: 将鼠标悬停在任何文件或文件夹上，即可看到包含详细元数据（尺寸、时长、日期等）的浮窗，甚至还能 **直接在浮窗里预览图片、视频和音频**！
* **高级排序与筛选**: 按 **名称、日期、大小、类型、尺寸或时长** 对文件进行排序，并可选择隐藏文件夹，总能快速找到你想要的。

#### 2. 顶级的界面与用户体验

* **双视图模式：** 可在经典的 **图标视图**（支持缩放）和信息详尽的 **列表视图** 之间一键切换。
* **沉浸式全屏查看器：** 双击文件即可进入一个独立的媒体查看器。支持视频/音频播放、用于快速导航的胶片缩略图，并提供多种导航方式（鼠标滑动、滚轮、键盘方向键和长按按钮）。
* **全键盘导航：** 为高级用户设计，你可以完全通过键盘在文件夹之间穿梭、选择文件、打开预览、显示/隐藏工具提示，实现真正的“效率流”操作。
* **交互式路径与面包屑导航：** 通过可点击的面包屑轻松跳转目录，或直接输入路径进行导航。

#### 3. 无与伦比的性能与技术

* **高性能后端：** 基于 Python 的异步后端，利用 `ffmpeg` 和 `Pillow` 进行即时的缩略图与视频海报生成。
* **智能缓存：** 所有缩略图和元数据都会被智能缓存。浏览器仅在文件发生变化时才重新生成，确保后续加载瞬间完成。
* **懒加载与虚拟滚动：** 我们在图标视图中使用 `IntersectionObserver`，在列表视图中实现 **完全虚拟滚动**。这意味着浏览器永远只渲染屏幕上可见的内容，即使面对上万个文件也能保持极低的内存占用和零延迟。
* **无阻塞元数据分析：** 在列表视图中，浏览器会在后台批量分析视频时长、尺寸等元数据，并以进度条显示，完全不阻塞UI操作。

#### 4. 灵活的自定义选项

* 在设置面板中，你可以根据习惯隐藏文件夹侧边栏，或开关视频预览功能。所有设置都会被自动保存。

---

## 🚀 核心设计理念：为大规模生成而生

本节点套件的另一个核心设计初衷，是为了攻克困扰许多用户的技术难题，**特别是解决在生成长视频或处理大量图像帧时常见的内存不足（OOM）问题**。

### 🌟 告别OOM：为处理海量帧而生

处理成百上千张高清图像帧时，轻易就会耗尽 VRAM 和系统内存，导致工作流中断。AIIA 节点通过 **增量式处理（Incremental Processing）** 的策略从根本上解决了这个问题。

无论是从磁盘流式读取帧进行视频合并，还是将生成结果逐帧保存到磁盘，我们的节点都**永远不需要将所有图像一次性加载到内存中**。这意味着您可以轻松生成数千帧的视频，而无需再为内存限制而烦恼。

### ✨ 无缝与高效的平衡

我们提供了两种工作模式，以适应不同场景：

- **内存模式**: 对于短序列或测试，可以直接将上游节点的 `IMAGE` 张量输入，实现无缝、快速的内存内处理。
- **磁盘模式**: 对于长序列的最终生成，节点会高效地从磁盘流式读取/写入帧，保证了稳定性和极低的内存占用。

### 🔧 强大且可扩展的预设系统

通过简单的 **JSON 文件**，您可以完全自定义视频和音频的编码参数，并将其保存为可复用的格式预设。这使得高级用户可以轻松实现复杂的 FFmpeg 配置，而无需修改任何代码。

---

## 🛠️ 安装与先决条件

### 1. 安装 FFmpeg (视频与浏览器节点强依赖)

视频合并节点和媒体浏览器的视频功能依赖 `ffmpeg` 和 `ffprobe`。

- 访问 [FFmpeg 官网](https://ffmpeg.org/download.html) 下载并安装。
- **强烈建议**将 FFmpeg 的 `bin` 目录添加到您系统的 `PATH` 环境变量中。
- 在终端中运行 `ffmpeg -version` 和 `ffprobe -version` 来验证安装。

### 2. 安装 NeMo 模型 (音频AI节点必须)

音频处理节点（如说话人日志）依赖 NeMo 模型。

- 在 ComfyUI 的 `models` 目录下，创建一个名为 `nemo_models` 的子目录。最终路径应为 `ComfyUI/models/nemo_models/`。
- 从 [NVIDIA NeMo 目录](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/diar_sortformer_telephonic_tele_ft_marblenet) 下载所需的 `.nemo` 模型文件。
  - 基础模型: `diar_sortformer_4spk-v1.nemo`
  - 流式模型 (v2.1): `diar_streaming_sortformer_4spk-v2.1.nemo`
- 将下载的 `.nemo` 文件放入 `ComfyUI/models/nemo_models/` 目录中。

**使用 huggingface-cli 下载 (推荐)**:
`nvidia/nemo-models` 仓库很大，建议直接下载指定文件：

```bash
# 进入 ComfyUI/models 目录
cd ComfyUI/models
mkdir -p nemo_models

# 下载基础模型
hf download nvidia/nemo-models diar_sortformer_4spk-v1.nemo --local-dir nemo_models

# 下载流式模型
hf download nvidia/nemo-models diar_streaming_sortformer_4spk-v2.1.nemo --local-dir nemo_models
```

### 3. 安装本节点套件

进入 ComfyUI 的自定义节点目录，然后克隆本仓库：

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/havvk/ComfyUI_AIIA.git
```

最后，**重启 ComfyUI**。

---

## ✨ 可用节点

### 1. 媒体管理 (Media Management)

#### AIIA 媒体浏览器

1. 安装并重启后，一个 **"AIIA Browser"** 按钮会出现在 ComfyUI 主菜单中。
2. 点击即可启动浏览器。
3. 尽情探索你的 `output` 目录吧！

---

### 2. 视频生成与合成 (Video Generation & Compositing)

#### 视频合并 (AIIA, 图像或目录)

这是一个功能强大且高度可定制的视频合并节点，是您工作流中处理视频生成的终极解决方案。

![video_combine_node_ui](https://github.com/user-attachments/assets/4185c2a7-e1a6-4980-ac8a-3d326bff4b87)

**核心亮点**:

- **内存高效**: 通过 `frames_directory` 输入，可以处理几乎无限数量的图像帧，完美解决了 OOM 问题。
- **直接张量输入**: 接受上游节点的 `IMAGE` 张量，方便快速迭代和测试。
- **全面的音频控制**: 支持 `AUDIO` 张量和外部文件，并提供对编解码器和码率的精细控制。
- **智能自动配置**: `auto` 模式能自动应用格式预设中的音频参数，并能自动检测源文件的码率。

#### FLOAT 影片生成 (内存与磁盘模式)

这组节点封装了先进的 **FLOAT** 模型，能够根据参考图像和音频生成高质量的口型同步影片。我们提供了两种模式，以应对不同长度的生成需求。

**1. Float Process (AIIA In-Memory)**

- **用途**: 用于生成**短片**、快速预览或直接与其他内存节点（如视频合并的 `images` 输入）连接。
- **输出**: `IMAGE` 张量。
- **优势**: 速度快，流程无缝。
- **限制**: 受限于您的 VRAM 和系统内存大小，不适合生成长视频。

**2. Float Process (AIIA To-Disk for Long Audio)**

- **用途**: 专门用于**长音频**的影片生成，是**解决OOM问题的关键**。
- **输出**: `STRING` (包含所有生成帧的目录路径) 和 `INT` (帧总数)。
- **优势**: 在解码过程中，节点以小批量方式处理帧并**逐帧保存到磁盘**，内存占用极低，可以处理任意长度的音频。
- **工作流**: 此节点的输出目录可以直接作为 **视频合并节点** 的 `frames_directory` 输入，构建一个完整的、内存高效的 talking head 视频生成管线。

#### PersonaLive 视频驱动 (AIIA Integrated)

这组节点基于强大的 [PersonaLive](https://github.com/GVCLab/PersonaLive) 模型，专为生成高质量的 Talking Head 视频而设计。我们将原版代码完全重构并集成到 ComfyUI 中，通过特有的分块处理和磁盘流式技术，**彻底解决了长视频生成时的显存和内存溢出 (OOM) 问题**。

**1. PersonaLive Checkpoint Loader**

- **用途**: 加载所有必要的模型权重（Base Model, VAE, PersonaLive Weights）。
- **功能**: 首次运行时会自动从 HuggingFace 下载所需模型，无需手动配置。

**2. PersonaLive Photo Sampler (AIIA In-Memory)**

- **用途**: 标准生成模式，适合**短视频**或**中等长度**视频。
- **输出**: `IMAGE` 张量（所有生成的帧）。
- **机制**: 节点会自动将长视频切分为多个 Chunk 进行推理，每推理完一个 Chunk 就会清空显存，从而允许你在有限的显存下生成较长的视频。

**3. PersonaLive Photo Sampler (AIIA To-Disk for Long Video)**

- **用途**: 专门用于**超长视频**生成。这是解决系统内存（System RAM）OOM 的终极方案。
- **输出**: `STRING` (包含生成帧的目录路径) 和 `INT` (帧数)。
- **最佳实践**: 将此节点的输出目录直接连接到 **AIIA Video Combine** 节点，即可实现从生成到合成的全流程 OOM-Safe。

---

### 3. 音频智能处理 (Intelligent Audio Processing)

#### 3.1 说话人日志 (Diarization)

这组节点利用 **NeMo Sortformer** E2E 模型，为您的音频提供先进的说话人识别功能。在4090RTX显卡上只需2秒钟就能完成10分钟音频的声纹分割聚类任务。

**1. AIIA Generate Speaker Segments**

- **用途**: 对一段音频进行分析，找出“**谁在什么时候说话**”。
- **输入**: `AUDIO` 张量。
- **输出**: `WHISPER_CHUNKS` (一个结构化的数据，包含一系列带有说话人标签的时间片段，如 `SPEAKER_00`, `SPEAKER_01` 等)。
- **亮点**: 提供多种**后处理配置**（从宽松到严格），让您可以微调分割的灵敏度，以适应不同质量的音频。

**2. AIIA E2E Speaker Diarization**

- **用途**: 将 `Generate Speaker Segments` 的识别结果**精确地应用**到由 Whisper 等工具生成的、带有文本的 `WHISPER_CHUNKS` 上。
- **输入**: `WHISPER_CHUNKS` (来自文本转录节点) 和 `AUDIO` 张量。
- **输出**: 更新后的 `WHISPER_CHUNKS`，其中每个文本块都已被赋予了最匹配的说话人标签。
- **工作流**: `(音频) -> Whisper -> (文本Chunks)` + `(音频)  => E2E Diarizer` => **最终带有说话人标签的文本稿**。

#### 3.2 说话人隔离与合并 (Speaker Isolation & Merger)

**Audio Speaker Isolator (AIIA)**

- **用途**: 根据说话人日志（Diarization）产生的 JSON 数据，从原始音轨中精确提取属于特定说话人的声音。
- **输入**:
  - `audio`: 原始音频张量。
  - `whisper_chunks`: 由 Diarization 节点生成的 JSON 片段数据。
  - `speaker_label`: 要提取的说话人 ID（例如 "SPEAKER_00"）。
  - `isolation_mode`:
    - **Maintain Duration (推荐)**: 输出与原音频等长的音轨，非目标说话人部分填充静音。这对于 **视频驱动（Talking Head）** 工作流至关重要，能确保口型与原视频时间轴严格对齐。
    - **Concatenate**: 将属于该说话人的所有片段无缝拼接在一起，去除中间的空隙。
- **亮点**:
  - **防爆音**: 内置微小的淡入淡出（Fade In/Out）处理，确保片段边缘自然顺滑。
    - **时间对齐**: 专门针对 AIIA 视频节点套件优化，保证音画同步。

**Audio Speaker Merger (AIIA)**

- **用途**: 将两段不同的音频流合并为一条。通常用于在分别对不同说话人进行视频驱动后，将各自的音轨重新混缩。
- **输入**:
  - `audio_1`, `audio_2`: 待合并的两段音频。
  - `duration_mode`:
    - `Longest`: 输出时长等于两段音频中的最大值。
    - `Shortest`: 输出时长等于两段音频中的最小值。
    - `Audio 1 / Audio 2`: 严格跟随特定输入段的时长。
    - `Specified`: 手动指定输出秒数。
    - `normalize`: 开启后将自动防止音量叠加导致的破音。

#### 3.3 智能切片 (Smart Chunking)

**Audio Smart Chunker (Silence-based)**

- **用途**: 全量扫描长音频，寻找最优切片方案。
- **机制**:
  - **全局静音扫描**: 自动识别音频中的天然停顿区间。
  - **贪心优化算法**: 在不超出设定时长（如 27s）的前提下，计算出能让每一刀都切在静音处的片段序列。
- **输出**: 供 `Voice Conversion (AIIA Unlimited)` 使用的 `whisper_chunks` 引导数据。
- **协同优势**: 预先规划好切点，避免转换节点在词句中间暴力切分，是彻底消除拼接毛刺的关键。

#### 3.4 Voice Conversion (AIIA Unlimited) (CosyVoice 增强)

**1. CosyVoice Model Loader (AIIA)**

- **用途**: 不需要安装任何其他第三方节点，直接加载 CosyVoice 模型。
- **功能**:
  - **自动安装依赖**: 首次运行时会自动安装 `cosyvoice` 及其依赖环境。
  - **自动下载模型**: 选择模型后，首次运行会自动从 HuggingFace 下载模型权重。
  - **支持模型**:
    - `Fun-CosyVoice3-0.5B-2512` (最新 CosyVoice 3.0, **推荐**)
    - `CosyVoice2-0.5B` (CosyVoice 2.0)
    - `CosyVoice-300M` 系列 (SFT, Instruct, TTSFRD)
- **输出**: `COSYVOICE_MODEL` (专为 AIIA Voice Conversion 节点优化)。
  - **⚠️ 模型下载问题 (Model Download Issues)**:
    如果遇到自动下载卡顿或失败，请**手工下载**模型文件夹，并将其放入 `ComfyUI/models/cosyvoice/` 目录中。

    **目录结构示例 (Directory Structure)**:
    请注意：文件夹名称建议与下方列表保持一致（即去除 `FunAudioLLM/` 前缀）。

    ```text
    ComfyUI/models/cosyvoice/
    ├── Fun-CosyVoice3-0.5B-2512/  <-- 对应选项 FunAudioLLM/Fun-CosyVoice3-0.5B-2512
    │   ├── cosyvoice.yaml
    │   ├── model.pt
    │   └── ...
    ├── CosyVoice2-0.5B/           <-- 对应选项 FunAudioLLM/CosyVoice2-0.5B
    └── CosyVoice-300M/            <-- 对应选项 CosyVoice-300M
    ```

    **下载地址 (Download Sources)**:

    * [ModelScope (魔搭社区)](https://www.modelscope.cn/organization/iic) (搜索 CosyVoice)
    * [HuggingFace - FunAudioLLM](https://huggingface.co/FunAudioLLM)

    **使用 huggingface-cli 下载 (CLI Example)**:

    ```bash
    # 进入 ComfyUI/models/cosyvoice 目录
    cd ComfyUI/models/cosyvoice

    # 下载 CosyVoice 3.0 (0.5B)
    hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local-dir Fun-CosyVoice3-0.5B-2512
    ```

**2. Voice Conversion (AIIA Unlimited)**

- **用途**: 增强版语音转换节点，解决了原版 CosyVoice3 只能转换 30 秒内语音的限制。
- **🔥 核心突破**: **全球首个突破时长限制的 ComfyUI 节点**。通过创新的静音点智能切片技术，本节点彻底突破了 CosyVoice 3 官方模型“建议 20 秒、最长 60 秒”的生成限制。它实现了**真正无限时长的音色克隆**，且拼接处自然无痕，完美满足长篇教学视频、有声书等专业制作需求。
- **创新设计 (创新亮点)**:
  - **语义感知切片 (Semantic-Aware Chunking)**: 节点可选接入 `whisper_chunks` 数据。它能智能识别每句话之间的“缝隙”，优先在说话人停顿的天然空隙处进行切分，从根本上避免了“在单词中间切开”导致的违和感。
  - **牺牲上下文无缝拼接 (Sacrificial Context Stitching)**: 针对生成式模型在重叠区域的相位不稳定性，我们在拼接时采用“牺牲上下文”策略。即利用重叠区域作为生成引导，但在最终拼接时丢弃前一片段的“尾部”，仅保留极短的 Cross-fade（约 50ms），从而彻底消除长音频拼接处的“发虚”和回声感。
  - **双层寻点算法**: 在语义缝隙确定的候选范围内，算法会自动进行微秒级的**物理静音探测**（基于能量平滑曲线），确保切点处于绝对静默状态。
- **输入**:
  - `model`: 连接 `CosyVoice Model Loader (AIIA)` 的输出。
  - `source_audio`: 待转换的源音频（支持任意时长）。
  - `whisper_chunks` (可选): 接入 Diarization 节点数据，开启语义感知切片。
  - `target_audio`: 目标音色参考音频（自动截取前 30 秒）。
  - `chunk_size`: 目标切片大小（默认 25 秒）。
  - `overlap_size`: 重叠大小，用于平滑衔接。

#### 3.5 智能音频增强 (Audio AI Enhance) (推荐)

- **用途**: 专为 CosyVoice 等生成式语音设计的增强器。它基于 `resemble-enhance` 强大的 **Conditional Flow Matching (CFM)** 模型，能同时完成后处理降噪和超分辨率（Bandwidth Extension）。
- **核心能力**: 将任意低采样率（如 22kHz, 16kHz）的输入音频，重构为 **44.1kHz** 的高保真音频。

**参数详解**:

1. **mode**:

   - `Enhance (Denoise + Bandwidth Ext)`: **(推荐)** 同时去除底噪并提升音质。
   - `Denoise Only`: 仅去除噪声，不改变音质。
2. **solver**: 推理求解器。

   - `Midpoint` (默认): 速度与质量的最佳平衡。
   - `RK4`: 质量最高，但速度慢 2 倍。
   - `Euler`: 速度最快，但可能产生条纹。
3. **nfe (Steps)**: 迭代步数。

   - **32**: 快速预览。可能会有残留条纹。
   - **64**: **(推荐)** 标准质量。
   - **128**: 极高画质，消除所有细微伪影，但耗时。
4. **tau (Temperature)**: 先验温度 (默认 0.5)。

   - 控制生成过程的“创造力”。
   - 0.5 为平衡点。如果声音发虚，尝试降低到 0.3。
5. **denoise_strength (去噪强度)** :

   - 范围 `0.0` - `1.0`。默认 `0.5`。
   - `0.0`: 保留所有原始底噪。
   - `1.0`: 强力去噪。
   - **Tip**: 如果你在频谱图的低频部分看到 **水平条纹 (Hum/Buzz)**，请将此值提高到 **0.8 - 1.0**。
6. **high_pass_hz (高通滤波) 🔥 新增 (v1.4.82)**:

   - **范围**: `0 - 1000` Hz。**默认 `0` (关闭)**。
   - **作用**: 在 AI 增强**之前**，直接切除指定频率以下的低频信号。
   - **为什么需要它？**: 如果 `denoise_strength=1.0` 仍无法消除低频条纹，说明 AI 把这些噪音当成了“真实信号”进行增强。
   - **最佳实践**: 遇到顽固的低频条纹时，设置 **60 - 80 Hz**。
   - **安全**: **只要保持默认为 0，此功能完全禁用，对原音质零影响。**
7. **chunk_seconds / overlap_seconds**:

   - **30s / 1s**: **(推荐)** 4090 等显卡的最佳平衡点。
   - **注意**: 首次运行时，模型会进行 **JIT 编译（约60秒）**。请耐心等待，这只发生一次。

- **依赖**: 首次运行会自动安装 `resemble-enhance`。

#### 3.6 智能音频降噪 (VoiceFixer) (Legacy)

- **用途**: 用于修复**严重受损**的音频（如老电影、严重削波或极强背景噪）。
- **注意**: 对于 CosyVoice 生成的较干净语音，**不推荐**使用此节点，因为它通过重构方式修复，容易在干净语音上引入伪影（Spectral Stripes）。请优先使用 `Audio AI Enhance`。
- **参数**:

  - `mode`:
    - **0 (Full)**: 去噪 + 去混响 + 去破音 + 超分。适合质量最差的音频。
    - **1 (No DeClip)**: 去噪 + 去混响 + 超分。适合大多数情况。
    - **2 (Denoise Only)**: 仅去噪。
  - `use_cuda`: 是否使用 GPU (推荐)。
- **依赖**: 首次运行会自动安装 `voicefixer` 库。
- **⚠️ 模型下载问题 (Model Download Issues)**:
  如果遇到下载卡顿或 `PytorchStreamReader` 报错，请尝试**手工下载**模型文件，并按以下结构放入 `ComfyUI/models/voicefixer` 目录中：

  ```text
  ComfyUI/models/voicefixer/
  ├── analysis_module/
  │   └── checkpoints/
  │       └── vf.ckpt  (466MB)
  └── synthesis_module/
      └── 44100/
          └── model.ckpt-1490000_trimed.pt (135MB)
  ```

  **下载地址 (Download Links)**:

  * **vf.ckpt**: [HuggingFace Mirror](https://huggingface.co/Diogodiogod/VoiceFixer-vf.ckpt/resolve/main/vf.ckpt) 或 [Zenodo](https://zenodo.org/record/5600188/files/vf.ckpt?download=1)
  * **model.ckpt-1490000_trimed.pt**: [HuggingFace Mirror](https://huggingface.co/Diogodiogod/VoiceFixer-model.ckpt-1490000_trimed.pt/resolve/main/model.ckpt-1490000_trimed.pt) 或 [Zenodo](https://zenodo.org/record/5513378/files/model.ckpt-1490000_trimed.pt?download=1)

  **使用 huggingface-cli 下载 (CLI Example)**:

  ```bash
  # 进入 ComfyUI/models/voicefixer 目录
  cd ComfyUI/models/voicefixer

  # 下载 vf.ckpt (注意路径)
  hf download Diogodiogod/VoiceFixer-vf.ckpt vf.ckpt --local-dir analysis_module/checkpoints

  # 下载 model.ckpt (注意路径)
  hf download Diogodiogod/VoiceFixer-model.ckpt-1490000_trimed.pt model.ckpt-1490000_trimed.pt --local-dir synthesis_module/44100
  ```

#### 3.7 Audio Post-Process (Resample/Fade/Norm)

- **用途**: 音频后期处理“母带”节点。
- **功能**:
  - **Resample**: 使用高精度算法（sinc_interp_hann）将音频上采样至 44.1kHz 或 48kHz，改善听感清晰度。
  - **LowPass Filter**: 可选的低通滤波器（**默认 11000 Hz**）。采用 **6级级联（Cascaded）** 设计，像“砖墙”一样强力消除由 CosyVoice 带来的高频混叠（Resampling Artifacts）。
  - **HighPass Filter**: 可选的高通滤波器（**默认 60 Hz**）。采用 **单级平滑** 设计，在去除低频轰鸣声和直流偏移的同时，完美保留人声的低音厚度。
  - **Fade In/Out**: 对音频首尾添加淡入淡出，消除由于切割或拼接可能残留的爆音。
  - **Normalize**: 将音量标准化至 -1dB，确保输出响度饱满且不破音。
- **场景**: 建议串联在 `Voice Conversion` 节点之后使用。

#### 3.8 Audio Splice Analyzer (AIIA Debug)

- **用途**: 可视化验证音频拼接质量。
- **功能**:
  - 生成 Log-Mel 语谱图 (Spectrogram)。
  - 如果接入了 `Voice Conversion` 节点的 `SPLICE_INFO` 输出，会自动在图上用红线标记出所有拼接点的位置。
  - 帮助用户直观地检查拼接点是否位于静音区，以及是否有明显的频谱断裂。
- **依赖**: 需要 `matplotlib` 库。如果未安装，节点会生成一张提示错误的图片，不会导致工作流崩溃。

#### 3.9 音频信息查看 (Audio Info & Metadata)

- **用途**: 实时查看音频流的元数据，确保你的采样率符合预期。
- **输入**: `AUDIO` 张量。
- **输出**:
  - `info_text`: 包含采样率、时长、通道数、BatchSize等详细信息的文本报告。
  - `sample_rate`: 采样率 (INT)。
  - `duration`: 时长秒数 (FLOAT)。
  - `channels`: 通道数 (INT)。

#### 3.10 Microsoft VibeVoice (Beta)

- **用途**: 微软最新的 1.5B TTS/音色克隆模型。音质非常惊人，但对环境要求较高。
- **环境要求**:
  - **Flash Attention 2**: 强烈推荐安装（否则速度较慢）。
  - **Transformers**: `>= 4.51`（重要: 旧版本不支持该模型）。
- **节点**:
  - `VibeVoice Loader`: 加载模型。支持从 HuggingFace 自动下载，也支持加载本地模型。
  - `VibeVoice TTS`: 支持 Zero-shot 音色克隆（输入 `reference_audio` 即可）。
- **模型准备 (Model Preparation)**:
  如果遇到下载问题或分词器报错，请手动下载模型文件到 `models/vibevoice` 目录。
  
  **必须的文件结构**:
  ```text
  ComfyUI/models/vibevoice/microsoft/VibeVoice-1.5B/
  ├── model-00001-of-00003.safetensors ... (模型权重)
  ├── config.json
  ├── modeling_vibevoice_*.py (Python代码)
  └── [Tokenizer Files] (必须包含以下 Qwen 文件!)
      ├── tokenizer.json
      ├── tokenizer_config.json
      ├── vocab.json
      └── merges.txt
  ```

  **⚠️ 重要提示**: VibeVoice 依赖 **Qwen2.5-1.5B** 的分词器。如果你下载的模型包里没有上面列出的 tokenizer 文件，请手动从 [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main) 下载这 4 个文件并放入模型目录。

  **手动下载命令**:
  ```bash
  mkdir -p models/vibevoice/microsoft/VibeVoice-1.5B
  # 1. 下载模型权重和代码
  hf download microsoft/VibeVoice-1.5B --local-dir models/vibevoice/microsoft/VibeVoice-1.5B
  # 2. 补全 Tokenizer 文件 (如果没有)
  wget https://huggingface.co/Qwen/Qwen2.5-1.5B/resolve/main/tokenizer.json -P models/vibevoice/microsoft/VibeVoice-1.5B/
  wget https://huggingface.co/Qwen/Qwen2.5-1.5B/resolve/main/tokenizer_config.json -P models/vibevoice/microsoft/VibeVoice-1.5B/
  wget https://huggingface.co/Qwen/Qwen2.5-1.5B/resolve/main/vocab.json -P models/vibevoice/microsoft/VibeVoice-1.5B/
  wget https://huggingface.co/Qwen/Qwen2.5-1.5B/resolve/main/merges.txt -P models/vibevoice/microsoft/VibeVoice-1.5B/
  ```

### 4. 图像工具 (Image Utilities)

#### Image Concatenate (AIIA Utils, Disk)

- **用途**: 将两个图像序列（来自两个不同的目录）逐帧拼接在一起，非常适合创建**对比视频**或**多面板视频**。
- **核心亮点 (OOM-Safe)**: 此节点**逐帧读取、处理和保存**，从不将整个图像序列加载到内存中，因此可以处理任意数量的帧。
- **功能**:

  - 支持上下左右四个方向的拼接。
  - 可自动调整其中一个图像序列的尺寸以匹配另一个，并保持宽高比。
  - 可自定义背景填充颜色。
- **输出**: `STRING` (包含所有拼接后帧的新目录路径)。
- **输出**: `STRING` (包含所有拼接后帧的新目录路径)。

---

## ❓ 故障排查

- **错误: "FFmpeg not found" / "NeMo model not found"**
  - 请返回阅读 [安装与先决条件](#-安装与先决条件) 部分，确保所有依赖都已正确安装和放置。
- **错误: "帧目录验证失败"**
  - 请确保您填写的路径是**绝对路径**且真实存在。

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

---

## Changelog

### [1.3.0] - 2025-07-07

本次更新为浏览器带来了关键的新功能，并解决了一系列UI/UX问题和Bug，旨在提供一个更稳定、更直观的用户体验。

#### ✨ 新功能与优化

- **工作流加载与文件下载**: 现在你可以直接从浏览器内部加载包含工作流的图片或视频，或一键下载任何文件，极大地简化了你的创作流程。
- **目录刷新功能**:
  - 在导航栏添加了手动刷新按钮 (🔄)。
  - 现在当在图标和列表视图之间切换时，浏览器会自动刷新目录内容，确保文件列表永远是最新状态。
- **优化的图标按钮布局**: 图标视图中的下载和加载工作流按钮现在改为在鼠标悬停时，在缩略图顶部显示为一个干净的浮层。
- **后端增强**: 包含对Python后端的若干改进，使缩略图生成和元数据提取更加健壮。

#### 🐞 Bug修复

- **修复列表视图按钮渲染问题**: 解决了一个因竞态条件导致的“加载工作流”按钮在列表视图中不显示的Bug。
- **修复加载工作流时的名称设置**: 通过为标准格式和API格式的工作流调用正确的API，现在加载工作流后能正确地在ComfyUI标题栏设置其名称。
- **修复图标缩放滑块**: 图标大小滑块现在可以通过触发一次完整的、干净的视图重绘来即时、正确地更新视图。
