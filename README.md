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

### 2. 安装 SoX (VibeVoice 变速不变调必须)

VibeVoice 节点的 `speed` 参数依赖系统级 `sox` 命令。

- **Ubuntu/Debian**: `sudo apt-get update && sudo apt-get install -y libsox-dev sox`
- **macOS**: `brew install sox`
- **Windows**: 下载 [SoX 编译版](https://sourceforge.net/projects/sox/files/sox/) 并将目录添加到 `PATH`。

### 3. 安装 NeMo 模型 (音频AI节点必须)

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

#### 2.1 视频合并 (AIIA, 图像或目录)

这是一个功能强大且高度可定制的视频合并节点，是您工作流中处理视频生成的终极解决方案。

![video_combine_node_ui](https://github.com/user-attachments/assets/4185c2a7-e1a6-4980-ac8a-3d326bff4b87)

**核心亮点**:

- **内存高效**: 通过 `frames_directory` 输入，可以处理几乎无限数量的图像帧，完美解决了 OOM 问题。
- **直接张量输入**: 接受上游节点的 `IMAGE` 张量，方便快速迭代和测试。
- **全面的音频控制**: 支持 `AUDIO` 张量和外部文件，并提供对编解码器和码率的精细控制。
- **智能自动配置**: `auto` 模式能自动应用格式预设中的音频参数，并能自动检测源文件的码率。

#### 2.2 FLOAT 影片生成 (内存与磁盘模式)

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

#### 2.3 PersonaLive 视频驱动 (AIIA Integrated)

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


#### 2.4 EchoMimic V3 (AIIA Integrated)

这组节点集成了最新的 **EchoMimic V3** (1.3B Parameters) 模型，它是目前开源界效果最惊艳的 Talking Head 解决方案之一。

**特点**:
- **多模态驱动**: 支持 **Audio Only** (仅音频驱动) 和 **Audio + Reference Pose** (音频+参考姿态) 驱动。
- **自然度极高**: 相比 float 等早期模型，V3 在头部运动、表情微表情的自然度上有巨大提升。
- **ComfyUI 原生**: 我们将其封装为标准的 Loader 和 Sampler 节点，支持流式生成和内存优化。

**1. EchoMimic V3 Loader**
- **用途**: 加载模型权重 (Transformer, VAE, Wav2Vec, etc.)。
- **参数**: 
  - `model_subfolder`: 模型子目录名 (默认 `Wan2.1-Fun-V1.1-1.3B-InP`)。
  - `device`: 指定运行设备 (CUDA)。

**2. EchoMimic V3 Sampler**
- **用途**: 执行推理生成。
- **输入**:
  - `ref_image`: 参考人物图片 (建议 1:1 比例，如 768x768)。
  - `ref_audio`: 驱动音频。
- **参数**:
  - `cfg`: 视觉引导系数 (默认 4.0)。
  - `audio_cfg`: 音频引导系数 (默认 2.9)。数值越高，口型越准，但运动可能变僵硬。

**🛠️ 模型下载指南 (Manual Download Guide)**

由于 EchoMimic V3 模型较大且组件较多，目前**不支持自动下载**，请按以下步骤手动准备模型。

目标目录: `ComfyUI/models/EchoMimicV3/`

**目录结构**:
```text
ComfyUI/models/EchoMimicV3/
├── Wan2.1-Fun-V1.1-1.3B-InP/     <-- 主模型目录
│   ├── transformer/
│   │   ├── config.json
│   │   └── diffusion_pytorch_model.safetensors
│   ├── vae/
│   │   ├── config.json
│   │   └── diffusion_pytorch_model.safetensors
│   ├── text_encoder/
│   ├── tokenizer/
│   ├── image_encoder/
│   └── scheduler/
└── wav2vec2-base-960h/           <-- 音频编码器 (必需)
    ├── config.json
    ├── pytorch_model.bin
    └── ...
```

**下载地址**:
1. **主模型 (Wan2.1-Fun-V1.1-1.3B-InP)**:
   - HuggingFace: [alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP)
   - **下载命令**:
     ```bash
     git clone https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP models/EchoMimicV3/Wan2.1-Fun-V1.1-1.3B-InP
     ```
   - *注意：目前代码默认寻找 `Wan2.1-Fun-V1.1-1.3B-InP` 文件夹。*

2. **音频编码器 (wav2vec2-base-960h)**:
   - HuggingFace: [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h)
   - 下载命令: `git clone https://huggingface.co/facebook/wav2vec2-base-960h models/EchoMimicV3/wav2vec2-base-960h`

**环境依赖**:
- 请确保安装了 `requirements.txt` 中的依赖，如 `diffusers>=0.30.1`。节点加载时会尝试自动引用，但如果报错缺包，请手动安装。


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

    **使用命令行快速下载 (CLI Examples)**:

    > [!TIP]
    > **国内用户推荐使用 ModelScope (魔搭)**，下载速度更快且无需代理。
    >

    **1. 使用 ModelScope (推荐):**

    ```bash
    # 进入 ComfyUI/models/cosyvoice 目录
    cd ComfyUI/models/cosyvoice

    # 下载 CosyVoice 3.0 (0.5B - 推荐)
    modelscope download --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local_dir Fun-CosyVoice3-0.5B-2512

    # 下载 CosyVoice 2.0 (0.5B)
    modelscope download --model iic/CosyVoice2-0.5B --local_dir CosyVoice2-0.5B

    # 下载 CosyVoice 300M 系列 (V1)
    modelscope download --model iic/CosyVoice-300M --local_dir CosyVoice-300M
    modelscope download --model iic/CosyVoice-300M-SFT --local_dir CosyVoice-300M-SFT
    modelscope download --model iic/CosyVoice-300M-Instruct --local_dir CosyVoice-300M-Instruct
    ```

    **2. 使用 HuggingFace:**

    ```bash
    # 下载 CosyVoice 3.0 (0.5B)
    huggingface-cli download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local-dir Fun-CosyVoice3-0.5B-2512

    # 下载 CosyVoice 2.0 (0.5B)
    huggingface-cli download FunAudioLLM/CosyVoice2-0.5B --local-dir CosyVoice2-0.5B

    # 下载 CosyVoice 300M 系列 (V1)
    huggingface-cli download FunAudioLLM/CosyVoice-300M --local-dir CosyVoice-300M
    huggingface-cli download FunAudioLLM/CosyVoice-300M-SFT --local-dir CosyVoice-300M-SFT
    huggingface-cli download FunAudioLLM/CosyVoice-300M-Instruct --local-dir CosyVoice-300M-Instruct
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

#### 3.5 CosyVoice 3.0 TTS (AIIA) (全版本集成文本转语音)

- **用途**: 一个节点通连 CosyVoice 全系列模型 (V1, V2, V3)，支持从纯文字描述生成到高保真音色克隆的全场景需求。
- **🔥 核心架构优势**:
  - **全版本自适应 (Version-Agnostic)**: 自动根据 `Model Loader` 加载的模型版本切换底层推理逻辑。无论是老牌的 300M 系列还是最新的 V3 0.5B 模型，均能获得最佳表现。
  - **300M-Instruct (V1) 专项修复**:
    - **指令跟随 (Fixed)**: 已修复指令读出问题。
    - **能力矩阵**:
      - ✅ **支持**: 情感 (Emotion), 语速 (Speed), 基础性别 (Gender)。
      - ⚠️ **注意**: 请务必使用 **英文指令** (如 "Sad tone", "Fast speed") 以确保生效。中文指令可能被忽略。
      - ❌ **不支持**: 方言 (Dialect)。300M-Instruct 模型无法通过指令修改方言。
  - **V3 稳定性引擎**: 深度适配 V3 模型的推理协议。内置自动补全机制，确保在任何配置下（即使缺少参考音频或内部音色）都能稳定运行，杜绝 `AudioDecoder` 和 `KeyError` 等常见崩溃。
  - **无限长文本支持 (Streaming Architecture)**: 与 VibeVoice (类 GPT) 存在 8K/32K Token 上限不同，CosyVoice 采用流式架构，**理论支持无限长文本输入的连续生成**。系统会自动按句切分流式合成，适合长篇有声读物生成。
- **🚀 五种核心生成模式**:
  1. **风格/情感建模 (Instruct)**: **文字 + 描述**。通过文字描述（如“非常开心地说”、“四川话”、“语速极快”）来控制生成的表现力。
  2. **音色克隆 (Zero-Shot)**: **文字 + 音频**。通过极短的参考音频，完美复现目标说话人的身份和音色特征。
  3. **混合控制 (Hybrid)**: **音频 + 描述**。在克隆音色的基础上，通过描述词改变情感或方言。
  4. **跨语言 (Cross-Lingual)**: 让你的克隆音色流利说出 9 种语言。
  5. **固定 ID (SFT)**: 直接选取模型内置的高保真音色（如 V3 的原生 01 序列）。
- **💡 进阶使用建议**:
  - **V3 推荐用法**: 优先通过 `reference_audio` 提供一个高音质片段。V3 模型对比 V1 有质的飞跃，其对音质的还原度和表现力极高。
  - **描述词技巧**: 尽量使用具体的场景化描述（例如“在大雨中嘶吼”或“温柔的枕边话”），V3 模型能捕捉到极其细微的情感波动。
  - **种子音频回退**: 在“描述生成 (Instruct)”模式下，若不提供参考音频，系统将使用内置的高保真种子作为底色。可以通过切换 `base_gender` 来获取不同的基础人声方向。

#### 3.6 智能音频增强 (Audio AI Enhance) (推荐)

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

#### 3.7 智能音频降噪 (VoiceFixer) (Legacy)

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

#### 3.8 Audio Post-Process (Resample/Fade/Norm)

- **用途**: 音频后期处理“母带”节点。
- **功能**:
  - **Resample**: 使用高精度算法（sinc_interp_hann）将音频上采样至 44.1kHz 或 48kHz，改善听感清晰度。
  - **LowPass Filter**: 可选的低通滤波器（**默认 11000 Hz**）。采用 **6级级联（Cascaded）** 设计，像“砖墙”一样强力消除由 CosyVoice 带来的高频混叠（Resampling Artifacts）。
  - **HighPass Filter**: 可选的高通滤波器（**默认 60 Hz**）。采用 **单级平滑** 设计，在去除低频轰鸣声和直流偏移的同时，完美保留人声的低音厚度。
  - **Fade In/Out**: 对音频首尾添加淡入淡出，消除由于切割或拼接可能残留的爆音。
  - **Normalize**: 将音量标准化至 -1dB，确保输出响度饱满且不破音。
- **场景**: 建议串联在 `Voice Conversion` 节点之后使用。

#### 3.9 Audio Splice Analyzer (AIIA Debug)

- **用途**: 可视化验证音频拼接质量。
- **功能**:
  - 生成 Log-Mel 语谱图 (Spectrogram)。
  - 如果接入了 `Voice Conversion` 节点的 `SPLICE_INFO` 输出，会自动在图上用红线标记出所有拼接点的位置。
  - 帮助用户直观地检查拼接点是否位于静音区，以及是否有明显的频谱断裂。
- **依赖**: 需要 `matplotlib` 库。如果未安装，节点会生成一张提示错误的图片，不会导致工作流崩溃。

#### 3.10 音频信息查看 (Audio Info & Metadata)

- **用途**: 实时查看音频流的元数据，确保你的采样率符合预期。
- **输入**: `AUDIO` 张量。
- **输出**:
  - `info_text`: 包含采样率、时长、通道数、BatchSize等详细信息的文本报告。
  - `sample_rate`: 采样率 (INT)。
  - `duration`: 时长秒数 (FLOAT)。
  - `channels`: 通道数 (INT)。

#### 3.11 Microsoft VibeVoice (Beta)

- **用途**: 微软最新的 TTS/音色克隆模型，支持 1.5B 和 7B 两种规格。
- **当前状态**: ✅ 可用（已通过测试）
- **支持语言**: **英文 (en) 和 中文 (zh)**（官方仅在这两种语言数据集上训练）
- **可选模型**:
  - `microsoft/VibeVoice-Realtime-0.5B`: **最新实时版**，极致速度，支持多种语言（如日、韩、英、中等），上下文 8K。推荐用于低延迟对话场景。
  - `microsoft/VibeVoice-1.5B`: 轻量版，64K 上下文（~3GB 显存）。
  - `vibevoice/VibeVoice-7B`: 高质量版，32K 上下文（~14GB 显存）。推荐用于生产环境。
- **特点**:
  - **多语言支持**: 0.5B 版本支持比 1.5B/7B 更多的语言种类。
  - **即时启动**: 无需预热或编译，首次运行即可使用（CosyVoice 首次需 ~1 分钟编译）。
  - **语言自动识别**: 模型会自动识别中英文文本。
  - **零样本音色克隆**: 输入 `reference_audio` 即可克隆声音。参考音频会自动重采样到 24000Hz。
- **节点参数 (推荐参数建议)**:
  - `cfg_scale` (默认: 1.3): CFG 引导强度。建议使用 **1.3**。
  - `ddpm_steps` (默认: 20): 扩散步数。
  - `do_sample` (默认: "auto"): **智能采样开关**。

    - `"auto"`: 自动适配。**1.5B 模型默认关闭**（保证稳定性），**7B 模型默认开启**（释放表达力）。
    - `"true"`: 强制开启。如果 1.5B 开启后出现电音或逻辑混乱，请切换回 `"auto"`。
    - `"false"`: 强制关闭（即 Greed Search）。
  - `temperature` (默认: 0.8): 采样温度。仅在 `do_sample` 开启时有效。
  - `top_k / top_p`: 采样约束。
  - `speed` (默认: 1.0): 播放速度。
  - `do_sample`: **"auto"** (或 `false`) - 保证极速和绝对稳定。
  - `normalize_text`: **True** - 帮助处理各种语言的特殊符号。
  - **特点**: 该模型支持包括**韩语、日语**在内的更多语种，速度极快。

#### 🔬 详细观察与进阶技巧

1. **0.5B 实时版**:

   - **频谱特征**: 在静音区域（Silence）可能在全部频率范围内观察到较高的能量分布，尤其在中频区域。
   - **听感**: 尽管频谱显示有能量，但**实际听感比较干净**，只有轻微的底噪。这表明其声码器可能存在某种特征泄露，但不影响实用性。
   - **迭代次数**: 由于基于音素（Phoneme）和流式切片处理，其显示的迭代次数（Total Steps）通常多于标准版，这是正常的。
2. **1.5B vs 7B 对比**:

   - **音质与风格**: 在参数调整得当（如开启 `do_sample`）的情况下，1.5B 模型生成的语音内容、风格和音质与 7B 模型**几乎无法区分**。
   - **性价比**: 1.5B 模型的推理速度约为 7B 的 **2倍**，显存占用仅为 1/4。除非对极细微的表达有极致要求，否则 **1.5B 是生产环境的最佳选择**。
   - **迭代差异**: 1.5B 的迭代次数（Token数）可能略少于 7B，这反映了不同模型对同一文本编码的简洁程度差异，属正常现象。

### 💡 7B 模型音质“全开”指南

要达到官方 Benchmark 的水准，请对 7B 模型尝试以下组合：

- `do_sample`: **True** (开启采样)
- `temperature`: **0.8 - 0.9**
- `cfg_scale`: **1.3 - 1.8**
- `ddpm_steps`: **20 - 50**
- `normalize_text`: **False**
- **环境要求**:

  - **Flash Attention 2**: 强烈推荐安装（否则速度较慢）。
  - **Transformers**: `>= 4.51`（重要: 旧版本不支持该模型）。
- **节点**:

  - `VibeVoice Loader`: 加载模型。支持从 HuggingFace 自动下载，也支持加载本地模型。
  - `VibeVoice TTS`: 支持 Zero-shot 音色克隆（输入 `reference_audio` 即可）。
- **模型准备 (Model Preparation)**:
  如果遇到下载问题或分词器报错，请手动下载模型文件到 `models/vibevoice` 目录。

  **必须的文件结构** (以 1.5B 为例，7B 类似):

  ```text
  ComfyUI/models/vibevoice/microsoft/VibeVoice-1.5B/  # 或 VibeVoice-7B/
  ├── model-*.safetensors (模型权重)
  ├── config.json
  └── [Tokenizer Files] (必须包含以下 Qwen 文件!)
      ├── tokenizer.json
      ├── tokenizer_config.json
      ├── vocab.json
      └── merges.txt
  ```

  **💡 说明**: 插件已内置并修复了所有 VibeVoice 的 Python 核心代码 (`vibevoice_core`)。你**不需要**也不建议在模型目录中保留 `modeling_vibevoice_*.py` 等 Python 脚本，以避免潜在的冲突。

  **⚠️ 重要提示**: VibeVoice 依赖 **Qwen2.5** 的分词器。如果模型包里没有 tokenizer 文件，请手动补全：

  - 1.5B 模型使用 [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main) 的 tokenizer
  - 7B 模型使用 [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B/tree/main) 的 tokenizer

### 💡 VibeVoice 使用指南 (Node Usage Guide)

由于模型架构不同，我们现在提供 **两个独立的 TTS 节点** 以优化体验：

#### 1. 🗣️ VibeVoice TTS (Standard)

- **适用模型**: `VibeVoice-1.5B`, `VibeVoice-7B`
- **参考音频 (Reference Audio)**: 可选 (`optional`)。如果不连接，将自动使用内置的高品质女声种子 (Fallback Seed) 进行生成。
- **功能**: 支持零样本音色克隆 (Zero-shot Cloning)。输入任何音频，它都会模仿该音色。
- **不支持**: `voice_preset` (预设)。

#### 2. 🗣️ VibeVoice TTS (Realtime 0.5B)

- **适用模型**: `VibeVoice-Realtime-0.5B`
- **必选参数**: `voice_preset` (音色预设) - **必须选择**。
- **功能**: 极速实时生成。基于预计算的 `.pt` 缓存文件生成语音。
- **不支持**: `reference_audio` (直接克隆)。
- **特点**:
  - **极低延迟**: 首包延迟极低，适合即时交互。
  - **BF16 加速**: 自动使用 Bloat16 精度进行推理（如果硬件支持），大幅提升速度。
  - **多语言支持**: 官方预设涵盖英、日、韩、法、德等。

#### 3. 🎤 VibeVoice Preset Maker (0.5B) (Experimental ⚠️)

- **用途**: 尝试制作 0.5B 模型专用的 `.pt` 音色预设。
- **现状**: **极不稳定**。
- **原因**: 社区反馈和测试表明，VibeVoice-Realtime-0.5B 模型的权重似乎对自定义音色进行了限制或未进行充分的 Zero-Shot 泛化训练。即使使用长达 1 分钟的高质量音频，生成时也极易出现**死循环、胡言乱语或噪音**。
- **建议**:

  - **首选**: 请直接下载并在 `Realtime 0.5B` 节点中使用 **微软官方提供的预设** (Carter, Emma 等)。
  - **尝试**: 如果您一定要克隆音色，请使用 **VibeVoice 1.5B / 7B (Standard)** 节点，它们原生支持完美的 Zero-Shot 克隆。
  - **仅供研究**: 此节点保留给开发者进行研究调试，普通用户**不推荐**使用。

  **手动下载命令**:

  ```bash
  # ===== 0.5B 实时多语言模型 (Realtime) =====
  mkdir -p models/vibevoice/microsoft/VibeVoice-Realtime-0.5B
  hf download microsoft/VibeVoice-Realtime-0.5B --local-dir models/vibevoice/microsoft/VibeVoice-Realtime-0.5B
  # 补全 Tokenizer (使用 Qwen2.5-0.5B)
  wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json -P models/vibevoice/microsoft/VibeVoice-Realtime-0.5B/
  wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer_config.json -P models/vibevoice/microsoft/VibeVoice-Realtime-0.5B/
  wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/vocab.json -P models/vibevoice/microsoft/VibeVoice-Realtime-0.5B/
  wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/merges.txt -P models/vibevoice/microsoft/VibeVoice-Realtime-0.5B/

  # 🔥 [重要] 下载 0.5B 官方音色库 (Voices Presets) 🔥
  # 0.5B 模型必须配合官方音色预设使用，不支持零样本克隆。
  # 请务必将以下文件下载到 `models/vibevoice/voices/streaming_model` 目录：
  mkdir -p models/vibevoice/voices/streaming_model
  cd models/vibevoice/voices/streaming_model

  # 下载核心音色 (仅示例，全部音色请参考官方 GitHub)
  # ⚠️ 注意：官方仓库目前暂未提供中文 (.pt) 预设，建议使用英文或日韩文测试，或自行制作预设。

  # 英文 (English)
  wget -N --no-check-certificate https://github.com/microsoft/VibeVoice/raw/main/demo/voices/streaming_model/en-Carter_man.pt
  wget -N --no-check-certificate https://github.com/microsoft/VibeVoice/raw/main/demo/voices/streaming_model/en-Emma_woman.pt
  # 日语 (Japanese)
  wget -N --no-check-certificate https://github.com/microsoft/VibeVoice/raw/main/demo/voices/streaming_model/jp-Spk0_man.pt
  wget -N --no-check-certificate https://github.com/microsoft/VibeVoice/raw/main/demo/voices/streaming_model/jp-Spk1_woman.pt
  # 韩语 (Korean)
  wget -N --no-check-certificate https://github.com/microsoft/VibeVoice/raw/main/demo/voices/streaming_model/kr-Spk0_woman.pt
  wget -N --no-check-certificate https://github.com/microsoft/VibeVoice/raw/main/demo/voices/streaming_model/kr-Spk1_man.pt
  # 更多语言 (德语 de, 法语 fr, 意大利语 it, 西班牙语 sp/es, 葡萄牙语 pt 等) 均支持！

  # ===== 1.5B 基础模型 =====
  mkdir -p models/vibevoice/microsoft/VibeVoice-1.5B
  hf download microsoft/VibeVoice-1.5B --local-dir models/vibevoice/microsoft/VibeVoice-1.5B
  # 补全 Tokenizer
  wget https://huggingface.co/Qwen/Qwen2.5-1.5B/resolve/main/tokenizer.json -P models/vibevoice/microsoft/VibeVoice-1.5B/
  wget https://huggingface.co/Qwen/Qwen2.5-1.5B/resolve/main/tokenizer_config.json -P models/vibevoice/microsoft/VibeVoice-1.5B/
  wget https://huggingface.co/Qwen/Qwen2.5-1.5B/resolve/main/vocab.json -P models/vibevoice/microsoft/VibeVoice-1.5B/
  wget https://huggingface.co/Qwen/Qwen2.5-1.5B/resolve/main/merges.txt -P models/vibevoice/microsoft/VibeVoice-1.5B/

  # ===== 7B 模型 =====
  mkdir -p models/vibevoice/vibevoice/VibeVoice-7B
  hf download vibevoice/VibeVoice-7B --local-dir models/vibevoice/vibevoice/VibeVoice-7B
  # 补全 Tokenizer (如果没有)
  wget https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/tokenizer.json -P models/vibevoice/vibevoice/VibeVoice-7B/
  wget https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/tokenizer_config.json -P models/vibevoice/vibevoice/VibeVoice-7B/
  wget https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/vocab.json -P models/vibevoice/vibevoice/VibeVoice-7B/
  wget https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/merges.txt -P models/vibevoice/vibevoice/VibeVoice-7B/
  ```

#### 3.12 VoxCPM 1.5 TTS (Beta)

- **用途**: 下一代 Tokenizer-free TTS 模型，提供 44.1kHz 原生高保真音质。
- **状态**: **Beta (骨架已上线，推理逻辑完善中)**
- **手动下载指南 (Manual Download)**:
  如果节点无法自动下载模型，请手动下载 `openbmb/VoxCPM1.5` 并放入以下目录：

  ```text
  ComfyUI/models/voxcpm/VoxCPM1.5/
  ├── model.safetensors
  ├── config.json
  └── ... (其他相关文件)
  ```

  **HuggingFace 下载命令**:

  ```bash
  mkdir -p models/voxcpm/VoxCPM1.5
  hf download openbmb/VoxCPM1.5 --local-dir models/voxcpm/VoxCPM1.5
  ```

  **降噪模型 (Optional Denoiser - ZipEnhancer)**:
  默认开启 `enable_denoiser` 会自动从 ModelScope 下载 `speech_zipenhancer_ans_multiloss_16k_base` 到以下目录：

  ```text
  ComfyUI/models/voxcpm/speech_zipenhancer_ans_multiloss_16k_base/
  ```

  如需手动下载（或离线使用）：

  ```bash
  pip install modelscope
  modelscope download --model iic/speech_zipenhancer_ans_multiloss_16k_base --local_dir models/voxcpm/speech_zipenhancer_ans_multiloss_16k_base
  ```

  > [!NOTE]
  > **频谱特征说明 (Spectral Analysis Note)**:
  > 用户实测发现，尽管该模型输出 44.1kHz 格式，但在频谱图上可能观察到以下特征：
  >
  > 1. **静音区能量较高 (High Noise Floor)**: 并非绝对静默。
  > 2. **水平条纹 (Horizontal Stripes)**: 特别是在低频区域，这通常是 Neural Upsampling (VAE/GAN) 重构波形的典型痕迹。
  > 3. **听感**: 这种“升频痕迹”可能会带来轻微的机械感或金属音，这属于模型权重的固有特性。
  >

#### 💡 用户实测与选型指南 (Model Comparison & Selection)

经过深度测试，我们在三个主流模型中整理了以下对比，助您选择最适合的引擎：

| 维度                                     | **VoxCPM 1.5** (800M)                                                                                                                                                   | **CosyVoice 3.0** (0.5B/1.5B)                                         | **VibeVoice** (1.5B/7B)                                  |
| :--------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------- | :------------------------------------------------------------- |
| **音质 (Fidelity)**                | **44.1kHz 格式**`<br>`虽然物理格式为 44.1k，但因采用 **Neural Upsampling** (神经升频) 技术，听感上会有**含混 (Muffled)** 或**金属感**，且伴有底噪。 | **优秀**`<br>`听感最自然，但采样率稍低 (22/24kHz)，有时需 AI 增强。 | **良好**`<br>`主要强在语气自然度，纯音质略逊。         |
| **推理速度 (Speed)**               | **🚀 冠军 (RTF ~0.17)**`<br>`得益于 Tokenizer-free，极其高效。                                                                                                        | **极快**`<br>`流式响应仅 150ms，且支持 TensorRT 加速。              | **一般/较慢**`<br>`7B 版本较重，更适合离线生成。       |
| **克隆能力 (Cloning)**             | **SOTA** (Zero-Shot)`<br>`只需 3-10秒，对**音色质感**还原极高。                                                                                                 | **SOTA** (稳定性)`<br>`对**说话韵律/口音**的捕捉最准。        | **良好**`<br>`适合克隆特定语气，而非纯粹音色。         |
| **多语言/方言**                    | **中/英** (双语优化)                                                                                                                                                    | **👑 霸主** (9种语言 + 18种方言)                                      | **中/英**                                                |
| **语音转换 (VC)** (Audio-to-Audio) | ❌**不支持**`<br>`仅支持 TTS (Text-to-Speech)。无法改变已有音频的音色。                                                                                               | ✅**支持**`<br>`可以将任意音频转换为任意音色 (保留语调/停顿)。      | ❌**不支持**`<br>`纯 TTS 模型。仅支持 Text-to-Speech。 |

**选型建议**:

- **追求“听起来最像真人” (音质+音色)**: 选 **VoxCPM 1.5**。它的 Tokenizer-free 架构带来了质的飞跃。
- **追求“方言/多语言/稳定性”**: 选 **CosyVoice 3.0**。目前依然是生产环境最稳的选择。
- **要做“长篇广播剧/播客”**: 选 **VibeVoice**。它的长窗口上下文优势依然不可替代。

### 4. 播客与对话生成 (Podcast & Dialogue Generation)

https://github.com/user-attachments/assets/9a5502c5-79e3-4fc8-8a2d-2cbdbdbbc860

> 🎬 **点击观看演示视频** (GitHub 限制，需要手工取消静音)

🚀 **新增功能**：这是专门为生成双人对话、相声、广播剧设计的完整工作流节点。能够自动解析剧本、调度多角色 TTS，并实现长音频的智能拼接。

#### 4.1 AIIA Podcast Script Parser (脚本解析器)

负责将自然语言剧本转换为机器可读的结构化数据。

- **Script Text**: 剧本输入区域。
  - **基本格式**: `角色名: 台词` (例如 `A: 大家好`)
  - **暂停控制**: `(Pause N)` (例如 `(Pause 0.5)` 表示暂停 0.5 秒)
  - **情感标签** (仅 CosyVoice): `[Emotion] 台词` (例如 `[Happy] 大家好`)
- **Speaker Mapping**: 角色映射配置 (可选)。
  - 格式: `原剧本角色名=A`, `原剧本角色名=B`
  - 示例: `Teacher=A`, `Student=B`

#### 4.2 AIIA Dialogue TTS (对话生成引擎)

核心调度与生成节点，支持自动角色切换和长音频拼接。

- **TTS Engine**: 后端引擎选择。
  - **CosyVoice**: 精准控制型。
  - **VibeVoice**: 自然演绎型。
- **Speaker A/B/C**:
  - **Ref Audio**: 参考音频 (用于 Zero-Shot 克隆)。
  - **ID**: 内置音色 ID (如 CosyVoice 的 `Chinese Female`)。
- **Batch Mode**: 生成模式控制。
  - `Natural (Hybrid)`: 混合批处理。仅在 `(Pause)` 处断开。语流最自然，但可能发生音色泄漏。
  - `Strict (Per-Speaker)`: 严格模式。每句话都会强制断开重置。彻底杜绝音色泄漏，但对话流畅度略低。
  - `Whole (Single Batch)`: 全量模式。无视所有暂停，一次性生成整本剧本。连贯性最强，但无法控制停顿时间。

#### 4.3 AIIA Subtitle Gen (字幕生成器)

**[v1.7.0 New]** 无需 STT，直接从生成过程中提取精准时间轴。

- **Input**:
  - `segments_info`: 来自 `AIIA Dialogue TTS` 的输出。
- **Output**:
  - `SRT`: 通用字幕格式。
  - `ASS`: 高级排版字幕格式 (自动区分角色颜色)。
- **原理**:
  - **CosyVoice**: 使用生成时的精确时长。
  - **VibeVoice**: 使用**智能插值算法 (Smart Interpolation)**，根据字符长度自动计算长音频段内的单句时间轴。

#### 4.4 AIIA Subtitle Preview (字幕预览)

**[v1.7.1 New]** 实时校验音画同步效果。

- **Input**:
  - `subtitle_content`: 来自 `Subtitle Gen` 的 `srt_content` 或 `ass_content`。
  - `audio` (Optional): 待预览的音频。
- **Output**:
  - **交互式界面**: 提供 Web 播放器，按时间轴滚动显示字幕。
  - **ASS 样式渲染**: 尝试还原 ASS 字幕的字体颜色、大小和描边效果。

#### 4.5 Interactive Teaching (Web Export) (互动式教学导出)

**[v1.8.1 New]** 将播客升级为视听同步的互动网页。支持“读写分离”的缓存优化，修改 Visual 标签无需重跑 TTS。

- **工作流 (Workflow)**:
    1.  `Script Parser` 输出 `tts_data` (连接到 TTS) 和 `full_script` (连接到 Merge)。
    2.  `AIIA Dialogue TTS` 生成音频和 `segments_info`。
    3.  `AIIA Segment Merge` 将 `full_script` 中的 Visual 标签重新贴回到 `segments_info` 时间轴上。
    4.  `AIIA Web Export` 生成最终 HTML。
- **Input**:
    - `audio`: 音频信号。
    - `segments_info`: 来自 Merge 节点的包含 Visual 信息的 JSON。
    - `template`: `Split Screen` (适合宽屏) 或 `Presentation` (适合演示)。
- **Visual Tag 语法**:
    - 在剧本中插入 `(Visual: url)`。
    - 支持绝对 URL: `(Visual: https://example.com)`
    - 支持相对路径: `(Visual: ./slides/01.jpg)` (相对于导出 HTML 的位置)

#### 💡 引擎选型与最佳实践 (Best Practices)

| 特性               | **CosyVoice**                        | **VibeVoice**                                                                              |
| :----------------- | :----------------------------------------- | :----------------------------------------------------------------------------------------------- |
| **核心优势** | **精准控制 (Instruction)**           | **自然演绎 (Context-Aware)**                                                               |
| **情感控制** | ✅**支持** (使用 `[Happy]` 等标签) | ❌ 不支持显式标签 (依赖上下文)                                                                   |
| **生成逻辑** | **逐句生成** (严格遵循每句话的指令)  | **混合批处理** (Hybrid Batching)                                                           |
| **最佳场景** | 需要精确指定某句话语气、方言时             | 长篇对话、广播剧、闲聊                                                                           |
| **使用建议** | 可以在剧本中详细标注情感。                 | **尽量减少 `(Pause)`**！`<br>`让多句对话连在一起，模型能更好地联系上下文产生自然语气。 |

#### 📝 综合测试剧本 (Example Script)

复制以下内容到 Script Parser 进行测试，能够充分体现两种引擎的特性：

```text
# 这是一个展示 CosyVoice 和 VibeVoice 能力的综合剧本
# 角色映射建议：A=男声, B=女声

A: 大家好，欢迎来到 AIIA 播客直播间。
B: [开心] 哇，今天的人气好高啊！看到这么多朋友在线，我太激动了。

(Pause 0.5)

A: 呵呵，淡定一点。即使是 VibeVoice 这种基于 LLM 的模型，也需要你保持从容。
B: [疑惑] 为什么？难道它不喜欢太吵闹的声音吗？
A: 不是不喜欢，而是因为它会“读取”上下文。你越自然，它演得越像。

(Pause 0.8)

A: 比如说，如果我们用 CosyVoice，我可以强行指定你现在的状态。
B: [悲伤] 比如让我突然变得很伤心？
A: 对，就像这样。CosyVoice 是“指哪打哪”，非常听话。

(Pause 0.5)

B: [机器人的方式] 那如果我变成一个机器人呢？可以吗？
A: 哈哈，完全没问题。

(Pause 1.0)

A: 但是，如果你想演一场相声，或者很自然的闲聊，VibeVoice 的“混合批处理”就是神技了。
B: [excited] 也就是它会把我们现在说的这一长串话，一口气生成出来？
A: 没错！只要我们中间不加 Pause，它就会一口气读完，语气极其连贯，就像真人对话一样。
B: 太神奇了！那我们快去生成试试吧！
```

### 5. 图像工具 (Image Utilities)

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

### [1.8.4] - 2026-01-19

- **VibeVoice Speed Control**: 实现了基于系统 `sox` 命令的**变速不变调**（Time Stretching）。
- **稳定性修复**: 
    - 修复了 VibeVoice 在调整速度时由于张量类型不匹配（Half vs Float）导致的崩溃。
    - 强制所有音频输出为 `float32`，解决了在 `speed=1.0` 且有参考音频时，下游节点（如 Resemble Enhance）报错的问题。
- **依赖更新**: 新增 `sox` 依赖。Linux 服务器用户请确保安装系统库：`sudo apt-get install libsox-dev sox`。

### [1.8.3] - 2026-01-07

- **VibeVoice TTS (Standard)**: `reference_audio` 变为可选参数。如果不输入，节点会自动加载内置的高品质女声种子，方便快速测试。
- **Fix**: 修复 GitHub Actions 发布的子模块错误。

### [1.8.1] - 2026-01-05

- **互动式教学 (Interactive Teaching)**: 新增 `AIIA Web Export` 节点，一键生成包含播放器、字幕和同步 Visual 展示的 HTML 页面。
- **Visual Tag 支持**: 剧本解析器支持 `(Visual: url)` 标签，并支持相对路径。
- **缓存优化 (Caching)**: 实现了 Script Parser 的读写分离和 `Segment Merge` 节点，支持修改 Visual 标签而不触发 TTS 重生成。
- **字幕增强**: 修复了 ASS 字幕的多角色颜色显示问题 (自动分配颜色)。

### [1.3.1] - 2026-01-03

本次更新标志着 CosyVoice 架构的**完全大一统**。通过“手术级”精准推理逻辑，我们成功解决了 300M 系列模型的所有顽疾。

#### ✅ 已完成 (Perfectly Supported)

- **V3 系列 (0.5B)**: 完美支持。性别、方言、情感以及指令文本均能精准跟随。
- **V2 系列 (0.5B)**: 完美支持。
- **300M-Instruct (V1)**: **彻底修复 (Fully Fixed)**。
  - **指令跟随修复**: 强制注入官方缺失的 `<|endofprompt|>` 边界标识，完美解决了模型将指令文本读出来的问题。
  - **能力矩阵**:
    - ✅ **支持**: 情感控制 (Happy/Sad/Angry)、语速控制 (Fast/Slow)、基础性别 (Male/Female)。
      > **Note**: 为了确保指令生效，**请务必使用英文描述** (如 "Sad tone", "Male speaker")。中文描述虽然不会导致报错，但模型极有可能忽略其语义。
      >
    - ❌ **不支持**: 方言指令 (Dialect) - 此模型架构原生不支持通过指令更改方言 (无论中文或英文)，请使用 V3 或 V2 模型获取方言能力。
- **300M-SFT / Base (V1)**: 恢复原生巅峰音质。
- **稳定性**: 杜绝了 `AudioDecoder`、`KeyError` 以及 `llm_embedding` 缺失导致的各种崩溃。

#### 🔈 资源更新

- **HQ 种子音色**: 内置从 300M-SFT 提取的高保真男声/女声种子音频。在零参考音频的“指令模式”下，V2/V3 将自动调用这些高保真资源，消除破碎感。
