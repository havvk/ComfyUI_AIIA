![image](https://github.com/user-attachments/assets/7e38d3fd-2084-4d0c-bf86-4b500eba5ced)

# AIIA Nodes for ComfyUI

欢迎来到 AIIA Nodes for ComfyUI 仓库！这是一个旨在为 ComfyUI 提供一系列强大、直观且高度可定制的节点的集合。这些节点专注于简化复杂的工作流，并为创意工作者提供最大的灵活性，**特别是解决在生成长视频或处理大量图像帧时常见的内存不足（OOM）问题**。

---

## 🚀 核心设计理念：为大规模生成而生

本节点套件的一个核心设计初衷，就是为了攻克困扰许多用户的技术难题。

### 🌟 告别OOM：为处理海量帧而生
处理成百上千张高清图像帧时，轻易就会耗尽 VRAM 和系统内存，导致工作流中断。AIIA 节点通过 **增量式处理（Incremental Processing）** 的策略从根本上解决了这个问题。

无论是从磁盘流式读取帧进行视频合并，还是将生成结果逐帧保存到磁盘，我们的节点都**永远不需要将所有图像一次性加载到内存中**。这意味着您可以轻松生成数千帧的视频，而无需再为内存限制而烦恼。

### ✨ 无缝与高效的平衡
我们提供了两种工作模式，以适应不同场景：
-   **内存模式**: 对于短序列或测试，可以直接将上游节点的 `IMAGE` 张量输入，实现无缝、快速的内存内处理。
-   **磁盘模式**: 对于长序列的最终生成，节点会高效地从磁盘流式读取/写入帧，保证了稳定性和极低的内存占用。

### 🔧 强大且可扩展的预设系统
通过简单的 **JSON 文件**，您可以完全自定义视频和音频的编码参数，并将其保存为可复用的格式预设。这使得高级用户可以轻松实现复杂的 FFmpeg 配置，而无需修改任何代码。

---

## 🛠️ 安装与先决条件

### 1. 安装 FFmpeg (视频节点必须)
视频节点依赖 `ffmpeg` 和 `probe`。
-   访问 [FFmpeg 官网](https://ffmpeg.org/download.html) 下载并安装。
-   **强烈建议**将 FFmpeg 的 `bin` 目录添加到您系统的 `PATH` 环境变量中。
-   在终端中运行 `ffmpeg -version` 和 `ffprobe -version` 来验证安装。

### 2. 安装 NeMo 模型 (音频AI节点必须)
音频处理节点（如说话人日志）依赖 NeMo 模型。
-   在 ComfyUI 的 `models` 目录下，创建一个名为 `nemo_models` 的子目录。最终路径应为 `ComfyUI/models/nemo_models/`。
-   从 [NVIDIA NeMo 目录](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/diar_sortformer_telephonic_tele_ft_marblenet) 下载所需的 `.nemo` 模型文件（例如 `diar_sortformer_4spk-v1.nemo`）。
-   将下载的 `.nemo` 文件放入 `ComfyUI/models/nemo_models/` 目录中。

### 3. 安装本节点套件
进入 ComfyUI 的自定义节点目录，然后克隆本仓库：
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/havvk/ComfyUI_AIIA.git
```
最后，**重启 ComfyUI**。

---

## ✨ 可用节点

### 视频合并 (AIIA, 图像或目录)

这是一个功能强大且高度可定制的视频合并节点，是您工作流中处理视频生成的终极解决方案。

![video_combine_node_ui](https://github.com/user-attachments/assets/4185c2a7-e1a6-4980-ac8a-3d326bff4b87)

#### 核心亮点
-   **内存高效**: 通过 `frames_directory` 输入，可以处理几乎无限数量的图像帧，完美解决了 OOM 问题。
-   **直接张量输入**: 接受上游节点的 `IMAGE` 张量，方便快速迭代和测试。
-   **全面的音频控制**: 支持 `AUDIO` 张量和外部文件，并提供对编解码器和码率的精细控制。
-   **智能自动配置**: `auto` 模式能自动应用格式预设中的音频参数，并能自动检测源文件的码率。

---

### FLOAT 影片生成 (内存与磁盘模式)

这组节点封装了先进的 **FLOAT** 模型，能够根据参考图像和音频生成高质量的口型同步影片。我们提供了两种模式，以应对不同长度的生成需求。

#### 1. Float Process (AIIA In-Memory)
-   **用途**: 用于生成**短片**、快速预览或直接与其他内存节点（如视频合并的 `images` 输入）连接。
-   **输出**: `IMAGE` 张量。
-   **优势**: 速度快，流程无缝。
-   **限制**: 受限于您的 VRAM 和系统内存大小，不适合生成长视频。

#### 2. Float Process (AIIA To-Disk for Long Audio)
-   **用途**: 专门用于**长音频**的影片生成，是**解决OOM问题的关键**。
-   **输出**: `STRING` (包含所有生成帧的目录路径) 和 `INT` (帧总数)。
-   **优势**: 在解码过程中，节点以小批量方式处理帧并**逐帧保存到磁盘**，内存占用极低，可以处理任意长度的音频。
-   **工作流**: 此节点的输出目录可以直接作为 **视频合并节点** 的 `frames_directory` 输入，构建一个完整的、内存高效的 talking head 视频生成管线。

---

### 音频处理：说话人日志 (Diarization)

这组节点利用 **NeMo Sortformer** E2E 模型，为您的音频提供先进的说话人识别功能。在4090RTX显卡上只需2秒钟就能完成10分钟音频的声纹分割聚类任务。

#### 1. AIIA Generate Speaker Segments
-   **用途**: 对一段音频进行分析，找出“**谁在什么时候说话**”。
-   **输入**: `AUDIO` 张量。
-   **输出**: `WHISPER_CHUNKS` (一个结构化的数据，包含一系列带有说话人标签的时间片段，如 `SPEAKER_00`, `SPEAKER_01` 等)。
-   **亮点**: 提供多种**后处理配置**（从宽松到严格），让您可以微调分割的灵敏度，以适应不同质量的音频。

#### 2. AIIA E2E Speaker Diarization
-   **用途**: 将 `Generate Speaker Segments` 的识别结果**精确地应用**到由 Whisper 等工具生成的、带有文本的 `WHISPER_CHUNKS` 上。
-   **输入**: `WHISPER_CHUNKS` (来自文本转录节点) 和 `AUDIO` 张量。
-   **输出**: 更新后的 `WHISPER_CHUNKS`，其中每个文本块都已被赋予了最匹配的说话人标签。
-   **工作流**: `(音频) -> Whisper -> (文本Chunks)` + `(音频)  => `E2E Diarizer` => **最终带有说话人标签的文本稿**。

---

### 实用工具 (Utilities)

#### Image Concatenate (AIIA Utils, Disk)
-   **用途**: 将两个图像序列（来自两个不同的目录）逐帧拼接在一起，非常适合创建**对比视频**或**多面板视频**。
-   **核心亮点 (OOM-Safe)**: 此节点**逐帧读取、处理和保存**，从不将整个图像序列加载到内存中，因此可以处理任意数量的帧。
-   **功能**:
    -   支持上下左右四个方向的拼接。
    -   可自动调整其中一个图像序列的尺寸以匹配另一个，并保持宽高比。
    -   可自定义背景填充颜色。
-   **输出**: `STRING` (包含所有拼接后帧的新目录路径)。

---

## ❓ 故障排查

-   **错误: "FFmpeg not found" / "NeMo model not found"**
    -   请返回阅读 [安装与先决条件](#-安装与先决条件) 部分，确保所有依赖都已正确安装和放置。
-   **错误: "帧目录验证失败"**
    -   请确保您填写的路径是**绝对路径**且真实存在。

## 许可证

本项目采用 [MIT 许可证](LICENSE)。
