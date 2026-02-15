import json
import os
import re
import warnings
import torch
import numpy as np


class AIIA_Podcast_Stitcher:
    """
    将分轨生成的多角色音频按原始对话顺序精确拼接。
    
    利用 ASR 词级时间戳找到每句话在音频中的边界，切分后交错拼接。
    """

    NODE_NAME = "AIIA Podcast Stitcher"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "split_map": ("STRING", {"forceInput": True}),
                "audio_A": ("AUDIO",),
                "audio_B": ("AUDIO",),
                "asr_A": ("ASR_RESULT",),
                "asr_B": ("ASR_RESULT",),
            },
            "optional": {
                "gap_duration": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "说话人交替时插入的过渡时长（秒）"
                }),
                "padding": ("FLOAT", {
                    "default": 0.10, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "每个切片前后保留的呼吸/尾音余量（秒）"
                }),
                "fade_ms": ("INT", {
                    "default": 30, "min": 5, "max": 100, "step": 5,
                    "tooltip": "切片首尾的余弦淡入淡出时长（毫秒），越长越平滑"
                }),
                "use_vad": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "启用 Silero VAD 模型精确检测语音边界（首次使用自动下载 ~2MB 模型）"
                }),
                "use_forced_align": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "启用 MMS Forced Alignment 字级强制对齐（需要 ~1.2GB 模型，精度最高）"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING",)
    RETURN_NAMES = ("audio", "segments_info",)
    FUNCTION = "stitch"
    CATEGORY = "AIIA/Podcast"

    def _audio_to_numpy(self, audio: dict) -> tuple:
        """将 ComfyUI AUDIO 转为 numpy 数组和采样率。"""
        waveform = audio["waveform"]
        sr = audio["sample_rate"]

        if waveform.ndim == 3:
            wav = waveform[0]
        else:
            wav = waveform

        if wav.ndim == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0)
        elif wav.ndim == 2:
            wav = wav.squeeze(0)

        return wav.cpu().numpy().astype(np.float32), sr

    def _find_sentence_boundaries(self, asr_words: list, sentences: list, total_duration: float) -> list:
        """
        将 ASR 词级时间戳与原始句子列表对齐，找到每句话在音频中的时间范围。
        
        三层匹配策略：
        1. 精确子串匹配（去标点后）
        2. 编辑距离模糊匹配（滑动窗口，容忍 ASR 错字/漏字）
        3. 间隙填补 / 等分回退
        """
        log = f"[{self.NODE_NAME}]"

        if not asr_words:
            print(f"{log} ASR 结果为空，使用等分策略")
            return self._fallback_equal_split(sentences, total_duration)

        if not sentences:
            return []

        # 构建 ASR 文本和字符到词索引的映射
        asr_full_text = ""
        char_to_word_idx = []  # char_to_word_idx[i] = 该字符属于哪个 word
        for word_idx, w in enumerate(asr_words):
            word_text = w["word"]
            for ch in word_text:
                char_to_word_idx.append(word_idx)
            asr_full_text += word_text

        print(f"{log} ASR 全文 ({len(asr_full_text)} 字): {asr_full_text[:100]}...")

        # 为每句话找到在 ASR 文本中的匹配位置
        boundaries = []
        search_start = 0  # 保证顺序匹配

        for sent_idx, sentence in enumerate(sentences):
            # 清理句子文本（去除标点符号和空格，与 ASR 输出对齐）
            clean_sent = self._clean_text_for_matching(sentence)

            if not clean_sent:
                print(f"{log} 句子 {sent_idx} 清理后为空: '{sentence}'")
                boundaries.append(None)
                continue

            # === 第 1 层：精确子串匹配 ===
            match_pos = asr_full_text.find(clean_sent, search_start)

            if match_pos != -1:
                match_end = match_pos + len(clean_sent) - 1
                match_quality = "精确"
            else:
                # === 第 2 层：编辑距离模糊匹配 ===
                match_pos, match_end, edit_dist = self._fuzzy_find(
                    asr_full_text, clean_sent, search_start
                )

                if match_pos != -1:
                    match_quality = f"模糊(ed={edit_dist})"
                else:
                    print(f"{log} 句子 {sent_idx} 无法匹配: '{clean_sent[:30]}...'")
                    boundaries.append(None)
                    continue

            # 映射字符位置到词索引
            start_word_idx = char_to_word_idx[match_pos] if match_pos < len(char_to_word_idx) else len(asr_words) - 1
            end_word_idx = char_to_word_idx[min(match_end, len(char_to_word_idx) - 1)]

            start_time = asr_words[start_word_idx]["start"]
            end_time = asr_words[end_word_idx]["end"]

            print(f"{log} 句子 {sent_idx} [{match_quality}]: "
                  f"'{clean_sent[:15]}' → pos={match_pos}-{match_end}, "
                  f"time={start_time:.2f}-{end_time:.2f}s")

            boundaries.append({
                "start": start_time,
                "end": end_time,
                "start_word_idx": start_word_idx,
                "end_word_idx": end_word_idx,
            })

            # 更新搜索起点
            search_start = match_end + 1

        # 填补未匹配的句子（使用前后句子的时间插值）
        boundaries = self._fill_missing_boundaries(boundaries, asr_words, total_duration)

        # 扩展边界到句间间隙的中点（避免截断尾音）
        boundaries = self._expand_to_midpoints(boundaries, total_duration)

        return boundaries

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """计算两个字符串的编辑距离（Levenshtein distance），使用空间优化的 DP。"""
        m, n = len(s1), len(s2)
        if m == 0:
            return n
        if n == 0:
            return m

        # 只需两行
        prev = list(range(n + 1))
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            curr[0] = i
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    curr[j] = prev[j - 1]
                else:
                    curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
            prev, curr = curr, prev

        return prev[n]

    def _fuzzy_find(self, haystack: str, needle: str, search_start: int = 0,
                    max_error_ratio: float = 0.4) -> tuple:
        """
        在 haystack 中从 search_start 开始，用滑动窗口+编辑距离找到与 needle 最相似的子串。
        
        参数:
            haystack: ASR 全文
            needle: 待匹配的原始句子（已去标点）
            search_start: 搜索起始位置
            max_error_ratio: 允许的最大错误率（编辑距离 / needle 长度）
        
        返回:
            (match_pos, match_end, edit_distance)  或  (-1, -1, -1) 表示失败
        """
        needle_len = len(needle)
        if needle_len == 0:
            return (-1, -1, -1)

        max_errors = int(needle_len * max_error_ratio)
        remaining = haystack[search_start:]
        remaining_len = len(remaining)

        if remaining_len == 0:
            return (-1, -1, -1)

        best_pos = -1
        best_end = -1
        best_dist = needle_len + 1  # 初始化为一个大值

        # 尝试多种窗口大小（needle 长度的 ±30%），处理 ASR 漏字/多字的情况
        window_sizes = set()
        for ratio in [1.0, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2]:
            ws = max(1, int(needle_len * ratio))
            if ws <= remaining_len:
                window_sizes.add(ws)

        # 限制搜索范围以避免 O(n²) 爆炸
        # 在合理的搜索范围内：从 search_start 开始，最多搜到 needle 长度的 3 倍
        max_search_len = min(remaining_len, needle_len * 3 + 20)

        for window_size in sorted(window_sizes):
            for i in range(0, max_search_len - window_size + 1):
                candidate = remaining[i:i + window_size]
                dist = self._edit_distance(needle, candidate)

                if dist < best_dist:
                    best_dist = dist
                    best_pos = search_start + i
                    best_end = search_start + i + window_size - 1

                    # 如果编辑距离为 0 或 1，可以提前退出
                    if dist <= 1:
                        break

            if best_dist <= 1:
                break

        # 只接受错误率在阈值内的匹配
        if best_dist <= max_errors:
            return (best_pos, best_end, best_dist)
        else:
            return (-1, -1, -1)

    def _clean_text_for_matching(self, text: str) -> str:
        """清理文本用于与 ASR 输出匹配：去除标点、空格、英文转小写。"""
        import re
        # 去除常见中英文标点和空格
        cleaned = re.sub(r'[，。！？、；：""''「」【】（）《》\s,\.!?\-\;\:\"\'\(\)\[\]\{\}…—~～·]', '', text)
        # 英文转小写（ASR 可能输出不同大小写）
        cleaned = cleaned.lower()
        return cleaned

    def _fallback_equal_split(self, sentences: list, total_duration: float) -> list:
        """回退策略：按句子字符数等比例分配时间。"""
        if not sentences:
            return []

        total_chars = sum(len(s) for s in sentences)
        if total_chars == 0:
            segment_duration = total_duration / len(sentences)
            return [{"start": i * segment_duration, "end": (i + 1) * segment_duration}
                    for i in range(len(sentences))]

        boundaries = []
        current_time = 0.0
        for sent in sentences:
            ratio = len(sent) / total_chars
            duration = ratio * total_duration
            boundaries.append({
                "start": round(current_time, 3),
                "end": round(current_time + duration, 3),
            })
            current_time += duration

        return boundaries

    def _fill_missing_boundaries(self, boundaries: list, asr_words: list, total_duration: float) -> list:
        """填补未能匹配的句子边界。"""
        filled = list(boundaries)

        for i in range(len(filled)):
            if filled[i] is not None:
                continue

            # 找前一个已知边界
            prev_end = 0.0
            for j in range(i - 1, -1, -1):
                if filled[j] is not None:
                    prev_end = filled[j]["end"]
                    break

            # 找后一个已知边界
            next_start = total_duration
            for j in range(i + 1, len(filled)):
                if filled[j] is not None:
                    next_start = filled[j]["start"]
                    break

            # 在空隙中均匀分配
            gap_count = 0
            gap_start_idx = i
            for j in range(i, len(filled)):
                if filled[j] is None:
                    gap_count += 1
                else:
                    break

            gap_duration = (next_start - prev_end) / gap_count
            for k in range(gap_count):
                filled[gap_start_idx + k] = {
                    "start": round(prev_end + k * gap_duration, 3),
                    "end": round(prev_end + (k + 1) * gap_duration, 3),
                }

        return filled

    # ========== Forced Alignment (MMS_FA) ==========
    _fa_model = None
    _fa_tokenizer = None
    _fa_aligner = None

    @classmethod
    def _load_fa_model(cls):
        """懒加载 MMS_FA 模型（类级单例）。优先从 models/mms_fa/ 读取本地权重。"""
        if cls._fa_model is not None:
            return cls._fa_model, cls._fa_tokenizer, cls._fa_aligner
        try:
            from torchaudio.pipelines import MMS_FA as bundle

            # 确保本地模型文件在 hub cache 中（symlink）
            import folder_paths
            local_model = os.path.join(folder_paths.models_dir, "mms_fa", "model.pt")
            hub_cache = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints", "model.pt")
            if os.path.exists(local_model) and not os.path.exists(hub_cache):
                os.makedirs(os.path.dirname(hub_cache), exist_ok=True)
                os.symlink(local_model, hub_cache)
                print(f"[{cls.__name__}] 已链接本地模型: {local_model} -> {hub_cache}")

            warnings.filterwarnings('ignore', message='.*forced_align has been deprecated.*')
            model = bundle.get_model().to('cuda' if torch.cuda.is_available() else 'cpu')
            tokenizer = bundle.get_tokenizer()
            aligner = bundle.get_aligner()
            cls._fa_model = model
            cls._fa_tokenizer = tokenizer
            cls._fa_aligner = aligner
            print(f"[{cls.__name__}] MMS Forced Alignment 模型加载成功")
            return model, tokenizer, aligner
        except Exception as e:
            print(f"[{cls.__name__}] MMS FA 加载失败: {e}")
            return None, None, None

    @staticmethod
    def _chinese_to_pinyin(text):
        """将中文文本转为拼音字符串（MMS_FA 只接受小写拉丁字符）。"""
        from pypinyin import lazy_pinyin, Style
        # 去除标点和特殊字符，保留中文、字母、数字、空格
        clean = re.sub(r'[^\u4e00-\u9fff\w\s]', '', text)
        if not clean.strip():
            return text.lower()
        # lazy_pinyin 会把中文转拼音，非中文字符原样保留
        result = ' '.join(lazy_pinyin(clean, style=Style.NORMAL))
        # 全部小写 + 只保留 MMS_FA tokenizer 支持的字符 [a-z, space, ', -]
        result = result.lower()
        result = re.sub(r"[^a-z\s'\-]", '', result)
        # 合并多余空格
        result = re.sub(r'\s+', ' ', result).strip()
        return result if result else 'a'

    def _forced_align_sentences(self, wav_np, sr, sentences):
        """
        对完整音频做 MMS Forced Alignment，返回每句的精确 {start, end} 时间。
        
        工作流程：
        1. 将所有句子拼为完整 pinyin 转录
        2. MMS_FA 模型生成 emission
        3. aligner 做 CTC 强制对齐，得到每个 word 的 token_spans
        4. 根据句子→word 映射还原每句的 start/end
        """
        model, tokenizer, aligner = self._fa_model, self._fa_tokenizer, self._fa_aligner
        if model is None:
            return None
        
        log = f"[{self.NODE_NAME}]"
        
        # 1. 准备 pinyin 转录（以 word 为单位，用空格分隔）
        all_words = []       # pinyin words 列表
        sentence_word_ranges = []  # 每句对应的 [start_word_idx, end_word_idx)
        
        for sent in sentences:
            pinyin_str = self._chinese_to_pinyin(sent)
            words = pinyin_str.split()
            if not words:
                words = ['a']  # 占位符，避免空句子
            start_idx = len(all_words)
            all_words.extend(words)
            sentence_word_ranges.append((start_idx, len(all_words)))
        
        if not all_words:
            return None
            
        # 2. 音频重采样到 16kHz（MMS_FA 要求）
        import torchaudio
        FA_SR = 16000
        wav_tensor = torch.from_numpy(wav_np).float().unsqueeze(0)  # [1, T]
        if sr != FA_SR:
            wav_tensor = torchaudio.functional.resample(wav_tensor, sr, FA_SR)
        
        device = next(model.parameters()).device
        wav_tensor = wav_tensor.to(device)
        
        # 3. 模型推理
        try:
            with torch.inference_mode():
                emission, _ = model(wav_tensor)
            
            token_spans = aligner(emission[0], tokenizer(all_words))
        except Exception as e:
            print(f"{log} FA 对齐失败: {e}")
            return None
        
        # 4. 将 token_spans 映射回每句的时间范围
        num_frames = emission.size(1)
        ratio = wav_tensor.size(1) / num_frames / FA_SR  # frame → 秒
        
        results = []
        for sent_idx, (w_start, w_end) in enumerate(sentence_word_ranges):
            if w_start >= len(token_spans) or w_end > len(token_spans):
                # 回退：无对齐结果
                results.append(None)
                continue
            
            # 句子的第一个 word 的第一个 token → start
            # 句子的最后一个 word 的最后一个 token → end
            first_spans = token_spans[w_start]
            last_spans = token_spans[w_end - 1]
            
            if not first_spans or not last_spans:
                results.append(None)
                continue
            
            t_start = first_spans[0].start * ratio
            t_end = last_spans[-1].end * ratio
            
            # 计算对齐置信度
            all_span_scores = []
            for wi in range(w_start, w_end):
                for s in token_spans[wi]:
                    all_span_scores.append(s.score)
            avg_score = sum(all_span_scores) / len(all_span_scores) if all_span_scores else 0
            
            results.append({
                'start': round(t_start, 4),
                'end': round(t_end, 4),
                'score': round(avg_score, 3)
            })
            print(f"{log} FA 句子 {sent_idx}: [{t_start:.3f}s - {t_end:.3f}s] score={avg_score:.3f} '{sentences[sent_idx][:20]}'")
        
        return results

    @staticmethod
    def _compute_iou(start1, end1, start2, end2):
        """计算两个时间区间的 IoU（Intersection over Union）。"""
        inter_start = max(start1, start2)
        inter_end = min(end1, end2)
        intersection = max(0, inter_end - inter_start)
        union = max(end1, end2) - min(start1, start2)
        return round(intersection / union, 3) if union > 0 else 0.0

    # ========== Silero VAD ==========
    _vad_model = None
    _vad_utils = None

    @classmethod
    def _load_vad_model(cls):
        """懒加载 Silero VAD 模型（类级单例，只下载一次）。"""
        if cls._vad_model is not None:
            return cls._vad_model, cls._vad_utils
        try:
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            cls._vad_model = model
            cls._vad_utils = utils
            print(f"[{cls.__name__}] Silero VAD 模型加载成功")
            return model, utils
        except Exception as e:
            print(f"[{cls.__name__}] Silero VAD 加载失败: {e}")
            return None, None

    def _get_vad_timestamps(self, wav_np, sr, vad_model, vad_utils):
        """
        对完整音频运行 Silero VAD，返回语音区间列表 [{start: float, end: float}, ...]（单位：秒）。
        """
        get_speech_timestamps = vad_utils[0]
        
        # Silero VAD 要求 16kHz
        vad_sr = 16000
        wav_tensor = torch.from_numpy(wav_np).float()
        if sr != vad_sr:
            import torchaudio
            wav_tensor = torchaudio.functional.resample(wav_tensor, sr, vad_sr)
        
        # 运行 VAD
        timestamps = get_speech_timestamps(
            wav_tensor, vad_model,
            sampling_rate=vad_sr,
            threshold=0.3,              # 灵敏度（TTS 干净音频可以略低）
            min_speech_duration_ms=100,  # 最短语音段 100ms
            min_silence_duration_ms=50,  # 最短静音段 50ms
            speech_pad_ms=20,            # 语音两侧填充 20ms
            return_seconds=True
        )
        
        return timestamps  # [{start: float, end: float}, ...]

    def _refine_with_vad(self, cut_start, cut_end, vad_timestamps, search_margin=0.2):
        """
        用 VAD 区间精修 cut_start/cut_end。
        
        策略：找到与 [cut_start, cut_end] 重叠最大的 VAD 语音区间，
        用该区间的起止替代 ASR+扩展 得到的粗边界。
        """
        if not vad_timestamps:
            return cut_start, cut_end
        
        # 查找与当前切片重叠的 VAD 区间
        best_overlap = 0
        best_vad = None
        
        for vad in vad_timestamps:
            # 计算重叠
            overlap_start = max(cut_start - search_margin, vad['start'])
            overlap_end = min(cut_end + search_margin, vad['end'])
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_vad = vad
        
        if best_vad is None:
            return cut_start, cut_end
        
        # 如果 VAD 区间的边界在 ASR 边界附近，使用 VAD 的精确边界
        vad_start = best_vad['start']
        vad_end = best_vad['end']
        
        # cut_start: 取 VAD start 但不能比原始 cut_start 晚太多（避免切掉气口）
        if abs(vad_start - cut_start) < search_margin:
            cut_start = vad_start
        
        # cut_end: 取 VAD end 但不能比原始 cut_end 远太多（避免吃下一句）
        if abs(vad_end - cut_end) < search_margin:
            cut_end = vad_end
        
        return cut_start, cut_end

    # ========== 能量检测（Fallback） ==========
    def _refine_cut_point(self, wav, sr, time_s, search_radius=0.15, direction="both"):
        """
        在 time_s 附近找到能量最低的"静音山谷"作为切割位置。
        使用 20ms 窗口 + 滑动平均平滑，避免被瞬时低能量（清辅音等）欺骗。
        
        direction:
            "before" — 只在 [time_s - radius, time_s] 搜索（用于 cut_start）
            "after"  — 只在 [time_s, time_s + radius] 搜索
            "both"   — 在 ±radius 搜索（用于 cut_end，寻找最近的静音谷）
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

        # 搜索区间太短（<50ms）则不微调
        if end - start < int(sr * 0.05):
            return time_s
        
        segment = wav[start:end]
        
        # 20ms 窗口，10ms 步长（跨越大部分短暂闭气停顿）
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
        
        # 5-point 滑动平均平滑（把锯齿状毛刺抹平，寻找宽阔的静音带）
        kernel_size = min(5, len(energies))
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(energies, kernel, mode='same')
        
        if len(smoothed) == 0:
            return time_s
        
        # 找平滑后的能量最低点
        min_idx = np.argmin(smoothed)
        return positions[min_idx] / sr

    def _expand_to_midpoints(self, boundaries: list, total_duration: float) -> list:
        """将切割点扩展到相邻句子间隙中，但限制最大扩展量以避免吃进下一句。"""
        MAX_EXPAND_START = 0.15  # cut_start 向前扩展：最多 150ms（保留吸气/起音余量）
        MAX_EXPAND_END = 0.10    # cut_end 向后扩展：最多 100ms（补偿 ASR 尾部时间戳早退）

        if len(boundaries) <= 1:
            if boundaries:
                boundaries[0]["cut_start"] = 0.0
                boundaries[0]["cut_end"] = total_duration
            return boundaries

        for i in range(len(boundaries)):
            if i == 0:
                boundaries[i]["cut_start"] = 0.0
            else:
                # 与前一句的间隙中点，但不超过 MAX_EXPAND_START
                gap_mid = (boundaries[i - 1]["end"] + boundaries[i]["start"]) / 2
                boundaries[i]["cut_start"] = round(
                    max(gap_mid, boundaries[i]["start"] - MAX_EXPAND_START), 3)

            if i == len(boundaries) - 1:
                boundaries[i]["cut_end"] = total_duration
            else:
                # 与后一句的间隙中点，但不超过 MAX_EXPAND_END
                gap_mid = (boundaries[i]["end"] + boundaries[i + 1]["start"]) / 2
                boundaries[i]["cut_end"] = round(
                    min(gap_mid, boundaries[i]["end"] + MAX_EXPAND_END), 3)

        return boundaries

    def stitch(self, split_map, audio_A, audio_B, asr_A, asr_B,
               gap_duration=0.25, padding=0.10, fade_ms=30, use_vad=False, use_forced_align=False):
        log = f"[{self.NODE_NAME}]"

        # 解析 split_map
        try:
            map_items = json.loads(split_map)
        except json.JSONDecodeError as e:
            print(f"{log} split_map JSON 解析失败: {e}")
            return (audio_A, "[]")

        # 提取音频数据
        wav_A, sr_A = self._audio_to_numpy(audio_A)
        wav_B, sr_B = self._audio_to_numpy(audio_B)
        duration_A = len(wav_A) / sr_A
        duration_B = len(wav_B) / sr_B

        # 使用统一采样率
        sr = sr_A
        if sr_A != sr_B:
            print(f"{log} 警告: sr_A={sr_A} != sr_B={sr_B}, 使用 sr_A")

        # Forced Alignment 模式：对每个说话人做字级强制对齐
        fa_results_A = None
        fa_results_B = None
        if use_forced_align:
            fa_model, fa_tokenizer, fa_aligner = self._load_fa_model()
            if fa_model is not None:
                print(f"{log} 使用 MMS Forced Alignment 字级对齐...")
            else:
                print(f"{log} FA 模型加载失败，回退")
                use_forced_align = False

        # VAD 模式：提前对每个说话人的完整音频运行 VAD
        # 当 FA 启用时，如果 VAD 也启用则同时运行用于交叉验证
        vad_timestamps_A = None
        vad_timestamps_B = None
        if use_vad or (use_forced_align and use_vad):
            vad_model, vad_utils = self._load_vad_model()
            if vad_model is not None:
                print(f"{log} 使用 Silero VAD 精确边界检测...")
                vad_timestamps_A = self._get_vad_timestamps(wav_A, sr_A, vad_model, vad_utils)
                vad_timestamps_B = self._get_vad_timestamps(wav_B, sr_B, vad_model, vad_utils)
                print(f"{log} VAD 检测到 A={len(vad_timestamps_A)} 段语音, B={len(vad_timestamps_B)} 段语音")
            else:
                print(f"{log} VAD 加载失败")
                if not use_forced_align:
                    use_vad = False

        print(f"{log} Audio A: {duration_A:.2f}s, Audio B: {duration_B:.2f}s, SR: {sr}")

        # 收集每个说话人的句子列表
        sentences_A = [item["text"] for item in map_items if item.get("type") == "speech" and item.get("speaker") == "A"]
        sentences_B = [item["text"] for item in map_items if item.get("type") == "speech" and item.get("speaker") == "B"]

        print(f"{log} 句子数 - A: {len(sentences_A)}, B: {len(sentences_B)}")

        # ASR 对齐切分
        words_A = asr_A.get("words", []) if isinstance(asr_A, dict) else []
        words_B = asr_B.get("words", []) if isinstance(asr_B, dict) else []

        print(f"{log} ASR 词数 - A: {len(words_A)}, B: {len(words_B)}")

        boundaries_A = self._find_sentence_boundaries(words_A, sentences_A, duration_A)
        boundaries_B = self._find_sentence_boundaries(words_B, sentences_B, duration_B)

        print(f"{log} 边界数 - A: {len(boundaries_A)}, B: {len(boundaries_B)}")

        # FA 对齐（在 ASR 边界之后，用于替代/验证 ASR 边界）
        if use_forced_align:
            print(f"{log} --- Speaker A FA ---")
            fa_results_A = self._forced_align_sentences(wav_A, sr_A, sentences_A)
            print(f"{log} --- Speaker B FA ---")
            fa_results_B = self._forced_align_sentences(wav_B, sr_B, sentences_B)

        # 按 split_map 顺序拼接
        audio_segments = []
        segments_info = []
        current_time = 0.0
        idx_A = 0
        idx_B = 0
        prev_speaker = None
        # 跟踪每个说话人上一个片段的实际 cut_end，防止 padding 导致重叠
        prev_cut_end = {"A": 0.0, "B": 0.0}

        for item in map_items:
            if item.get("type") == "pause":
                # 显式暂停
                pause_dur = item.get("duration", 0.3)
                pause_samples = int(pause_dur * sr)
                audio_segments.append(np.zeros(pause_samples, dtype=np.float32))
                current_time += pause_dur
                continue

            if item.get("type") != "speech":
                continue

            speaker = item["speaker"]
            
            # 说话人切换时插入过渡间隙（带低频噪声底噪，避免死寂）
            if prev_speaker is not None and speaker != prev_speaker:
                gap_samples = int(gap_duration * sr)
                # 生成极低音量的粉噪声代替纯静音，听感更自然
                noise = np.random.randn(gap_samples).astype(np.float32) * 0.0003
                audio_segments.append(noise)
                current_time += gap_duration

            # 获取对应的边界和音频
            if speaker == "A":
                if idx_A >= len(boundaries_A):
                    print(f"{log} 警告: A 的句子索引 {idx_A} 超出边界数 {len(boundaries_A)}")
                    idx_A += 1
                    continue
                boundary = boundaries_A[idx_A]
                wav = wav_A
                idx_A += 1
            elif speaker == "B":
                if idx_B >= len(boundaries_B):
                    print(f"{log} 警告: B 的句子索引 {idx_B} 超出边界数 {len(boundaries_B)}")
                    idx_B += 1
                    continue
                boundary = boundaries_B[idx_B]
                wav = wav_B
                idx_B += 1
            else:
                continue

            # 切割音频片段（使用 cut_start/cut_end，带 padding）
            cut_start = boundary.get("cut_start", boundary["start"])
            cut_end = boundary.get("cut_end", boundary["end"])

            # 获取当前句子在说话人内的索引
            sent_local_idx = (idx_A - 1) if speaker == "A" else (idx_B - 1)

            # 边界微调：FA > VAD > Energy
            if use_forced_align:
                fa_results = fa_results_A if speaker == "A" else fa_results_B
                fa_entry = None
                if fa_results and sent_local_idx < len(fa_results):
                    fa_entry = fa_results[sent_local_idx]
                
                if fa_entry:
                    fa_start, fa_end = fa_entry['start'], fa_entry['end']
                    # 混合策略：FA 精确起点 + 能量检测自然收尾
                    cut_start = fa_start
                    cut_end = self._refine_cut_point(wav, sr, fa_end,
                        search_radius=0.15, direction="after")
                    
                    # 交叉验证：同时计算 VAD 和 Energy 的结果做对比
                    if use_vad and vad_timestamps_A is not None:
                        vad_ts = vad_timestamps_A if speaker == "A" else vad_timestamps_B
                        vad_start, vad_end = self._refine_with_vad(
                            boundary.get("cut_start", boundary["start"]),
                            boundary.get("cut_end", boundary["end"]),
                            vad_ts)
                        energy_start = self._refine_cut_point(wav, sr,
                            boundary.get("cut_start", boundary["start"]),
                            search_radius=0.15, direction="before")
                        energy_end = self._refine_cut_point(wav, sr,
                            boundary.get("cut_end", boundary["end"]),
                            search_radius=0.10, direction="both")
                        
                        iou_fa_vad = self._compute_iou(fa_start, fa_end, vad_start, vad_end)
                        iou_fa_energy = self._compute_iou(fa_start, fa_end, energy_start, energy_end)
                        iou_vad_energy = self._compute_iou(vad_start, vad_end, energy_start, energy_end)
                        
                        print(f"{log} 🔬 {speaker}[{sent_local_idx}] "
                              f"FA=[{fa_start:.3f},{fa_end:.3f}] "
                              f"VAD=[{vad_start:.3f},{vad_end:.3f}] "
                              f"Energy=[{energy_start:.3f},{energy_end:.3f}] | "
                              f"FA-VAD={iou_fa_vad:.3f} FA-Energy={iou_fa_energy:.3f} VAD-Energy={iou_vad_energy:.3f}")
                else:
                    # FA 结果缺失，回退到 VAD 或 Energy
                    if use_vad and vad_timestamps_A is not None:
                        vad_ts = vad_timestamps_A if speaker == "A" else vad_timestamps_B
                        cut_start, cut_end = self._refine_with_vad(cut_start, cut_end, vad_ts)
                    else:
                        cut_start = self._refine_cut_point(wav, sr, cut_start, search_radius=0.15, direction="before")
                        cut_end = self._refine_cut_point(wav, sr, cut_end, search_radius=0.10, direction="both")
            elif use_vad and vad_timestamps_A is not None:
                vad_ts = vad_timestamps_A if speaker == "A" else vad_timestamps_B
                cut_start, cut_end = self._refine_with_vad(cut_start, cut_end, vad_ts)
            else:
                cut_start = self._refine_cut_point(wav, sr, cut_start, search_radius=0.15, direction="before")
                cut_end = self._refine_cut_point(wav, sr, cut_end, search_radius=0.10, direction="both")

            # 应用 padding：cut_start 全额保留起音余量，cut_end 少量保留尾音衰减
            cut_start = max(0, cut_start - padding)
            cut_end = min(len(wav) / sr, cut_end + padding * 0.3)

            # 防重叠：确保 cut_start 不早于同一说话人上一个片段的 cut_end
            if cut_start < prev_cut_end[speaker]:
                cut_start = prev_cut_end[speaker]
            prev_cut_end[speaker] = cut_end

            start_sample = int(cut_start * sr)
            end_sample = int(cut_end * sr)
            end_sample = min(end_sample, len(wav))

            segment = wav[start_sample:end_sample]

            if len(segment) == 0:
                print(f"{log} 警告: 空片段 at {cut_start:.3f}-{cut_end:.3f}s")
                continue

            seg_duration = len(segment) / sr

            # 对片段首尾施加余弦淡入淡出，使拼接过渡平滑自然
            fade_seconds = fade_ms / 1000.0
            fade_samples = min(int(fade_seconds * sr), len(segment) // 4)
            if fade_samples > 1:
                # 余弦淡入淡出比线性更平滑——能量变化曲线更接近自然衰减
                fade_in = (0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_samples)))).astype(np.float32)
                fade_out = (0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_samples)))).astype(np.float32)
                segment = segment.copy()
                segment[:fade_samples] *= fade_in
                segment[-fade_samples:] *= fade_out

            audio_segments.append(segment)

            # 记录 segment info
            original_speaker = item.get("original_speaker", speaker)
            segments_info.append({
                "start": round(current_time, 3),
                "end": round(current_time + seg_duration, 3),
                "text": item["text"],
                "speaker": original_speaker,
            })

            current_time += seg_duration
            prev_speaker = speaker

        # 拼接所有片段
        if not audio_segments:
            print(f"{log} 错误: 没有任何音频片段")
            return (audio_A, "[]")

        final_audio = np.concatenate(audio_segments)
        total_duration = len(final_audio) / sr
        print(f"{log} 拼接完成: {total_duration:.2f}s, {len(segments_info)} 个语音段")

        # 转为 ComfyUI AUDIO 格式
        audio_tensor = torch.from_numpy(final_audio).unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
        audio_output = {"waveform": audio_tensor, "sample_rate": sr}

        segments_info_json = json.dumps(segments_info, ensure_ascii=False, indent=2)

        return (audio_output, segments_info_json)


# --- ComfyUI 节点注册 ---
NODE_CLASS_MAPPINGS = {
    "AIIA_Podcast_Stitcher": AIIA_Podcast_Stitcher,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_Podcast_Stitcher": "🧵 AIIA Podcast Stitcher",
}
