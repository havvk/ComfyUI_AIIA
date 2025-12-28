import torch
import numpy as np

class AIIA_Audio_Splice_Analyzer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "splice_info": ("SPLICE_INFO",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("spectrogram_image",)
    FUNCTION = "analyze_splice"
    CATEGORY = "AIIA/Debug"

    def analyze_splice(self, audio, splice_info=None):
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            import io
            from PIL import Image
            import torchaudio
        except ImportError:
            print("[AIIA Debug] Matplotlib not found. Generating error image.")
            return (self._create_error_image("Matplotlib not found"),)

        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        # Ensure 2D (channels, time)
        if waveform.ndim == 3:
            waveform = waveform.squeeze(0) # Remove batch
        
        # Convert to mono for spectrogram
        if waveform.shape[0] > 1:
            waveform_mono = torch.mean(waveform, dim=0, keepdim=True)
        else:
            waveform_mono = waveform

        # Compute Spectrogram
        n_fft = 1024
        win_length = None
        hop_length = 512
        n_mels = 128
        
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            onesided=True,
            n_mels=n_mels,
            mel_scale="htk",
        )(waveform_mono)

        mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
        
        # Plotting
        plt.figure(figsize=(12, 6))
        plt.imshow(mel_spectrogram_db.squeeze().numpy(), aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram with Splice Points')
        plt.xlabel('Time (frames)')
        plt.ylabel('Frequency (Hz)')

        # Create Custom Y-Ticks for Hz
        def hz_to_mel_htk(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz_htk(mel):
            return 700 * (10**(mel / 2595) - 1)

        # 1. Determine Freq Range
        f_min = 0.0
        f_max = sample_rate / 2.0
        
        # 2. Convert boundaries to Mel
        min_mel = hz_to_mel_htk(f_min)
        max_mel = hz_to_mel_htk(f_max)
        
        # 3. Choose tick locations in Hz
        # E.g. 0, 500, 1000, 2000, 4000, 8000, 16000...
        tick_freqs = [0, 500, 1000, 2000, 4000, 8000, 11025, 16000, 22050]
        tick_freqs = [f for f in tick_freqs if f <= f_max] # filter valid
        
        # 4. Convert ticks to Mel Bin Indices
        # bin = (mel - min_mel) / (max_mel - min_mel) * (n_mels - 1)
        tick_bins = []
        tick_labels = []
        for f in tick_freqs:
            m = hz_to_mel_htk(f)
            bin_idx = (m - min_mel) / (max_mel - min_mel) * (n_mels - 1)
            tick_bins.append(bin_idx)
            tick_labels.append(str(int(f)))
            
        plt.yticks(tick_bins, tick_labels)

        # Draw splice lines if info is present
        if splice_info and "splice_points" in splice_info:
            splice_points = splice_info["splice_points"]
            total_samples = splice_info.get("total_samples", waveform.shape[-1])
            # Convert sample indices to spectrogram frame indices
            # Frame index ~= sample_index / hop_length
            for sp in splice_points:
                frame_idx = sp / hop_length
                plt.axvline(x=frame_idx, color='r', linestyle='--', alpha=0.8, label='Splice Point')
            
            # Add legend strictly once to avoid clutter
            if splice_points:
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys())

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        # Convert to ComfyUI Image Tensor (Batch, Height, Width, Channels)
        image = Image.open(buf).convert('RGB')
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        
        return (image_tensor,)

    def _create_error_image(self, message):
        # Create a simple black image with text using PIL if possible, else just black
        try:
            from PIL import Image, ImageDraw, ImageFont
            img = Image.new('RGB', (512, 256), color=(0, 0, 0))
            d = ImageDraw.Draw(img)
            d.text((10,128), message, fill=(255, 255, 255))
            image_np = np.array(img).astype(np.float32) / 255.0
            return torch.from_numpy(image_np).unsqueeze(0)
        except:
             return torch.zeros((1, 256, 512, 3))

NODE_CLASS_MAPPINGS = {
    "AIIA_Audio_Splice_Analyzer": AIIA_Audio_Splice_Analyzer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_Audio_Splice_Analyzer": "Audio Splice Analyzer (AIIA Debug)"
}
