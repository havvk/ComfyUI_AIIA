import os
import sys
import torch
import torchaudio
import numpy as np

sys.path.insert(0, '/app/ComfyUI/custom_nodes/ComfyUI_AIIA/libs/CosyVoice')
sys.path.insert(0, '/app/ComfyUI/custom_nodes/ComfyUI_AIIA/libs/CosyVoice/third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import AutoModel

def normalize_audio(waveform, target_rms=0.22):
    if waveform.abs().max() == 0: return waveform
    current_rms = torch.sqrt(torch.mean(waveform**2))
    if current_rms > 0:
        scale = target_rms / current_rms.item()
        waveform = waveform * scale
    if waveform.abs().max() > 0.99:
        waveform = waveform / (waveform.abs().max() / 0.99)
    return waveform

def generate_grid():
    text = '系统正在进行语音身份验证，请仔细聆听当前的音色表现。'
    models = {
        'SFT': '/app/ComfyUI/models/cosyvoice/CosyVoice-300M-SFT',
        'Instruct': '/app/ComfyUI/models/cosyvoice/CosyVoice-300M-Instruct'
    }
    
    output_dir = '/tmp/cosy_grid'
    os.makedirs(output_dir, exist_ok=True)
    
    for m_type, m_path in models.items():
        print(f'Processing {m_type} model...')
        try:
            cosy = AutoModel(model_dir=m_path)
            spks = cosy.list_available_spks()
            
            for spk in spks:
                print(f'  Generating {spk}...')
                try:
                    for i, j in enumerate(cosy.inference_sft(text, spk, speed=1.0)):
                        wav = normalize_audio(j['tts_speech'])
                        # Use English-friendly filenames
                        filename = f'grid_{m_type}_{spk}_{i}.wav'.replace('中文', 'CN_').replace('女', 'FEMALE').replace('男', 'MALE').replace('日语', 'JP_').replace('粤语', 'CANTO_').replace('英文', 'EN_').replace('韩语', 'KR_')
                        out = os.path.join(output_dir, filename)
                        torchaudio.save(out, wav, cosy.sample_rate)
                except Exception as e:
                    print(f'    Failed {spk}: {e}')
                    
            if m_type == 'SFT':
                print('  Generating SFT CN_MALE at speed 1.1...')
                for i, j in enumerate(cosy.inference_sft(text, '中文男', speed=1.1)):
                    wav = normalize_audio(j['tts_speech'])
                    out = os.path.join(output_dir, 'grid_SFT_CN_MALE_speed11_0.wav')
                    torchaudio.save(out, wav, cosy.sample_rate)
        except Exception as e:
            print(f'Error loading {m_type}: {e}')

if __name__ == "__main__":
    generate_grid()
