import sys
import os
import torch
import torchaudio

# Explicitly add remote paths
base_dir = "/app/ComfyUI/custom_nodes/ComfyUI_AIIA"
sys.path.insert(0, os.path.join(base_dir, "libs/CosyVoice"))
sys.path.insert(0, os.path.join(base_dir, "libs/CosyVoice/third_party/Matcha-TTS"))

print(f"Debug: sys.path[0-2]: {sys.path[:3]}")

try:
    from cosyvoice.cli.cosyvoice import CosyVoice
    print("Success: Imported CosyVoice")
except ImportError as e:
    print(f"Error importing CosyVoice: {e}")
    # Listing dir to see what's wrong
    cv_path = os.path.join(base_dir, "libs/CosyVoice")
    print(f"Listing {cv_path}: {os.listdir(cv_path) if os.path.exists(cv_path) else 'Not Found'}")
    sys.exit(1)

def test_instruct():
    # Try multiple possible model locations
    possible_paths = [
        "/app/ComfyUI/models/cosyvoice/CosyVoice-300M-Instruct",
        "/app/ComfyUI/models/cosyvoice/FunAudioLLM/CosyVoice-300M-Instruct",
        "pretrained_models/CosyVoice-300M-Instruct",
        os.path.join(base_dir, "libs/CosyVoice/pretrained_models/CosyVoice-300M-Instruct")
    ]
    
    model_dir = None
    for p in possible_paths:
        if os.path.exists(p):
            model_dir = p
            break
            
    if not model_dir:
        print("Error: Could not find CosyVoice-300M-Instruct model.")
        print(f"Checked: {possible_paths}")
        return

    print(f"Loading model from {model_dir}")

    cosyvoice = CosyVoice(model_dir)
    print("Model loaded.")

    # List of test cases for Dialect Prompt Tuning
    test_cases = [
        {
            "name": "Dialect_01_CN_Term_Cantonese",
            "text": "雷好啊，今晚去边度食饭？",
            "spk": "中文女",
            "instruct": "A female speaker with a 广东话 accent.<|endofprompt|>"
        },
        {
            "name": "Dialect_02_CN_Instruction_Cantonese",
            "text": "雷好啊，今晚去边度食饭？",
            "spk": "中文女",
            "instruct": "A female speaker. 用广东话。<|endofprompt|>"
        },
        {
            "name": "Dialect_03_Sichuan_Native_Term",
            "text": "我们四川人说话就是这个味儿。",
            "spk": "中文男",
            "instruct": "A male speaker with a 四川话 accent.<|endofprompt|>"
        }
    ]

    for case in test_cases:
        print(f"Generating Case: {case['name']}")
        print(f"  Instruct: {case['instruct']}")
        try:
            output = cosyvoice.inference_instruct(case['text'], case['spk'], case['instruct'])
            for i, j in enumerate(output):
                # Ensure filename is safe
                safe_name = case['name'].replace(" ", "_").replace("/", "-")
                filename = f"sample_{safe_name}.wav"
                abs_save_path = os.path.join(base_dir, filename)
                torchaudio.save(abs_save_path, j['tts_speech'], cosyvoice.sample_rate)
                print(f"  Saved to: {abs_save_path}")
        except Exception as e:
            print(f"  Failed {case['name']}: {e}")


if __name__ == "__main__":
    test_instruct()

