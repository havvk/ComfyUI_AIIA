
import re

def get_speaker_key(speaker_name):
    spk_key = speaker_name.strip()
    if spk_key.upper() in ["A", "B", "C"]: return spk_key.upper()
    clean = re.sub(r'speaker[ _-]*', '', spk_key, flags=re.IGNORECASE).strip()
    if clean and clean[0].upper() in ["A", "B", "C"]: return clean[0].upper()
    if spk_key[-1].upper() in ["A", "B", "C"]: return spk_key[-1].upper()
    return spk_key[0].upper()

test_cases = [
    "Host A", "A", "speaker_A", "Speaker A", "Guest B", "C-Speaker",
    "Alice", "Bob" # User might expect these to NOT map to A/B? 
]

print("Mapping Results:")
for t in test_cases:
    print(f"'{t}' -> '{get_speaker_key(t)}'")
