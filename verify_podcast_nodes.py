
import sys
import os
import json
import unittest
from unittest.mock import MagicMock

# Mock dependencies BEFORE importing nodes
sys.modules['torch'] = MagicMock()
sys.modules['folder_paths'] = MagicMock()
sys.modules['torchaudio'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['soundfile'] = MagicMock()

# Add current dir to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from aiia_podcast_nodes import AIIA_Podcast_Script_Parser, AIIA_Dialogue_TTS

class TestPodcastNodes(unittest.TestCase):
    def test_parser(self):
        parser = AIIA_Podcast_Script_Parser()
        script = """A: Hello.
(Pause 0.5)
B: [Happy] Hi!
"""
        mapping = "A=Teacher\nB=Student"
        
        json_out, speakers = parser.parse_script(script, mapping)
        print(f"\n[Parser Output]\n{json_out}")
        
        data = json.loads(json_out)
        self.assertEqual(len(data), 3)
        self.assertEqual(data[0]['speaker'], 'Teacher')
        self.assertEqual(data[1]['type'], 'pause')
        self.assertEqual(data[2]['emotion'], 'Happy')
        
        self.assertIn("Student", speakers)
        self.assertIn("Teacher", speakers)

    def test_dialogue_scheduler(self):
        # Test Speaker Mapping Logic
        # We instantiate the class and test a mocked process_dialogue or just extract logic?
        # Since logic is inside process_dialogue, we can mock the generator calls and check what args they got.
        
        # Mocking generators
        sys.modules['aiia_cosyvoice_nodes'] = MagicMock()
        sys.modules['aiia_vibevoice_nodes'] = MagicMock()
        
        node = AIIA_Dialogue_TTS()
        # Mock _load_fallback_audio to not fail
        node._load_fallback_audio = MagicMock(return_value={"waveform": MagicMock(), "sample_rate": 22050})

        dialogue = [
            {"type": "speech", "speaker": "Speaker_A", "text": "Test A"},
            {"type": "speech", "speaker": "Role B", "text": "Test B"},
            {"type": "speech", "speaker": "C", "text": "Test C"}
        ]
        
        # We want to verify that for Speaker_A, it tries to access input 'speaker_A_ref' keys in kwargs.
        # But process_dialogue does kwargs.get(f"speaker_{spk_key}_ref") internally.
        # We can pass specific mock audio objects and see if they are passed to generate.
        
        mock_audio_A = {"waveform": torch.zeros(1), "sample_rate": 22050, "name": "AudioA"}
        mock_audio_B = {"waveform": torch.zeros(1), "sample_rate": 22050, "name": "AudioB"}
        mock_audio_C = {"waveform": torch.zeros(1), "sample_rate": 22050, "name": "AudioC"}
        
        # Mock generate to print received ref audio
        def mock_gen(*args, **kwargs):
            ref = kwargs.get('reference_audio')
            print(f"Generate called with ref: {ref.get('name') if ref else 'None'}")
            return ({"waveform": torch.zeros(1, 16000), "sample_rate": 16000},)

        with unittest.mock.patch('aiia_cosyvoice_nodes.AIIA_CosyVoice_TTS') as MockCosy:
            instance = MockCosy.return_value
            instance.generate.side_effect = mock_gen
            
            node.process_dialogue(
                json.dumps(dialogue), 
                "CosyVoice", 
                0.5, 1.0, 
                cosyvoice_model="Model",
                speaker_A_ref=mock_audio_A,
                speaker_B_ref=mock_audio_B,
                speaker_C_ref=mock_audio_C
            )
            
            # We expect 3 calls
            # Call 1: Speaker_A -> Should find mock_audio_A
            # Call 2: Role B -> Should find mock_audio_B
            # Call 3: C -> Should find mock_audio_C
            pass

if __name__ == '__main__':
    unittest.main()
