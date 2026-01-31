import unittest
import json
import math

class SubtitleCalibrator:
    """Extract of the calibration logic from aiia_subtitle_nodes.py for testing."""
    def calibrate_segments(self, segments, chunks):
        if not chunks:
            return segments

        # 1. Ensure chunks are sorted chronologically
        sorted_chunks = sorted(chunks, key=lambda x: x["timestamp"][0])
        
        calibrated = []
        chunk_idx = 0
        num_chunks = len(sorted_chunks)
        
        def normalize_spk(s):
            if not s: return ""
            return str(s).lower().replace("speaker_", "").replace("speaker ", "").strip()

        for i, seg in enumerate(segments):
            seg_start = seg["start"]
            seg_end = seg["end"]
            
            # --- Speaker-Centric Magic (v1.10.5) ---
            speaker_overlaps = {}
            scan_idx = chunk_idx
            while scan_idx < num_chunks:
                c = sorted_chunks[scan_idx]
                c_start, c_end = c["timestamp"]
                if c_start > seg_end + 3.0: break
                
                overlap = min(seg_end, c_end) - max(seg_start, c_start)
                if overlap > 0:
                    spk = normalize_spk(c.get("speaker", "unknown"))
                    speaker_overlaps[spk] = speaker_overlaps.get(spk, 0.0) + overlap
                scan_idx += 1
            
            winner_spk = None
            if speaker_overlaps:
                winner_spk = max(speaker_overlaps, key=speaker_overlaps.get)
            
            matched_chunks = []
            find_idx = chunk_idx
            lookahead_count = 0
            while find_idx < num_chunks and lookahead_count < 15:
                chunk = sorted_chunks[find_idx]
                c_start, c_end = chunk["timestamp"]
                c_spk = normalize_spk(chunk.get("speaker", "unknown"))
                
                overlap = min(seg_end, c_end) - max(seg_start, c_start)
                is_overlap = overlap > 0.05
                # Special case: tiny gap exactly at boundaries or start of video
                if not is_overlap and i == 0 and find_idx == 0:
                    if abs(c_start - seg_start) < 0.5 and c_end > seg_start - 0.1:
                        is_overlap = True
                
                if is_overlap:
                    if winner_spk is None or c_spk == winner_spk:
                        matched_chunks.append(find_idx)
                
                if c_start > seg_end + 1.5: break
                find_idx += 1
                lookahead_count += 1
            
            if matched_chunks:
                actual_starts = [sorted_chunks[idx]["timestamp"][0] for idx in matched_chunks]
                actual_ends = [sorted_chunks[idx]["timestamp"][1] for idx in matched_chunks]
                
                min_s = min(actual_starts)
                max_e = max(actual_ends)
                
                chunk_idx = max(matched_chunks) + 1
                
                seg["start"] = round(min_s, 3)
                seg["end"] = round(max_e, 3)
            else:
                if i > 0:
                    prev_end = calibrated[-1]["end"]
                    if seg["start"] < prev_end:
                         diff = prev_end - seg["start"]
                         seg["start"] += diff
                         seg["end"] += diff
            
            calibrated.append(seg)
            
        return calibrated

class TestSubtitleCalibration(unittest.TestCase):
    def setUp(self):
        self.calibrator = SubtitleCalibrator()

    def test_empty_inputs(self):
        """Test with empty segments or chunks."""
        self.assertEqual(self.calibrator.calibrate_segments([], [{"timestamp": [1, 2]}]), [])
        segments = [{"start": 1, "end": 2, "text": "Hi"}]
        self.assertEqual(self.calibrator.calibrate_segments(segments, []), segments)

    def test_ideal_match(self):
        """Test 1-to-1 perfect match."""
        segments = [{"start": 1.0, "end": 2.0, "speaker": "A"}]
        chunks = [{"timestamp": [0.9, 2.1], "speaker": "SPEAKER_00"}]
        result = self.calibrator.calibrate_segments(segments, chunks)
        self.assertEqual(result[0]["start"], 0.9)
        self.assertEqual(result[0]["end"], 2.1)

    def test_v1_10_4_sorting(self):
        """Test that unsorted chunks (NeMo grouped) are handled correctly."""
        segments = [
            {"start": 0.0, "end": 2.0, "speaker": "A"},
            {"start": 5.0, "end": 7.0, "speaker": "B"}
        ]
        # Chunks are grouped by speaker, unsorted by time
        chunks = [
            {"timestamp": [5.1, 6.9], "speaker": "01"},
            {"timestamp": [0.1, 1.9], "speaker": "00"}
        ]
        result = self.calibrator.calibrate_segments(segments, chunks)
        self.assertEqual(result[0]["start"], 0.1)
        self.assertEqual(result[1]["start"], 5.1)

    def test_v1_10_5_speaker_centric(self):
        """Test that cross-speaker overlaps are ignored by winner-takes-all logic."""
        segments = [
            {"start": 0.0, "end": 6.6, "speaker": "A"},
            {"start": 6.6, "end": 9.0, "speaker": "B"}
        ]
        chunks = [
            {"timestamp": [0.2, 5.0], "speaker": "00"}, # Belongs to A
            {"timestamp": [6.5, 8.9], "speaker": "01"}  # Belongs to B, but overlaps A by 0.1s
        ]
        # Without speaker-centric, Seg 0 might grab Chunk 1 because 6.5 < 6.6
        result = self.calibrator.calibrate_segments(segments, chunks)
        
        # Segment 0 should ONLY match Chunk 0 (00)
        self.assertEqual(result[0]["start"], 0.2)
        self.assertEqual(result[0]["end"], 5.0)
        
        # Segment 1 should match Chunk 1 (01)
        self.assertEqual(result[1]["start"], 6.5)
        self.assertEqual(result[1]["end"], 8.9)

    def test_boundary_overlap_threshold(self):
        """Test the 0.05s overlap threshold."""
        segments = [{"start": 1.0, "end": 2.0}]
        # Case 1: Overlap = 0.05s (Exactly on boundary)
        chunks_match = [{"timestamp": [1.95, 3.0]}]
        result = self.calibrator.calibrate_segments(segments, chunks_match)
        self.assertEqual(result[0]["start"], 1.95)

        # Case 2: Overlap = 0.04s (Just below)
        segments_fallback = [{"start": 1.0, "end": 2.0}]
        chunks_fail = [{"timestamp": [1.96, 3.0]}]
        result = self.calibrator.calibrate_segments(segments_fallback, chunks_fail)
        self.assertEqual(result[0]["start"], 1.0) # Fallback to original

    def test_multi_chunk_merge(self):
        """Test that multiple chunks for a single segment are merged."""
        segments = [{"start": 1.0, "end": 5.0}]
        chunks = [
            {"timestamp": [1.1, 2.0]}, # pause
            {"timestamp": [3.0, 4.9]}
        ]
        result = self.calibrator.calibrate_segments(segments, chunks)
        self.assertEqual(result[0]["start"], 1.1)
        self.assertEqual(result[0]["end"], 4.9)

    def test_lookahead_limit(self):
        """Test that it doesn't scan too many chunks ahead (limit 15)."""
        segments = [{"start": 1.0, "end": 2.0}]
        # Many small chunks before the target
        chunks = [{"timestamp": [0.0, 0.1]} for _ in range(20)]
        chunks.append({"timestamp": [1.1, 1.9]})
        result = self.calibrator.calibrate_segments(segments, chunks)
        # Even if it matches, it might have stopped before reaching the target if lookahead is exhausted
        # But lookahead only increments if we FAIL to match significant overlap?
        # Actually in code: `lookahead_count < 15` always increments.
        # So it should FAIL to reach chunk 20 if it starts at 0.
        self.assertEqual(result[0]["start"], 1.0) # Fallback because target chunk was too far

    def test_reversed_time_protection(self):
        """Test that min/max prevents backwards time ranges."""
        # Simulated corrupted input or weird behavior
        segments = [{"start": 5.0, "end": 10.0}]
        chunks = [
            {"timestamp": [9.0, 11.0]},
            {"timestamp": [4.0, 6.0]}
        ]
        # sorting happens first -> [4, 6], [9, 11]
        # min_s = 4.0, max_e = 11.0
        result = self.calibrator.calibrate_segments(segments, chunks)
        self.assertTrue(result[0]["start"] < result[0]["end"])
        self.assertEqual(result[0]["start"], 4.0)
        self.assertEqual(result[0]["end"], 11.0)

    def test_real_world_calibration_sequence(self):
        """Test with actual data segment that previously had issues."""
        # Data from user's report
        segments = [
            {"start": 0.0, "end": 6.62, "speaker": "A"},
            {"start": 6.62, "end": 9.48, "speaker": "B"}
        ]
        chunks = [
            {"timestamp": [0.19, 2.85], "speaker": "SPEAKER_00"},
            {"timestamp": [3.15, 5.97], "speaker": "SPEAKER_00"},
            {"timestamp": [6.51, 9.09], "speaker": "SPEAKER_01"} # Overlaps A slightly, but winner for B
        ]
        result = self.calibrator.calibrate_segments(segments, chunks)
        
        # Segment 0 (A) should match first two chunks of SPEAKER_00
        # Start should be 0.19, End should be 5.97
        self.assertEqual(result[0]["start"], 0.19)
        self.assertEqual(result[0]["end"], 5.97)
        
        # Segment 1 (B) should match SPEAKER_01 chunk
        # Start should be 6.51, End 9.09
        self.assertEqual(result[1]["start"], 6.51)
        self.assertEqual(result[1]["end"], 9.09)

if __name__ == '__main__':
    unittest.main()
