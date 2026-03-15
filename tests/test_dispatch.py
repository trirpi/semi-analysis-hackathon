import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from confidence_dispatch.dispatch import build_dispatched_transcript, collect_low_confidence_spans


class TestDispatchSpans(unittest.TestCase):
    def test_collect_low_confidence_spans_merges_nearby_tokens(self):
        analysis = {
            "duration_sec": 4.0,
            "segments": [
                {
                    "id": 0,
                    "token_details": [
                        {"token_id": 1, "token_text": " hello", "start": 0.0, "end": 0.4, "probability": 0.95},
                        {"token_id": 2, "token_text": " uncertain", "start": 0.45, "end": 0.8, "probability": 0.4},
                        {"token_id": 3, "token_text": " token", "start": 0.82, "end": 1.1, "probability": 0.5},
                        {"token_id": 4, "token_text": " world", "start": 1.5, "end": 1.9, "probability": 0.99},
                    ],
                }
            ],
        }

        spans = collect_low_confidence_spans(
            analysis,
            threshold=0.6,
            merge_gap_sec=0.05,
            context_sec=0.2,
        )

        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["local_text"], " uncertain token")
        self.assertEqual(spans[0]["token_count"], 2)
        self.assertAlmostEqual(spans[0]["dispatch_start"], 0.25)
        self.assertAlmostEqual(spans[0]["dispatch_end"], 1.3)

    def test_build_dispatched_transcript_replaces_low_confidence_span(self):
        analysis = {
            "segments": [
                {
                    "id": 0,
                    "token_details": [
                        {"token_text": " hello"},
                        {"token_text": " uncertain"},
                        {"token_text": " token"},
                        {"token_text": " world"},
                    ],
                }
            ]
        }
        dispatched = [
            {
                "segment_id": 0,
                "start_token_index": 1,
                "end_token_index": 3,
                "local_text": " uncertain token",
                "remote_text": " accurate phrase",
            }
        ]

        transcript = build_dispatched_transcript(analysis, dispatched)
        self.assertEqual(transcript, "hello accurate phrase world")


if __name__ == "__main__":
    unittest.main()
