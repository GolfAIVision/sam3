"""Optional real-model lifecycle checks: SAM3_INTEGRATION=1 enables local weights."""

import os
import unittest
from pathlib import Path

import torch


@unittest.skipUnless(
    os.getenv("SAM3_INTEGRATION") == "1" and torch.cuda.is_available(),
    "Set SAM3_INTEGRATION=1 with CUDA and local weights",
)
class TextStreamAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sam3.model_builder import build_sam3_video_predictor

        cls.predictor = build_sam3_video_predictor(
            checkpoint_path="/data/model_weights/segment_anything/sam3/sam3.pt",
            bpe_path="/data/model_weights/segment_anything/sam3/bpe_simple_vocab_16e6.txt.gz",
            inference_mode="text_stream",
            kernel_backend="auto",
        )

    @classmethod
    def tearDownClass(cls):
        cls.predictor.shutdown()

    def test_request_lifecycle_and_cancellation(self):
        p = self.predictor
        response = p.handle_request(
            dict(
                type="start_session",
                resource_path="testset",
                output_fields=["out_binary_masks"],
            )
        )
        session = response["session_id"]
        state = p._get_session(session)["state"]
        source = state["input_batch"].img_batch
        try:
            with self.assertRaises(ValueError):
                p.handle_request(
                    dict(
                        type="add_prompt",
                        session_id=session,
                        frame_index=1,
                        text="person",
                    )
                )
            self.assertIsNone(state["text_prompt"])
            first = p.handle_request(
                dict(
                    type="add_prompt", session_id=session, frame_index=0, text="person"
                )
            )
            self.assertEqual(
                set(first["outputs"]), {"out_obj_ids", "out_probs", "out_binary_masks"}
            )
            with self.assertRaises(ValueError):
                p.handle_request(
                    dict(
                        type="add_prompt", session_id=session, frame_index=0, text="dog"
                    )
                )
            stream = p.handle_stream_request(
                dict(type="propagate_in_video", session_id=session)
            )
            self.assertEqual(next(stream)["frame_index"], 0)
            with self.assertRaises(ValueError):
                p.handle_request(dict(type="reset_session", session_id=session))
            stream.close()
            self.assertFalse(source.thread.is_alive())
            self.assertFalse(state["feature_cache"])
            self.assertFalse(state["tracker_inference_states"])
            with self.assertRaises(ValueError):
                next(
                    p.handle_stream_request(
                        dict(type="propagate_in_video", session_id=session)
                    )
                )
            p.handle_request(dict(type="reset_session", session_id=session))
            self.assertIsNone(state["text_prompt"])
            p.handle_request(
                dict(
                    type="add_prompt", session_id=session, frame_index=0, text="person"
                )
            )
            stream = p.handle_stream_request(
                dict(
                    type="propagate_in_video",
                    session_id=session,
                    max_frame_num_to_track=2,
                )
            )
            self.assertEqual([item["frame_index"] for item in stream], [0, 1, 2])
        finally:
            p.handle_request(dict(type="close_session", session_id=session))
        self.assertFalse(source.thread.is_alive())


if __name__ == "__main__":
    unittest.main()
