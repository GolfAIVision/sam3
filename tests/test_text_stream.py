import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from sam3.model.text_stream import (
    LazyFrames,
    output_fields,
    prune_tracker_states,
    StreamingVideoFrames,
)
from sam3.perflib.stream_outputs import resize_masks


class StreamTests(unittest.TestCase):
    def test_lazy_frames_bound_and_indices(self):
        values = LazyFrames(10000, lambda i: i)
        for i in range(10000):
            self.assertEqual(values[i], i)
        self.assertEqual(len(values.values), 8)
        with self.assertRaises(IndexError):
            values[10000]

    def test_output_fields(self):
        self.assertEqual(output_fields([]), {"out_obj_ids", "out_probs"})
        with self.assertRaises(ValueError):
            output_fields(["not_a_field"])

    def test_reader_reset_and_preprocessing(self):
        from sam3.model.io_utils import load_video_frames_from_image_folder

        with tempfile.TemporaryDirectory() as directory:
            for i in range(4):
                Image.fromarray(np.full((9, 13, 3), i * 40, dtype=np.uint8)).save(
                    Path(directory) / f"{i}.jpg"
                )
            expected, _, _ = load_video_frames_from_image_folder(
                directory, 8, True, (0.5,) * 3, (0.5,) * 3, False
            )
            source = StreamingVideoFrames(directory, 8, (0.5,) * 3, (0.5,) * 3, "cpu")
            try:
                for i in range(4):
                    torch.testing.assert_close(source[i], expected[i], rtol=0, atol=0)
                self.assertLessEqual(len(source.recent), 2)
                with self.assertRaises(ValueError):
                    source[0]
                source.reset()
                torch.testing.assert_close(source[0], expected[0], rtol=0, atol=0)
            finally:
                source.close()
            self.assertFalse(source.thread.is_alive())

    def test_video_preprocessing_and_early_close(self):
        import cv2
        from sam3.model.io_utils import load_video_frames_from_video_file_using_cv2

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.avi"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (16, 12)
            )
            for i in range(8):
                writer.write(np.full((12, 16, 3), 15 * i, dtype=np.uint8))
            writer.release()
            expected, _, _ = load_video_frames_from_video_file_using_cv2(
                str(path), 8, offload_video_to_cpu=True
            )
            source = StreamingVideoFrames(path, 8, (0.5,) * 3, (0.5,) * 3, "cpu")
            try:
                torch.testing.assert_close(source[0], expected[0], rtol=0, atol=0)
                torch.testing.assert_close(source[1], expected[1], rtol=0, atol=0)
            finally:
                source.close()
            self.assertFalse(source.thread.is_alive())

    def test_stats_only_outputs_and_boxes_before_overlap(self):
        from sam3.perflib.stream_outputs import render_outputs

        model = SimpleNamespace(kernel_backend="torch")
        state = {"orig_height": 3, "orig_width": 5, "output_fields": {"out_boxes_xywh"}}
        out = {
            "obj_id_to_mask": {9: torch.ones(1, 3, 5), 2: -torch.ones(1, 3, 5)},
            "obj_id_to_score": {9: 0.8, 2: 0.5},
            "obj_id_to_tracker_score": {9: 0.0},
        }
        result = render_outputs(model, state, out)
        self.assertEqual(set(result), {"out_obj_ids", "out_probs", "out_boxes_xywh"})
        np.testing.assert_array_equal(result["out_obj_ids"], [9])
        np.testing.assert_allclose(result["out_boxes_xywh"], [[0, 0, 4 / 5, 2 / 3]])

    def test_tracker_history_10000_frames(self):
        state = {
            "output_dict": {"non_cond_frame_outputs": {}},
            "output_dict_per_obj": {0: {"non_cond_frame_outputs": {}}},
            "consolidated_frame_inds": {"non_cond_frame_outputs": set()},
            "frames_already_tracked": {},
        }
        for i in range(10000):
            state["output_dict"]["non_cond_frame_outputs"][i] = {"obj_ptr": i}
            state["output_dict_per_obj"][0]["non_cond_frame_outputs"][i] = {
                "obj_ptr": i
            }
            state["frames_already_tracked"][i] = False
            prune_tracker_states([state], i, 128)
            self.assertLessEqual(
                len(state["output_dict"]["non_cond_frame_outputs"]), 128
            )
        self.assertEqual(
            set(state["output_dict"]["non_cond_frame_outputs"]), set(range(9872, 10000))
        )
        self.assertEqual(len(state["frames_already_tracked"]), 128)

    def test_low_score_selection_scans_only_retained_history(self):
        from sam3.model.sam3_tracker_base import Sam3TrackerBase

        class CountedOutputs(dict):
            checks = 0

            def __contains__(self, key):
                self.checks += 1
                return super().__contains__(key)

        history = CountedOutputs(
            {i: {"eff_iou_score": 0.0} for i in range(9873, 10001)}
        )
        owner = SimpleNamespace(max_obj_ptrs_in_encoder=16, mf_threshold=0.1)
        output = {"non_cond_frame_outputs": history}
        expected = Sam3TrackerBase.frame_filter(owner, output, False, 10001, 10002, 1)
        history.checks = 0
        output["non_cond_frame_min_idx"] = 9873
        actual = Sam3TrackerBase.frame_filter(owner, output, False, 10001, 10002, 1)
        self.assertEqual(actual, expected)
        self.assertEqual(history.checks, 128)

    def test_generated_stream_10000_frames(self):
        import os

        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "0.jpg"
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(first)
            for i in range(1, 10000):
                os.link(first, Path(directory) / f"{i}.jpg")
            source = StreamingVideoFrames(directory, 8, (0.5,) * 3, (0.5,) * 3, "cpu")
            try:
                for i in range(10000):
                    self.assertEqual(tuple(source[i].shape), (3, 8, 8))
                    self.assertLessEqual(len(source.recent), 2)
                    self.assertLessEqual(source.pending.qsize(), 2)
            finally:
                source.close()
            self.assertFalse(source.thread.is_alive())

    def test_shared_positional_storage_after_removal(self):
        # Exercise the actual nested removal path, without constructing model weights.
        import ast

        from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor

        tree = ast.parse(Path("sam3/model/sam3_tracking_predictor.py").read_text())
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_slice_state"
        )
        pos = torch.zeros(1, 64, 3, 3)
        state = {"constants": {"maskmem_pos_enc": [pos]}}
        fake = SimpleNamespace(
            use_memory_selection=False, _add_output_per_object=lambda *a, **kw: None
        )
        fake._get_maskmem_pos_enc = (
            lambda s, o: Sam3TrackerPredictor._get_maskmem_pos_enc(fake, s, o)
        )
        scope = dict(self=fake, inference_state=state, remain_old_obj_inds=[0])
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), "<removal>", "exec"),
            scope,
        )
        outputs = {
            "non_cond_frame_outputs": {
                i: {
                    "obj_ptr": torch.zeros(2, 4),
                    "object_score_logits": torch.ones(2, 1),
                    "maskmem_pos_enc": [pos.expand(2, -1, -1, -1)],
                }
                for i in range(32)
            }
        }
        scope["_slice_state"](outputs, "non_cond_frame_outputs")
        for out in outputs["non_cond_frame_outputs"].values():
            self.assertEqual(out["maskmem_pos_enc"][0].data_ptr(), pos.data_ptr())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class KernelTests(unittest.TestCase):
    def test_overlap_ties_and_zero_scores(self):
        from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor
        from sam3.perflib.triton.stream_outputs import resolve_overlaps

        # No learned layers are used by this operation.
        tracker = Sam3TrackerPredictor.__new__(Sam3TrackerPredictor)
        for n in (2, 3, 16, 64, 128):
            masks = torch.rand(n, 13, 17, device="cuda") > 0.25
            scores = torch.rand(n, device="cuda")
            scores[:2] = 0
            if n > 2:
                scores[1:3] = 0.5
            expected = (
                tracker._apply_object_wise_non_overlapping_constraints(
                    masks[:, None], scores[:, None], background_value=0
                )[:, 0]
                > 0
            )
            actual = resolve_overlaps(masks.clone(), scores)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_compile_integration(self):
        from sam3.perflib.triton.mask_iou import mask_iou_triton
        from sam3.perflib.triton.stream_outputs import resize_masks_triton

        mask = torch.rand(3, 13, 17, device="cuda") > 0.5
        compiled = torch.compile(mask_iou_triton, fullgraph=True)
        torch.testing.assert_close(
            compiled(mask, mask), mask_iou_triton(mask, mask), atol=0, rtol=0
        )
        logits = torch.randn(1, 1, 13, 17, device="cuda")
        compiled = torch.compile(resize_masks_triton, fullgraph=True)
        actual = compiled(logits, (27, 31), False, True)
        expected = resize_masks_triton(logits, (27, 31), False, True)
        for index, (a, b) in enumerate(zip(actual, expected)):
            torch.testing.assert_close(a, b, atol=1e-6 if index == 1 else 0, rtol=0)

    def test_iou_exact(self):
        from sam3.perflib.masks_ops import mask_iou
        from sam3.perflib.triton.mask_iou import mask_iou_triton

        torch.manual_seed(2)
        for n, m, h, w in [
            (0, 3, 13, 17),
            (1, 1, 1, 1),
            (3, 5, 13, 17),
            (16, 9, 63, 63),
            (64, 7, 288, 288),
        ]:
            a = torch.rand(n, h, w, device="cuda") > 0.5
            b = torch.rand(m, h, w, device="cuda") > 0.6
            for packed in (False, True):
                actual = mask_iou_triton(a, b, packed)
                torch.testing.assert_close(actual, mask_iou(a, b), atol=0, rtol=0)
        a = torch.zeros(3, 7, 9, dtype=torch.bool, device="cuda").transpose(1, 2)
        torch.testing.assert_close(
            mask_iou_triton(a, a), torch.zeros(3, 3, device="cuda")
        )

    def test_resize_and_optional_outputs(self):
        torch.manual_seed(9)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            logits = torch.randn(3, 1, 9, 13, device="cuda", dtype=dtype)
            for shape in ((17, 23), (5, 7), (9, 13)):
                expected = resize_masks(logits, shape, backend="torch")
                actual = resize_masks(logits, shape, backend="triton")
                torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
                torch.testing.assert_close(
                    actual[1],
                    expected[1],
                    atol=0.008 if dtype == torch.bfloat16 else 0.001,
                    rtol=0.001,
                )
                torch.testing.assert_close(actual[2], expected[2], atol=0, rtol=0)
                sparse = resize_masks(logits, shape, False, False, "triton")
                self.assertEqual(sparse[0].numel() + sparse[1].numel(), 0)
                torch.testing.assert_close(sparse[2], actual[2], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
