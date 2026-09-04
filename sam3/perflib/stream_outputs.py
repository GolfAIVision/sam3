"""Render only requested fields from compact, delayed text-stream predictions."""

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sam3.perflib.backend import use_triton
from sam3.perflib.masks_ops import masks_to_boxes


def resize_masks(logits, size, binary=True, probability=True, backend="torch"):
    if use_triton(logits, backend):
        from sam3.perflib.triton.stream_outputs import resize_masks_triton

        return resize_masks_triton(logits, size, binary, probability)
    values = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
    masks = values[:, 0] > 0
    boxes = masks_to_boxes(masks, list(range(len(masks))))
    stats = torch.cat((boxes, masks.sum((1, 2)).unsqueeze(1)), dim=1).float()
    probs = values[:, 0].sigmoid().float() if probability else logits.new_empty(0)
    return masks if binary else masks.new_empty(0), probs, stats


def render_outputs(model, state, out, removed=None, suppressed=None, unconfirmed=None):
    from sam3.model.text_stream import output_fields

    fields = output_fields(state.get("output_fields"))
    h, w = state["orig_height"], state["orig_width"]
    hide = set(removed or ()) | set(suppressed or ()) | set(unconfirmed or ())
    ids = sorted(obj for obj in out["obj_id_to_mask"] if obj not in hide)
    need_binary, need_prob = "out_binary_masks" in fields, "out_prob_masks" in fields
    groups = defaultdict(list)
    for obj in ids:
        mask = out["obj_id_to_mask"][obj]
        groups[(mask.shape, mask.dtype)].append(obj)
    rendered = {}
    for objects in groups.values():
        logits = torch.stack([out["obj_id_to_mask"][obj] for obj in objects])
        masks, probs, stats = resize_masks(
            logits, (h, w), need_binary, need_prob, model.kernel_backend
        )
        stats_cpu = stats.cpu().numpy()
        for index, obj in enumerate(objects):
            if stats_cpu[index, 4] > 0:
                rendered[obj] = (
                    masks[index] if need_binary else None,
                    probs[index] if need_prob else None,
                    stats_cpu[index, :4],
                )
    ids = [obj for obj in ids if obj in rendered]
    result = {
        "out_obj_ids": np.asarray(ids, dtype=np.int64),
        "out_probs": np.asarray(
            [out["obj_id_to_score"][obj] for obj in ids], dtype=np.float32
        ),
    }
    if "out_boxes_xywh" in fields:
        boxes = np.asarray([rendered[obj][2] for obj in ids], dtype=np.float32).reshape(
            -1, 4
        )
        boxes[:, 2:] -= boxes[:, :2]
        boxes /= np.asarray([w, h, w, h], dtype=np.float32)
        result["out_boxes_xywh"] = boxes
    if need_binary:
        if ids:
            masks = torch.stack([rendered[obj][0] for obj in ids])
            if len(ids) > 1:
                scores = torch.tensor(
                    [out["obj_id_to_tracker_score"].get(obj, 0.0) for obj in ids],
                    device=masks.device,
                )
                if use_triton(masks, model.kernel_backend):
                    from sam3.perflib.triton.stream_outputs import resolve_overlaps

                    masks = resolve_overlaps(masks, scores)
                else:
                    masks = (
                        model.tracker._apply_object_wise_non_overlapping_constraints(
                            masks.unsqueeze(1), scores[:, None], background_value=0
                        ).squeeze(1)
                        > 0
                    )
            result["out_binary_masks"] = masks.cpu().numpy()
        else:
            result["out_binary_masks"] = np.empty((0, h, w), dtype=bool)
    if need_prob:
        result["out_prob_masks"] = (
            torch.stack([rendered[obj][1] for obj in ids]).cpu().numpy()
            if ids
            else np.empty((0, h, w), dtype=np.float32)
        )
    if "frame_stats" in fields:
        result["frame_stats"] = out.get("frame_stats")
    return result
