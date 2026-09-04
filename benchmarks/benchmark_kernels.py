"""Shape-specific kernel timings on the current visible GPU."""

import argparse
import json
import time

import torch
from sam3.perflib.masks_ops import mask_iou
from sam3.perflib.stream_outputs import resize_masks
from sam3.perflib.triton.mask_iou import mask_iou_triton


def measure(fn, warmup, runs):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(runs):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return sorted(values)[len(values) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()
    compiled_iou = torch.compile(mask_iou, fullgraph=True)
    torch.manual_seed(0)
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__}))
    for n in (1, 3, 16, 64, 128):
        a = torch.rand(n, 288, 288, device="cuda") > 0.5
        for name, fn in [
            ("torch", lambda: mask_iou(a, a)),
            ("compiled_torch", lambda: compiled_iou(a, a)),
            ("tiled", lambda: mask_iou_triton(a, a, False)),
            ("packed", lambda: mask_iou_triton(a, a, True)),
        ]:
            print(
                json.dumps(
                    {
                        "op": "iou",
                        "n": n,
                        "backend": name,
                        "ms": measure(fn, args.warmup, args.runs),
                    }
                ),
                flush=True,
            )
    for n in (1, 3, 16):
        logits = torch.randn(n, 1, 288, 288, device="cuda")
        for probability in (False, True):
            for backend in ("torch", "triton"):
                fn = lambda: resize_masks(
                    logits, (1472, 1936), True, probability, backend
                )
                print(
                    json.dumps(
                        {
                            "op": "render",
                            "n": n,
                            "probability": probability,
                            "backend": backend,
                            "ms": measure(fn, args.warmup, args.runs),
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
