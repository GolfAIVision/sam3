"""Local video regression/throughput runner; prediction artifacts are opt-in."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import psutil
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="testset")
    parser.add_argument(
        "--checkpoint", default="/data/model_weights/segment_anything/sam3/sam3.pt"
    )
    parser.add_argument(
        "--bpe",
        default="/data/model_weights/segment_anything/sam3/bpe_simple_vocab_16e6.txt.gz",
    )
    parser.add_argument(
        "--mode", default="standard", choices=["standard", "text_stream"]
    )
    parser.add_argument(
        "--backend", default="torch", choices=["torch", "triton", "auto", "cuda"]
    )
    parser.add_argument(
        "--history", type=int, default=128, help="0 retains unbounded reference history"
    )
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--prompt", default="person")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--warmup-pass", action="store_true")
    parser.add_argument("--binary-only", action="store_true")
    parser.add_argument("--save", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    from sam3.model_builder import build_sam3_video_model

    torch.manual_seed(0)
    start = time.perf_counter()
    extra = {}
    if args.mode != "standard":
        extra = dict(
            inference_mode=args.mode,
            kernel_backend=args.backend,
            tracker_history_frames=args.history or None,
        )
    model = build_sam3_video_model(
        checkpoint_path=args.checkpoint,
        bpe_path=args.bpe,
        load_from_HF=False,
        compile=args.compile,
        **extra,
    )
    state = model.init_state(args.video, offload_video_to_cpu=True)
    if args.binary_only:
        state["output_fields"] = {"out_obj_ids", "out_probs", "out_binary_masks"}
    model.add_prompt(state, 0, text_str=args.prompt)
    torch.cuda.synchronize()
    init_seconds = time.perf_counter() - start
    warmup_seconds = 0.0
    if args.warmup_pass:
        warmup_start = time.perf_counter()
        for _ in model.propagate_in_video(state, 0, args.frames - 1):
            pass
        model.reset_state(state)
        model.add_prompt(state, 0, text_str=args.prompt)
        torch.cuda.synchronize()
        warmup_seconds = time.perf_counter() - warmup_start
        torch.cuda.reset_peak_memory_stats()
    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
    timings, memories, comparisons = [], [], []
    process = psutil.Process()
    profiler = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        )
        if args.profile
        else None
    )
    if profiler:
        profiler.start()
    start = previous = time.perf_counter()
    try:
        for frame, out in model.propagate_in_video(
            state, start_frame_idx=0, max_frame_num_to_track=args.frames - 1
        ):
            torch.cuda.synchronize()
            now = time.perf_counter()
            timings.append(now - previous)
            memories.append(
                (
                    torch.cuda.memory_allocated(),
                    torch.cuda.memory_reserved(),
                    process.memory_info().rss,
                )
            )
            if args.save:
                np.savez_compressed(
                    args.save / f"{frame}.npz",
                    **{k: v for k, v in out.items() if isinstance(v, np.ndarray)},
                )
            if args.compare:
                with np.load(args.compare / f"{frame}.npz") as ref:
                    same_ids = np.array_equal(out["out_obj_ids"], ref["out_obj_ids"])
                    entry = {"frame": frame, "same_ids": same_ids}
                    if same_ids and "out_binary_masks" in out:
                        a, b = out["out_binary_masks"], ref["out_binary_masks"]
                        intersection, union = int((a & b).sum()), int((a | b).sum())
                        entry.update(
                            intersection=intersection,
                            union=union,
                            iou=intersection / union if union else 1.0,
                        )
                    if same_ids and "out_prob_masks" in out:
                        error = np.abs(out["out_prob_masks"] - ref["out_prob_masks"])
                        entry.update(
                            prob_mae=float(error.mean()) if error.size else 0,
                            prob_p99=(
                                float(np.quantile(error, 0.99)) if error.size else 0
                            ),
                        )
                    comparisons.append(entry)
            previous = time.perf_counter()
    finally:
        if profiler:
            profiler.stop()
            profiler.export_chrome_trace(str(args.profile))
            print(
                profiler.key_averages().table(
                    sort_by="self_cuda_time_total", row_limit=20
                )
            )
        images = state["input_batch"].img_batch
        if hasattr(images, "close"):
            images.close()
    result = dict(
        mode=args.mode,
        backend=args.backend,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        frames=len(timings),
        init_seconds=init_seconds,
        warmup_seconds=warmup_seconds,
        compute_fps=len(timings) / sum(timings),
        first_yield_seconds=timings[0],
        median_seconds=float(np.median(timings)),
        p95_seconds=float(np.quantile(timings, 0.95)),
        peak_allocated=torch.cuda.max_memory_allocated(),
        memory_samples=memories,
        comparisons=comparisons,
    )
    if comparisons:
        ids_ok = all(c["same_ids"] for c in comparisons)
        union = sum(c.get("union", 0) for c in comparisons)
        mean_iou = (
            sum(c.get("intersection", 0) for c in comparisons) / union if union else 1.0
        )
        minimum_iou = min(c.get("iou", 0.0) for c in comparisons)
        result["quality_gate"] = dict(
            passed=ids_ok
            and mean_iou >= 0.995
            and minimum_iou >= 0.98
            and all(
                c.get("prob_mae", 0) <= 0.001 and c.get("prob_p99", 0) <= 0.01
                for c in comparisons
            ),
            ids_match=ids_ok,
            aggregate_iou=mean_iou,
            minimum_frame_iou=minimum_iou,
        )
    if args.report:
        args.report.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("memory_samples", "comparisons")
            }
        )
    )


if __name__ == "__main__":
    main()
