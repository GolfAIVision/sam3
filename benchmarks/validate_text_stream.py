"""Compare independent one-GPU workers without archiving proprietary predictions."""

import argparse
import json
import multiprocessing as mp
import os
import queue
import time
import traceback
from pathlib import Path


def worker(args, gpu, history, backend, results):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["RANK"], os.environ["WORLD_SIZE"] = "0", "1"
    try:
        import numpy as np
        import psutil
        import torch
        from sam3.model_builder import build_sam3_video_model

        torch.manual_seed(0)
        model = build_sam3_video_model(
            checkpoint_path=args.checkpoint,
            bpe_path=args.bpe,
            load_from_HF=False,
            inference_mode="text_stream",
            kernel_backend=backend,
            tracker_history_frames=history,
        )
        state = model.init_state(args.video)
        model.add_prompt(state, 0, text_str="person")
        process = psutil.Process()
        elapsed = []
        previous = time.perf_counter()
        for frame, out in model.propagate_in_video(state, 0, args.frames - 1):
            elapsed.append(time.perf_counter() - previous)
            mask = out["out_binary_masks"]
            out["mask_shape"] = mask.shape
            out["out_binary_masks"] = np.packbits(mask)
            states = state["tracker_inference_states"]
            counts = {
                "non_cond": sum(
                    len(s["output_dict"]["non_cond_frame_outputs"]) for s in states
                ),
                "cond": sum(
                    len(s["output_dict"]["cond_frame_outputs"]) for s in states
                ),
            }
            results.put(
                (
                    frame,
                    out,
                    {
                        "allocated": torch.cuda.memory_allocated(),
                        "rss": process.memory_info().rss,
                        **counts,
                    },
                )
            )
            previous = time.perf_counter()
        results.put(
            (
                "done",
                {
                    "frames": len(elapsed),
                    "inference_fps": len(elapsed) / sum(elapsed),
                    "peak_allocated": torch.cuda.max_memory_allocated(),
                },
            )
        )
    except BaseException:
        results.put(("error", traceback.format_exc()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames", type=int, default=1113)
    parser.add_argument("--gpus", type=int, nargs=2, default=[0, 1])
    parser.add_argument(
        "--checkpoint", default="/data/model_weights/segment_anything/sam3/sam3.pt"
    )
    parser.add_argument(
        "--bpe",
        default="/data/model_weights/segment_anything/sam3/bpe_simple_vocab_16e6.txt.gz",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np

    ctx = mp.get_context("spawn")
    queues = [ctx.Queue(maxsize=1), ctx.Queue(maxsize=1)]
    processes = [
        ctx.Process(target=worker, args=(args, gpu, history, backend, q))
        for gpu, history, backend, q in zip(
            args.gpus, [None, 128], ["torch", "triton"], queues
        )
    ]
    for process in processes:
        process.start()
    records, timings = [], []
    try:
        while True:
            pair = []
            for q, process in zip(queues, processes):
                while True:
                    try:
                        pair.append(q.get(timeout=5))
                        break
                    except queue.Empty:
                        if not process.is_alive():
                            raise RuntimeError(
                                f"Validation worker exited unexpectedly: {process.exitcode}"
                            )
            if any(item[0] == "error" for item in pair):
                raise RuntimeError(
                    "\n".join(item[1] for item in pair if item[0] == "error")
                )
            if any(item[0] == "done" for item in pair):
                if not all(item[0] == "done" for item in pair):
                    raise RuntimeError("Worker frame counts differ")
                timings = [item[1] for item in pair]
                break
            (frame, ref, ref_mem), (other, candidate, cand_mem) = pair
            assert frame == other
            same_ids = np.array_equal(ref["out_obj_ids"], candidate["out_obj_ids"])
            row = dict(
                frame=frame, same_ids=same_ids, reference=ref_mem, candidate=cand_mem
            )
            if same_ids:
                a, b = ref["out_binary_masks"], candidate["out_binary_masks"]
                # Packed masks have zero padding, so all bits can be counted.
                intersection = int(np.unpackbits(a & b).sum())
                union = int(np.unpackbits(a | b).sum())
                diff = np.abs(ref["out_prob_masks"] - candidate["out_prob_masks"])
                row.update(
                    intersection=intersection,
                    union=union,
                    iou=intersection / union if union else 1.0,
                    prob_mae=float(diff.mean()) if diff.size else 0.0,
                    prob_p99=float(np.quantile(diff, 0.99)) if diff.size else 0.0,
                )
            records.append(row)
            if frame % 100 == 0:
                print(json.dumps(row), flush=True)
    finally:
        for process in processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join()
        for q in queues:
            q.close()
    union = sum(row.get("union", 0) for row in records)
    iou = sum(row.get("intersection", 0) for row in records) / union if union else 1.0
    passed = (
        len(records) == args.frames
        and iou >= 0.995
        and all(
            row["same_ids"]
            and row["iou"] >= 0.98
            and row["prob_mae"] <= 0.001
            and row["prob_p99"] <= 0.01
            for row in records
        )
    )
    result = dict(
        passed=passed,
        aggregate_iou=iou,
        frames=len(records),
        timings=timings,
        records=records,
    )
    args.report.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
