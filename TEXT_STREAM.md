# Text-prompt streaming inference

`text_stream` is an opt-in, forward-only mode for one text prompt at frame zero.
It supports video files and directories of contiguous, unpadded JPEG names
(`0.jpg`, `1.jpg`, ...). Use one GPU per process:

```bash
CUDA_VISIBLE_DEVICES=0 python your_application.py
```

```python
from contextlib import closing
from sam3.model_builder import build_sam3_video_predictor

predictor = build_sam3_video_predictor(
    checkpoint_path="/path/to/sam3.pt",
    bpe_path="/path/to/bpe_simple_vocab_16e6.txt.gz",
    inference_mode="text_stream",
    kernel_backend="auto",
    tracker_history_frames=128,
    max_num_objects=128,
    compile=False,
)
session = predictor.handle_request({
    "type": "start_session",
    "resource_path": "/path/to/video.mp4",
    "output_fields": ["out_binary_masks", "out_boxes_xywh"],
})["session_id"]
try:
    predictor.handle_request({
        "type": "add_prompt", "session_id": session,
        "frame_index": 0, "text": "person",
    })
    with closing(predictor.handle_stream_request({
        "type": "propagate_in_video", "session_id": session,
    })) as stream:
        for result in stream:
            consume(result)  # application-defined; do not accumulate all outputs
finally:
    predictor.handle_request({"type": "close_session", "session_id": session})
```

The builder selects the single-process predictor; it does not spawn distributed
workers or require `RANK`, `WORLD_SIZE`, or rendezvous settings. Device indices are
local to `CUDA_VISIBLE_DEVICES`. Conflicting distributed settings are rejected.
Run additional processes with their own visible GPU selection.

## Contract and options

- `inference_mode="standard"` remains the default. Standard mode retains historical
  outputs for interactive editing/replay; its memory usage can grow with video length.
- Text mode accepts one nonempty text prompt at frame zero, then one forward
  propagation. Point/box edits, replay, reverse propagation, and user object removal
  are rejected before changing state. Automatic tracker object removal remains active.
- `reset_session` allows a new prompt and pass, after closing any active stream.
  Closing the propagation generator releases the decoder and tracker state.
  Closing a session also closes its active propagation. Sessions are owned by their
  predictor instance.
- `output_fields` is a session-level selection from `out_boxes_xywh`,
  `out_binary_masks`, `out_prob_masks`, and `frame_stats`. IDs (`out_obj_ids`) and
  scores (`out_probs`) are always returned. Omit the option for all existing outputs;
  use an empty list for IDs and scores alone. Unrequested keys are absent.
- Outputs retain the existing NumPy formats and sorted ID order. Boxes are computed
  **before** overlap suppression; probability masks preserve their existing semantics.
  Binary masks still use object-score overlap resolution, including first-index ties.
- `tracker_history_frames=128` bounds non-conditioning state by frame age. `None`
  disables tracker-history pruning for reference comparisons. Positive history
  values must cover both memory and pointer horizons. A positive object cap is
  required. Detection thresholds and reconditioning cadence are unchanged.
- `kernel_backend="torch"` uses reference operators. `"triton"` explicitly selects
  custom kernels on CUDA. `"auto"` enables them on validated compute capability 8.9
  and otherwise uses PyTorch; missing optional Triton also falls back in auto mode.
  **RTX 5090 execution/tuning is pending**: it uses the reference fallback in auto
  mode; explicit Triton is available for validation on that hardware.
- `"cuda"` currently raises an explicit unsupported-backend error. The measured
  remaining hotspots did not justify adding a compiled CUDA extension.
- `compile=True` remains explicit and experimental. On this validation clip it
  failed the strict per-frame regression gate, and cold max-autotune took several
  minutes. It is not enabled by text mode or automatic kernel dispatch.

## Memory and implementation

Video decoding uses a two-frame CPU prefetch queue, at most two recently consumed
CPU frames, and one shared GPU input frame. Only bounded staging tensors are pinned.
Frame metadata is lazy, not allocated as one tensor/object per video frame. Detector
feature indexing also avoids a video-sized GPU index map. Video preprocessing
matches the corresponding existing loader, including the existing OpenCV loader's
pixel scale; normalization changes are intentionally separate from this work.

Hot-start buffering retains compact low-resolution logits and score/removal
snapshots. Dense masks/probabilities are rendered only when yielded. There is no
historical output cache in text mode. Non-conditioning memories expire by age;
conditioning memories retain the first frame, the nearest attention candidates
plus margin, and the pointer horizon. Associated inputs, per-object mirrors,
frame scores, and retired-object metadata are retired consistently. Object removal
preserves shared positional-encoding storage.

Memory therefore depends on the object cap, history, image sizes, hot-start delay,
and model/allocator caches, rather than elapsed video length. Caller-retained output
arrays are outside this bound. An unbounded reference history intentionally grows.
Deleting older candidates can change tracking on other footage: the local agreement
results below are not a general bit-identity or tracking-quality guarantee.

Custom Triton operations provide tiled mask IoU, packed-bit/popcount IoU for larger
pair counts, fused resize/threshold/sigmoid/statistics, and in-place output overlap
resolution. Integer intersections/unions and tie rules are tested exactly. Scratch
storage scales with tiles rather than pair-by-pixel intermediates. PyTorch compile
integration is tested for the custom IoU and renderer. Association reuses a single
CPU IoU transfer and avoids repeated scalar score transfers.

## Measured results

Measured on RTX 4090, PyTorch 2.13.0+cu130, local weights, prompt `person`.
Proprietary test footage and prediction artifacts are not committed.

The 64-frame branch-reference comparison passed. A separate **1,113-frame**
comparison ran two independent GPU processes: unbounded-history/PyTorch reference
versus 128-frame-history/Triton. Every frame had identical object IDs and binary
masks (aggregate and minimum mask IoU **1.0**); maximum frame probability MAE was
**4.38e-10**. Near frame 1,100, the candidate retained **120 non-conditioning + 7
conditioning** entries versus **1,043 + 70**, and host RSS was approximately
**2.73 GiB versus 5.62 GiB**. Candidate RSS plateaued after the history filled.

Three warmed 64-frame repeats per configuration, run on separate visible GPUs:

| Configuration | FPS range | Peak allocated VRAM |
| --- | ---: | ---: |
| Standard, all outputs | 7.77–8.06 | 5.59 GiB |
| Text stream, Triton, all outputs | 8.22–8.31 | 4.49 GiB |
| Text stream, PyTorch, binary only | 8.45–8.56 | 4.49 GiB |
| Text stream, Triton, binary only | 8.50–8.52 | 4.49 GiB |

This mostly single-person workload does **not** meet the aspirational 20% end-to-end
speedup target. Kernel gains are much larger at crowded object counts:

| Mask IoU, 288×288 | Eager PyTorch | Compiled PyTorch | Selected Triton |
| --- | ---: | ---: | ---: |
| 1×1 pairs | 0.044 ms | 0.022 ms | 0.019 ms |
| 16×16 pairs | 0.871 ms | 0.064 ms | 0.025 ms |
| 64×64 pairs | 13.83 ms | 0.688 ms | 0.032 ms |
| 128×128 pairs | 55.07 ms | 2.697 ms | 0.075 ms |

Rendering three full-resolution masks with probabilities and statistics took
approximately **0.039 ms versus 0.302 ms** for the PyTorch reference operator.
Timing used 10 warmups and 30 CUDA-event iterations per operator configuration.
These operator timings do not imply equivalent whole-model gains.

Steady GPU profiling shows existing BF16 GEMMs and Flash Attention dominating;
the remaining individual custom-CUDA candidates did not pass the 10% workload gate.
Compiled model inference had aggregate mask IoU **0.99823**, but a minimum frame
IoU **0.97959**, below the required 0.98. Its warm median frame latency was about
115 ms versus 127 ms eager; this does not override the failed quality gate.

## Reproduction and acceptance gates

Use `PYTHONPATH=.` with the benchmark entrypoints:

```bash
# Exact integer, numerical, overlap, compile, loader, and 10k-frame state tests
.venv/bin/python -m unittest discover -s tests -v

# Optional public API lifecycle test with the local checkpoint paths
SAM3_INTEGRATION=1 .venv/bin/python -m unittest discover -s tests -v

PYTHONPATH=. .venv/bin/python benchmarks/benchmark_kernels.py --warmup 10 --runs 30
PYTHONPATH=. .venv/bin/python benchmarks/benchmark_text_stream.py \
  --mode text_stream --backend triton --frames 64 --warmup-pass

# Full video comparison, one process per selected GPU, no prediction archive
PYTHONPATH=. .venv/bin/python benchmarks/validate_text_stream.py \
  --video /path/to/numbered-jpegs --frames 1113 --gpus 0 1 --report /tmp/comparison.json

# Four configurations on four GPUs, three warm repeats each
PYTHONPATH=. .venv/bin/python benchmarks/benchmark_text_suite.py \
  --output-dir /tmp/text-stream-timings
```

`benchmark_text_stream.py` also supports `--save`, `--compare`, `--profile`,
`--binary-only`, `--history 0` (unbounded), and `--report`. Benchmark `--frames`
specifies the number of output frames; the existing request API's
`max_frame_num_to_track` retains its inclusive-start behavior.

Regression gates: identical emitted IDs/counts; aggregate foreground IoU ≥0.995;
every frame IoU ≥0.98; per-frame probability MAE ≤0.001 and p99 error ≤0.01.
These are baseline agreement checks, not labeled HOTA/MOTA accuracy measurements.
Tests include 10,000 generated loader frames and 10,000 synthetic tracker-state
updates, not 10,000 full neural-network inference frames. The actual model was
validated for 1,113 frames. RTX 5090 and crowded real-world tracking-quality
validation remain pending.

Formatting follows `ufmt==2.8.0` with `ruff-api==0.1.0` and `usort==1.0.2`, as
configured in `pyproject.toml` and CI; Black is not the selected formatter.
