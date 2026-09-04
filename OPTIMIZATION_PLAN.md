# sam3 memory optimization plan (long-video det-track on 24 GB GPUs)

> Historical branch plan. The implemented forward-only text-stream mode, current
> retention rules, kernel dispatch, and measured results are documented in
> [TEXT_STREAM.md](TEXT_STREAM.md). In particular, standard interactive mode now
> retains full historical outputs, while text-stream mode uses bounded deletion
> rather than the slim-entry archive described below.

Grounded in a measured failure: video det-track with a text prompt on ~1100-frame
1936x1472 videos grows ~11.06 MiB/frame on GPU and OOMs a 24 GB card around frame ~970
(22.1 GiB allocated). The ledger at that point: 3.36 GiB weights + 6.32 GiB whole-video
fp16 `img_batch` + ~7.9 GiB `cached_frame_outputs` (3 live objects) + ~1.2-5 GiB tracker
states + transients. Line numbers below were verified against sam3 0.1.0 and upstream
main as of 2026-08-26; re-grep before patching.

## Growth mechanisms (verified)

- **G1 - `cached_frame_outputs` (dominant)**: `propagate_in_video` calls
  `_cache_frame_outputs` every yielded frame (`sam3/model/sam3_video_inference.py`,
  def ~L538, call ~L336). Values are full-video-resolution bool CUDA masks
  (1, H, W) per object per frame (~2.85 MB/obj/frame), built via `F.interpolate` in
  `build_outputs` (`sam3/model/sam3_video_base.py` ~L953-1018). Nothing prunes by frame
  index; only `remove_object` (~L1320-1327) deletes per-object entries. The cache is
  only *read* by interactive refinement (`_build_tracker_output` ~L565 and the
  `propagation_fetch` branch ~L1059) - pure waste for non-interactive use.
- **G2 - tracker-state accumulation**: `_tracker_add_new_objects`
  (`sam3_video_base.py` ~L1521) creates a new SAM2 tracker state on every frame with new
  detections. Each state stores per remaining frame in
  `output_dict["non_cond_frame_outputs"][frame_idx]` (`sam3_tracking_predictor.py`
  `_run_single_frame_inference` ~L1046): `maskmem_features` bf16 [B,64,63,63]
  (~0.5 MB/obj), `pred_masks` fp32 [B,1,288,288] (~0.33 MB/obj), `obj_ptr`,
  `object_score_logits`, mirrored in `output_dict_per_obj`. Community-measured
  ~3 MB/frame/object (issue #408). States live until emptied by object removal.
- **G3 - `cond_frame_outputs` buildup**: every birth frame plus reconditioning every
  `recondition_every_nth_frame=16` (`_recondition_masklets` ~L454) adds conditioning
  entries that persist even though attention selects at most
  `max_cond_frames_in_attn=4`.
- **G4 - fixed base**: `init_state` -> `load_video_frames_from_array`
  (`io_utils.py` ~L338) converts to fp16 (N,3,1008,1008);
  `_construct_initial_input_batch` (~L113-143) does an unconditional
  `copy_data_to_device(input_batch, device)` (~L150), so the whole video sits on GPU
  (~6.7 GiB at 1103 frames) even with `offload_video_to_cpu=True` (the kwarg only avoids
  a load-time duplicate).
- **G5 - transients**: detector (ViT-L + DETR decoder at 1008^2) and tracker FPN
  activations set the per-frame peak; the OOM site is
  `_prepare_memory_conditioned_features` (`sam3_tracker_base.py` ~L560-783) building the
  cross-attn prompt (<=4 cond + 6 mem frames x 5184 tokens + obj-ptrs per state).

## Patches, in implementation order

### P1 - disable `cached_frame_outputs` (biggest win, zero risk here)
Make `_cache_frame_outputs` return early (or keep only a rolling window of ~8 frames),
and guard `_build_tracker_output`'s cache assertion for the interactive path.
Upstream #408 recommends exactly this. Reclaims ~2.85 MB/obj/frame (~28 MB/frame at 10
objects). Non-interactive video inference is unaffected.

### P2 - rolling-window prune of tracker `non_cond_frame_outputs`
After each frame, pop entries older than `frame_idx - (num_maskmem + r)` from
`output_dict["non_cond_frame_outputs"]` and from every
`output_dict_per_obj[obj]["non_cond_frame_outputs"]`, across ALL tracker states (the
det-track path maintains a list of states). Good insertion points: end of
`_det_track_one_frame` (`sam3_video_base.py` ~L151) or right after
`output_dict[storage_key][frame_idx] = current_out` (`sam3_tracking_predictor.py`
~L860). Only memories older than the 7-frame memory window are dropped; community-
verified on 5500+ frame runs. State becomes O(window) instead of O(video).

### P6 - keep `img_batch` on CPU, move per frame
In `_construct_initial_input_batch`, skip the wholesale `copy_data_to_device`; move
only `img_batch[frame_idx]` where consumed (detector call + `_get_image_feature`),
pinned + non_blocking. Reclaims the fixed ~6.7 GiB on 1100-frame videos at a small H2D
cost per frame.

### W4 - cap `max_num_objects`
`Sam3VideoBase.__init__(max_num_objects=-1)` resolves to 10000/"unlimited"
(`sam3_video_base.py` ~L119-126); `_drop_new_det_with_obj_limit` (~L562-570) already
drops new detections by score when the cap is hit. Default to ~128 for generic prompts.
Caveat: `num_obj_for_compile = ceil(max_num_objects/world_size)` - with compile enabled,
warm-up scales with the cap.

### P4 - thread `offload_state_to_cpu=True` into det-track state creation
`Sam3TrackerPredictor.init_state` supports it (docstring ~L53-75, incl. measured fps
cost ~27->24 at 1 obj) but the det-track calls
(`sam3_video_inference.py` ~L1001, `sam3_video_base.py` ~L1540) never pass it. The
tracker already re-uploads memories per read
(`prev["maskmem_features"].cuda(non_blocking=True)`, `sam3_tracker_base.py` ~L646), so
the change is small. Moves nearly all G2 growth to host RAM.

### P3 - prune stale `cond_frame_outputs` (optional, medium quality risk)
Keep the first + most recent K (e.g. 3) detection-conditioning frames per state, drop
the rest. Upstream warns this affects tracking (#408: "some logic is needed to decide
which ones to keep"); measure quality before shipping.

> **Implementation addendum (flag-gated, default-off).** Implemented as
> `Sam3VideoBase._prune_tracker_cond_outputs` (sam3/model/sam3_video_base.py), called
> from the det-track step next to the P2 non-cond prune, gated by the additive
> `prune_tracker_cond_outputs` kwarg on `Sam3VideoBase` and
> `build_sam3_video_model` (default `None` = legacy unbounded conditioning memory,
> bit-identical). Corrections vs. the sketch above:
> - **K correction:** K = `max_cond_frames_in_attn + 2` (= 6 with the shipped
>   `max_cond_frames_in_attn=4`, model_builder.py), **not** the suggested 3. The
>   memory attention selects up to `max_cond_frames_in_attn` conditioning frames
>   temporally closest to the current frame (`select_closest_cond_frames`,
>   sam3_tracker_utils.py:270-324); tracking forwards these are the
>   `max_cond_frames_in_attn` largest cond indices, so keeping only the 3 newest
>   would prune a frame the attention would have selected and change tracking.
>   K = 4 + 2 keeps the selectable set plus margin.
> - **Prune horizon:** entries are deleted only when `t <= frame_idx -
>   max_obj_ptrs_in_encoder` (reverse-symmetric: `t >= frame_idx +
>   max_obj_ptrs_in_encoder`) AND outside the kept set. The obj-ptr loop and the
>   mask-attention window never read cond entries more than
>   `max_obj_ptrs_in_encoder - 1` frames behind the current frame
>   (sam3_tracker_base.py:645-656, :724-739), so pruned entries are unread on the
>   forward pass -> opt-in forward bit-identity by construction.
> - **Invariants:** keep the FIRST cond frame (protects the predictor's
>   `first_ann_frame_idx` fallback / default propagation start,
>   sam3_tracking_predictor.py:745-766) and never prune a state's cond dict to
>   empty (guards at sam3_tracker_base.py:591 and sam3_tracking_predictor.py:809);
>   per-object cond dicts are pruned in lockstep, stale
>   `consolidated_frame_inds["cond_frame_outputs"]` entries are discarded, and the
>   pruned frames' registered point/mask inputs are dropped in lockstep (required by
>   the preflight bookkeeping assert `consolidated_frame_inds == frames with
>   point/mask inputs`, sam3_tracking_predictor.py:728-739 -- caught by the real-data
>   flag-on verification run; the dropped inputs were already consumed when the
>   pruned cond frames' memories were encoded).
> - **Default-off rationale:** forward det-track runs are bit-identical by
>   construction with the flag on, but the P3 quality gate still applies to
>   (1) reverse re-propagation composition (the temporally-closest selection can
>   reach beyond the kept set on sparse cond timelines in reverse) and (2)
>   interactive refinement on pruned cond frames. Both need a tracking-quality
>   metric (HOTA/MOTA) before the flag is enabled in production, so it ships
>   opt-in only.
> - **Measured motivation (real 64-frame testset, prompt "person"):** cond keys
>   [0, 16, 32, 48] after 64 frames (~1.9 MiB GPU + ~2.85 MiB host per cond event
>   per state) -- unbounded on long videos. With the flag on, the bound is
>   ~1 + K + horizon/cadence events per state
>   (`horizon = max_obj_ptrs_in_encoder`), O(1) at any
>   `recondition_every_nth_frame` cadence; exactly 1 + K = 7 only at the shipped
>   cadence 16 (measured 9 events at cadence 2 -- the within-horizon cond events
>   are legitimately retained).
> - **Flag-on verification (same testset):** against the same-schedule unpruned
>   control (identical `recondition_every_nth_frame` schedule), 0/64 frames
>   mismatch with the flag on at cadence 4; against the shipped cadence-16
>   baseline, 59/64 frames mismatch, but the flag-off cadence-4 control shows
>   the identical divergence profile -- i.e. the divergence is attributable to
>   the cadence change, not to the prune.

### P7 / P8 / W5 - wrapper-side complements
- `remove_object(inference_state, obj_id, is_user_action=False)` (~L1294) drops emptied
  tracker states and cache entries; expire objects not seen for N frames (mind
  re-ID of occluded people; built-in hotstart: `hotstart_delay=15`,
  `max_trk_keep_alive=30`).
- Chunked sessions (PR #602 pattern): fresh `init_state` + text prompt per N-frame
  chunk; ran 850 frames on a 12 GB GPU. Hard memory bound; seam artifacts at chunk
  boundaries. Good no-surgery fallback.
- Raising `score_threshold_detection` (default 0.5) / `new_det_thresh` (default 0.7)
  in `build_sam3_video_model` (~L684-688) reduces spurious IDs and state creation for
  generic concepts (#413).

## Do NOT do (verified failures/degradations on the det-track path)

- `offload_output_to_cpu_for_eval=True`: the det-track path splits propagation
  (`run_mem_encoder=False`) from memory encoding; the trim block
  (`sam3_tracker_base.py` ~L1048-1063) omits the `maskmem_features` key in that mode and
  `sam3_tracking_predictor._run_single_frame_inference` reads it unconditionally
  (~L1085) -> KeyError on the first propagated frame. Works only for semi-supervised VOS.
- `trim_past_non_cond_mem_for_eval=True`: drops `eff_iou_score` from trimmed entries,
  and `frame_filter` (~L518-556) skips entries lacking it, shrinking the obj-ptr
  selection pool (~16 -> ~6 candidates); also a stray debug
  `print(past_out.get("eff_iou_score", 0))` fires every frame per state (~L1083).
- `image_size=1008` is load-bearing (positional encodings, decoder buffers, checkpoint
  compatibility) - do not change.
- There are no `VOS_OPT_*`-style env vars in sam3; SAM2's do not carry over.
- bf16 autocast is already active for tracker and det-track frames; `maskmem_features`
  are stored bf16. Optionally cast stored `pred_masks` to bf16 (P9, ~1-line, minor).

## Expected end state and verification

After P1+P2+P6 (+W4), GPU residency should be ~weights + one frame of features +
windowed state (a few GiB regardless of video length) vs 22 GiB today.

Measure with:
- end-state growth or median frame-to-frame delta of
  `torch.cuda.max_memory_allocated()` (raw peaks false-alarm on transients),
  before/after, on a fixed ~200-frame clip: pre-fix ~11 MiB/frame total,
  ~2.9 MiB/frame with only the output cache cleared; target < 1 MiB/frame with P2.
- mask equality on a fixed seed clip (before/after each patch) to prove
  behavior preservation; P3/P7 need a tracking-quality metric (HOTA/MOTA on a labeled
  clip) rather than mask equality.

References: upstream issues #408 (streaming playbook, ~3 MB/frame/object measurement),
#413, #169, #481, #495; PR #602 (chunked sessions, 12 GB defaults).
