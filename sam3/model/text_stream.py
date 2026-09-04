"""Bounded, sequential inputs and state maintenance for text-only video sessions."""

import queue
import threading
from collections import OrderedDict
from pathlib import Path

import torch

OUTPUT_FIELDS = frozenset(
    {
        "out_obj_ids",
        "out_probs",
        "out_boxes_xywh",
        "out_binary_masks",
        "out_prob_masks",
        "frame_stats",
    }
)


def output_fields(fields=None):
    fields = OUTPUT_FIELDS if fields is None else frozenset(fields)
    if fields - OUTPUT_FIELDS:
        raise ValueError(f"Unknown output fields: {sorted(fields - OUTPUT_FIELDS)}")
    return fields | {"out_obj_ids", "out_probs"}


class LazyFrames:
    """Length-aware default values with bounded recent overrides (no video-sized list)."""

    def __init__(self, length, factory, capacity=8):
        self.length, self.factory, self.capacity = length, factory, capacity
        self.values = OrderedDict()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not 0 <= index < self.length:
            raise IndexError(index)
        if index not in self.values:
            self[index] = self.factory(index)
        return self.values[index]

    def __setitem__(self, index, value):
        if not 0 <= index < self.length:
            raise IndexError(index)
        self.values.pop(index, None)
        self.values[index] = value
        while len(self.values) > self.capacity:
            self.values.popitem(last=False)

    def clear(self):
        self.values.clear()


class StreamingVideoFrames:
    """Two-frame CPU prefetch queue with one shared GPU frame.

    Folder inputs use contiguous, unpadded numeric JPEG names starting at zero.
    Video preprocessing deliberately matches the existing cv2 loader, including
    its pixel scale; changing normalization is a separate accuracy change.
    """

    def __init__(self, resource, image_size, mean, std, device):
        import cv2

        self.path = Path(resource)
        self.image_size, self.device = image_size, torch.device(device)
        self.mean = torch.tensor(mean, dtype=torch.float16)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float16)[:, None, None]
        self.folder = self.path.is_dir()
        if self.folder:
            import os

            self.suffix = next(
                (
                    s
                    for s in (".jpg", ".jpeg", ".JPG", ".JPEG")
                    if (self.path / ("0" + s)).exists()
                ),
                None,
            )
            if self.suffix is None:
                raise ValueError(
                    "text_stream folders require JPEGs numbered 0.jpg, 1.jpg, ..."
                )
            with os.scandir(self.path) as files:
                self.length = sum(
                    p.is_file() and p.name.endswith(self.suffix) for p in files
                )
            from PIL import Image

            with Image.open(self.path / ("0" + self.suffix)) as im:
                self.video_width, self.video_height = im.size
        else:
            cap = cv2.VideoCapture(str(self.path))
            try:
                if not cap.isOpened():
                    raise ValueError(f"Cannot open video: {self.path}")
                self.length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            finally:
                cap.release()
        if self.length <= 0:
            raise ValueError(
                "text_stream requires a nonempty video with a known frame count"
            )
        self._start()

    def _start(self):
        self.pending = queue.Queue(maxsize=2)
        self.stop = threading.Event()
        self.recent = OrderedDict()
        self.gpu_index, self.gpu_image, self.transfer = None, None, None
        self.next_index = 0
        self.thread = threading.Thread(
            target=self._produce, daemon=True, name="sam3-frame-reader"
        )
        self.thread.start()

    def _put(self, value):
        while not self.stop.is_set():
            try:
                self.pending.put(value, timeout=0.1)
                return
            except queue.Full:
                pass

    @torch.inference_mode()
    def _produce(self):
        import cv2
        import numpy as np

        cap = None
        try:
            if not self.folder:
                cap = cv2.VideoCapture(str(self.path))
            for index in range(self.length):
                if self.stop.is_set():
                    break
                if self.folder:
                    from sam3.model.io_utils import _load_img_as_tensor

                    tensor, h, w = _load_img_as_tensor(
                        str(self.path / f"{index}{self.suffix}"), self.image_size
                    )
                    if (h, w) != (self.video_height, self.video_width):
                        raise ValueError(
                            "All stream frames must have the same dimensions"
                        )
                    tensor = tensor.half()
                else:
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(
                            f"Video ended before metadata frame count at frame {index}"
                        )
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(
                        rgb,
                        (self.image_size, self.image_size),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    tensor = torch.from_numpy(resized.astype(np.float32)).permute(
                        2, 0, 1
                    )
                tensor = ((tensor - self.mean) / self.std).contiguous()
                if self.device.type == "cuda":
                    tensor = tensor.pin_memory()
                self._put((index, tensor))
        except Exception as exc:
            self._put(exc)
        finally:
            if cap is not None:
                cap.release()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index in self.recent:
            return self.recent[index]
        if self.stop.is_set():
            raise RuntimeError("Stream is closed; reset before reusing it")
        if index != self.next_index or index >= self.length:
            raise ValueError(
                f"text_stream requires sequential frames; expected {self.next_index}, got {index}"
            )
        value = self.pending.get()
        if isinstance(value, Exception):
            raise value
        loaded_index, tensor = value
        assert loaded_index == index
        self.recent[index] = tensor
        self.next_index += 1
        while len(self.recent) > 2:
            self.recent.popitem(last=False)
        return tensor

    def get_gpu_frame(self, index):
        if self.gpu_index != index:
            if self.transfer is not None:
                self.transfer.synchronize()
            frame = self[index]
            self.gpu_image = frame.to(self.device, non_blocking=True)
            self.gpu_index = index
            if self.device.type == "cuda":
                self.transfer = torch.cuda.Event()
                self.transfer.record(torch.cuda.current_stream(self.device))
        return self.gpu_image

    def close(self):
        self.stop.set()
        self.thread.join()
        if self.transfer is not None:
            self.transfer.synchronize()
        self.gpu_image = None
        self.gpu_index = None
        self.recent.clear()
        while not self.pending.empty():
            self.pending.get_nowait()

    def reset(self):
        self.close()
        self._start()


def prune_tracker_states(states, frame, history):
    """Delete expired non-cond history in insertion order; cond pruning is separate."""
    if history is None:
        return
    cutoff = frame - history
    for state in states:
        state["output_dict"]["non_cond_frame_min_idx"] = max(cutoff + 1, 0)
        outputs = state["output_dict"]["non_cond_frame_outputs"]
        # Forward-only insertion order makes eviction proportional to expired frames.
        while outputs:
            oldest = next(iter(outputs))
            if oldest > cutoff:
                break
            del outputs[oldest]
            state["consolidated_frame_inds"]["non_cond_frame_outputs"].discard(oldest)
            for obj in state["output_dict_per_obj"].values():
                obj["non_cond_frame_outputs"].pop(oldest, None)
        tracked = state["frames_already_tracked"]
        while tracked and next(iter(tracked)) <= cutoff:
            del tracked[next(iter(tracked))]


def prune_metadata(metadata, frame, delay):
    """Keep only metadata still read by active tracks or delayed output consumers."""
    live = set(metadata["obj_ids_all_gpu"])
    cutoff = frame - max(delay, 2) - 2
    rank = metadata["rank0_metadata"]
    for mapping in (
        metadata["obj_id_to_tracker_score_frame_wise"],
        rank["suppressed_obj_ids"],
    ):
        while mapping and next(iter(mapping)) < cutoff:
            del mapping[next(iter(mapping))]
    for name in ("obj_id_to_last_occluded", "obj_id_to_score"):
        for obj in list(metadata[name]):
            if obj not in live:
                del metadata[name][obj]
    for name in ("obj_first_frame_idx", "unmatched_frame_inds", "trk_keep_alive"):
        for obj in list(rank[name]):
            if obj not in live:
                del rank[name][obj]
    for pair in list(rank["overlap_pair_to_frame_inds"]):
        if not set(pair) <= live:
            del rank["overlap_pair_to_frame_inds"][pair]
    # Removed IDs cannot reappear; delayed outputs own their removal snapshots.
    rank["removed_obj_ids"].intersection_update(live)
