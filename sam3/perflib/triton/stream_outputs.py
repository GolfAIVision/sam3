"""Resize, optional dense outputs, and mask statistics without full-size logits."""

import torch
import triton
import triton.language as tl


@triton.jit
def _render(
    Input,
    Binary,
    Prob,
    Partial,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    TILES: tl.constexpr,
    WRITE_BINARY: tl.constexpr,
    WRITE_PROB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile, obj = tl.program_id(0), tl.program_id(1)
    pixel = tile * BLOCK + tl.arange(0, BLOCK)
    valid = pixel < OH * OW
    x, y = pixel % OW, pixel // OW
    fx = tl.maximum((x.to(tl.float32) + 0.5) * (IW / OW) - 0.5, 0.0)
    fy = tl.maximum((y.to(tl.float32) + 0.5) * (IH / OH) - 0.5, 0.0)
    x0, y0 = fx.to(tl.int32), fy.to(tl.int32)
    x1, y1 = tl.minimum(x0 + 1, IW - 1), tl.minimum(y0 + 1, IH - 1)
    dx, dy = fx - x0, fy - y0
    base = obj * IH * IW
    v00 = tl.load(Input + base + y0 * IW + x0, valid, 0).to(tl.float32)
    v01 = tl.load(Input + base + y0 * IW + x1, valid, 0).to(tl.float32)
    v10 = tl.load(Input + base + y1 * IW + x0, valid, 0).to(tl.float32)
    v11 = tl.load(Input + base + y1 * IW + x1, valid, 0).to(tl.float32)
    value = (1.0 - dy) * ((1.0 - dx) * v00 + dx * v01) + dy * (
        (1.0 - dx) * v10 + dx * v11
    )
    value = value.to(Input.dtype.element_ty).to(tl.float32)
    foreground = valid & (value > 0.0)
    if WRITE_BINARY:
        tl.store(Binary + obj * OH * OW + pixel, foreground, valid)
    if WRITE_PROB:
        prob = (1.0 / (1.0 + tl.exp(-value))).to(Input.dtype.element_ty).to(tl.float32)
        tl.store(Prob + obj * OH * OW + pixel, prob, valid)
    base_out = (obj * TILES + tile) * 5
    tl.store(Partial + base_out, tl.min(tl.where(foreground, x, OW), 0))
    tl.store(Partial + base_out + 1, tl.min(tl.where(foreground, y, OH), 0))
    tl.store(Partial + base_out + 2, tl.max(tl.where(foreground, x, 0), 0))
    tl.store(Partial + base_out + 3, tl.max(tl.where(foreground, y, 0), 0))
    tl.store(Partial + base_out + 4, tl.sum(foreground.to(tl.int32), 0))


@triton.jit
def _stats(
    Partial,
    Stats,
    TILES: tl.constexpr,
    OW: tl.constexpr,
    OH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    obj = tl.program_id(0)
    tile = tl.arange(0, BLOCK)
    base = (obj * TILES + tile) * 5
    tl.store(Stats + obj * 5, tl.min(tl.load(Partial + base, tile < TILES, OW), 0))
    tl.store(
        Stats + obj * 5 + 1, tl.min(tl.load(Partial + base + 1, tile < TILES, OH), 0)
    )
    tl.store(
        Stats + obj * 5 + 2, tl.max(tl.load(Partial + base + 2, tile < TILES, 0), 0)
    )
    tl.store(
        Stats + obj * 5 + 3, tl.max(tl.load(Partial + base + 3, tile < TILES, 0), 0)
    )
    tl.store(
        Stats + obj * 5 + 4, tl.sum(tl.load(Partial + base + 4, tile < TILES, 0), 0)
    )


def resize_masks_triton(logits, size, binary=True, probability=True):
    n, _, ih, iw = logits.shape
    oh, ow = size
    dense_shape = (n, oh, ow)
    masks = torch.empty(
        dense_shape if binary else (0,), dtype=torch.bool, device=logits.device
    )
    probs = torch.empty(
        dense_shape if probability else (0,), dtype=torch.float32, device=logits.device
    )
    stats = torch.empty((n, 5), dtype=torch.float32, device=logits.device)
    if n:
        tiles = triton.cdiv(oh * ow, 1024)
        partial = torch.empty((n, tiles, 5), dtype=torch.int32, device=logits.device)
        with torch.cuda.device(logits.device):
            _render[(tiles, n)](
                logits.contiguous(),
                masks,
                probs,
                partial,
                ih,
                iw,
                oh,
                ow,
                tiles,
                binary,
                probability,
                1024,
                enable_fp_fusion=False,
            )
            _stats[(n,)](partial, stats, tiles, ow, oh, triton.next_power_of_2(tiles))
    return masks, probs, stats


@triton.jit
def _overlap(
    Masks,
    Scores,
    P: tl.constexpr,
    N: tl.constexpr,
    OBJECTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pixel = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    obj = tl.arange(0, OBJECTS)
    valid = (obj[:, None] < N) & (pixel[None, :] < P)
    masks = tl.load(Masks + obj[:, None] * P + pixel[None, :], valid, 0)
    scores = tl.load(Scores + obj, obj < N, 0)
    values = tl.where(masks, scores[:, None], 0.0)
    maximum = tl.max(values, axis=0)
    winner = tl.min(tl.where(values == maximum[None, :], obj[:, None], OBJECTS), axis=0)
    output = masks & (obj[:, None] == winner[None, :]) & (scores[:, None] > 0.0)
    tl.store(Masks + obj[:, None] * P + pixel[None, :], output, valid)


def resolve_overlaps(masks, scores):
    """In-place on an owned contiguous output batch, preserving first-index ties."""
    n, h, w = masks.shape
    with torch.cuda.device(masks.device):
        _overlap[(triton.cdiv(h * w, 128),)](
            masks, scores.contiguous(), h * w, n, triton.next_power_of_2(n), 128
        )
    return masks
