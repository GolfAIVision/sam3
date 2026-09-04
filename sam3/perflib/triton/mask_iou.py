"""Exact integer mask intersection/union with bounded intermediate storage."""

import torch
import triton
import triton.language as tl


@triton.jit
def _count(
    A,
    B,
    Partial,
    P: tl.constexpr,
    M: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair, chunk = tl.program_id(0), tl.program_id(1)
    pixel = chunk * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + (pair // M) * P + pixel, pixel < P, 0).to(tl.int32)
    b = tl.load(B + (pair % M) * P + pixel, pixel < P, 0).to(tl.int32)
    offset = (pair * CHUNKS + chunk) * 2
    tl.store(Partial + offset, tl.sum(a & b, 0))
    tl.store(Partial + offset + 1, tl.sum(a | b, 0))


@triton.jit
def _pack(A, Packed, P: tl.constexpr, WORDS: tl.constexpr, BLOCK: tl.constexpr):
    obj, block = tl.program_id(0), tl.program_id(1)
    words = block * BLOCK + tl.arange(0, BLOCK)
    bit = tl.arange(0, 32)
    pixel = words[:, None] * 32 + bit[None, :]
    value = tl.load(A + obj * P + pixel, pixel < P, 0).to(tl.uint32)
    packed = tl.sum(value << bit[None, :], 1)
    tl.store(Packed + obj * WORDS + words, packed, words < WORDS)


@triton.jit
def _popcount(x):
    return tl.inline_asm_elementwise(
        "popc.b32 $0, $1;",
        constraints="=r,r",
        args=[x],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _count_packed(
    A,
    B,
    Partial,
    WORDS: tl.constexpr,
    M: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair, chunk = tl.program_id(0), tl.program_id(1)
    word = chunk * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + (pair // M) * WORDS + word, word < WORDS, 0)
    b = tl.load(B + (pair % M) * WORDS + word, word < WORDS, 0)
    offset = (pair * CHUNKS + chunk) * 2
    tl.store(Partial + offset, tl.sum(_popcount(a & b), 0))
    tl.store(Partial + offset + 1, tl.sum(_popcount(a | b), 0))


@triton.jit
def _finish(Partial, Out, CHUNKS: tl.constexpr, BLOCK: tl.constexpr):
    pair = tl.program_id(0)
    chunk = tl.arange(0, BLOCK)
    offset = (pair * CHUNKS + chunk) * 2
    intersection = tl.sum(tl.load(Partial + offset, chunk < CHUNKS, 0), 0).to(
        tl.float32
    )
    union = tl.sum(tl.load(Partial + offset + 1, chunk < CHUNKS, 0), 0).to(tl.float32)
    tl.store(Out + pair, tl.div_rn(intersection, tl.maximum(union, 1)))


def mask_iou_triton(a, b, packed=None):
    n, h, w = a.shape
    m = b.shape[0]
    out = torch.empty((n, m), dtype=torch.float32, device=a.device)
    if n == 0 or m == 0:
        return out
    a, b = a.contiguous(), b.contiguous()
    pixels = h * w
    if pixels == 0:
        return out.zero_()
    if packed is None:
        packed = n * m >= 16 and pixels >= 4096
    with torch.cuda.device(a.device):
        if packed:
            words = triton.cdiv(pixels, 32)
            aa = torch.empty((n, words), dtype=torch.uint32, device=a.device)
            bb = (
                aa
                if a is b
                else torch.empty((m, words), dtype=torch.uint32, device=b.device)
            )
            _pack[(n, triton.cdiv(words, 32))](a, aa, pixels, words, 32)
            if bb is not aa:
                _pack[(m, triton.cdiv(words, 32))](b, bb, pixels, words, 32)
            chunks = triton.cdiv(words, 1024)
            partial = torch.empty(
                (n * m, chunks, 2), dtype=torch.int32, device=a.device
            )
            _count_packed[(n * m, chunks)](aa, bb, partial, words, m, chunks, 1024)
        else:
            chunks = triton.cdiv(pixels, 4096)
            partial = torch.empty(
                (n * m, chunks, 2), dtype=torch.int32, device=a.device
            )
            _count[(n * m, chunks)](a, b, partial, pixels, m, chunks, 4096)
        _finish[(n * m,)](partial, out, chunks, triton.next_power_of_2(chunks))
    return out
