"""Per-call kernel selection; independent models do not change global dispatch."""

from contextlib import contextmanager
from contextvars import ContextVar

import torch

_backend = ContextVar("sam3_kernel_backend", default="torch")


@contextmanager
def kernel_scope(backend):
    token = _backend.set(backend)
    try:
        yield
    finally:
        _backend.reset(token)


def use_triton(tensor, backend=None):
    if backend is None and torch.compiler.is_compiling():
        # ContextVar access is Python-only; leave compiled reference graphs to Inductor.
        return False
    backend = _backend.get() if backend is None else backend
    if backend == "torch" or not tensor.is_cuda:
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        if backend == "triton":
            raise RuntimeError(
                "The requested Triton backend is not installed"
            ) from None
        return False
    # Auto is conservative until each GPU architecture has passed validation.
    if backend == "auto":
        return torch.cuda.get_device_capability(tensor.device) == (8, 9)
    return backend == "triton"
