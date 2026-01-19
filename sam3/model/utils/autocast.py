# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

from contextlib import nullcontext
from typing import ContextManager

import torch


def bf16_autocast_context(
    device: torch.device, enabled: bool = True
) -> ContextManager:
    """Return a scoped bf16 autocast context for CUDA devices.

    This helper avoids leaking autocast globally and safely no-ops on non-CUDA
    devices or when disabled.
    """
    if not enabled or device is None:
        return nullcontext()
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()
