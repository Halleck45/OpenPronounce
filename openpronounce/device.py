"""Torch device selection: CUDA when available, CPU otherwise, ``OPENPRONOUNCE_DEVICE`` to force."""

import logging
import os
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_device():
    """Return the torch device models run on.

    ``OPENPRONOUNCE_DEVICE`` (``cpu``, ``cuda``, ``cuda:1``, ``mps``...) wins; otherwise
    CUDA if available, else CPU. Resolved once per process.
    """
    name = os.environ.get("OPENPRONOUNCE_DEVICE")
    if not name:
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    logger.info("Using device %s", device)
    return device
