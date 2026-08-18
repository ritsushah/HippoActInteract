"""Compute-device selection for host MPS vs Docker CPU."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_ALLOWED_DEVICES: frozenset[str] = frozenset({"cpu", "mps"})


def running_in_docker() -> bool:
    """Return True when the process is inside a Docker container."""
    return Path("/.dockerenv").exists()


def get_device() -> torch.device:
    """Return MPS on a capable macOS host, otherwise CPU.

    ``HIPPO_DEVICE`` may force ``cpu`` or ``mps``. Forcing ``mps`` inside
    Docker raises ``RuntimeError`` because Metal is not available in Linux VMs.
    """
    override: str = os.environ.get("HIPPO_DEVICE", "").strip().lower()
    if override:
        if override not in _ALLOWED_DEVICES:
            raise ValueError(
                f"HIPPO_DEVICE={override!r} is invalid; expected one of {sorted(_ALLOWED_DEVICES)}"
            )
        if override == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError(
                "HIPPO_DEVICE=mps but torch.backends.mps.is_available() is False. "
                "Metal cannot run inside Docker; omit HIPPO_DEVICE on macOS host, "
                "or use HIPPO_DEVICE=cpu in the container."
            )
        device = torch.device(override)
        logger.info("device override HIPPO_DEVICE=%s docker=%s", override, running_in_docker())
        return device

    if torch.backends.mps.is_available():
        logger.info("selected device=mps")
        return torch.device("mps")

    logger.info("selected device=cpu docker=%s", running_in_docker())
    return torch.device("cpu")
