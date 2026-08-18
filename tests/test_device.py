"""Hardware profiling: MPS on macOS host, CPU smoke test inside Docker."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.device import get_device, running_in_docker

logger = logging.getLogger(__name__)

_SMOKE_DIM: int = 64


def _log_torch_environment() -> None:
    """Emit interpreter and PyTorch build metadata for hardware verification."""
    logger.info("python=%s", sys.version.replace("\n", " "))
    logger.info("torch=%s", torch.__version__)
    logger.info("docker=%s", running_in_docker())
    logger.info("mps.is_built=%s", torch.backends.mps.is_built())
    logger.info("mps.is_available=%s", torch.backends.mps.is_available())
    logger.info("selected_device=%s", get_device())


def test_torch_imports() -> None:
    """Fail if the container/host cannot import a working PyTorch build."""
    _log_torch_environment()
    assert torch.__version__, "torch.__version__ is empty"


def test_smoke_matmul_on_selected_device() -> None:
    """Allocate on the selected device, run a small matmul, verify finite CPU result."""
    device: torch.device = get_device()
    try:
        left: torch.Tensor = torch.randn(_SMOKE_DIM, _SMOKE_DIM, device=device, dtype=torch.float32)
        right: torch.Tensor = torch.randn(_SMOKE_DIM, _SMOKE_DIM, device=device, dtype=torch.float32)
        product: torch.Tensor = left @ right
        result_cpu: torch.Tensor = product.cpu()
    except RuntimeError as exc:
        logger.exception("smoke matmul failed on device=%s", device)
        pytest.fail(f"Compute failed on {device}: {exc}")

    logger.info(
        "smoke ok device=%s shape=%s dtype=%s",
        device,
        tuple(result_cpu.shape),
        result_cpu.dtype,
    )
    assert result_cpu.shape == (_SMOKE_DIM, _SMOKE_DIM)
    assert torch.isfinite(result_cpu).all(), "matmul produced non-finite values"
    assert result_cpu.device.type == "cpu"


@pytest.mark.skipif(
    running_in_docker() or not torch.backends.mps.is_available(),
    reason="MPS is a macOS Metal API; skipped inside Docker and on machines without MPS",
)
def test_mps_is_available_on_host() -> None:
    """Host-only: Metal must be usable when this test is not skipped."""
    assert torch.backends.mps.is_built()
    assert torch.backends.mps.is_available()
    assert get_device().type == "mps"
