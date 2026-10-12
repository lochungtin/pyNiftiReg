import os

import pytest
import torch

HAS_CUDA = torch.cuda.is_available()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_gpu = pytest.mark.skip(reason="no CUDA device")
    skip_slow = pytest.mark.skip(reason="slow; set PYNIFTIREG_SLOW=1 to run")
    for item in items:
        if "slow" in item.keywords and os.environ.get("PYNIFTIREG_SLOW") != "1":
            item.add_marker(skip_slow)
        if "gpu" in item.keywords and not HAS_CUDA:
            item.add_marker(skip_gpu)


@pytest.fixture(params=[2, 3], ids=["2d", "3d"])
def ndim(request: pytest.FixtureRequest) -> int:
    """Spatial dimensionality; math tests take this fixture so they run in both 2D and 3D."""
    return int(request.param)
