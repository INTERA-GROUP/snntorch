import pytest
import torch


def pytest_collection_modifyitems(config, items):
    """Skip tests marked ``cuda`` when no GPU is available."""
    if torch.cuda.is_available():
        return
    skip_cuda = pytest.mark.skip(reason="CUDA is not available")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip_cuda)
