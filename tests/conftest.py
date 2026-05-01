from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.make_toy_surgwmbench import make_toy_surgwmbench


@pytest.fixture()
def toy_root(tmp_path: Path) -> Path:
    return make_toy_surgwmbench(tmp_path / "SurgWMBench", num_clips=2)
