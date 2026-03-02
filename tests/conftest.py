"""Make granite_asr importable for tests."""

import sys
from pathlib import Path

# Add project root (EEVA-edge/) to sys.path so `from granite_asr import ...` works
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
