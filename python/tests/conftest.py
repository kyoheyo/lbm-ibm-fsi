"""pytest configuration for the lbm Python pre/post-processing tests."""
import sys
from pathlib import Path

# Make the lbm_pre / lbm_post packages importable without installing
sys.path.insert(0, str(Path(__file__).parent.parent))
