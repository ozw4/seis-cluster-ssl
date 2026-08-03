"""Pytest configuration for local source-tree imports."""

from __future__ import annotations

import os
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent / 'src'

if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

existing_pythonpath = os.environ.get('PYTHONPATH')
pythonpath_parts = [str(SRC_ROOT)]
if existing_pythonpath:
	pythonpath_parts.append(existing_pythonpath)
os.environ['PYTHONPATH'] = os.pathsep.join(pythonpath_parts)
