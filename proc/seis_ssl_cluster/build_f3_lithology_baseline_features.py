"""Build F3 lithology baseline feature token datasets."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from proc.seis_ssl_cluster.build_f3_lithology_baseline_token_dataset import (  # noqa: E402
	main,
)

if __name__ == '__main__':
	main()
