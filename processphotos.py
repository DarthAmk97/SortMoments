"""Backward-compatible wrapper for the packaged Sort Moments pipeline.

The installable package now lives under ``src/sortmoments`` following the
PyPA packaging tutorial style. This wrapper keeps existing GUI/build imports
(``import processphotos``) working from a source checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sortmoments.pipeline import *  # noqa: F401,F403
from sortmoments.pipeline import main as main


if __name__ == "__main__":
    raise SystemExit(main())
