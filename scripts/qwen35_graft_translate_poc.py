#!/usr/bin/env python3
"""CLI entrypoint for the Qwen3.5 graft-translation PoC."""
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.qwen35_translation_poc import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
