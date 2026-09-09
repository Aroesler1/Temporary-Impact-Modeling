#!/usr/bin/env python3
"""Withdrawn empirical schedule runner. Historical tables are retained.

Use scripts/audit_kernel_response.py --check for the supported diagnostic.
No fit, data access, output directory creation, or overwrite occurs here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from execution import SCHEDULE_WITHDRAWAL


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Accept the old switches so existing commands receive the scientific
    # withdrawal message instead of an unrelated argument-parser failure.
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--starts", type=int, default=40)
    parser.add_argument("--kappa", type=float, default=0.005)
    parser.parse_args()
    print(SCHEDULE_WITHDRAWAL, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
