#!/usr/bin/env python3
"""Run the existing builders over every session in the panel.

The path from a raw MBO extract to the two committed series has four steps and
this runs all of them, in order, for all fifteen symbol-days:

    <SYMBOL>/<DATE>.mbo.dbn.zst
      -> databento_to_lobster.py   LOBSTER message stream
      -> lob_engine --depth 1      reconstructed L1 book
      -> build_1s_bars.py          data/<KEY>_1s.csv
      -> build_metaorders.py       data/<KEY>_metaorders.csv

The two intermediates are large (about 100 MB a session) and are deleted after
each session unless --keep-intermediates is passed, so the disk cost is one
session at a time rather than fifteen.

MSFT 2024-06-03 and INTC 2024-08-02 are rebuilt like the rest, and MSFT is
diffed against the committed series with `build_1s_bars.py --compare`. If that
prints anything but IDENTICAL the panel is not being built on the convention
the earlier results were built on, and the run stops.

Paths to the sibling engine come from $LOB_ENGINE_REPO, never from a committed
absolute path.

Usage:
    python scripts/build_all_sessions.py
    python scripts/build_all_sessions.py --symbol AAPL --keep-intermediates
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sessions import raw_root, sessions  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def engine_repo() -> Path:
    root = os.environ.get("LOB_ENGINE_REPO")
    if not root:
        raise SystemExit(
            "LOB_ENGINE_REPO is not set. It must point at the sibling "
            "lob-engine-cpp checkout, which supplies databento_to_lobster.py "
            "and the built lob_engine binary.")
    path = Path(root).expanduser()
    if not (path / "build" / "lob_engine").exists():
        raise SystemExit(f"{path}/build/lob_engine not found; build it first")
    return path


def run(cmd: list[str], quiet: bool = True) -> None:
    result = subprocess.run(cmd, capture_output=quiet, text=True)
    if result.returncode != 0:
        if quiet:
            sys.stderr.write(result.stdout or "")
            sys.stderr.write(result.stderr or "")
        raise SystemExit(f"failed: {' '.join(cmd)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", action="append", default=None)
    ap.add_argument("--out-dir", type=Path, default=REPO / "data")
    ap.add_argument("--work-dir", type=Path, default=None,
                    help="where the message and book intermediates go "
                         "(default: a temporary directory)")
    ap.add_argument("--keep-intermediates", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="rebuild sessions whose outputs already exist")
    args = ap.parse_args()

    engine = engine_repo()
    root = raw_root()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    work_root = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(
        prefix="tim_build_"))
    work_root.mkdir(parents=True, exist_ok=True)

    built = 0
    for sess in sessions():
        if args.symbol and sess.symbol not in set(args.symbol):
            continue
        bars = args.out_dir / f"{sess.key}_1s.csv"
        metas = args.out_dir / f"{sess.key}_metaorders.csv"
        if bars.exists() and metas.exists() and not args.force:
            print(f"  {sess.key:<16} already built")
            continue

        mbo = sess.raw_path("mbo", root)
        if not mbo.exists():
            print(f"  {sess.key:<16} MISSING {mbo.name}", file=sys.stderr)
            continue

        work = work_root / sess.key
        work.mkdir(parents=True, exist_ok=True)
        message = work / "message.csv"
        book = work / "l1.csv"

        print(f"  {sess.key:<16} converting ...", flush=True)
        run([sys.executable, "-W", "ignore",
             str(engine / "scripts" / "databento_to_lobster.py"),
             str(mbo), "--out", str(message)])
        run([str(engine / "build" / "lob_engine"), str(message),
             "--backend", "map", "--depth", "1", "--book-out", str(book)])

        cmd = [sys.executable, str(HERE / "build_1s_bars.py"),
               "--messages", str(message), "--book", str(book), "--out", str(bars)]
        # the one session with a committed predecessor is the convention check
        if sess.key == "MSFT_2024-06-03" and (args.out_dir / f"{sess.key}_1s.csv").exists():
            reference = work / "reference_1s.csv"
            shutil.copy(args.out_dir / f"{sess.key}_1s.csv", reference)
            cmd += ["--compare", str(reference)]
        run(cmd, quiet=False)

        run([sys.executable, str(HERE / "build_metaorders.py"),
             "--messages", str(message), "--book", str(book), "--out", str(metas)],
            quiet=False)

        if not args.keep_intermediates:
            shutil.rmtree(work, ignore_errors=True)
        built += 1

    if not args.keep_intermediates and not args.work_dir:
        shutil.rmtree(work_root, ignore_errors=True)
    print(f"\nbuilt {built} session(s) -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
