#!/usr/bin/env python3
"""Pull every session in `sessions.SESSIONS` into the shared raw directory.

Two hard rules, both enforced here rather than left to the operator:

1. `metadata.get_cost` is queried for every symbol-day and schema BEFORE any
   bytes are downloaded, and a non-zero cost aborts the pull unless
   --allow-cost is passed explicitly. The programme entitlement covers
   XNAS.ITCH outright, so anything other than $0.0000 means the request is not
   the one that was intended.
2. The destination is $DATABENTO_RAW_DIR, never a path written into the repo.

The window is the full Nasdaq extended session, 04:00-20:00 exchange time,
matching the two extracts the earlier study used. Expressing it in exchange
time keeps it correct across the DST boundary between February and December.

Needs the `databento` SDK, which is an extraction-time dependency only: the
committed derived series reproduce every published result without it.

Usage:
    python scripts/fetch_sessions.py                 # price the whole plan
    python scripts/fetch_sessions.py --confirm       # pull it
    python scripts/fetch_sessions.py --confirm --symbol AAPL
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sessions import DATASET, SCHEMAS, raw_root, sessions  # noqa: E402

_EXCHANGE_TZ = ZoneInfo("America/New_York")
_SESSION_OPEN = time(4, 0)
_SESSION_CLOSE = time(20, 0)


def session_window(date_str: str) -> tuple[datetime, datetime]:
    """UTC-aware bounds of the extended session on `date_str`, in exchange time."""
    day = datetime.strptime(date_str, "%Y-%m-%d").date()
    return (datetime.combine(day, _SESSION_OPEN, tzinfo=_EXCHANGE_TZ),
            datetime.combine(day, _SESSION_CLOSE, tzinfo=_EXCHANGE_TZ))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirm", action="store_true",
                    help="actually download; without it this only prices the plan")
    ap.add_argument("--symbol", action="append", default=None,
                    help="restrict to these symbols (repeatable)")
    ap.add_argument("--schemas", nargs="+", default=list(SCHEMAS))
    ap.add_argument("--allow-cost", action="store_true",
                    help="permit a non-zero get_cost; off by default on purpose")
    ap.add_argument("--cost-out", type=Path, default=None,
                    help="write the priced plan to this CSV, for DATA.md")
    args = ap.parse_args()

    if not os.environ.get("DATABENTO_API_KEY"):
        print("DATABENTO_API_KEY is not set (it lives in ~/.zshrc; use `zsh -ic`)",
              file=sys.stderr)
        return 1

    import databento as db

    client = db.Historical()
    root = raw_root()
    plan, total = [], 0.0
    wanted = [s for s in sessions()
              if args.symbol is None or s.symbol in set(args.symbol)]

    for sess in wanted:
        start, end = session_window(sess.date)
        for schema in args.schemas:
            out = sess.raw_path(schema, root)
            if out.exists():
                print(f"  {sess.key:<16} {schema:<6} present "
                      f"({out.stat().st_size / 1e6:.1f} MB), not re-priced")
                plan.append((sess, schema, out, None))
                continue
            cost = client.metadata.get_cost(
                dataset=DATASET, symbols=[sess.symbol], schema=schema,
                start=start, end=end, stype_in="raw_symbol",
            )
            total += cost
            plan.append((sess, schema, out, cost))
            print(f"  {sess.key:<16} {schema:<6} ${cost:>9.4f}  -> "
                  f"{out.relative_to(root)}")

    print(f"\n  {'TOTAL':<16} {'':<6} ${total:>9.4f}  over "
          f"{sum(1 for *_, c in plan if c is not None)} new requests")

    if args.cost_out is not None:
        import pandas as pd
        pd.DataFrame([{"symbol": s.symbol, "date": s.date, "schema": sch,
                       "get_cost_usd": c,
                       "status": "already present" if c is None else "priced"}
                      for s, sch, _, c in plan]).to_csv(args.cost_out, index=False)
        print(f"  plan written to {args.cost_out}")

    if total > 0 and not args.allow_cost:
        print("\nget_cost is not $0. Aborting: the entitlement covers XNAS.ITCH "
              "outright, so a non-zero price means the request is wrong.",
              file=sys.stderr)
        return 2
    if not args.confirm:
        print("\ndry run; nothing downloaded. re-run with --confirm to fetch.")
        return 0

    print()
    for sess, schema, out, cost in plan:
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        start, end = session_window(sess.date)
        client.timeseries.get_range(
            dataset=DATASET, symbols=[sess.symbol], schema=schema,
            start=start, end=end, stype_in="raw_symbol", path=str(out),
        )
        print(f"  {sess.key:<16} {schema:<6} {out.stat().st_size / 1e6:>8.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
