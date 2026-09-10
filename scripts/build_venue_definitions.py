#!/usr/bin/env python3
"""Two venue-volume definitions against consolidated volume, paired.

The metaorder builder's V_D is RTH-CONTINUOUS Nasdaq volume:
`[09:30:00, 16:00:00)` local exchange time, which is exactly the filter
`build_cross_section_metaorders.py` applies before a trade is ever binned.
CRSP's consolidated daily volume is FULL-DAY: every print, every venue,
including both auction crosses. Dividing a full-day consolidated figure by an
RTH-continuous venue figure is not like for like, and this script reports the
comparison two ways so the mismatch is visible rather than implicit:

  * consolidated / RTH-continuous venue volume  (what the README calls 7.60)
  * consolidated / full-day venue volume        (like for like)

Both use the SAME 108 stratified names, the SAME raw trades files, and the
SAME external CRSP cache the rest of the normaliser comparison uses. Full-day
venue volume is computed here directly from the raw trades files, since no
committed table carries it; RTH-continuous is recomputed from the same raw
files rather than re-read from `venue_vs_consolidated_volume.csv`, so both
numbers in every ratio come from one pass over one row set.

Usage:
    python scripts/build_venue_definitions.py --crsp-cache <dir>
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import cross_section as cs  # noqa: E402
import crsp  # noqa: E402
from fetch_cross_section_trades import raw_path  # noqa: E402
from sessions import raw_root  # noqa: E402

MIN_TRADES_PER_SESSION = 500     # matches build_cross_section_metaorders.py


def load_raw(symbol: str, root: Path) -> pd.DataFrame:
    """Every print in the pull window, no time-of-day filter at all."""
    import databento as db

    path = raw_path(symbol, root)
    if not path.exists():
        raise FileNotFoundError(path)
    arr = db.DBNStore.from_file(str(path)).to_ndarray()
    local = pd.to_datetime(arr["ts_event"].astype(np.int64), unit="ns", utc=True
                           ).tz_convert("America/New_York")
    sec = (local.hour * 3600 + local.minute * 60 + local.second).to_numpy(np.int64)
    return pd.DataFrame({"date": local.strftime("%Y-%m-%d").to_numpy(),
                         "sec": sec,
                         "price": arr["price"].astype(float) * 1e-9,
                         "size": arr["size"].astype(float)})


def daily_volumes(symbol: str, root: Path) -> pd.DataFrame:
    """Per-session RTH-continuous and full-day volume, on the same session
    filter `build_cross_section_metaorders.py` applies (minimum RTH trade
    count and a finite five-minute realised volatility). Volume-only, so no
    proxy metaorder needs to be reconstructed here."""
    trades = load_raw(symbol, root)
    rows = []
    for date, day in trades.groupby("date", sort=True):
        rth_mask = (day.sec.to_numpy() >= cs.RTH_OPEN_SEC) & \
            (day.sec.to_numpy() < cs.RTH_CLOSE_SEC)
        if int(rth_mask.sum()) < MIN_TRADES_PER_SESSION:
            continue
        rth_day = day[rth_mask]
        sigma = cs.realised_vol_5min(rth_day.sec.to_numpy(),
                                     rth_day.price.to_numpy())
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        rth_volume, full_day_volume = cs.venue_volume_definitions(
            day.sec.to_numpy(), day["size"].to_numpy())
        if rth_volume <= 0 or full_day_volume <= 0:
            continue
        rows.append({"symbol": symbol, "date": date,
                     "volume_rth_continuous": rth_volume,
                     "volume_full_day": full_day_volume})
    return pd.DataFrame(rows)


def quartiles(values: pd.Series) -> tuple[float, float, float]:
    return (float(values.quantile(0.25)), float(values.median()),
           float(values.quantile(0.75)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=Path,
                    default=Path("data/cross_section/sample.csv"))
    ap.add_argument("--crsp-cache", type=Path,
                    default=os.environ.get("IMPACT_CRSP_CACHE_DIR"),
                    required=os.environ.get("IMPACT_CRSP_CACHE_DIR") is None)
    ap.add_argument("--out", type=Path,
                    default=Path("reports/cross_section/venue_definitions.csv"))
    args = ap.parse_args()

    sample = pd.read_csv(args.sample)
    symbols = sample.loc[sample["role"] == "stratified", "symbol"].tolist()
    root = raw_root()

    daily = pd.concat([daily_volumes(symbol, root) for symbol in symbols],
                      ignore_index=True)

    # Pass every cached symbol (108 stratified plus the MSFT/INTC comparison
    # names), not just the 108 we compute volumes for: the identifier check
    # in `crsp.load_consolidated_cache` requires its `symbols` argument to
    # match the cache's mapping file one to one, and this external cache
    # (shared with other WRDS pulls) currently resolves all 110 pull names,
    # not just the stratified subset. `required_pairs` still restricts what
    # must actually be found to the 108 stratified symbol-dates below.
    consolidated = crsp.load_consolidated_cache(
        sample["symbol"].tolist(), daily["date"].min(), daily["date"].max(),
        required_pairs=daily[["symbol", "date"]], cache_dir=args.crsp_cache)
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    merged = daily.merge(consolidated[["symbol", "date", "volume_crsp"]],
                         on=["symbol", "date"], how="left", validate="many_to_one")
    if merged["volume_crsp"].isna().any():
        missing = merged.loc[merged["volume_crsp"].isna(), ["symbol", "date"]]
        raise ValueError(f"{len(missing)} symbol-dates lack consolidated "
                         f"volume: {missing.head(5).to_dict('records')}")

    merged["ratio_rth_continuous"] = (merged["volume_crsp"] /
                                      merged["volume_rth_continuous"])
    merged["ratio_full_day"] = merged["volume_crsp"] / merged["volume_full_day"]
    merged["full_day_over_rth_continuous"] = (merged["volume_full_day"] /
                                              merged["volume_rth_continuous"])

    # CRSP's own exchcd (from the same verified mapping the consolidated
    # cache resolves identifiers against) turns out to explain almost all of
    # the per-name spread below: it is not loaded for the ratio computation,
    # only attached afterward as a diagnostic column.
    mapping = pd.read_csv(Path(args.crsp_cache) / "impact_resolved_permnos_verified.csv")
    exchcd_by_symbol = mapping.set_index("ticker")["exchcd"]

    per_symbol = []
    for symbol, group in merged.groupby("symbol", sort=True):
        q25_rth, med_rth, q75_rth = quartiles(group["ratio_rth_continuous"])
        q25_full, med_full, q75_full = quartiles(group["ratio_full_day"])
        q25_vr, med_vr, q75_vr = quartiles(group["full_day_over_rth_continuous"])
        per_symbol.append({
            "symbol": symbol, "n_sessions": len(group),
            "crsp_exchcd": int(exchcd_by_symbol.get(symbol, -1)),
            "median_ratio_rth_continuous": med_rth,
            "q25_ratio_rth_continuous": q25_rth,
            "q75_ratio_rth_continuous": q75_rth,
            "median_ratio_full_day": med_full,
            "q25_ratio_full_day": q25_full,
            "q75_ratio_full_day": q75_full,
            "median_full_day_over_rth_continuous": med_vr,
            "q25_full_day_over_rth_continuous": q25_vr,
            "q75_full_day_over_rth_continuous": q75_vr,
        })
    out = pd.DataFrame(per_symbol)

    q25_pooled_rth, med_pooled_rth, q75_pooled_rth = quartiles(
        out["median_ratio_rth_continuous"])
    q25_pooled_full, med_pooled_full, q75_pooled_full = quartiles(
        out["median_ratio_full_day"])
    q25_pooled_vr, med_pooled_vr, q75_pooled_vr = quartiles(
        out["median_full_day_over_rth_continuous"])
    pooled_row = {
        "symbol": "POOLED_median_of_name_medians",
        "n_sessions": int(out["n_sessions"].sum()),
        "crsp_exchcd": -1,
        "median_ratio_rth_continuous": med_pooled_rth,
        "q25_ratio_rth_continuous": q25_pooled_rth,
        "q75_ratio_rth_continuous": q75_pooled_rth,
        "median_ratio_full_day": med_pooled_full,
        "q25_ratio_full_day": q25_pooled_full,
        "q75_ratio_full_day": q75_pooled_full,
        "median_full_day_over_rth_continuous": med_pooled_vr,
        "q25_full_day_over_rth_continuous": q25_pooled_vr,
        "q75_full_day_over_rth_continuous": q75_pooled_vr,
    }
    out = pd.concat([out, pd.DataFrame([pooled_row])], ignore_index=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"{len(symbols)} names, {len(merged)} symbol-days")
    print(f"pooled ratio, consolidated / RTH-continuous venue volume: "
          f"median {med_pooled_rth:.3f}  [{q25_pooled_rth:.3f}, "
          f"{q75_pooled_rth:.3f}]")
    print(f"pooled ratio, consolidated / full-day venue volume:       "
          f"median {med_pooled_full:.3f}  [{q25_pooled_full:.3f}, "
          f"{q75_pooled_full:.3f}]")
    print(f"pooled ratio, full-day / RTH-continuous venue volume:     "
          f"median {med_pooled_vr:.3f}  [{q25_pooled_vr:.3f}, "
          f"{q75_pooled_vr:.3f}]")
    by_exchcd = out[out["symbol"] != "POOLED_median_of_name_medians"].groupby(
        "crsp_exchcd")["median_full_day_over_rth_continuous"].agg(["size", "median"])
    print("full-day / RTH-continuous by CRSP exchcd (1=NYSE, 3=Nasdaq):")
    print(by_exchcd.to_string())
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
