#!/usr/bin/env python3
"""
The same test on filings that came after a freeze (freeze.py).

Before running: raise LAST_Q in config.py to the newest quarter the SEC has published, then run ingest.py,
prices.py --retry and events.py again. This script takes the company-days and insider events after the frozen
last_public_day, computes the same tables and puts them next to the frozen numbers:

  forecasts/{label}/forward_test.csv   table, clock, side, group, horizon, frozen and new mean (bps), new n,
                                       z = difference / its standard error

A group needs at least 30 events and a full holding period after day 0, so the long horizons fill in last.

  python src/forward_test.py --label 2026-10
"""

import argparse
import json

import numpy as np
import pandas as pd

from config import ROOT

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    args = ap.parse_args()
    out = ROOT / "forecasts" / args.label
    meta = json.loads((out / "meta.json").read_text())
    frozen = pd.read_csv(out / "results.csv")
    cut = pd.Timestamp(meta["last_public_day"])

    import analysis as A        # loads data/events.csv and data/issuer_days.csv as they are now
    A.E = A.E[A.E.avail_date > cut]
    A.D = A.D[A.D.avail_date > cut]
    print(f"after {cut.date()}: {len(A.E):,} insider events, {len(A.D):,} company-days")
    new = A.results(save=False)

    # the new filings are one sample, compared with the frozen numbers of the whole sample
    key = ["table", "clock", "bench", "side", "group", "horizon"]
    frozen, new = frozen[frozen["sample"] == A.ALL], new[new["sample"] == A.ALL]
    m = frozen[key + ["mean_bps", "se_bps"]].merge(new[key + ["n", "mean_bps", "se_bps", "t"]], on=key,
                                                   suffixes=("_frozen", "_new"))
    m = m.dropna(subset=["mean_bps_new"])
    m["z"] = (m.mean_bps_new - m.mean_bps_frozen) / np.sqrt(m.se_bps_new ** 2 + m.se_bps_frozen ** 2)
    m.to_csv(out / "forward_test.csv", index=False)

    pd.set_option("display.width", 200)
    show = m[m.table.isin(["horizons", "timing"]) & (m.bench == "SPY")]
    print(show[["clock", "side", "group", "horizon", "n", "mean_bps_frozen", "mean_bps_new", "t", "z"]].round(1)
          .to_string(index=False))
    print(f"-> {out / 'forward_test.csv'}")
