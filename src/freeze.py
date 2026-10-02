#!/usr/bin/env python3
"""
Freeze the results, before the next quarters of filings exist. forward_test.py later runs the same test on the
filings that came after, and that only counts if the numbers it is compared with could not be changed.

Writes forecasts/{label}/ (in the repo, not in data/, so git tracks it):

  results.csv   data/results.csv as it is now: every table, both clocks, all horizons
  meta.json     when, which git commit, the last public day in the data, sha256 of results.csv

Commit + push right after, the commit time is the proof it came first.
Won't overwrite an existing freeze (--force if you really mean it).

  python src/freeze.py                 # label = this month, e.g. forecasts/2026-10/
  python src/freeze.py --label v2
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone

import pandas as pd

from config import DATA_DIR, LAST_Q, ROOT


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=datetime.now().strftime("%Y-%m"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = ROOT / "forecasts" / args.label
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already exists, that's the frozen one. --force to overwrite (and say why)")
    out.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(DATA_DIR / "results.csv", out / "results.csv")
    d = pd.read_csv(DATA_DIR / "issuer_days.csv", usecols=["avail_date"], parse_dates=["avail_date"])
    meta = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit_at_freeze": git_commit(),
        "sec_data_through": f"{LAST_Q[0]} Q{LAST_Q[1]}",
        # the forward test only uses company-days after this one
        "last_public_day": str(d.avail_date.max().date()),
        "claim": "insider clock: buys beat SPY in the first week. public clock: nothing after day 1, buys or sales",
        "sha256": {"results.csv": hashlib.sha256((out / "results.csv").read_bytes()).hexdigest()},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    r = pd.read_csv(out / "results.csv")
    h = r[(r.table == "horizons") & (r["sample"] == "2006-2026") & (r.bench == "SPY")] \
        .pivot_table(index=["side", "clock"], columns="horizon", values="mean_bps")
    print(f"frozen -> {out}, last public day {meta['last_public_day']}")
    print(h.round(0).to_string())
    print("\nnow commit + push, the commit time is the timestamp:")
    print(f'  git add forecasts && git commit -m "freeze {args.label}"')
