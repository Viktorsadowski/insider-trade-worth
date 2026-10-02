"""
Does a price series belong to the company whose insiders traded? Used by prices.py and events.py.

A ticker can be reused by another company, a rename can leave the old ticker empty, and two companies can
share a ticker over time (Allergan Inc and the later Allergan plc). The check uses the prices the insiders
wrote on their own Form 4s, year by year:

  level   median of log(Form 4 price / close) in the year. 0 when it is the same stock
  spread  median distance from that level. Small when it is the same stock, also if the level is off

A level that is off but the same two years in a row is fine: a spin-off the price source did not adjust for
(Howmet), or insiders trading another share class (Berkshire A against the B price).
"""

import numpy as np
import pandas as pd

MAX_SPREAD = 0.03                       # within the year they stay within 3% of each other (median)
MAX_LEVEL = 0.05                        # and are at the same level, or at a fixed distance from it
SAME_LEVEL = 0.02                       # (fixed = within 2% of the year before or after)
LOOSE_SPREAD, LOOSE_LEVEL = 0.15, 0.10  # for a volatile year between good years


def year_fit(dates, prices, close: pd.Series) -> pd.DataFrame:
    """year, n, level, spread, valid for one company against one price series (close: date -> traded price)"""
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = np.log(np.asarray(prices, float) / close.reindex(dates).values)
    fin = np.isfinite(dev)
    if not fin.any():
        return pd.DataFrame(columns=["year", "n", "level", "spread", "valid"])
    d = pd.DataFrame({"year": pd.DatetimeIndex(dates).year[fin], "dev": dev[fin]})
    f = d.groupby("year").dev.agg(n="size", level="median")
    f["spread"] = (d.dev - d.year.map(f["level"])).abs().groupby(d.year).median()
    f = f.reset_index()

    def next_to(good, max_level):
        # the year before / after with trades, at most 3 years away, itself good, and at about the same level
        out = pd.Series(False, index=f.index)
        for k in (1, -1):
            out |= ((f["year"] - f["year"].shift(k)).abs() <= 3) & good.shift(k).eq(True) \
                   & ((f["level"] - f["level"].shift(k)).abs() < max_level)
        return out

    tight = f["spread"] < MAX_SPREAD
    valid = tight & ((f["level"].abs() < MAX_LEVEL) | next_to(tight, SAME_LEVEL))
    # a wild year (2008, March 2020) moves a stock several percent between the trade and the close. Such a
    # year still counts when a good year right next to it is at about the same level
    f["valid"] = valid | ((f["spread"] < LOOSE_SPREAD) & next_to(valid, LOOSE_LEVEL))
    return f
