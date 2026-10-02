#!/usr/bin/env python3
"""
Daily prices for every S&P 500 ticker since 2006 + SPY -> data/prices/{TICKER}.csv, data/prices.parquet

  date, close, adj_close, volume (volume only in the csv files)
  close      the price as it traded that day (not adjusted), to compare with the price on the Form 4
  adj_close  adjusted for splits and dividends, this is what returns are computed from

Yahoo Finance first (free, no key, but only companies that are still listed). What Yahoo does not have goes to
Tiingo, which keeps part of the delisted companies. The free Tiingo tier allows 50 requests an hour, so the
first full run takes a few hours: it waits when the limit is hit, and it can be stopped and started again at
any time.

A ticker can belong to another company today than when it was in the index (CCU, EP, ADT). So every series is
checked twice: it must cover the days the ticker was in the S&P 500, and the prices insiders wrote on their
Form 4s must follow its close (the rule is in pricefit.py). A series that fits no company in any year is not
used.

  data/prices/_status.csv   ticker, source, first, last, rows, coverage (share of its S&P 500 days with a
                            price), fit_n (Form 4 trades compared) and fit_ok (of those, in years that fit)

The csv files are kept as they came. prices.parquet is the tidy version (tidy() below): both sources repeat the
last price for weeks or months after a stock stopped trading, and a ticker that was reused comes as one series
with a hole in it.

  python src/prices.py            # skips tickers that are already done
  python src/prices.py --retry    # try yahoo again for the ones that had no data
  python src/prices.py --yahoo-only
  python src/prices.py --try-unlisted   # also ask tiingo for tickers that are not in its own ticker list
  python src/prices.py --combine        # only rebuild prices.parquet from the csv files (safe while another run is going)
"""

import argparse
import io
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import requests

from config import BENCHMARK, DATA_DIR, PRICE_DIR, PRICE_START, RAW_DIR, TIINGO_API_KEY
from pricefit import year_fit

STATUS = PRICE_DIR / "_status.csv"
COLS = ["date", "close", "adj_close", "volume"]
TIINGO_LIST = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
# windows does not allow files with these names
RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}


def fname(ticker: str) -> str:
    t = ticker.replace(".", "-")
    return (t + "_" if t in RESERVED else t) + ".csv"


# ── the two sources ────────────────────────────────────────────────────────────

def yahoo(ticker: str):
    import logging
    import yfinance as yf
    # yfinance prints a "possibly delisted" line for every ticker it does not have, a few hundred of them here
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    h = None
    for attempt in range(3):
        try:
            h = yf.Ticker(ticker.replace(".", "-")).history(start=PRICE_START, auto_adjust=False, actions=True)
            break
        except Exception as e:      # a delisted ticker normally just comes back empty, this is the network
            limited = "Too Many Requests" in str(e) or "429" in str(e) or "Rate" in str(e)
            print(f"    yahoo {ticker}: {type(e).__name__}, waiting {60 if limited else 5}s")
            time.sleep(60 if limited else 5)
    if h is None or h.empty or "Close" not in h:
        return None
    h = h[h["Close"].notna()]
    d = pd.DataFrame({"date": pd.to_datetime(h.index).tz_localize(None).normalize()})
    # yahoo's Close is already adjusted for splits. Multiply the later splits back in to get the traded price
    later = 1.0
    if "Stock Splits" in h:
        sp_ = h["Stock Splits"].where(h["Stock Splits"] > 0, 1.0)
        later = sp_[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    d["close"] = (h["Close"] * later).values
    d["adj_close"] = h["Adj Close"].values if "Adj Close" in h else h["Close"].values
    d["volume"] = h["Volume"].values
    return d[COLS]


class TiingoLimit(Exception):
    pass


def tiingo(ticker: str):
    url = f"https://api.tiingo.com/tiingo/daily/{ticker.replace('.', '-')}/prices"
    r = requests.get(url, params={"startDate": PRICE_START, "token": TIINGO_API_KEY, "format": "json"},
                     headers={"Content-Type": "application/json"}, timeout=60)
    if r.status_code == 429:
        raise TiingoLimit(r.text[:200])
    if r.status_code == 404:
        return None
    r.raise_for_status()
    js = r.json()
    if not isinstance(js, list) or not js:
        return None
    h = pd.DataFrame(js)
    d = pd.DataFrame({"date": pd.to_datetime(h["date"]).dt.tz_localize(None).dt.normalize(),
                      "close": h["close"], "adj_close": h["adjClose"], "volume": h["volume"]})
    return d[COLS]


def tiingo_wait(ticker: str):
    """tiingo with the hourly limit handled: wait and try again, up to a day"""
    for _ in range(30):
        try:
            return tiingo(ticker)
        except TiingoLimit:
            # the allocation is per clock hour, so wait until a bit past the next full hour
            wait = 3600 - (time.time() % 3600) + 90
            print(f"    tiingo hourly limit, waiting {wait / 60:.0f} min (stop with ctrl-c, a rerun continues here)")
            time.sleep(wait)
        except requests.RequestException as e:
            print(f"    tiingo error for {ticker}: {e}")
            return None
    return None


def tiingo_listing():
    """every ticker tiingo has, with first and last day. Saves a request (of 50 an hour) for each ticker that
    is not there at all. Returns None when the list cannot be read, then every ticker is simply tried"""
    path = RAW_DIR / "tiingo_tickers.csv"
    try:
        if not path.exists():
            r = requests.get(TIINGO_LIST, timeout=120)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                path.write_bytes(zf.read(zf.namelist()[0]))
        lst = pd.read_csv(path, dtype=str)
        lst.columns = [c.strip() for c in lst.columns]
        if not {"ticker", "startDate", "endDate"} <= set(lst.columns):
            return None
        lst["ticker"] = lst["ticker"].str.upper()
        lst["startDate"] = pd.to_datetime(lst["startDate"], errors="coerce")
        lst["endDate"] = pd.to_datetime(lst["endDate"], errors="coerce")
        lst = lst.dropna(subset=["startDate", "endDate"])
        # a list without these is not the list we think it is
        if len(lst) < 5000 or not {"AAPL", "MSFT", "SPY"} <= set(lst.ticker):
            return None
        return lst
    except Exception as e:
        print(f"  tiingo ticker list not available ({type(e).__name__}), every ticker is tried")
        return None


# ── the two checks ─────────────────────────────────────────────────────────────

def wanted_days(spells, days):
    want = np.zeros(len(days), bool)
    for s in spells.itertuples():
        end = s.end if pd.notna(s.end) else days[-1] + pd.Timedelta(days=1)
        want |= (days >= s.start) & (days < end)
    return want


def coverage(d, spells, days) -> float:
    """share of the trading days the ticker was in the S&P 500 (since the price start) that have a price"""
    if d is None or d.empty:
        return 0.0
    want = wanted_days(spells, days)
    if not want.any():
        return 0.0
    return float(pd.Index(d["date"]).isin(days[want]).sum() / want.sum())


def load_form4_prices() -> dict:
    """ticker -> [(trade dates, prices) per company] from the Form 4s of the companies behind that ticker"""
    p = DATA_DIR / "trades.parquet"
    if not p.exists():
        print("  data/trades.parquet not there yet, prices are not compared with the Form 4s this run")
        return {}
    t = pd.read_parquet(p, columns=["cik", "tickers", "trade_date", "price"])
    t = t[t.price > 0].dropna(subset=["tickers"])
    t["ticker"] = t.tickers.str.split("|")
    t = t.explode("ticker")
    out = {}
    for (tk, cik), g in t.groupby(["ticker", "cik"]):
        out.setdefault(tk, []).append((g.trade_date.values, g.price.values))
    return out


def fit(d, f4):
    """Form 4 trades on days the series has a price, and how many of them fall in a company-year that fits"""
    if d is None or d.empty or not f4:
        return 0, 0
    close = pd.Series(d["close"].values, index=d["date"].values)
    n = ok = 0
    for dates, prices in f4:
        f = year_fit(dates, prices, close)
        n += int(f.n.sum())
        ok += int(f.loc[f.valid.astype(bool), "n"].sum())
    return n, ok


STALE = 5       # this many equal closes in a row at the end: the stock stopped trading on the first of them
LONG_STALE = 20
HOLE = 30       # calendar days without a price: two different stretches, often two different companies


def tidy(d: pd.DataFrame) -> pd.DataFrame:
    """
    The longest stretch without a hole, with the repeated last price cut off.
    MHS (Medco, bought in April 2012) is the worst case: 248 days at 70.30, then 9 days at 71.71. So after
    cutting a run, a short tail that sits on top of a long run goes too, and the long run after it.
    """
    d = d.sort_values("date").reset_index(drop=True)
    piece = (d.date.diff().dt.days > HOLE).cumsum()
    d = d[piece == piece.value_counts().idxmax()].reset_index(drop=True)
    c = d.close.values

    def run_back(n):
        # where the run of equal closes ending at row n-1 starts
        k = n - 1
        while k > 0 and c[k - 1] == c[n - 1]:
            k -= 1
        return k

    n = len(c)
    while n > 1:
        k = run_back(n)
        if n - k >= STALE:
            n = k + 1                                   # keep the first day of the run
        elif k > 0 and k - run_back(k) >= LONG_STALE:
            n = k                                       # a few new prices on top of a long run: not trading
        else:
            break
    return d.iloc[:n]


def usable(cov, n, ok) -> bool:
    # with fewer than 5 trades to compare there is nothing to say. Otherwise some company must fit
    return cov > 0 and (n < 5 or ok > 0)


def good(cov, n, ok) -> bool:
    # nothing more to gain from asking tiingo
    return cov >= 0.9 and (n < 5 or ok >= 0.5 * n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retry", action="store_true", help="try yahoo again for tickers without data")
    ap.add_argument("--yahoo-only", action="store_true", help="skip tiingo (quick first pass)")
    ap.add_argument("--try-unlisted", action="store_true", help="also ask tiingo for tickers not in its list")
    ap.add_argument("--combine", action="store_true", help="only write prices.parquet from the csv files so far")
    a = ap.parse_args()
    PRICE_DIR.mkdir(parents=True, exist_ok=True)

    members = DATA_DIR / "sp500_members.csv"
    if not members.exists():
        sys.exit("data/sp500_members.csv is missing, run src/ingest.py first")
    sp = pd.read_csv(members, parse_dates=["start", "end"])
    tickers = sorted(sp.ticker.unique())
    if not a.yahoo_only and not a.combine and not TIINGO_API_KEY:
        sys.exit("TIINGO_API_KEY is not set (free key from tiingo.com). Put it in .env, or run with --yahoo-only.")

    status, asked = {}, {}
    if STATUS.exists():
        s = pd.read_csv(STATUS, dtype={"note": str, "tiingo": str})
        first_version = "tiingo" not in s
        for c in ("note", "tiingo"):
            if c not in s:
                s[c] = ""
            s[c] = s[c].fillna("")
        if "fit_ok" not in s:
            # numbers from the first version of the check mean something else, start them again
            s["fit_n"], s["fit_ok"] = 0, 0
        for c in ("fit_n", "fit_ok"):
            s[c] = s[c].fillna(0).astype(int)
        s = s.drop(columns=[c for c in ("fit_dev",) if c in s])
        if first_version:
            # before the tiingo column existed these three outcomes only happened after tiingo was asked
            s.loc[s.source.isin(["tiingo", "none"]) | s.note.eq("tiingo had less"), "tiingo"] = "tried"
        # the first version of the check compared a series with all companies behind a ticker at once, and
        # threw out tickers shared by two companies. Those are fetched again: from yahoo, and from tiingo too
        # when it was a tiingo series that was thrown out
        both = (s.source == "none") & s.note.eq("another company uses the ticker now")
        yahoo_again = (s.source == "none") & s.note.str.startswith("yahoo has nothing that fits")
        asked = dict(zip(s.ticker[yahoo_again], s.tiingo[yahoo_again]))
        s = s[~(both | yahoo_again)]
        status = s.set_index("ticker").to_dict("index")

    def combine():
        # one file with everything, easier to load (and to move around) than a thousand csv files.
        # volume stays in the csv files only
        parts = []
        for t, st in status.items():
            p = PRICE_DIR / fname(t)
            if st["source"] != "none" and p.exists():
                d = tidy(pd.read_csv(p, parse_dates=["date"], usecols=["date", "close", "adj_close"]))
                d.insert(0, "ticker", t)
                parts.append(d)
        allp = pd.concat(parts, ignore_index=True)
        allp.to_parquet(DATA_DIR / "prices.parquet", index=False)
        return allp

    if a.combine:
        allp = combine()
        print(f"prices.parquet: {len(allp):,} rows, {allp.ticker.nunique()} tickers")
        return

    def save_status():
        s = pd.DataFrame.from_dict(status, orient="index").rename_axis("ticker").reset_index()
        s.sort_values("ticker").to_csv(STATUS, index=False)

    def store(t, d, source, cov, n=0, ok=0, note="", tried=None):
        tried = status.get(t, {}).get("tiingo", "") if tried is None else tried
        if d is not None and len(d) and source != "none":
            d.to_csv(PRICE_DIR / fname(t), index=False)
            status[t] = {"source": source, "first": d.date.min().date(), "last": d.date.max().date(),
                         "rows": len(d), "coverage": round(cov, 3), "fit_n": n, "fit_ok": ok, "note": note,
                         "tiingo": tried}
        else:
            # a file that is already there stays on disk (to look at), but it is not used any more
            status[t] = {"source": "none", "first": "", "last": "", "rows": 0, "coverage": 0.0, "fit_n": n,
                         "fit_ok": ok, "note": note, "tiingo": tried}
        save_status()

    # the benchmark first, its dates are the trading calendar
    bpath = PRICE_DIR / fname(BENCHMARK)
    if not bpath.exists():
        b = yahoo(BENCHMARK)
        if b is None and not a.yahoo_only:
            b = tiingo_wait(BENCHMARK)
        if b is None:
            sys.exit(f"could not get {BENCHMARK} from yahoo or tiingo, nothing else makes sense without it")
        store(BENCHMARK, b, "yahoo/tiingo", 1.0)
    days = pd.DatetimeIndex(pd.read_csv(bpath, parse_dates=["date"]).date)
    print(f"{BENCHMARK}: {len(days)} trading days, {days[0].date()} to {days[-1].date()}")

    f4 = load_form4_prices()

    def check(t, d):
        return (coverage(d, sp[sp.ticker == t], days), *fit(d, f4.get(t)))

    def on_disk(t):
        st = status.get(t)
        if st and st["source"] != "none" and (PRICE_DIR / fname(t)).exists():
            return pd.read_csv(PRICE_DIR / fname(t), parse_dates=["date"])
        return None

    # the series already on disk, compared with the Form 4 prices again (trades.parquet may be newer)
    checked = set()
    if f4:
        for t in tickers:
            d = on_disk(t)
            if d is not None:
                cov, n, ok = check(t, d)
                status[t].update({"coverage": round(cov, 3), "fit_n": n, "fit_ok": ok})
                checked.add(t)
        save_status()

    # pass 1: yahoo for tickers that are new (or had nothing, with --retry)
    todo = [t for t in tickers if t not in status or (a.retry and status[t]["source"] == "none")]
    print(f"{len(tickers)} tickers, {len(tickers) - len(todo)} done, {len(todo)} to fetch from yahoo")
    for i, t in enumerate(todo, 1):
        d = yahoo(t)
        cov, n, ok = check(t, d)
        checked.add(t)
        tried = status.get(t, {}).get("tiingo", asked.get(t, ""))
        if good(cov, n, ok):
            store(t, d, "yahoo", cov, n, ok, tried=tried)
        elif usable(cov, n, ok):
            # part of the days, or only some of the companies behind the ticker: keep it, tiingo may have more
            store(t, d, "yahoo partial", cov, n, ok, tried=tried)
        else:
            store(t, None, "none", 0.0, n, ok, "yahoo: no data" if d is None else "yahoo: another company",
                  tried=tried)
        if i % 50 == 0 or i == len(todo):
            print(f"  yahoo {i}/{len(todo)}")
        time.sleep(0.25)

    # what there is so far, so the next scripts can already run while tiingo takes its hours
    allp = combine()
    print(f"prices.parquet: {allp.ticker.nunique()} tickers so far")

    # pass 2: tiingo for every ticker without a good series that tiingo was not asked about yet
    def is_good(st):
        return st["source"] != "none" and good(st["coverage"], st["fit_n"], st["fit_ok"])

    open_ = ("", "not listed") if a.try_unlisted else ("",)
    queue = [t for t in tickers if not is_good(status[t]) and status[t].get("tiingo", "") in open_]
    if a.yahoo_only:
        print(f"{len(queue)} tickers left for tiingo, run again without --yahoo-only")
        queue = []
    lst = tiingo_listing() if queue else None
    if lst is not None:
        # trust the list only if it knows the tickers tiingo already gave us
        got = [t.replace(".", "-") for t, st in status.items() if st["source"] == "tiingo"]
        known = sum(g in set(lst.ticker) for g in got)
        if got and known < 0.9 * len(got):
            print(f"  tiingo ticker list only has {known} of {len(got)} tickers tiingo already delivered, not used")
            lst = None
    if lst is not None and not a.try_unlisted:
        # only ask for tickers tiingo has, for days that overlap the time in the index
        skip = []
        for t in queue:
            rows = lst[lst.ticker == t.replace(".", "-")]
            want = days[wanted_days(sp[sp.ticker == t], days)]
            if rows.empty or len(want) == 0 or not ((rows.startDate <= want[-1]) & (rows.endDate >= want[0])).any():
                skip.append(t)
        for t in skip:
            status[t]["tiingo"] = "not listed"
        save_status()
        queue = [t for t in queue if t not in skip]
        print(f"tiingo has no listing for {len(skip)} of them")
    if queue:
        print(f"tiingo: {len(queue)} tickers, about {len(queue) / 50:.1f} hours at 50 requests an hour")
    for i, t in enumerate(queue, 1):
        d = tiingo_wait(t)
        cov, n, ok = check(t, d)
        checked.add(t)
        od = on_disk(t)
        ocov, on, ook = check(t, od) if od is not None else (0.0, 0, 0)
        # the one that fits more Form 4 trades wins, then the one that covers more days
        if usable(cov, n, ok) and (not usable(ocov, on, ook) or (ok, cov) >= (ook, ocov)):
            store(t, d, "tiingo", cov, n, ok, tried="tried")
        elif usable(ocov, on, ook):
            src = status[t]["source"] if good(ocov, on, ook) else "yahoo partial"
            store(t, od, "tiingo" if status[t]["source"] == "tiingo" else src, ocov, on, ook, "tiingo had less",
                  tried="tried")
        else:
            why = "no data on yahoo or tiingo" if d is None and od is None else "prices of another company"
            store(t, None, "none", 0.0, max(n, on), 0, why, tried="tried")
        print(f"  tiingo {i}/{len(queue)} {t}: {status[t]['source']}, coverage {status[t]['coverage']:.0%}")

    # a series that fits no company in any year is another company's: not used
    for t in sorted(checked):
        st = status[t]
        if st["source"] != "none" and not usable(st["coverage"], st["fit_n"], st["fit_ok"]):
            store(t, None, "none", 0.0, st["fit_n"], st["fit_ok"], "prices of another company")

    allp = combine()

    s = pd.DataFrame.from_dict(status, orient="index")
    s = s[s.index != BENCHMARK]
    print(f"prices.parquet: {len(allp):,} rows, {allp.ticker.nunique()} tickers")
    print(s.source.value_counts().to_string())
    print(f"tickers with at least 90% of their S&P 500 days: {(s.coverage >= 0.9).sum()} of {len(tickers)}")


if __name__ == "__main__":
    main()
