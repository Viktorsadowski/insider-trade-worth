#!/usr/bin/env python3
"""
trades.parquet + prices -> data/events.csv, data/issuer_days.csv, data/paths.csv

One event = one insider reporting buys (or sales) in one company on one filing day. Returns are measured on
two clocks:

  insider clock  from the close of his last trade day in the filing: what the insider himself earned
  public clock   from the close of the first trading day after the filing: what an outsider can earn
                 (the data has closing prices only, so the outsider reads the filing and buys at the next close)

Abnormal return = stock total return minus SPY total return over the same days (ar_*), and minus the
equal-weighted S&P 500 as a check (arew_*). Columns are named ar_pub_21, ar_ins_126 and so on.

The control for momentum (arm_*): the stock minus the S&P 500 members that had the same past return on day 0.
Every day the members are sorted into fifths by their return over the past year without its last month, and
into fifths by their return over the last month. A stock is matched with the members in the same fifth on
both, about 17 of them, the stock itself left out. mom12_* and mom1_* are the two fifths (1 = the losers).

  events.csv       one row per insider event, both clocks, all horizons, with the flags below
  issuer_days.csv  the same collapsed to company + day + direction, the unit for the public clock
  paths.csv        average abnormal return day by day, from 20 days before to 24 months after, for the charts
  calendar_time.csv  daily return of a portfolio that holds every stock for h days after a buy (or sale), on
                     both clocks. port_m and ctrl: the same portfolio for the company-days that have matched
                     stocks, and the portfolio of those matched stocks
  price_map.csv    which price series each company uses and how well it fits the prices on the Form 4s
  coverage.csv     events per year and side, and how many of them have a price series

What happens when insiders buy and sell at the same time:
  mixed_day  the same company has buys and sales becoming public on the same day. Both events stay in the data
             with the flag. net_side says which side was bigger in dollars that day
  opp_30d    insiders of the same company traded the other way in the 30 days before this filing
The analysis shows the results with and without them.

What happens when a stock stops trading inside the holding period (data/manual/delistings.csv):
  cash buyout  the deal price is the last price        stock merger  the last traded price
  failure      the stock goes to 0
After that the money sits in SPY for the rest of the period, so the abnormal return stops moving.

  python src/events.py
"""

import numpy as np
import pandas as pd

from config import BENCHMARK, DATA_DIR, HORIZONS, MANUAL_DIR, ROLES
from pricefit import year_fit

PRE, POST = 20, max(HORIZONS)          # days before and after day 0 in paths.csv
MOM_LONG, MOM_SHORT = 252, 21          # past return: the past year without its last month, and the last month
FIFTHS = 5
MIN_MATCH = 5                          # a stock needs at least this many matched members, or it has no control


# ── prices ─────────────────────────────────────────────────────────────────────

def load_prices():
    p = pd.read_parquet(DATA_DIR / "prices.parquet")
    p["date"] = pd.to_datetime(p["date"])
    days = pd.DatetimeIndex(sorted(p.loc[p.ticker == BENCHMARK, "date"].unique()))
    p = p[p.date.isin(days)]
    adj = p.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last").reindex(days)
    raw = p.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").reindex(days)
    return days, adj, raw


def fit_prices(t: pd.DataFrame, raw) -> pd.DataFrame:
    """cik, ticker, year, n, level, spread, valid: which price series fits which company in which year.
    The rule is in pricefit.py"""
    rows = []
    for cik, g in t.groupby("cik"):
        for tk in sorted(set("|".join(g.tickers.dropna().unique()).split("|")) - {""}):
            if tk in raw.columns:
                f = year_fit(g.trade_date.values, g.price.values, raw[tk])
                f.insert(0, "ticker", tk)
                f.insert(0, "cik", cik)
                rows.append(f)
    return pd.concat(rows, ignore_index=True)


def delisting_table(adj, raw, days) -> pd.DataFrame:
    """ticker, last_day, type, cash_price for every series that ends before the last day in the data"""
    man_path = MANUAL_DIR / "delistings.csv"
    man = pd.read_csv(man_path, comment="#", parse_dates=["last_day"]) if man_path.exists() else \
        pd.DataFrame(columns=["ticker", "last_day", "type", "cash_price"])
    man = man.set_index("ticker")
    rows = []
    for tk in adj.columns:
        last = adj[tk].last_valid_index()
        if last is None or last >= days[-5]:
            continue
        r = {"ticker": tk, "last_day": last, "last_close": raw[tk].get(last, np.nan), "type": "unknown",
             "cash_price": np.nan}
        if tk in man.index:
            m = man.loc[tk]
            # the table is about one company. If the series ends somewhere else it is not that company's end
            if abs((last - m["last_day"]).days) <= 15:
                r["type"], r["cash_price"] = m["type"], m["cash_price"]
            else:
                print(f"  {tk}: the series ends {last.date()}, delistings.csv says {m['last_day'].date()}, not used")
        rows.append(r)
    return pd.DataFrame(rows, columns=["ticker", "last_day", "last_close", "type", "cash_price"])


def extend(adj, raw, days, dl: pd.DataFrame):
    """
    total return index per ticker that keeps going after a delisting: the final payout, then SPY.
    A series that ends for an unknown reason stays empty after its last day, those events are dropped.
    """
    x = adj.copy()
    spy = adj[BENCHMARK].values
    for r in dl.itertuples():
        if r.type == "unknown":
            continue
        i = days.get_loc(r.last_day)
        base = x[r.ticker].values[i]
        if r.type == "failure":
            pay = 0.0
        elif r.type == "cash" and pd.notna(r.cash_price) and pd.notna(r.last_close) and r.last_close > 0:
            pay = float(r.cash_price) / r.last_close
        else:
            pay = 1.0           # stock or mixed deal: the last traded price is the value
        col = x[r.ticker].values.copy()
        col[i + 1:] = base * pay * spy[i + 1:] / spy[i]
        x[r.ticker] = col
    return x


# ── stocks with the same past return ───────────────────────────────────────────

def past_fifths(X: np.ndarray, P: np.ndarray):
    """
    q12, q1: for every day and stock, the fifth (0 = lowest, 4 = highest) of its return over the past year
    without the last month, and of its return over the last month. The breakpoints are those of the S&P 500
    members with a price that day (P), so a stock outside the index gets a fifth too. -1 when its prices do
    not go that far back
    """
    n, m = X.shape
    r12, r1 = np.full((n, m), np.nan), np.full((n, m), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        r12[MOM_LONG:] = X[MOM_LONG - MOM_SHORT: n - MOM_SHORT] / X[: n - MOM_LONG] - 1
        r1[MOM_SHORT:] = X[MOM_SHORT:] / X[:-MOM_SHORT] - 1
    out = []
    for r in (r12, r1):
        q = np.full((n, m), -1, dtype=np.int8)
        for t in range(n):
            ok = P[t] & ~np.isnan(r[t])
            if ok.sum() < 100:
                continue
            has = ~np.isnan(r[t])
            q[t, has] = np.searchsorted(np.quantile(r[t, ok], np.arange(1, FIFTHS) / FIFTHS), r[t, has],
                                        side="right")
        out.append(q)
    return out


def cell_returns(X: np.ndarray, P: np.ndarray, cell: np.ndarray, h: int):
    """F: return of every stock over the next h days. S, N: per day and group, the sum of F over the members
    in the group and how many they are"""
    n, m = X.shape
    F = np.full((n, m), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        F[: n - h] = X[h:] / X[: n - h] - 1
    S, N = np.zeros((n, FIFTHS * FIFTHS)), np.zeros((n, FIFTHS * FIFTHS))
    t, k = np.nonzero(P & (cell >= 0) & ~np.isnan(F))
    np.add.at(S, (t, cell[t, k]), F[t, k])
    np.add.at(N, (t, cell[t, k]), 1)
    return F, S, N


def matched_weights(P: np.ndarray, cell: np.ndarray, i0: np.ndarray, j: np.ndarray):
    """
    For the calendar-time portfolios: the matched stocks of every company-day, as three arrays of the same
    length. row: which company-day, col: the matched stock, w: 1 / the number of matched stocks, so every
    company-day puts the same money in its matched stocks as in the stock itself
    """
    rows, cols, w, members = [], [], [], {}
    for r, (t, k) in enumerate(zip(i0, j)):
        c = cell[t, k]
        if c < 0:
            continue
        if (t, c) not in members:
            members[(t, c)] = np.flatnonzero(P[t] & (cell[t] == c))
        mk = members[(t, c)]
        mk = mk[mk != k]
        if len(mk) >= MIN_MATCH:
            rows.append(np.full(len(mk), r))
            cols.append(mk)
            w.append(np.full(len(mk), 1 / len(mk)))
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(w)


# ── events ─────────────────────────────────────────────────────────────────────

def build_events(t: pd.DataFrame, days) -> pd.DataFrame:
    t = t[t.in_sp500].copy()
    t["value"] = t.shares * t.price
    rank = {r: i for i, r in enumerate(ROLES)}
    t["rank"] = t.role.map(rank)
    g = t.groupby(["cik", "owner_cik", "filing_date", "code"], sort=False)
    e = g.agg(issuer=("issuer", "first"), owner=("owner", "first"), rank=("rank", "min"),
              title=("title", "first"), first_trade=("trade_date", "min"), last_trade=("trade_date", "max"),
              shares=("shares", "sum"), value=("value", "sum"), n_trades=("shares", "size"),
              tickers=("tickers", "first")).reset_index()
    e["role"] = e["rank"].map(dict(enumerate(ROLES)))
    e["side"] = e.code.map({"P": "buy", "S": "sell"})

    # day 0 on each clock, as a position in the trading calendar
    d = days.values
    i_pub = np.searchsorted(d, e.filing_date.values, side="right")       # first trading day after the filing
    i_ins = np.searchsorted(d, e.last_trade.values, side="left")         # the trade day (or the next open day)
    e["i_pub"], e["i_ins"] = i_pub, i_ins
    # the close on the filing day itself: the last price before the market has seen the filing
    e["i_fil"] = np.maximum(i_pub - 1, i_ins)
    e = e[(e.i_pub < len(d)) & (e.i_ins < len(d))].copy()
    e["avail_date"] = d[e.i_pub.values]
    e["lag_days"] = e.i_pub - e.i_ins                                    # trading days from trade to public
    return e


def add_flags(e: pd.DataFrame) -> pd.DataFrame:
    # same company, same public day, both directions
    v = e.groupby(["cik", "avail_date", "side"]).value.sum().unstack(fill_value=0.0)
    for c in ("buy", "sell"):
        if c not in v:
            v[c] = 0.0
    v["mixed_day"] = (v.buy > 0) & (v.sell > 0)
    v["net_side"] = np.where(v.buy >= v.sell, "buy", "sell")
    e = e.join(v[["mixed_day", "net_side"]], on=["cik", "avail_date"])
    e["n_insiders_day"] = e.groupby(["cik", "avail_date", "side"]).owner_cik.transform("nunique")

    # the other side in the 30 days before: for every event, the latest filing of the opposite direction in
    # the same company that came strictly earlier
    e = e.sort_values("filing_date", kind="stable")
    e["opp_30d"] = False
    for side, other in (("buy", "sell"), ("sell", "buy")):
        a = e.loc[e.side == side, ["cik", "filing_date"]].reset_index()
        b = e.loc[e.side == other, ["cik", "filing_date"]].drop_duplicates().rename(columns={"filing_date": "opp"})
        m = pd.merge_asof(a, b.sort_values("opp"), left_on="filing_date", right_on="opp", by="cik",
                          allow_exact_matches=False)
        hit = (m.filing_date - m.opp).dt.days <= 30
        e.loc[m.loc[hit, "index"], "opp_30d"] = True
    return e


def add_returns(e: pd.DataFrame, x: pd.DataFrame, ew: np.ndarray, col_of: dict, P: np.ndarray, q12: np.ndarray,
                q1: np.ndarray) -> pd.DataFrame:
    """ar_{clock}_{h} for both clocks and all horizons, plus the move between the two clocks (ar_delay), and
    arm_{clock}_{h}, the same return against the members with the same past return"""
    X = x.values
    spy = x[BENCHMARK].values
    n = len(spy)
    j = e.price_ticker.map(col_of).values.astype(int)

    def bhar(i0, i1, bench):
        ok = (i1 < n) & (i0 >= 0)
        i0c, i1c = np.where(ok, i0, 0), np.where(ok, i1, 0)
        stock = X[i1c, j] / X[i0c, j] - 1
        b = bench[i1c] / bench[i0c] - 1
        return np.where(ok, stock - b, np.nan), np.where(ok, stock, np.nan)

    for clock, col in (("pub", "i_pub"), ("ins", "i_ins")):
        i0 = e[col].values
        for h in HORIZONS:
            e[f"ar_{clock}_{h}"], e[f"ret_{clock}_{h}"] = bhar(i0, i0 + h, spy)
            e[f"arew_{clock}_{h}"] = bhar(i0, i0 + h, ew)[0]
    # what happens between the insider's trade and the day an outsider can act, and its two parts: up to the
    # close on the filing day (nobody outside knows yet), and the first day the market has the filing
    e["ar_delay"] = bhar(e.i_ins.values, e.i_pub.values, spy)[0]
    e["ar_wait"] = bhar(e.i_ins.values, e.i_fil.values, spy)[0]
    e["ar_react"] = bhar(e.i_fil.values, e.i_pub.values, spy)[0]
    # the 20 days before the trade: did he buy after a drop, sell after a rise
    e["ar_before"] = bhar(e.i_ins.values - PRE, e.i_ins.values, spy)[0]

    # against the members with the same past return. The stock itself is taken out of its group's average
    cell = np.where((q12 >= 0) & (q1 >= 0), q12.astype(int) * FIFTHS + q1, -1)
    new = {}
    for clock, col in (("pub", "i_pub"), ("ins", "i_ins")):
        i0 = e[col].values
        new[f"mom12_{clock}"] = np.where(q12[i0, j] >= 0, q12[i0, j] + 1.0, np.nan)      # 1 = the losers
        new[f"mom1_{clock}"] = np.where(q1[i0, j] >= 0, q1[i0, j] + 1.0, np.nan)
    for h in HORIZONS:
        F, S, N = cell_returns(X, P, cell, h)
        for clock, col in (("pub", "i_pub"), ("ins", "i_ins")):
            i0 = e[col].values
            c = cell[i0, j]
            own = P[i0, j] & (c >= 0) & ~np.isnan(F[i0, j])
            cnt = N[i0, np.maximum(c, 0)] - own
            tot = S[i0, np.maximum(c, 0)] - np.where(own, F[i0, j], 0.0)
            ctrl = np.where((c >= 0) & (cnt >= MIN_MATCH), tot / np.maximum(cnt, 1), np.nan)
            new[f"arm_{clock}_{h}"] = e[f"ret_{clock}_{h}"].values - ctrl
    return pd.concat([e, pd.DataFrame(new, index=e.index)], axis=1)


def paths(ev: pd.DataFrame, x: pd.DataFrame, col_of: dict, clock: str, groups: dict, bench=None) -> pd.DataFrame:
    """mean abnormal return relative to day 0, for each offset from -PRE to POST. bench: SPY unless given"""
    X, spy = x.values, x[BENCHMARK].values if bench is None else bench
    n = len(spy)
    offs = np.arange(-PRE, POST + 1)
    rows = []
    for name, sub in groups.items():
        if sub.empty:
            continue
        i0 = sub[f"i_{clock}"].values
        j = sub.price_ticker.map(col_of).values.astype(int)
        for k in offs:
            i1 = i0 + k
            ok = (i1 >= 0) & (i1 < n)
            i1c = np.where(ok, i1, 0)
            rel = X[i1c, j] / X[i0, j] - spy[i1c] / spy[i0]        # same definition as ar_*: stock minus SPY
            rel = rel[ok & ~np.isnan(rel)]
            rows.append({"clock": clock, "group": name, "day": k, "mean": rel.mean() if len(rel) else np.nan,
                         "n": len(rel)})
    return pd.DataFrame(rows)


def calendar_time(d: pd.DataFrame, x: pd.DataFrame, col_of: dict, ew: np.ndarray, P: np.ndarray, cell: np.ndarray,
                  holds=HORIZONS[1:]) -> pd.DataFrame:
    """
    The same test without overlapping events, and as money: every day, hold every stock with an insider buy
    (or sale) in the last h trading days, equal weight per company-day, and write down that day's portfolio
    return next to SPY's and next to the equal-weighted S&P 500 (ew, the average stock). Once counted from the
    close on the insider's trade day and once from the close on the public day 0. analysis.py turns the daily
    series into an alpha and a compounded return per year.

    port_m and ctrl are the control for momentum: port_m is the same portfolio for the company-days that have
    matched stocks (a few have none, most of them in 2006), ctrl holds the matched stocks instead, for the
    same days.
    """
    R = x.pct_change(fill_method=None).values
    R0 = np.nan_to_num(R)
    spy = x[BENCHMARK].pct_change(fill_method=None).values
    ew_ret = np.r_[np.nan, ew[1:] / ew[:-1] - 1]
    n, m = R.shape
    first = int(d.i_pub.min()) - 1                  # the series starts on the first public day in the data

    def daily(i0, j, w, h):
        """return per day of the portfolio that puts w in stock j from the day after i0 and for h days"""
        # +w the day after entry, -w the day after the holding period ends, then a running sum per stock
        c = np.zeros((n + 1, m))
        np.add.at(c, (np.minimum(i0 + 1, n), j), w)
        np.add.at(c, (np.minimum(i0 + h + 1, n), j), -w)
        c = np.cumsum(c, axis=0)[:n]
        c[np.isnan(R) | (np.abs(c) < 1e-9)] = 0.0      # the running sum of fractions does not end at exactly 0
        held = c.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (c * R0).sum(axis=1) / held, held        # nan on days with nothing held

    rows = []
    for side in ("buy", "sell"):
        s = d[(d.side == side) & ~d.mixed_day]
        j = s.price_ticker.map(col_of).values.astype(int)
        for clock, i0 in (("insider", s.i_ins.values), ("public", s.i_pub.values)):
            mr, mk, mw = matched_weights(P, cell, i0, j)
            has = np.zeros(len(s), bool)
            has[mr] = True
            for h in holds:
                port, held = daily(i0, j, 1.0, h)
                port_m, _ = daily(i0[has], j[has], 1.0, h)
                ctrl, _ = daily(i0[mr], mk, mw, h)
                rows.append(pd.DataFrame({"date": x.index, "side": side, "clock": clock, "hold": h, "port": port,
                                          "spy": spy, "ew": ew_ret, "n_held": held, "port_m": port_m,
                                          "ctrl": ctrl}).iloc[first + 1:])
    return pd.concat(rows, ignore_index=True)


def main():
    t = pd.read_parquet(DATA_DIR / "trades.parquet")
    days, adj, raw = load_prices()
    print(f"{len(t):,} trades, {adj.shape[1]} price series, {days[0].date()} to {days[-1].date()}")

    # which series belongs to which company, year by year
    pm = fit_prices(t[t.price > 0], raw)
    pm.to_csv(DATA_DIR / "price_map.csv", index=False)
    good = pm[pm.valid]
    sp_ciks = set(t.loc[t.in_sp500, "cik"])
    print(f"price_map.csv: {len(sp_ciks & set(good.cik))} of {len(sp_ciks)} companies have a price series "
          f"that fits")

    dl = delisting_table(adj, raw, days)
    dl.to_csv(DATA_DIR / "delistings_used.csv", index=False)
    print(f"{len(dl)} series end before the last day: " + ", ".join(f"{k} {v}" for k, v in
                                                                      dl.type.value_counts().items()))
    unk = sorted(set(dl.ticker[dl.type == "unknown"]) & set(good.ticker))
    if unk:
        print(f"  no line in data/manual/delistings.csv for {' '.join(unk)}: returns that run past their last "
              f"day are left out")
    x = extend(adj, raw, days, dl)
    col_of = {tk: i for i, tk in enumerate(x.columns)}

    # equal-weighted S&P 500: average daily return of the members with a price that day
    sp = pd.read_csv(DATA_DIR / "sp500_members.csv", parse_dates=["start", "end"])
    ret = x.pct_change(fill_method=None)
    member = pd.DataFrame(False, index=days, columns=x.columns)
    for s in sp.itertuples():
        if s.ticker in member.columns:
            end = s.end if pd.notna(s.end) else days[-1] + pd.Timedelta(days=1)
            member.loc[(days >= s.start) & (days < end), s.ticker] = True
    # not the series of another company: drop tickers that never fit, and the years that do not
    never = [tk for tk in member.columns if tk not in set(good.ticker)]
    member[never] = False
    bad = pm.groupby(["ticker", "year"]).valid.max()
    for (tk, y), v in bad.items():
        if not v and tk in member.columns:
            member.loc[days.year == y, tk] = False
    ew = (1 + ret.where(member).mean(axis=1).fillna(0.0)).cumprod().values
    # the same members sorted by past return, for the control for momentum
    P = member.values & (x.values > 0)
    q12, q1 = past_fifths(x.values, P)
    cell = np.where((q12 >= 0) & (q1 >= 0), q12.astype(int) * FIFTHS + q1, -1)

    e = build_events(t, days)
    n0 = len(e)
    # the series to use: one that fits the company in the year of the trade (or the year next to it), has a
    # price on day 0 of both clocks, and of those the one that runs longest
    last = {tk: adj[tk].last_valid_index() for tk in adj.columns}
    best = {}
    for (cik, year), g in good.groupby(["cik", "year"]):
        best[(cik, year)] = sorted(g.ticker, key=lambda tk: last[tk], reverse=True)
    X = x.values
    pick = []
    for cik, year, i_pub, i_ins in zip(e.cik.values, e.last_trade.dt.year.values, e.i_pub.values, e.i_ins.values):
        cands = best.get((cik, year)) or best.get((cik, year - 1)) or best.get((cik, year + 1)) or []
        pick.append(next((c for c in cands if X[i_pub, col_of[c]] > 0 and X[i_ins, col_of[c]] > 0), None))
    e["price_ticker"] = pick
    no_price = e.price_ticker.isna()
    print(f"events: {n0:,}, without a usable price {no_price.sum():,} ({no_price.mean():.1%}), dropped")
    cov = e.assign(year=e.filing_date.dt.year, priced=~no_price).groupby(["year", "side"]).priced \
        .agg(events="size", with_price="sum").reset_index()
    cov.to_csv(DATA_DIR / "coverage.csv", index=False)
    e = e[~no_price].copy()

    e = add_flags(e)
    e = add_returns(e, x, ew, col_of, P, q12, q1)
    e = e.sort_values(["avail_date", "cik", "side", "owner_cik"]).reset_index(drop=True)
    e.drop(columns=["rank", "tickers"]).to_csv(DATA_DIR / "events.csv", index=False)

    # company + day + direction. The public clock is the same for everyone that day, so take the first row,
    # with the most senior role of the day
    rank = {r: i for i, r in enumerate(ROLES)}
    e["rank"] = e.role.map(rank)
    keep = [c for c in e.columns if c.startswith(("ar_pub", "arew_pub", "ret_pub", "arm_pub", "mom12_pub",
                                                   "mom1_pub"))]
    d = e.sort_values("rank", kind="stable").groupby(["cik", "avail_date", "side"], sort=False).agg(
        issuer=("issuer", "first"), price_ticker=("price_ticker", "first"), i_pub=("i_pub", "first"),
        i_ins=("i_ins", "max"), top_role=("role", "first"), n_insiders=("owner_cik", "nunique"),
        value=("value", "sum"), mixed_day=("mixed_day", "first"), net_side=("net_side", "first"),
        opp_30d=("opp_30d", "max"), **{c: (c, "first") for c in keep}).reset_index()
    d = d.sort_values(["avail_date", "cik", "side"])
    d.to_csv(DATA_DIR / "issuer_days.csv", index=False)

    clean = d[~d.mixed_day]
    gp = {"buy": clean[clean.side == "buy"], "sell": clean[clean.side == "sell"],
          "buy, mixed day": d[d.mixed_day & (d.side == "buy")]}
    for r in ROLES:
        gp[f"buy, {r}"] = clean[(clean.side == "buy") & (clean.top_role == r)]
    gi = {"buy": e[e.side == "buy"], "sell": e[e.side == "sell"]}
    # the same buys against the average S&P 500 stock instead of SPY
    gew = {"buy, equal-weighted": gp["buy"], "sell, equal-weighted": gp["sell"]}
    pd.concat([paths(d, x, col_of, "pub", gp), paths(e, x, col_of, "ins", gi),
               paths(d, x, col_of, "pub", gew, bench=ew)]).to_csv(DATA_DIR / "paths.csv", index=False)

    calendar_time(d, x, col_of, ew, P, cell).to_csv(DATA_DIR / "calendar_time.csv", index=False)

    print(f"events.csv: {len(e):,} insider events ({(e.side == 'buy').sum():,} buys, {(e.side == 'sell').sum():,} "
          f"sales), {e.cik.nunique()} companies")
    print(f"issuer_days.csv: {len(d):,} company-days, {d.mixed_day.sum():,} on a day with both buys and sales")
    for side in ("buy", "sell"):
        s = d[(d.side == side) & ~d.mixed_day]
        print(f"  {side:4} public clock, mean abnormal return in bps: " +
              "  ".join(f"{h}d {s[f'ar_pub_{h}'].mean() * 1e4:+.0f}" for h in HORIZONS))
        print(f"       against stocks with the same past return:  " +
              "  ".join(f"{h}d {s[f'arm_pub_{h}'].mean() * 1e4:+.0f}" for h in HORIZONS))


if __name__ == "__main__":
    main()
