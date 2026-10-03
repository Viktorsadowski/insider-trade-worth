#!/usr/bin/env python3
"""
What is an insider trade worth: the tables and charts.

  data/results.csv         every number in the report: table, clock, side, group, horizon, n, mean (bps),
                           standard error, t, share of events above 0
  data/calendar_alpha.csv  the calendar-time portfolios: compounded return per year, and alpha against SPY
  figures/sample.png      when insiders buy (per month), and how long from the trade to the public day
  figures/growth.png      the headline: $1 in every insider buy from the insider's close, from the filing, in SPY,
                          held for 3 months and for a week
  figures/holds.png       return per year of the portfolios by holding period: the edge is in the first weeks
  figures/timeline.png    buys: where the gain is. Before the trade, up to the filing, the day after, later
  figures/clocks.png      day by day for two years: the insider's own return next to what an outsider gets
  figures/roles.png       buys by role, after a week and after 3 months
  figures/mixed.png       buys and sales at the same time
  figures/since.png       the last ten years on their own: $1 in the stocks insiders bought and sold, from the
                          filing, and the return per year by holding period
  figures/periods.png     insider buys before and after 2017, the insider next to an outsider
  figures/horizons.png    not in the report: the per-trade table as bars

t-values are clustered by calendar month of day 0: events in the same month share the same market moves.
From 3 months up the holding periods of neighbouring months overlap too, so there the standard errors also count
the covariance between months (overlap_lags). The calendar-time alpha is the check that does not depend on this.

  python src/analysis.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (BLUE, DATA_DIR, FIG_DIR, GRID, HORIZONS, INK, LIGHT_GREY, MAIN_H, MUTED, ORANGE, ROLES,
                    SURFACE)

E = pd.read_csv(DATA_DIR / "events.csv", parse_dates=["filing_date", "avail_date", "first_trade", "last_trade"],
                low_memory=False)
D = pd.read_csv(DATA_DIR / "issuer_days.csv", parse_dates=["avail_date"], low_memory=False)
P = pd.read_csv(DATA_DIR / "paths.csv")
C = pd.read_csv(DATA_DIR / "calendar_time.csv", parse_dates=["date"])

E["month"] = E.avail_date.dt.to_period("M").astype(str)
E["month_ins"] = E.last_trade.dt.to_period("M").astype(str)
D["month"] = D.avail_date.dt.to_period("M").astype(str)
POST = max(HORIZONS)      # last day in the path charts, same as in events.py
HLAB = {1: "1 day", 5: "1 week", 21: "1 month", 63: "3 months", 126: "6 months", 252: "12 months", 504: "24 months"}
ALL = "2006-2026"
PERIODS = [("2006-2016", 2006, 2016), ("2017-2026", 2017, 2026)]
FOCUS = "2017-2026"       # the outsider's question is asked once more for the last ten years on their own
SAMPLES = [(ALL, 2006, 2026)] + PERIODS
BENCH = {"SPY": "ar", "equal-weighted": "arew"}      # benchmark -> prefix of the return columns in events.csv
LIGHT_BLUE = "#9dbfeb"    # BLUE at 45%, the outsider in the buy charts
LIGHT_ORANGE = "#f4b9a1"  # ORANGE at 45%, the outsider in the sale charts
MID_BLUE = "#5c93dd"      # LIGHT_BLUE is too pale to read as text, the labels of the outsider's lines use this
GROWTH_H = (MAIN_H, 5)    # the growth chart: every stock held for 3 months, and for a week (where the gain is)


# ── statistics ─────────────────────────────────────────────────────────────────

def overlap_lags(h) -> int:
    """how many months apart two events can be and still share part of an h-day holding period, plus one.
    Up to a month: 0, the plain month clusters are enough"""
    return 0 if h <= 21 else int(np.ceil(h / 21)) + 1


def ols_cluster(y, X, cluster, lags=0):
    """
    OLS with standard errors clustered on `cluster`. Returns beta, se.
    lags > 0: the clusters are months ("2009-03") and months up to `lags` apart are allowed to hang together
    (Bartlett weights, like Newey-West on the month sums). Needed when the holding period is longer than a
    month: a 3-month return from a January buy and one from a February buy share two months.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    u = y - X @ beta
    codes, uniq = pd.factorize(cluster)
    g = len(uniq)
    if lags:
        # months in calendar order, empty months keep their place
        no = np.array([int(str(m)[:4]) * 12 + int(str(m)[5:7]) for m in uniq])
        codes = (no - no.min())[codes]
    s = np.zeros((codes.max() + 1, k))
    np.add.at(s, codes, X * u[:, None])
    meat = s.T @ s
    for l in range(1, min(lags, len(s) - 1) + 1):
        c = s[l:].T @ s[:-l]
        meat += (1 - l / (lags + 1)) * (c + c.T)
    adj = (g / (g - 1)) * ((n - 1) / (n - k)) if g > 1 and n > k else np.nan
    v = xtx_inv @ meat @ xtx_inv * adj
    return beta, np.sqrt(np.diag(v))


def mean_stat(x, cluster, h=0) -> dict:
    """h: the holding period in days, for the overlap between months"""
    ok = ~np.isnan(np.asarray(x, float))
    x, cl = np.asarray(x, float)[ok], np.asarray(cluster)[ok]
    if len(x) < 30:
        return {"n": len(x), "mean_bps": np.nan, "se_bps": np.nan, "t": np.nan, "pos": np.nan}
    b, se = ols_cluster(x, np.ones((len(x), 1)), cl, overlap_lags(h))
    return {"n": len(x), "mean_bps": b[0] * 1e4, "se_bps": se[0] * 1e4, "t": b[0] / se[0], "pos": (x > 0).mean()}


def diff_stat(x, flag, cluster, h=0) -> dict:
    """mean of the flagged group minus mean of the rest"""
    ok = ~np.isnan(np.asarray(x, float))
    x, f, cl = np.asarray(x, float)[ok], np.asarray(flag, float)[ok], np.asarray(cluster)[ok]
    if f.sum() < 30 or (1 - f).sum() < 30:
        return {"diff_bps": np.nan, "diff_t": np.nan}
    b, se = ols_cluster(x, np.column_stack([np.ones(len(x)), f]), cl, overlap_lags(h))
    return {"diff_bps": b[1] * 1e4, "diff_t": b[1] / se[1]}


def newey_west_alpha(port, spy, lags=10) -> dict:
    """daily port = a + b * spy. Alpha per year, with a t-value that allows for autocorrelation"""
    y, x = np.asarray(port, float), np.asarray(spy, float)
    ok = ~np.isnan(y) & ~np.isnan(x)
    y, x = y[ok], x[ok]
    X = np.column_stack([np.ones(len(y)), x])
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    h = X * (y - X @ beta)[:, None]
    s = h.T @ h
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        g = h[l:].T @ h[:-l]
        s += w * (g + g.T)
    se = np.sqrt(np.diag(xtx_inv @ s @ xtx_inv))
    return {"days": len(y), "alpha_year": beta[0] * 252, "t": beta[0] / se[0], "beta": beta[1],
            "excess_year": (y - x).mean() * 252}


# ── tables ─────────────────────────────────────────────────────────────────────

def results(save=True) -> pd.DataFrame:
    """
    Every table for the whole sample and for each period (by the public day 0), against SPY and against the
    equal-weighted S&P 500. One row = sample, table, clock, bench, side, group, horizon.
    diff_bps / diff_t: that group against the rest (a role against the other roles, buys after recent sales
    against buys alone, two or more insiders against one).
    """
    rows = []

    def add(table, clock, side, group, h, s, **extra):
        rows.append({"sample": sample, "table": table, "clock": clock, "bench": bench, "side": side,
                     "group": group, "horizon": h, **s, **extra})

    for sample, y0, y1 in SAMPLES:
        Ds, Es = D[D.avail_date.dt.year.between(y0, y1)], E[E.avail_date.dt.year.between(y0, y1)]
        clean = Ds[~Ds.mixed_day]
        for side in ("buy", "sell"):
            d, e, a = clean[clean.side == side], Es[Es.side == side], Ds[Ds.side == side]
            for bench, pre in BENCH.items():
                for h in HORIZONS:
                    pub, ins = f"{pre}_pub_{h}", f"{pre}_ins_{h}"
                    # the two clocks. Public: one row per company and day. Insider: one row per insider
                    add("horizons", "public", side, "all", h, mean_stat(d[pub], d.month, h))
                    add("horizons", "insider", side, "all", h, mean_stat(e[ins], e.month_ins, h))
                    # roles, on both clocks, per insider event
                    for r in ROLES:
                        er = e[e.role == r]
                        add("roles", "public", side, r, h, mean_stat(er[pub], er.month, h),
                            **diff_stat(e[pub], e.role == r, e.month, h))
                        add("roles", "insider", side, r, h, mean_stat(er[ins], er.month_ins, h),
                            **diff_stat(e[ins], e.role == r, e.month_ins, h))
                    # buying and selling at the same time
                    parts = {"alone": a[~a.mixed_day & ~a.opp_30d],
                             "other side in the 30 days before": a[~a.mixed_day & a.opp_30d],
                             "other side the same day, this side bigger": a[a.mixed_day & (a.net_side == side)],
                             "other side the same day, other side bigger": a[a.mixed_day & (a.net_side != side)]}
                    for name, g in parts.items():
                        extra = diff_stat(d[pub], d.opp_30d, d.month, h) if name.endswith("before") else {}
                        add("mixed", "public", side, name, h, mean_stat(g[pub], g.month, h), **extra)
                    # more than one insider the same day
                    add("several insiders", "public", side, "1 insider", h,
                        mean_stat(d.loc[d.n_insiders == 1, pub], d.month[d.n_insiders == 1], h))
                    add("several insiders", "public", side, "2 or more insiders", h,
                        mean_stat(d.loc[d.n_insiders >= 2, pub], d.month[d.n_insiders >= 2], h),
                        **diff_stat(d[pub], d.n_insiders >= 2, d.month, h))
            # before the trade and between trade and public
            bench = "SPY"
            add("timing", "insider", side, "20 days before the trade", 0, mean_stat(e.ar_before, e.month_ins))
            add("timing", "insider", side, "from trade to public", 0, mean_stat(e.ar_delay, e.month_ins))
            add("timing", "insider", side, "from trade to the close on the filing day", 0,
                mean_stat(e.ar_wait, e.month_ins))
            add("timing", "insider", side, "the day after the filing", 0, mean_stat(e.ar_react, e.month_ins))
    r = pd.DataFrame(rows)
    if save:
        r.to_csv(DATA_DIR / "results.csv", index=False)
    return r


def pick(R, table, sample=ALL, bench="SPY", **kw):
    """the rows of one table. The whole sample against SPY unless something else is asked for"""
    r = R[(R.table == table) & (R["sample"] == sample) & (R.bench == bench)]
    for k, v in kw.items():
        r = r[r[k] == v]
    return r


def growth(g: pd.DataFrame) -> pd.DataFrame:
    """$1 in the portfolio, in SPY and in the average S&P 500 stock, day by day. On a day with nothing to hold
    the money is in SPY"""
    g = g.sort_values("date")
    port = g.port.fillna(g.spy).fillna(0.0)
    return pd.DataFrame({"date": g.date.values, "port": (1 + port).cumprod().values,
                         "spy": (1 + g.spy.fillna(0.0)).cumprod().values,
                         "ew": (1 + g.ew.fillna(0.0)).cumprod().values})


def calendar_alpha() -> pd.DataFrame:
    """
    One line per portfolio and period. per_year, spy_per_year and ew_per_year are compounded (geometric)
    returns, excess_per_year and over_ew_per_year the growth of the portfolio relative to SPY and to the
    equal-weighted S&P 500, also compounded. alpha, t and beta are from the daily regression on SPY, alpha_ew,
    t_ew and beta_ew from the one on the equal-weighted S&P 500.
    """
    rows = []
    for (side, clock, hold), g in C.groupby(["side", "clock", "hold"]):
        for name, y0, y1 in SAMPLES:
            p = g[g.date.dt.year.between(y0, y1)]
            w = growth(p)
            years = len(w) / 252
            per_year = lambda v: v ** (1 / years) - 1
            e = newey_west_alpha(p.port, p.ew)
            rows.append({"side": side, "clock": clock, "hold": hold, "period": name, "avg_held": p.n_held.mean(),
                         "per_year": per_year(w.port.iloc[-1]), "spy_per_year": per_year(w.spy.iloc[-1]),
                         "ew_per_year": per_year(w.ew.iloc[-1]),
                         "excess_per_year": per_year(w.port.iloc[-1] / w.spy.iloc[-1]),
                         "over_ew_per_year": per_year(w.port.iloc[-1] / w.ew.iloc[-1]),
                         **newey_west_alpha(p.port, p.spy),
                         "alpha_ew": e["alpha_year"], "t_ew": e["t"], "beta_ew": e["beta"]})
    # bought minus sold: the daily difference between the two portfolios, as a return per year
    for (clock, hold), g in C.groupby(["clock", "hold"]):
        w = g.pivot(index="date", columns="side", values=["port", "spy"])
        diff, spy = w["port"]["buy"] - w["port"]["sell"], w["spy"]["buy"]
        for name, y0, y1 in SAMPLES:
            m = (diff.index.year >= y0) & (diff.index.year <= y1)
            x = newey_west_alpha(diff[m], spy[m])
            rows.append({"side": "buy minus sell", "clock": clock, "hold": hold, "period": name,
                         "per_year": diff[m].mean() * 252, **x})
    a = pd.DataFrame(rows)
    a.to_csv(DATA_DIR / "calendar_alpha.csv", index=False)
    return a


# ── charts ─────────────────────────────────────────────────────────────────────

def style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def new(w=8.4, h=5.0, **kw):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200, **kw)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / name, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def title(ax, t):
    ax.set_title(t, loc="left", fontsize=11, color=INK, pad=10)


def month_ticks(ax, lo, hi):
    lab = {0: "day 0", 63: "3 m", 126: "6 m", 252: "12 m", 378: "18 m", 504: "24 m"}
    ticks = [t for t in lab if lo <= t <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([lab[t] for t in ticks])


def fig_clocks():
    """day by day for two years. Colour says who (the insider or an outsider), dashed is the outsider measured
    against the average S&P 500 stock instead of SPY: the slow drift at the end is the benchmark"""
    fig, axes = new(10.0, 3.9, ncols=2, sharey=True)
    for ax, side, col, light in zip(axes, ("buy", "sell"), (BLUE, ORANGE), (LIGHT_BLUE, LIGHT_ORANGE)):
        style(ax)
        ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
        for clock, grp, c, ls, lw, lab in (("ins", side, col, "-", 2, "the insider, from his trade"),
                                           ("pub", side, light, "-", 2, "an outsider, from the filing"),
                                           ("pub", f"{side}, equal-weighted", light, (0, (4, 2)), 1.5,
                                            "an outsider, against the average stock")):
            p = P[(P.clock == clock) & (P.group == grp) & (P.day >= 0)].sort_values("day")
            ax.plot(p.day, p["mean"] * 100, color=c, lw=lw, ls=ls, label=lab, zorder=3 if clock == "ins" else 2)
        month_ticks(ax, 0, POST)
        ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="upper left")
        title(ax, "After insider buys" if side == "buy" else "After insider sales")
    lo, hi = axes[0].get_ylim()
    axes[0].set_ylim(lo, hi + 0.36 * (hi - lo))      # room for the legend above the lines
    axes[0].set_ylabel("stock return minus SPY since day 0, %", fontsize=8, color=MUTED)
    fig.subplots_adjust(wspace=0.06)
    save(fig, "clocks.png")


def bars_ci(ax, x, r, col, width=0.38, label=None, fmt="{:+.0f}"):
    ax.bar(x, r.mean_bps, width, color=col, label=label, zorder=3)
    ax.errorbar(x, r.mean_bps, yerr=1.96 * r.se_bps, fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5, zorder=4)
    for xi, m, s in zip(x, r.mean_bps, r.se_bps):
        if np.isnan(m):
            continue
        off = 1.96 * (0 if np.isnan(s) else s)
        ax.annotate(fmt.format(m), (xi, m + off if m >= 0 else m - off), xytext=(0, 3 if m >= 0 else -9),
                    textcoords="offset points", ha="center", fontsize=7.5, color=INK)


def fig_horizons(R):
    fig, axes = new(9.8, 4.5, ncols=2, sharey=True)      # same scale on purpose: the sales side is flat
    x = np.arange(len(HORIZONS))
    lo = hi = 0.0
    for ax, side, col in zip(axes, ("buy", "sell"), (BLUE, ORANGE)):
        style(ax)
        for k, (clock, alpha, lab) in enumerate((("insider", 1.0, "the insider, from his trade"),
                                                  ("public", 0.45, "an outsider, from the filing"))):
            r = pick(R, "horizons", clock=clock, side=side).set_index("horizon").loc[HORIZONS]
            xs = x + (k - 0.5) * 0.4
            ax.bar(xs, r.mean_bps, 0.38, color=col, alpha=alpha, label=lab, zorder=3)
            ax.errorbar(xs, r.mean_bps, yerr=1.96 * r.se_bps, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2,
                        zorder=4)
            near = r[r.index <= 126]
            lo = min(lo, (near.mean_bps - 1.96 * near.se_bps).min())
            hi = max(hi, (near.mean_bps + 1.96 * near.se_bps).max())
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([HLAB[h].replace(" months", " m").replace(" month", " m") for h in HORIZONS])
        ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="upper left")
        title(ax, "After insider buys" if side == "buy" else "After insider sales")
    # the scale fits the intervals up to 6 months. At 12 and 24 months they are several times wider and run out of
    # the panel, with them inside the first week would be a few pixels high
    axes[0].set_ylim(lo * 1.15, hi * 1.45)
    axes[0].set_ylabel("stock return minus SPY, basis points", fontsize=8, color=MUTED)
    fig.subplots_adjust(wspace=0.06)
    save(fig, "horizons.png")


def fig_growth():
    """buys: $1 next to the insider, $1 the day after the filing, $1 in SPY. Held 3 months, and held a week.
    Each panel has its own scale: on the scale of the week the 3-month lines sit in the bottom third"""
    fig, axes = new(10.4, 4.4, ncols=2)
    for ax, hold in zip(axes, GROWTH_H):
        style(ax)
        ends = []
        for clock, col, lab in (("insider", BLUE, "at the insider's close"), ("public", LIGHT_BLUE, "after the filing")):
            w = growth(C[(C.side == "buy") & (C.hold == hold) & (C.clock == clock)])
            ax.plot(w.date, w.port, color=col, lw=2)
            years = len(w) / 252
            v = w.port.iloc[-1]
            # the label has the colour of its line, or nobody can tell which line is which
            ends.append([v, f"{lab}\n${v:,.0f}, {v ** (1 / years) - 1:.1%}" if v >= 100 else
                         f"{lab}\n${v:,.2f}, {v ** (1 / years) - 1:.1%}", MID_BLUE if col == LIGHT_BLUE else col])
        ax.plot(w.date, w.spy, color=MUTED, lw=1.3)
        ends.append([w.spy.iloc[-1], f"SPY\n${w.spy.iloc[-1]:,.2f}, {w.spy.iloc[-1] ** (1 / years) - 1:.1%}", MUTED])
        ax.set_yscale("log")
        ticks = [1, 10, 100, 1000] if max(e[0] for e in ends) > 50 else [1, 2, 5, 10]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"${t:,}" for t in ticks])
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        # labels at the end of the lines, pushed apart (in log space) when the lines end close to each other
        lo, hi = ax.get_ylim()
        gap = (hi / lo) ** 0.095
        ends.sort(key=lambda e: e[0])
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] * gap)
        for y, lab, col in ends:
            ax.annotate(lab, (w.date.iloc[-1], y), xytext=(5, 0), textcoords="offset points", fontsize=7.5,
                        color=col, va="center")
        ax.set_xlim(w.date.iloc[0], w.date.iloc[-1] + pd.Timedelta(days=int(365 * 9.5)))
        ax.set_xticks([pd.Timestamp(y, 1, 1) for y in range(2006, 2027, 5)])
        ax.set_xticklabels([str(y) for y in range(2006, 2027, 5)])
        title(ax, f"Held for {HLAB[hold]}")
    fig.subplots_adjust(wspace=0.12)
    save(fig, "growth.png")


def fig_timeline(R):
    """buys only: the same average stock, cut into the stretches before and after the filing"""
    t = pick(R, "timing", side="buy").set_index("group")
    h = pick(R, "horizons", side="buy", clock="public").set_index("horizon")
    rows = [("the month\nbefore the trade", t.loc["20 days before the trade"], LIGHT_GREY),
            ("trade to\nfiling day", t.loc["from trade to the close on the filing day"], BLUE),
            ("the day after\nthe filing", t.loc["the day after the filing"], BLUE),
            # the outsider's three all start at his close, so the day is inside the week and the week in the month
            ("outsider:\nfirst day", h.loc[1], LIGHT_BLUE), ("outsider:\nfirst week", h.loc[5], LIGHT_BLUE),
            ("outsider:\nfirst month", h.loc[21], LIGHT_BLUE)]
    fig, ax = new(8.4, 4.0)
    style(ax)
    x = np.arange(len(rows))
    r = pd.DataFrame([v for _, v, _ in rows])
    ax.bar(x, r.mean_bps, 0.62, color=[c for _, _, c in rows], zorder=3)
    ax.errorbar(x, r.mean_bps, yerr=1.96 * r.se_bps, fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5, zorder=4)
    for xi, m, se in zip(x, r.mean_bps, r.se_bps):
        off = 1.96 * se
        ax.annotate(f"{m:+.0f}", (xi, m + off if m >= 0 else m - off), xytext=(0, 3 if m >= 0 else -10),
                    textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _, _ in rows], fontsize=8)
    ax.set_ylabel("stock return minus SPY, basis points", fontsize=8, color=MUTED)
    title(ax, "Insider buys: where the gain is")
    save(fig, "timeline.png")


def fig_roles(R):
    """buys by role, after a week and after 3 months. Each panel has its own scale, the week is the small one"""
    roles = [r for r in ROLES if r != "Other"]          # Other is a handful of events, it only stretches the axis
    fig, axes = new(9.8, 3.9, ncols=2, sharey=True)
    y = np.arange(len(roles))[::-1]
    for ax, h in zip(axes, (5, MAIN_H)):
        style(ax, "x")
        for k, (clock, col, lab) in enumerate((("insider", BLUE, "the insider"),
                                               ("public", LIGHT_BLUE, "an outsider"))):
            r = pick(R, "roles", clock=clock, side="buy", horizon=h).set_index("group").loc[roles]
            ys = y - (k - 0.5) * 0.36
            ax.barh(ys, r.mean_bps, 0.34, color=col, label=lab, zorder=3)
            ax.errorbar(r.mean_bps, ys, xerr=1.96 * r.se_bps, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2,
                        zorder=4)
            if clock == "insider":
                n = r.n if h == 5 else n
                # the number on the insider's bars only, at the end of the interval
                for m, se, yy in zip(r.mean_bps, r.se_bps, ys):
                    ax.annotate(f"{m:+.0f}", (m + 1.96 * se, yy), xytext=(4, 0), textcoords="offset points",
                                fontsize=7.5, color=INK, va="center")
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.margins(x=0.08)
        ax.set_xlabel("stock return minus SPY, basis points", fontsize=8, color=MUTED)
        title(ax, f"After {HLAB[h]}")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{g}\n{int(v):,} buys" for g, v in zip(roles, n)], fontsize=8, color=INK)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "roles.png")


def fig_mixed(R):
    fig, axes = new(9.6, 3.8, ncols=2, sharey=True)
    order = ["alone", "other side in the 30 days before", "other side the same day, this side bigger",
             "other side the same day, other side bigger"]
    for ax, side, col in zip(axes, ("buy", "sell"), (BLUE, ORANGE)):
        style(ax)
        this, other = ("buys", "sales") if side == "buy" else ("sales", "buys")
        # the two same-day groups are the same company-days in both panels
        labs = ["alone", f"{other} in the\n30 days before", f"{other} the same\nday, {this} bigger",
                f"{other} the same\nday, {other} bigger"]
        r = pick(R, "mixed", side=side, horizon=MAIN_H).set_index("group").loc[order]
        x = np.arange(len(r))
        bars_ci(ax, x, r, col, width=0.6)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{l}\n({int(n):,})" for l, n in zip(labs, r.n)], fontsize=7.5)
        title(ax, "Buys" if side == "buy" else "Sales")
    axes[0].set_ylabel(f"stock return minus SPY after {HLAB[MAIN_H]}, bps", fontsize=8, color=MUTED)
    fig.subplots_adjust(wspace=0.06)
    save(fig, "mixed.png")


def hold_lines(ax, a, series, holds):
    """return per year by holding period, one line per portfolio. A filled point: the alpha has a t-value above 2
    (column `tcol` of the portfolio), a hollow one: it has not"""
    x = np.arange(len(holds))
    for key, col, lab, tcol in series:
        r = a.loc[key].loc[holds]
        v, sig = r.per_year.values * 100, (r[tcol].abs() > 2).values
        ax.plot(x, v, color=col, lw=2, label=lab, zorder=3)
        ax.scatter(x[sig], v[sig], s=34, color=col, zorder=4)
        ax.scatter(x[~sig], v[~sig], s=34, facecolor=SURFACE, edgecolor=col, linewidth=1.5, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([HLAB[h].replace(" months", " m").replace(" month", " m") for h in holds])
    ax.set_ylabel("return per year, %", fontsize=8, color=MUTED)
    return x


def level(ax, v, lab, col=MUTED):
    """a benchmark as a flat line with its name at the right end"""
    ax.axhline(v, color=col, lw=1.2, zorder=2)
    ax.annotate(lab, (1, v), xycoords=("axes fraction", "data"), xytext=(4, 0), textcoords="offset points",
                fontsize=7.5, color=MUTED, va="center")


def fig_holds(A):
    """the whole sample: the four portfolios by holding period. The insider's edge is in the first weeks, on both
    sides, and everything else ends at SPY"""
    fig, ax = new(8.4, 3.7)
    style(ax)
    a = A[A.period == ALL].set_index(["side", "clock", "hold"]).sort_index()
    holds = HORIZONS[1:]
    series = ((("buy", "insider"), BLUE, "bought, from the insider's close", "t"),
              (("buy", "public"), LIGHT_BLUE, "bought, from the filing", "t"),
              (("sell", "insider"), ORANGE, "sold, from the insider's close", "t"),
              (("sell", "public"), LIGHT_ORANGE, "sold, from the filing", "t"))
    hold_lines(ax, a, series, holds)
    level(ax, a.spy_per_year.iloc[0] * 100, f"SPY {a.spy_per_year.iloc[0]:.1%}")
    # the numbers where the lines are far from SPY
    for key, hs in ((("buy", "insider"), (5, 21, 63)), (("sell", "insider"), (5,))):
        for h in hs:
            v = a.loc[key].loc[h, "per_year"] * 100
            ax.annotate(f"{v:.1f}%", (holds.index(h), v), xytext=(7, 4 if key[0] == "buy" else -11),
                        textcoords="offset points", fontsize=7.5, color=INK)
    ax.set_ylim(0, None)
    ax.set_xlim(-0.3, len(holds) - 0.55)
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="upper right")
    title(ax, "Return per year by holding period, 2006 to 2026")
    save(fig, "holds.png")


def fig_since(A):
    """the focus period, public clock. Left: $1 in the stocks insiders bought, in the ones they sold, in the
    average S&P 500 stock and in SPY, every stock held for 3 months after the filing. Right: the same two
    portfolios for every holding period, as a return per year"""
    fig, axes = new(11.0, 4.2, ncols=2, gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    style(ax)
    y0, y1 = [(a, b) for name, a, b in PERIODS if name == FOCUS][0]
    ends = []
    for side, col, lab in (("buy", BLUE, "stocks insiders bought"), ("sell", ORANGE, "stocks insiders sold")):
        c = C[(C.side == side) & (C.clock == "public") & (C.hold == MAIN_H) & C.date.dt.year.between(y0, y1)]
        w = growth(c)
        ax.plot(w.date, w.port, color=col, lw=2)
        years = len(w) / 252
        ends.append([w.port.iloc[-1], f"{lab}\n${w.port.iloc[-1]:.2f}, {w.port.iloc[-1] ** (1 / years) - 1:.1%} a year", col])
    ax.plot(w.date, w.ew, color=MUTED, lw=1.4)
    ends.append([w.ew.iloc[-1], f"the average S&P 500 stock\n${w.ew.iloc[-1]:.2f}, {w.ew.iloc[-1] ** (1 / years) - 1:.1%} a year", MUTED])
    ax.plot(w.date, w.spy, color=LIGHT_GREY, lw=1.4)
    ends.append([w.spy.iloc[-1], f"SPY\n${w.spy.iloc[-1]:.2f}, {w.spy.iloc[-1] ** (1 / years) - 1:.1%} a year", MUTED])
    lo, hi = ax.get_ylim()
    gap = 0.115 * (hi - lo)
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, lab, col in ends:
        ax.annotate(lab, (w.date.iloc[-1], y), xytext=(6, 0), textcoords="offset points", fontsize=8, color=col,
                    va="center")
    ax.set_xlim(w.date.iloc[0], w.date.iloc[-1] + pd.Timedelta(days=int(365 * 4.6)))
    ax.set_xticks([pd.Timestamp(y, 1, 1) for y in range(y0, y1 + 1, 2)])
    ax.set_xticklabels([str(y) for y in range(y0, y1 + 1, 2)])
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["$1", "$2", "$3", "$4"])
    title(ax, f"$1 in every stock from the filing, held for {HLAB[MAIN_H]}")

    ax = axes[1]
    style(ax)
    a = A[(A.period == FOCUS) & (A.clock == "public")].set_index(["side", "hold"]).sort_index()
    hold_lines(ax, a, (("buy", BLUE, "bought", "t_ew"), ("sell", ORANGE, "sold", "t_ew")), HORIZONS[1:])
    level(ax, a.ew_per_year.iloc[0] * 100, "average\nstock")
    level(ax, a.spy_per_year.iloc[0] * 100, "SPY", LIGHT_GREY)
    ax.set_ylim(a.loc[["buy", "sell"]].per_year.min() * 100 - 3.5, None)
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="lower right")
    title(ax, "The same for every holding period")
    fig.subplots_adjust(wspace=0.2)
    save(fig, "since.png")


def fig_periods(R):
    """insider buys before and after 2017, the insider next to an outsider, after a week and after 3 months"""
    fig, axes = new(8.6, 3.6, ncols=2)
    x = np.arange(len(PERIODS))
    for ax, h in zip(axes, (5, MAIN_H)):
        style(ax)
        for k, (clock, col, lab) in enumerate((("insider", BLUE, "the insider"),
                                               ("public", LIGHT_BLUE, "an outsider"))):
            r = pd.concat([pick(R, "horizons", name, clock=clock, side="buy", horizon=h) for name, _, _ in PERIODS])
            bars_ci(ax, x + (k - 0.5) * 0.4, r, col, label=lab)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([name for name, _, _ in PERIODS])
        ax.margins(y=0.12)
        title(ax, f"After {HLAB[h]}")
    axes[0].set_ylabel("stock return minus SPY, basis points", fontsize=8, color=MUTED)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="upper right")
    fig.subplots_adjust(wspace=0.22)
    save(fig, "periods.png")


def fig_sample():
    """the sample in two pictures. Left: insider buys per month, they come in bursts when stocks have fallen.
    Right: how long the insider is ahead, trading days from his last trade to the first day an outsider can act"""
    fig, axes = new(10.4, 3.3, ncols=2, gridspec_kw={"width_ratios": [1.9, 1]})
    ax = axes[0]
    style(ax)
    n = E[E.side == "buy"].avail_date.dt.to_period("M").value_counts().sort_index()
    ax.bar(n.index.to_timestamp(how="start") + pd.Timedelta(days=15), n.values, 26, color=BLUE, zorder=3)
    # the three biggest months by name, at least two years apart. When two are close to each other the later
    # one gets its name to the right of the bar
    named = []
    for m in n.sort_values(ascending=False).index:
        if all(abs((m - o).n) >= 24 for o in named):
            named.append(m)
        if len(named) == 3:
            break
    named.sort()
    for i, m in enumerate(named):
        close = i > 0 and (m - named[i - 1]).n < 60
        ax.annotate(m.strftime("%B %Y"), (m.to_timestamp() + pd.Timedelta(days=15), n[m]),
                    xytext=(3 if close else 0, 3), textcoords="offset points", ha="left" if close else "center",
                    fontsize=7.5, color=INK)
    ax.set_xlim(pd.Timestamp(2005, 9, 1), pd.Timestamp(2026, 12, 31))
    ax.set_xticks([pd.Timestamp(y, 1, 1) for y in range(2006, 2027, 4)])
    ax.set_xticklabels([str(y) for y in range(2006, 2027, 4)])
    ax.margins(y=0.12)
    title(ax, "Insider buys per month")

    ax = axes[1]
    style(ax)
    lag = E.lag_days.clip(upper=6)
    share = lag.value_counts(normalize=True).reindex(range(1, 7), fill_value=0) * 100
    ax.bar(share.index, share.values, 0.7, color=[BLUE if i <= 3 else LIGHT_GREY for i in share.index], zorder=3)
    for i, v in share.items():
        ax.annotate(f"{v:.0f}%", (i, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5,
                    color=INK)
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels([str(i) for i in range(1, 6)] + ["6+"])
    ax.set_xlabel("trading days from the trade to the public day", fontsize=8, color=MUTED)
    ax.set_yticks([0, 20, 40])
    ax.set_yticklabels(["0%", "20%", "40%"])
    ax.margins(y=0.12)
    title(ax, "How long the insider is ahead")
    fig.subplots_adjust(wspace=0.16)
    save(fig, "sample.png")


def main():
    R = results()
    A = calendar_alpha()
    fig_sample(); fig_growth(); fig_holds(A); fig_timeline(R); fig_clocks(); fig_roles(R); fig_mixed(R)
    fig_since(A); fig_periods(R)
    fig_horizons(R)      # not in the report

    pd.set_option("display.width", 200)
    for sample, _, _ in SAMPLES:
        for bench in BENCH:
            h = pick(R, "horizons", sample, bench)
            v = h.assign(v=[f"{m:+.0f} ({t:.1f})" for m, t in zip(h.mean_bps, h.t)])
            print(f"{sample}, against {bench}: mean abnormal return in bps (t)\n",
                  v.pivot_table(index=["side", "clock"], columns="horizon", values="v", aggfunc="first").to_string())
    r = pick(R, "roles", horizon=MAIN_H)
    print(f"roles after {MAIN_H} days\n", r[["clock", "side", "group", "n", "mean_bps", "t", "diff_bps", "diff_t"]]
          .round(1).to_string(index=False))
    print("timing\n", pick(R, "timing")[["side", "group", "n", "mean_bps", "t"]].round(1).to_string(index=False))
    cols = ["side", "clock", "hold", "per_year", "spy_per_year", "ew_per_year", "excess_per_year", "alpha_year", "t",
            "over_ew_per_year", "alpha_ew", "t_ew"]
    for sample, _, _ in SAMPLES:
        print(f"calendar-time portfolios, per year, {sample}\n",
              A.loc[A.period == sample, cols].round(3).to_string(index=False))
    # $ per $1M: 1 bp of 1,000,000 is 100
    for clock, who in (("insider", "by the insider on his trade day"), ("public", "by an outsider the day after the filing")):
        b = pick(R, "horizons", clock=clock, side="buy").set_index("horizon").mean_bps
        print(f"an insider buy, in $ per $1M put in {who}: " +
              ", ".join(f"{HLAB[h]} {b[h] * 100:+,.0f}" for h in HORIZONS))


if __name__ == "__main__":
    main()
