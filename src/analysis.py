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
  figures/past.png        the control for momentum: where the stocks insiders buy and sell stood in the index
  figures/benchmarks.png  the main results against SPY, the average stock and stocks with the same past return
  figures/lookalikes.png  2017-2026: $1 in the stocks insiders bought and sold, and in the stocks that had the
                          same past return
  figures/horizons.png    not in the report: the per-trade table as bars

t-values are clustered by calendar month of day 0: events in the same month share the same market moves.
From 3 months up the holding periods of neighbouring months overlap too, so there the standard errors also count
the covariance between months (overlap_lags). The calendar-time alpha is the check that does not depend on this.

  python src/analysis.py
"""

import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.legend_handler import HandlerTuple

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
# benchmark -> prefix of the return columns in events.csv. matched: the S&P 500 members with the same past return
BENCH = {"SPY": "ar", "equal-weighted": "arew", "matched": "arm"}
LIGHT_BLUE = "#9dbfeb"    # BLUE at 45%, the outsider in the buy charts
LIGHT_ORANGE = "#f4b9a1"  # ORANGE at 45%, the outsider in the sale charts
MID_BLUE = "#5c93dd"      # LIGHT_BLUE is too pale to read as text, the labels of the outsider's lines use this
MID_ORANGE = "#f0916b"    # the same for LIGHT_ORANGE
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
    w = pd.DataFrame({"date": g.date.values, "port": (1 + port).cumprod().values,
                      "spy": (1 + g.spy.fillna(0.0)).cumprod().values,
                      "ew": (1 + g.ew.fillna(0.0)).cumprod().values})
    # the control for momentum: the company-days with matched stocks, and the matched stocks
    for c in ("port_m", "ctrl"):
        w[c] = (1 + g[c].fillna(g.spy).fillna(0.0)).cumprod().values
    return w


def calendar_alpha() -> pd.DataFrame:
    """
    One line per portfolio and period. per_year, spy_per_year and ew_per_year are compounded (geometric)
    returns, excess_per_year and over_ew_per_year the growth of the portfolio relative to SPY and to the
    equal-weighted S&P 500, also compounded. alpha, t and beta are from the daily regression on SPY, alpha_ew,
    t_ew and beta_ew from the one on the equal-weighted S&P 500.
    The control for momentum: m_per_year is the portfolio of the company-days that have matched stocks,
    ctrl_per_year the portfolio of those matched stocks, over_ctrl_per_year the first relative to the second,
    alpha_ctrl, t_ctrl and beta_ctrl the daily regression of the first on the second.
    """
    rows = []
    for (side, clock, hold), g in C.groupby(["side", "clock", "hold"]):
        for name, y0, y1 in SAMPLES:
            p = g[g.date.dt.year.between(y0, y1)]
            w = growth(p)
            years = len(w) / 252
            per_year = lambda v: v ** (1 / years) - 1
            e = newey_west_alpha(p.port, p.ew)
            k = newey_west_alpha(p.port_m, p.ctrl)
            rows.append({"side": side, "clock": clock, "hold": hold, "period": name, "avg_held": p.n_held.mean(),
                         "per_year": per_year(w.port.iloc[-1]), "spy_per_year": per_year(w.spy.iloc[-1]),
                         "ew_per_year": per_year(w.ew.iloc[-1]),
                         "excess_per_year": per_year(w.port.iloc[-1] / w.spy.iloc[-1]),
                         "over_ew_per_year": per_year(w.port.iloc[-1] / w.ew.iloc[-1]),
                         **newey_west_alpha(p.port, p.spy),
                         "alpha_ew": e["alpha_year"], "t_ew": e["t"], "beta_ew": e["beta"],
                         "m_per_year": per_year(w.port_m.iloc[-1]), "ctrl_per_year": per_year(w.ctrl.iloc[-1]),
                         "over_ctrl_per_year": per_year(w.port_m.iloc[-1] / w.ctrl.iloc[-1]),
                         "alpha_ctrl": k["alpha_year"], "t_ctrl": k["t"], "beta_ctrl": k["beta"]})
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

# Calibri on windows, its metric clone Carlito on linux: the same widths, so a chart looks the same on both
_fonts = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next((f for f in ("Calibri", "Carlito") if f in _fonts), "DejaVu Sans")
FS, FS_NUM, FS_PANEL, FS_HEAD, FS_SUB = 9.5, 9, 11, 13.5, 10      # text sizes: axes, numbers, panel title, headline, the lines under it
WHISK = dict(fmt="none", ecolor=MUTED, elinewidth=1.0, capsize=0, zorder=4)      # 95% interval, quiet


def style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=FS, length=3)


def new(w=8.4, h=5.0, **kw):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200, **kw)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / name, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def title(ax, t):
    """the title of one panel"""
    ax.set_title(t, loc="left", fontsize=FS_PANEL, color=INK, pad=7)


def head(fig, t, sub, panels=True):
    """
    The headline of a chart and, under it, what is plotted in a sentence or two: the sample, the unit and how to
    read it. With these a chart can be read on its own, without the report around it. panels: the axes have
    titles of their own, leave room for them.
    """
    fig.canvas.draw()
    # flush with the left edge of everything that is drawn (axis names, tick labels), not with the axes
    boxes = [a.get_tightbbox().transformed(fig.transFigure.inverted()) for a in fig.axes]
    left, right = min(b.x0 for b in boxes), max(b.x1 for b in boxes)
    top = max(a.get_position().y1 for a in fig.axes)
    w, h = fig.get_size_inches()
    sub, t = sub.replace("$", r"\$"), t.replace("$", r"\$")      # two $ on a line and matplotlib reads it as a formula

    def too_wide(width):
        probe = fig.text(0, 0, textwrap.fill(sub, width), fontsize=FS_SUB)
        wide = probe.get_window_extent().width > (right - left) * fig.bbox.width
        probe.remove()
        return wide

    # as many letters on a line as fit under the chart, measured
    width = 220
    while too_wide(width):
        width -= 2
    # then as narrow as it gets with the same number of lines: no last line of one word
    n = len(textwrap.wrap(sub, width))
    while len(textwrap.wrap(sub, width - 1)) == n:
        width -= 1
    sub = textwrap.fill(sub, width)
    y = top + (0.36 if panels else 0.12) / h
    fig.text(left, y, sub, fontsize=FS_SUB, color=MUTED, ha="left", va="bottom", linespacing=1.3)
    fig.text(left, y + (0.19 * (sub.count("\n") + 1) + 0.07) / h, t, fontsize=FS_HEAD, color=INK, ha="left",
             va="bottom")


def ylab(ax, t):
    ax.set_ylabel(t, fontsize=FS, color=MUTED)


def xlab(ax, t):
    ax.set_xlabel(t, fontsize=FS, color=MUTED)


def legend(ax, **kw):
    return ax.legend(frameon=False, fontsize=FS, labelcolor=MUTED, **kw)


def legend_below(fig, ax, handles=None, **kw):
    """the names in one row under the chart, for when there is no empty corner inside a panel"""
    fig.canvas.draw()
    low = min(a.get_tightbbox().transformed(fig.transFigure.inverted()).y0 for a in fig.axes)
    h, lab = ax.get_legend_handles_labels()
    kw = {"handlelength": 1.4, **kw}
    fig.legend(handles or h, lab, loc="upper center", bbox_to_anchor=(0.5, low - 0.01), ncol=4, frameon=False,
               fontsize=FS, labelcolor=MUTED, columnspacing=2.2, **kw)


def no_ticks(ax):
    """the right panel shares the scale of the left one, the gridlines are enough"""
    ax.tick_params(axis="y", length=0)


def month_ticks(ax, lo, hi):
    lab = {0: "day 0", 63: "3 m", 126: "6 m", 252: "12 m", 378: "18 m", 504: "24 m"}
    ticks = [t for t in lab if lo <= t <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([lab[t] for t in ticks])


def end_labels(ax, x, ends, d, size=FS, log=False):
    """names at the right end of the lines, [value, text, colour] each. Lines that end close to each other get
    their names pushed apart, half the way each, so the names stay around the ends.
    d: the smallest distance between two names, as a share of the height of the panel"""
    lo, hi = ax.get_ylim()
    f, back = (np.log, np.exp) if log else (float, float)
    ends = sorted(ends, key=lambda e: e[0])
    pos = [(f(e[0]) - f(lo)) / (f(hi) - f(lo)) for e in ends]
    for _ in range(100):
        for k in range(1, len(pos)):
            short = d - (pos[k] - pos[k - 1])
            if short > 1e-9:
                pos[k - 1] -= short / 2
                pos[k] += short / 2
    for q, (_, lab, col) in zip(pos, ends):
        # annotation_clip: a name that is pushed over the top of the panel is still drawn
        ax.annotate(lab, (x, back(f(lo) + q * (f(hi) - f(lo)))), xytext=(6, 0), textcoords="offset points",
                    fontsize=size, color=col, va="center", linespacing=1.15, annotation_clip=False)
    # no gridlines through the names
    ax.axvspan(x, x + pd.Timedelta(days=365 * 40), color=SURFACE, lw=0, zorder=1)
    ax.set_ylim(lo, hi)


def room(ax, first, last, years):
    """space to the right of the lines for their names. The axis itself stops at the last day"""
    ax.set_xlim(first, last + pd.Timedelta(days=int(365 * years)))
    ax.spines["bottom"].set_bounds(*mdates.date2num([first, last]))


def fig_sample():
    """the sample in two pictures. Left: insider buys per month, they come in bursts when stocks have fallen.
    Right: how long the insider is ahead, trading days from his last trade to the first day an outsider can act"""
    fig, axes = new(10.4, 3.2, ncols=2, gridspec_kw={"width_ratios": [1.9, 1]})
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
                    fontsize=FS_NUM, color=INK)
    ax.set_xlim(pd.Timestamp(2005, 9, 1), pd.Timestamp(2026, 12, 31))
    ax.set_xticks([pd.Timestamp(y, 1, 1) for y in range(2006, 2027, 4)])
    ax.set_xticklabels([str(y) for y in range(2006, 2027, 4)])
    ax.margins(y=0.12)
    ylab(ax, "buys per month")
    title(ax, "Insider buys per month")

    ax = axes[1]
    style(ax)
    lag = E.lag_days.clip(upper=6)
    share = lag.value_counts(normalize=True).reindex(range(1, 7), fill_value=0) * 100
    ax.bar(share.index, share.values, 0.7, color=[BLUE if i <= 3 else LIGHT_GREY for i in share.index], zorder=3)
    for i, v in share.items():
        ax.annotate(f"{v:.0f}%", (i, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=FS_NUM,
                    color=INK)
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels([str(i) for i in range(1, 6)] + ["6+"])
    xlab(ax, "trading days")
    ylab(ax, "share of all trades")
    ax.set_yticks([0, 20, 40])
    ax.set_yticklabels(["0%", "20%", "40%"])
    ax.margins(y=0.12)
    title(ax, "From the trade to the public day")
    fig.subplots_adjust(wspace=0.2)
    head(fig, "When insiders buy, and how fast their trades are public",
         f"Open-market trades by insiders in S&P 500 companies, 2006 to mid 2026. Left: the {int((E.side == 'buy').sum()):,} "
         f"buys by the month they became public. Right: all {len(E):,} buys and sales by the number of trading days "
         f"from the insider's trade to the first close an outsider can act on, the day after the filing.")
    save(fig, "sample.png")


def fig_growth():
    """buys: $1 next to the insider, $1 the day after the filing, $1 in SPY. Held 3 months, and held a week.
    Each panel has its own scale: on the scale of the week the 3-month lines sit in the bottom third"""
    fig, axes = new(10.4, 3.8, ncols=2)
    for ax, hold in zip(axes, GROWTH_H):
        style(ax)
        ends = []
        for clock, col, lab in (("insider", BLUE, "from the insider's close"), ("public", LIGHT_BLUE, "from the filing")):
            w = growth(C[(C.side == "buy") & (C.hold == hold) & (C.clock == clock)])
            ax.plot(w.date, w.port, color=col, lw=2)
            years = len(w) / 252
            v = w.port.iloc[-1]
            # the label has the colour of its line, or nobody can tell which line is which
            ends.append([v, f"{lab}\n${v:,.0f}, {v ** (1 / years) - 1:.1%} a year" if v >= 100 else
                         f"{lab}\n${v:,.2f}, {v ** (1 / years) - 1:.1%} a year", MID_BLUE if col == LIGHT_BLUE else col])
        ax.plot(w.date, w.spy, color=MUTED, lw=1.3)
        ends.append([w.spy.iloc[-1], f"SPY\n${w.spy.iloc[-1]:,.2f}, {w.spy.iloc[-1] ** (1 / years) - 1:.1%} a year",
                     MUTED])
        ax.set_yscale("log")
        ticks = [1, 10, 100, 1000] if max(e[0] for e in ends) > 50 else [1, 2, 5, 10]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"${t:,}" for t in ticks])
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.tick_params(which="minor", length=0)
        end_labels(ax, w.date.iloc[-1], ends, 0.12, log=True)
        room(ax, w.date.iloc[0], w.date.iloc[-1], 10.5)
        ax.set_xticks([pd.Timestamp(y, 1, 1) for y in range(2006, 2027, 5)])
        ax.set_xticklabels([str(y) for y in range(2006, 2027, 5)])
        title(ax, f"Every stock held for {HLAB[hold]}")
    fig.subplots_adjust(wspace=0.12)
    head(fig, "A dollar that follows every insider buy",
         "$1 in a portfolio of every S&P 500 stock an insider bought, 2006 to 2026, equal weight: bought at the close "
         "on the insider's own trade day, or at the first close after the filing. Next to $1 in SPY. Log scale, no "
         "trading costs.")
    save(fig, "growth.png")


def hold_lines(ax, a, series, holds, short=False):
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
    ax.set_xticklabels([HLAB[h].replace(" months", " m").replace(" month", " m") if short else HLAB[h] for h in holds])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    return x


def level(ax, v, lab, col=MUTED):
    """a benchmark as a flat line with its name at the right end"""
    ax.axhline(v, color=col, lw=1.2, zorder=2)
    ax.annotate(lab, (1, v), xycoords=("axes fraction", "data"), xytext=(4, 0), textcoords="offset points",
                fontsize=FS_NUM, color=MUTED, va="center", linespacing=1.1)


def fig_holds(A):
    """the whole sample: the four portfolios by holding period. The insider's edge is in the first weeks, on both
    sides, and everything else ends at SPY"""
    fig, ax = new(8.4, 3.5)
    style(ax)
    a = A[A.period == ALL].set_index(["side", "clock", "hold"]).sort_index()
    holds = HORIZONS[1:]
    series = ((("buy", "insider"), BLUE, "stocks insiders bought, from the insider's close", "t"),
              (("buy", "public"), LIGHT_BLUE, "stocks insiders bought, from the filing", "t"),
              (("sell", "insider"), ORANGE, "stocks insiders sold, from the insider's close", "t"),
              (("sell", "public"), LIGHT_ORANGE, "stocks insiders sold, from the filing", "t"))
    hold_lines(ax, a, series, holds)
    level(ax, a.spy_per_year.iloc[0] * 100, f"SPY {a.spy_per_year.iloc[0]:.1%}")
    # the numbers where the lines are far from SPY
    for key, hs in ((("buy", "insider"), (5, 21, 63)), (("sell", "insider"), (5,))):
        for h in hs:
            v = a.loc[key].loc[h, "per_year"] * 100
            ax.annotate(f"{v:.1f}%", (holds.index(h), v), xytext=(7, 4 if key[0] == "buy" else -12),
                        textcoords="offset points", fontsize=FS_NUM, color=INK)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.set_xlim(-0.3, len(holds) - 0.55)
    xlab(ax, "how long every stock is held")
    ylab(ax, "return per year")
    legend(ax, loc="upper right")
    head(fig, "The insider's edge is gone within three months",
         "Return per year of four portfolios, 2006 to 2026: every S&P 500 stock insiders bought, or sold, held from "
         "the close on the insider's trade day or from the first close after the filing. Equal weight. A filled "
         "point is different from SPY (t-value of the alpha above 2).", panels=False)
    save(fig, "holds.png")


def fig_timeline(R):
    """buys only: the same average stock, cut into the stretches before and after the filing"""
    t = pick(R, "timing", side="buy").set_index("group")
    h = pick(R, "horizons", side="buy", clock="public").set_index("horizon")
    rows = [("the month\nbefore the trade", t.loc["20 days before the trade"], LIGHT_GREY),
            ("from the trade to\nthe filing day", t.loc["from trade to the close on the filing day"], BLUE),
            ("the day after\nthe filing", t.loc["the day after the filing"], BLUE),
            # the outsider's three all start at his close, so the day is inside the week and the week in the month
            ("an outsider's\nfirst day", h.loc[1], LIGHT_BLUE), ("an outsider's\nfirst week", h.loc[5], LIGHT_BLUE),
            ("an outsider's\nfirst month", h.loc[21], LIGHT_BLUE)]
    fig, ax = new(8.4, 3.2)
    style(ax)
    x = np.arange(len(rows))
    r = pd.DataFrame([v for _, v, _ in rows])
    ax.bar(x, r.mean_bps, 0.56, color=[c for _, _, c in rows], zorder=3)
    ax.errorbar(x, r.mean_bps, yerr=1.96 * r.se_bps, **WHISK)
    for xi, m, se in zip(x, r.mean_bps, r.se_bps):
        off = 1.96 * se
        ax.annotate(f"{m:+.0f}", (xi, m + off if m >= 0 else m - off), xytext=(0, 3 if m >= 0 else -11),
                    textcoords="offset points", ha="center", fontsize=FS_NUM, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _, _ in rows])
    ax.margins(y=0.1)
    ylab(ax, "stock minus SPY, basis points")
    head(fig, "Insider buys: where the gain is",
         "Average return of the stock minus SPY in each stretch around an insider buy, in basis points (100 is 1%), "
         "with 95% intervals. S&P 500 companies, 2006 to 2026. An outsider buys at the close of the day after the "
         "filing, and his three bars are all counted from there.", panels=False)
    save(fig, "timeline.png")


def fig_clocks():
    """day by day for two years. Colour says who (the insider or an outsider), dashed is the outsider measured
    against the average S&P 500 stock instead of SPY: the slow drift at the end is the benchmark"""
    fig, axes = new(10.0, 3.3, ncols=2, sharey=True)
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
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.0f}%" if v else "0%")
        title(ax, "After insider buys" if side == "buy" else "After insider sales")
    no_ticks(axes[1])
    ylab(axes[0], "stock minus SPY since day 0")
    xlab(axes[0], "months since day 0")
    xlab(axes[1], "months since day 0")
    fig.subplots_adjust(wspace=0.06)
    # the names once under both panels, each with the blue line of the buys and the orange one of the sales
    pairs = list(zip(axes[0].get_legend_handles_labels()[0], axes[1].get_legend_handles_labels()[0]))
    legend_below(fig, axes[0], pairs, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)}, handlelength=4.2)
    head(fig, "Two years after an insider trade",
         "Average return of the stock minus SPY, day by day from day 0: the insider's trade day for the insider, the "
         "first close after the filing for an outsider. Dashed: against the average S&P 500 stock instead of SPY. "
         "S&P 500 companies, 2006 to 2026.")
    save(fig, "clocks.png")


def bars_ci(ax, x, r, col, width=0.38, label=None, fmt="{:+.0f}"):
    ax.bar(x, r.mean_bps, width, color=col, label=label, zorder=3)
    ax.errorbar(x, r.mean_bps, yerr=1.96 * r.se_bps, **WHISK)
    for xi, m, s in zip(x, r.mean_bps, r.se_bps):
        if np.isnan(m):
            continue
        off = 1.96 * (0 if np.isnan(s) else s)
        ax.annotate(fmt.format(m), (xi, m + off if m >= 0 else m - off), xytext=(0, 3 if m >= 0 else -11),
                    textcoords="offset points", ha="center", fontsize=FS_NUM, color=INK)


def fig_roles(R):
    """buys by role, after a week and after 3 months. Each panel has its own scale, the week is the small one"""
    roles = [r for r in ROLES if r != "Other"]          # Other is a handful of events, it only stretches the axis
    fig, axes = new(9.8, 3.4, ncols=2, sharey=True)
    y = np.arange(len(roles))[::-1]
    for ax, h in zip(axes, (5, MAIN_H)):
        style(ax, "x")
        for k, (clock, col, lab) in enumerate((("insider", BLUE, "the insider, from his trade"),
                                               ("public", LIGHT_BLUE, "an outsider, from the filing"))):
            r = pick(R, "roles", clock=clock, side="buy", horizon=h).set_index("group").loc[roles]
            ys = y - (k - 0.5) * 0.34
            ax.barh(ys, r.mean_bps, 0.3, color=col, label=lab, zorder=3)
            ax.errorbar(r.mean_bps, ys, xerr=1.96 * r.se_bps, **WHISK)
            if clock == "insider":
                n = r.n if h == 5 else n
                # the number on the insider's bars only, at the end of the interval
                for m, se, yy in zip(r.mean_bps, r.se_bps, ys):
                    ax.annotate(f"{m:+.0f}", (m + 1.96 * se, yy), xytext=(4, 0), textcoords="offset points",
                                fontsize=FS_NUM, color=INK, va="center")
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.margins(x=0.08)
        ax.tick_params(axis="y", length=0)
        xlab(ax, "stock minus SPY, basis points")
        title(ax, "One week later" if h == 5 else "Three months later")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{g}\n{int(v):,} buys" for g, v in zip(roles, n)], fontsize=FS, color=INK)
    fig.subplots_adjust(wspace=0.08)
    legend_below(fig, axes[1])
    head(fig, "Insider buys by the role of the insider",
         "Average return of the stock minus SPY, in basis points (100 is 1%), with 95% intervals: for the insider "
         "from his trade day, and for an outsider from the first close after the filing. S&P 500 companies, 2006 to "
         "2026. The two panels have different scales.")
    save(fig, "roles.png")


def fig_mixed(R):
    fig, axes = new(9.6, 3.3, ncols=2, sharey=True)
    order = ["alone", "other side in the 30 days before", "other side the same day, this side bigger",
             "other side the same day, other side bigger"]
    for ax, side, col in zip(axes, ("buy", "sell"), (BLUE, ORANGE)):
        style(ax)
        this, other = ("buys", "sales") if side == "buy" else ("sales", "buys")
        # the two same-day groups are the same company-days in both panels
        labs = [f"no {other}\naround", f"{other} in the\n30 days before", f"{other} the same\nday, {this} bigger",
                f"{other} the same\nday, {other} bigger"]
        r = pick(R, "mixed", side=side, horizon=MAIN_H).set_index("group").loc[order]
        x = np.arange(len(r))
        bars_ci(ax, x, r, col, width=0.5)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{l}\n({int(n):,})" for l, n in zip(labs, r.n)], fontsize=FS_NUM)
        ax.margins(y=0.1)
        title(ax, "Days with insider buys" if side == "buy" else "Days with insider sales")
    ylab(axes[0], f"stock minus SPY after {HLAB[MAIN_H]}, basis points")
    no_ticks(axes[1])
    fig.subplots_adjust(wspace=0.06)
    head(fig, "Does it matter what the other insiders of the company did?",
         "Average return of the stock minus SPY three months after the filing, in basis points (100 is 1%), with "
         "95% intervals. Days with insider buys and days with insider sales, split by what the other insiders of the "
         "same company did. In brackets the number of days, bigger means in dollars. S&P 500 companies, 2006 to 2026.")
    save(fig, "mixed.png")


def fig_since(A):
    """the focus period, public clock. Left: $1 in the stocks insiders bought, in the ones they sold, in the
    average S&P 500 stock and in SPY, every stock held for 3 months after the filing. Right: the same two
    portfolios for every holding period, as a return per year"""
    fig, axes = new(11.0, 4.0, ncols=2, gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    style(ax)
    y0, y1 = [(a, b) for name, a, b in PERIODS if name == FOCUS][0]
    ends = []
    for side, col, lab in (("buy", BLUE, "stocks insiders bought"), ("sell", ORANGE, "stocks insiders sold")):
        c = C[(C.side == side) & (C.clock == "public") & (C.hold == MAIN_H) & C.date.dt.year.between(y0, y1)]
        w = growth(c)
        ax.plot(w.date, w.port, color=col, lw=2)
        years = len(w) / 252
        per_year = lambda v: v ** (1 / years) - 1
        ends.append([w.port.iloc[-1], f"{lab}\n${w.port.iloc[-1]:.2f}, {per_year(w.port.iloc[-1]):.1%} a year", col])
    ax.plot(w.date, w.ew, color=MUTED, lw=1.4)
    ends.append([w.ew.iloc[-1], f"the average S&P 500 stock\n${w.ew.iloc[-1]:.2f}, {per_year(w.ew.iloc[-1]):.1%} a year",
                 MUTED])
    ax.plot(w.date, w.spy, color=LIGHT_GREY, lw=1.4)
    ends.append([w.spy.iloc[-1], f"SPY\n${w.spy.iloc[-1]:.2f}, {per_year(w.spy.iloc[-1]):.1%} a year", MUTED])
    end_labels(ax, w.date.iloc[-1], ends, 0.125)
    room(ax, w.date.iloc[0], w.date.iloc[-1], 4.6)
    ax.set_xticks([pd.Timestamp(y, 1, 1) for y in range(y0, y1 + 1, 2)])
    ax.set_xticklabels([str(y) for y in range(y0, y1 + 1, 2)])
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["$1", "$2", "$3", "$4"])
    title(ax, f"$1, every stock held for {HLAB[MAIN_H]}")

    ax = axes[1]
    style(ax)
    a = A[(A.period == FOCUS) & (A.clock == "public")].set_index(["side", "hold"]).sort_index()
    hold_lines(ax, a, (("buy", BLUE, "stocks insiders bought", "t_ew"), ("sell", ORANGE, "stocks insiders sold", "t_ew")),
               HORIZONS[1:], short=True)
    level(ax, a.ew_per_year.iloc[0] * 100, "average\nstock")
    level(ax, a.spy_per_year.iloc[0] * 100, "SPY", LIGHT_GREY)
    ax.set_ylim(a.loc[["buy", "sell"]].per_year.min() * 100 - 4.5, None)
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    xlab(ax, "how long every stock is held")
    legend(ax, loc="lower right")
    title(ax, "Return per year by holding period")
    fig.subplots_adjust(wspace=0.16)
    head(fig, f"After the filing, {y0} to {y1}: the stocks insiders sold did better than the ones they bought",
         "Two portfolios: every S&P 500 stock insiders bought, and every stock they sold, held from the first close "
         "after the filing, equal weight. Next to the average S&P 500 stock and SPY. In the right panel a filled "
         "point is different from the average stock (t-value of the alpha above 2).")
    save(fig, "since.png")


def fig_periods(R):
    """insider buys before and after 2017, the insider next to an outsider, after a week and after 3 months"""
    fig, axes = new(8.6, 3.1, ncols=2)
    x = np.arange(len(PERIODS))
    for ax, h in zip(axes, (5, MAIN_H)):
        style(ax)
        for k, (clock, col, lab) in enumerate((("insider", BLUE, "the insider, from his trade"),
                                               ("public", LIGHT_BLUE, "an outsider, from the filing"))):
            r = pd.concat([pick(R, "horizons", name, clock=clock, side="buy", horizon=h) for name, _, _ in PERIODS])
            bars_ci(ax, x + (k - 0.5) * 0.38, r, col, width=0.34, label=lab)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace("-", " to ") for name, _, _ in PERIODS])
        ax.margins(y=0.14)
        title(ax, "One week later" if h == 5 else "Three months later")
    ylab(axes[0], "stock minus SPY, basis points")
    fig.subplots_adjust(wspace=0.2)
    legend_below(fig, axes[1])
    head(fig, "Insider buys before and after 2017",
         "Average return of the stock minus SPY after an insider buy, in basis points (100 is 1%), with 95% "
         "intervals: for the insider from his trade day, and for an outsider from the first close after the filing. "
         "S&P 500 companies. The two panels have different scales.")
    save(fig, "periods.png")


def fig_past():
    """the control for momentum, the starting point: where the stocks stood on the public day. Share of the
    company-days with buys, and with sales, in each fifth of past return among the S&P 500 members"""
    fig, axes = new(8.6, 2.7, ncols=2, sharey=True)
    clean = D[~D.mixed_day]
    x = np.arange(5)
    for ax, col, name in zip(axes, ("mom12_pub", "mom1_pub"), ("the past year", "the past month")):
        style(ax)
        ax.axhline(20, color=MUTED, lw=1.0, zorder=2)        # where every bar would be if past return did not matter
        for k, (side, colr, lab) in enumerate((("buy", BLUE, "days with insider buys"),
                                               ("sell", ORANGE, "days with insider sales"))):
            v = clean.loc[clean.side == side, col].dropna()
            share = v.value_counts(normalize=True).reindex(range(1, 6), fill_value=0).values * 100
            xs = x + (k - 0.5) * 0.38
            ax.bar(xs, share, 0.34, color=colr, label=lab, zorder=3)
            for i in (0, 4):      # the numbers on the two ends only
                ax.annotate(f"{share[i]:.0f}%", (xs[i], share[i]), xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=FS_NUM, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(["fell most\n(lowest fifth)", "2", "3", "4", "rose most\n(highest fifth)"])
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.margins(y=0.14)
        xlab(ax, f"S&P 500 members ranked by return over {name}")
        title(ax, f"By return over {name}")
    ylab(axes[0], "share of the days")
    axes[1].annotate("20%: no tilt", (1, 20), xycoords=("axes fraction", "data"), xytext=(4, 0),
                     textcoords="offset points", fontsize=FS_NUM, color=MUTED, va="center")
    legend(axes[1], loc="upper center")
    no_ticks(axes[1])
    fig.subplots_adjust(wspace=0.06)
    head(fig, "Insiders buy stocks that have fallen and sell stocks that have risen",
         "Every day the S&P 500 members are sorted into fifths by past return. The bars show in which fifth the "
         "stock was on the days with insider buys, and on the days with insider sales, 2006 to 2026. The past year "
         "is counted without its last month.")
    save(fig, "past.png")


def fig_benchmarks(R):
    """the main results once per benchmark: SPY, the average stock, the stocks with the same past return.
    If momentum explained a result, its blue point would sit at zero"""
    rows = [(ALL, "insider", "buy", 5, "the insider, buys, after 1 week"),
            (ALL, "insider", "buy", MAIN_H, "the insider, buys, after 3 months"),
            (ALL, "public", "buy", 5, "an outsider, buys, after 1 week"),
            (ALL, "public", "buy", MAIN_H, "an outsider, buys, after 3 months"),
            (FOCUS, "public", "buy", MAIN_H, "an outsider, buys, after 3 months"),
            (FOCUS, "public", "sell", MAIN_H, "an outsider, sales, after 3 months")]
    benches = (("SPY", LIGHT_GREY, "minus SPY"), ("equal-weighted", MUTED, "minus the average S&P 500 stock"),
               ("matched", BLUE, "minus S&P 500 stocks with the same past return"))
    fig, ax = new(8.0, 3.3)
    style(ax, "x")
    # one line per result, a gap between the whole sample and the last ten years
    y = np.array([-(i + (0.9 if sample == FOCUS else 0)) for i, (sample, *_) in enumerate(rows)])
    for k, (bench, col, lab) in enumerate(benches):
        r = pd.concat([pick(R, "horizons", sample, bench, clock=clock, side=side, horizon=h)
                       for sample, clock, side, h, _ in rows])
        ys = y + (1 - k) * 0.24
        ax.errorbar(r.mean_bps, ys, xerr=1.96 * r.se_bps, fmt="o", color=col, ecolor=col, ms=4.5, elinewidth=1.3,
                    label=lab, zorder=3 + k)
        if bench == "matched":
            for m, se, yy in zip(r.mean_bps, r.se_bps, ys):
                ax.annotate(f"{m:+.0f}", (m + 1.96 * se, yy), xytext=(5, 0), textcoords="offset points",
                            fontsize=FS_NUM, color=INK, va="center")
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for *_, lab in rows], fontsize=FS, color=INK)
    ax.tick_params(axis="y", length=0)
    for sample, yy in ((ALL, y[0]), (FOCUS, y[4])):
        ax.annotate(sample.replace("-", " to "), (0, yy + 0.62), xycoords=("axes fraction", "data"), xytext=(4, 0),
                    textcoords="offset points", fontsize=FS, color=MUTED, va="center")
    ax.set_ylim(y[-1] - 0.6, y[0] + 0.95)
    xlab(ax, "return of the stock minus the benchmark, basis points")
    legend(ax, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3, columnspacing=1.2, handletextpad=0.3)
    head(fig, "The same results against three benchmarks",
         "Average return of the stock minus a benchmark, in basis points (100 is 1%), with 95% intervals. S&P 500 "
         "companies. The number is the result against the stocks with the same past return: if momentum explained "
         "a result, its blue point would be at zero.")
    save(fig, "benchmarks.png")


def fig_lookalikes():
    """the focus period, public clock, 3 months: $1 in the stocks insiders bought and in the ones they sold,
    and (light) $1 in the stocks that had the same past return on the same days"""
    fig, ax = new(7.6, 3.4)
    style(ax)
    y0, y1 = [(a, b) for name, a, b in PERIODS if name == FOCUS][0]
    ends = []
    for side, col, light, mid, lab in (("buy", BLUE, LIGHT_BLUE, MID_BLUE, "bought"),
                                       ("sell", ORANGE, LIGHT_ORANGE, MID_ORANGE, "sold")):
        c = C[(C.side == side) & (C.clock == "public") & (C.hold == MAIN_H) & C.date.dt.year.between(y0, y1)]
        w = growth(c)
        years = len(w) / 252
        ax.plot(w.date, w.ctrl, color=light, lw=1.8, zorder=2)
        ax.plot(w.date, w.port_m, color=col, lw=2, zorder=3)
        # the names with the return per year only, the dollars are on the axis
        for v, name, c_ in ((w.port_m.iloc[-1], f"stocks insiders {lab}", col),
                            (w.ctrl.iloc[-1], f"same past return as the {lab}", mid)):
            ends.append([v, f"{name}: {v ** (1 / years) - 1:.1%} a year", c_])
    end_labels(ax, w.date.iloc[-1], ends, 0.075)
    room(ax, w.date.iloc[0], w.date.iloc[-1], 5.6)
    ax.set_xticks([pd.Timestamp(y, 1, 1) for y in range(y0, y1 + 1, 2)])
    ax.set_xticklabels([str(y) for y in range(y0, y1 + 1, 2)])
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["$1", "$2", "$3", "$4"])
    head(fig, f"{y0} to {y1}: past return does not explain the gap",
         "$1 in every S&P 500 stock insiders bought, and $1 in every stock they sold, from the first close after "
         "the filing. Every stock held for 3 months, equal weight. Light lines: the same $1 in the S&P 500 stocks "
         "that had the same past return on the same days.", panels=False)
    save(fig, "lookalikes.png")


def fig_horizons(R):
    """not in the report: the per-trade table as bars"""
    fig, axes = new(9.8, 4.0, ncols=2, sharey=True)      # same scale on purpose: the sales side is flat
    x = np.arange(len(HORIZONS))
    lo = hi = 0.0
    for ax, side, col, light in zip(axes, ("buy", "sell"), (BLUE, ORANGE), (LIGHT_BLUE, LIGHT_ORANGE)):
        style(ax)
        for k, (clock, c, lab) in enumerate((("insider", col, "the insider, from his trade"),
                                             ("public", light, "an outsider, from the filing"))):
            r = pick(R, "horizons", clock=clock, side=side).set_index("horizon").loc[HORIZONS]
            xs = x + (k - 0.5) * 0.4
            ax.bar(xs, r.mean_bps, 0.36, color=c, label=lab, zorder=3)
            ax.errorbar(xs, r.mean_bps, yerr=1.96 * r.se_bps, **WHISK)
            near = r[r.index <= 126]
            lo = min(lo, (near.mean_bps - 1.96 * near.se_bps).min())
            hi = max(hi, (near.mean_bps + 1.96 * near.se_bps).max())
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([HLAB[h].replace(" months", " m").replace(" month", " m") for h in HORIZONS])
        legend(ax, loc="upper left")
        title(ax, "After insider buys" if side == "buy" else "After insider sales")
    # the scale fits the intervals up to 6 months. At 12 and 24 months they are several times wider and run out of
    # the panel, with them inside the first week would be a few pixels high
    axes[0].set_ylim(lo * 1.15, hi * 1.45)
    ylab(axes[0], "stock minus SPY, basis points")
    no_ticks(axes[1])
    fig.subplots_adjust(wspace=0.06)
    head(fig, "After an insider trade, by holding period",
         "Average return of the stock minus SPY, in basis points (100 is 1%), with 95% intervals, from the insider's "
         "trade day and for an outsider from the first close after the filing. S&P 500 companies, 2006 to 2026. "
         "The intervals at 12 and 24 months run out of the panel.")
    save(fig, "horizons.png")


def main():
    R = results()
    A = calendar_alpha()
    fig_sample(); fig_growth(); fig_holds(A); fig_timeline(R); fig_clocks(); fig_roles(R); fig_mixed(R)
    fig_since(A); fig_periods(R); fig_past(); fig_benchmarks(R); fig_lookalikes()
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
            "over_ew_per_year", "alpha_ew", "t_ew", "m_per_year", "ctrl_per_year", "over_ctrl_per_year", "alpha_ctrl",
            "t_ctrl"]
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
