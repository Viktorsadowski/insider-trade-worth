#!/usr/bin/env python3
"""
Builds report/insider_trade_worth.pdf from the text below + the figures in figures/.

Numbers in the text are from the October 2026 run (filings 2006 Q1 to 2026 Q2, prices to 1 October 2026), written
by hand. The tables and figures are read from data/ and figures/, so they follow the data if it changes.

  python src/analysis.py
  python report/build_report.py
"""

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = ROOT / "figures"
DATA = ROOT / "data"
OUT = HERE / "insider_trade_worth.pdf"
TITLE = "What is an insider trade worth?"

# windows: Times New Roman + Calibri, linux: their metric clones Liberation Serif + Carlito (same as nba-moneyball)
F = "/usr/share/fonts/truetype"
FONTS = {"Serif": [r"C:\Windows\Fonts\times.ttf", f"{F}/liberation/LiberationSerif-Regular.ttf"],
         "Serif-Bold": [r"C:\Windows\Fonts\timesbd.ttf", f"{F}/liberation/LiberationSerif-Bold.ttf"],
         "Sans": [r"C:\Windows\Fonts\calibri.ttf", f"{F}/crosextra/Carlito-Regular.ttf"],
         "Sans-Bold": [r"C:\Windows\Fonts\calibrib.ttf", f"{F}/crosextra/Carlito-Bold.ttf"]}
for name, paths in FONTS.items():
    path = next((p for p in paths if Path(p).exists()), None)
    if path is None:
        raise SystemExit(f"no font found for {name}, tried {paths}")
    pdfmetrics.registerFont(TTFont(name, path))

INK = colors.HexColor("#0b0b0b")
MUTED = colors.HexColor("#52514e")
GRID = colors.HexColor("#e6e5e0")
PANEL = colors.HexColor("#f4f3ef")

W, H = A4
MARGIN = 2.1 * cm
TEXT_W = W - 2 * MARGIN

body = ParagraphStyle("body", fontName="Serif", fontSize=10.3, leading=14.2, textColor=INK, alignment=TA_JUSTIFY,
                      spaceAfter=6)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=3)
h1 = ParagraphStyle("h1", fontName="Sans-Bold", fontSize=14, leading=18, textColor=INK, spaceBefore=14, spaceAfter=6)
h2 = ParagraphStyle("h2", fontName="Sans-Bold", fontSize=11.2, leading=14, textColor=INK, spaceBefore=9,
                    spaceAfter=4)
title = ParagraphStyle("title", fontName="Sans-Bold", fontSize=24, leading=28, textColor=INK, alignment=TA_LEFT)
subtitle = ParagraphStyle("subtitle", fontName="Sans", fontSize=13, leading=17, textColor=MUTED, spaceBefore=4)
byline = ParagraphStyle("byline", fontName="Sans", fontSize=10, leading=13, textColor=MUTED, spaceBefore=10)
caption = ParagraphStyle("caption", fontName="Sans", fontSize=8.6, leading=11, textColor=MUTED, spaceBefore=3,
                         spaceAfter=10)
abstract = ParagraphStyle("abstract", parent=body, fontSize=10, leading=13.8)
cell = ParagraphStyle("cell", fontName="Sans", fontSize=8.6, leading=10.4, textColor=INK)
cell_head = ParagraphStyle("cell_head", parent=cell, fontName="Sans-Bold", textColor=MUTED)
mono = ParagraphStyle("mono", fontName="Courier", fontSize=8.2, leading=10.4, textColor=INK, leftIndent=8)


def P(t, s=body):
    return Paragraph(t.replace("S&P", "S&amp;P"), s)


def bullets(items):
    return [Paragraph(t, bullet, bulletText="•") for t in items]


def figure(path, cap, width=TEXT_W):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    img = Image(str(path), width=width, height=width * h / w)
    return KeepTogether([img, P(cap, caption)])


def table(rows, widths, cap=None):
    data = [[P(str(c), cell_head) for c in rows[0]]] + [[P(str(c), cell) for c in r] for r in rows[1:]]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    st = [("LINEBELOW", (0, 0), (-1, 0), 0.8, MUTED),
          ("LINEBELOW", (0, -1), (-1, -1), 0.5, GRID),
          ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    for i in range(1, len(data)):
        if i % 2 == 0:
            st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f7f6f3")))
    t.setStyle(TableStyle(st))
    parts = [t]
    if cap:
        parts.append(P(cap, caption))
    return KeepTogether(parts)


def box(flow):
    t = Table([[flow]], colWidths=[TEXT_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PANEL), ("LEFTPADDING", (0, 0), (-1, -1), 12),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 12), ("TOPPADDING", (0, 0), (-1, -1), 10),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    return t


def on_page(c, doc):
    c.saveState()
    c.setFont("Sans", 8)
    c.setFillColor(MUTED)
    if doc.page > 1:
        c.drawString(MARGIN, H - 1.25 * cm, TITLE)
        c.drawRightString(W - MARGIN, H - 1.25 * cm, "Viktor Sadowski, October 2026")
    c.drawCentredString(W / 2, 1.2 * cm, str(doc.page))
    c.restoreState()


# ── tables from the data ──────────────────────────────────────────────────────

RES = pd.read_csv(DATA / "results.csv")
ALL, FOCUS = "2006-2026", "2017-2026"


def res(table, sample=ALL, bench="SPY"):
    return RES[(RES.table == table) & (RES["sample"] == sample) & (RES.bench == bench)]


EV = pd.read_csv(DATA / "events.csv", usecols=["role", "side", "value", "owner_cik", "cik"])
COV = pd.read_csv(DATA / "coverage.csv")
CAL = pd.read_csv(DATA / "calendar_alpha.csv")
HORIZONS = [1, 5, 21, 63, 126, 252, 504]
HLAB = ["1 day", "1 week", "1 month", "3 months", "6 months", "12 months", "24 months"]
HOLDS = dict(zip(HORIZONS, HLAB))
ROLES = ["CEO", "CFO", "Other officer", "Director", "10% owner", "Other"]


def money(v):
    return f"${v / 1e6:.1f}m" if v >= 1e6 else f"${v / 1e3:.0f}k"


def bt(mean, t):
    # +64 (0.6). Round first, so a tiny negative number is not printed as -0
    m = round(mean)
    return f"{m:+d} ({t:.1f})" if m else f"0 ({abs(t) if round(t, 1) == 0 else t:.1f})"


def sample_table():
    rows = [["Role", "Buys", "Median buy", "Sales", "Median sale"]]
    g = EV.groupby(["role", "side"]).value.agg(["size", "median"])
    for r in ROLES:
        rows.append([r, f"{int(g.loc[(r, 'buy'), 'size']):,}", money(g.loc[(r, "buy"), "median"]),
                     f"{int(g.loc[(r, 'sell'), 'size']):,}", money(g.loc[(r, "sell"), "median"])])
    n = EV.groupby("side").value.agg(["size", "median"])
    rows.append(["All", f"{int(n.loc['buy', 'size']):,}", money(n.loc["buy", "median"]),
                 f"{int(n.loc['sell', 'size']):,}", money(n.loc["sell", "median"])])
    return rows


def horizon_table():
    rows = [["", ""] + HLAB]
    h = res("horizons")
    for side, sname in (("buy", "Buys"), ("sell", "Sales")):
        for clock, cname in (("insider", "the insider"), ("public", "an outsider")):
            x = h[(h.side == side) & (h.clock == clock)].set_index("horizon").loc[HORIZONS]
            rows.append([sname, cname] + [bt(m, t) for m, t in zip(x.mean_bps, x.t)])
    return rows


def role_table():
    rows = [["", "Events", "Insider: 1 week", "Insider: 3 months", "Outsider: 1 week", "Outsider: 3 months"]]
    r = res("roles")
    r = r[r.side == "buy"]
    for role in ROLES[:-1]:
        x = r[r.group == role].set_index(["clock", "horizon"])
        f = lambda c, h: bt(x.loc[(c, h), "mean_bps"], x.loc[(c, h), "t"])
        rows.append([role, f"{int(x.loc[('insider', 5), 'n']):,}", f("insider", 5), f("insider", 63),
                     f("public", 5), f("public", 63)])
    return rows


def focus_trade_table():
    rows = [["", ""] + HLAB[1:]]
    for side, sname in (("buy", "Buys"), ("sell", "Sales")):
        for bench, bname in (("SPY", "against SPY"), ("equal-weighted", "against the average stock")):
            h = res("horizons", FOCUS, bench)
            x = h[(h.side == side) & (h.clock == "public")].set_index("horizon").loc[HORIZONS[1:]]
            rows.append([sname, bname] + [bt(m, t) for m, t in zip(x.mean_bps, x.t)])
    return rows


def focus_portfolio_table():
    rows = [["", ""] + HLAB[1:]]
    a = CAL[(CAL.period == FOCUS) & (CAL.clock == "public")].set_index(["side", "hold"])
    pt = lambda v, t: f"{v:+.1%} ({t:.1f})"
    for side, sname in (("buy", "Bought"), ("sell", "Sold")):
        x = a.loc[side].loc[HORIZONS[1:]]
        rows.append([sname, "return per year"] + [f"{v:.1%}" for v in x.per_year])
        rows.append([sname, "over the average stock"] + [f"{v:+.1%}" for v in x.over_ew_per_year])
        rows.append([sname, "alpha against it (t)"] + [pt(v, t) for v, t in zip(x.alpha_ew, x.t_ew)])
    x = a.loc["buy minus sell"].loc[HORIZONS[1:]]
    rows.append(["Bought minus sold", "per year (t)"] + [pt(v, t) for v, t in zip(x.alpha_year, x.t)])
    return rows


def period_table():
    rows = [["", "With a price", "Insider: 1 week", "Outsider: 1 week", "Outsider: 3 months",
             "Outsider: 3 months, average stock", "Bought minus sold, 3 months"]]
    COV["period"] = pd.cut(COV.year, [2005, 2016, 2026], labels=["2006-2016", FOCUS])
    c = COV.groupby("period", observed=True)[["events", "with_price"]].sum()
    k = CAL[(CAL.side == "buy minus sell") & (CAL.clock == "public") & (CAL.hold == 63)].set_index("period")
    for g in c.index:
        f = lambda clock, h, bench="SPY": (lambda x: bt(x.mean_bps.iloc[0], x.t.iloc[0]))(
            res("horizons", g, bench).query("side == 'buy' and clock == @clock and horizon == @h"))
        rows.append([g, f"{c.loc[g, 'with_price'] / c.loc[g, 'events']:.0%}", f("insider", 5), f("public", 5),
                     f("public", 63), f("public", 63, "equal-weighted"),
                     f"{k.loc[g, 'alpha_year']:+.1%} ({k.loc[g, 't']:.1f})"])
    return rows


def portfolio_table():
    rows = [["", "", ""] + HLAB[1:]]
    a = CAL[CAL.period == ALL].set_index(["side", "clock", "hold"]).sort_index()
    pt = lambda v, t: f"{v:+.1%} ({t:.1f})"
    for side, sname in (("buy", "Bought"), ("sell", "Sold")):
        for clock, cname in (("insider", "at the insider's close"), ("public", "after the filing")):
            x = a.loc[(side, clock)].loc[HORIZONS[1:]]
            rows.append([sname, cname, "return per year"] + [f"{v:.1%}" for v in x.per_year])
            if side == "buy":
                rows.append([sname, cname, "over SPY"] + [f"{v:+.1%}" for v in x.excess_per_year])
            rows.append([sname, cname, "alpha (t)"] + [pt(v, t) for v, t in zip(x.alpha_year, x.t)])
    return rows


# ── content ───────────────────────────────────────────────────────────────────

s = []
s += [Spacer(1, 1.2 * cm),
      P(TITLE, title),
      P("Buys and sales by insiders in S&P 500 companies, 2006 to 2026: what the insider earns, and what is left "
        "when the filing is public", subtitle),
      P("Viktor Sadowski · October 2026 · code: insider-trade-worth", byline),
      Spacer(1, 0.6 * cm)]

s.append(box([
    P("Summary", ParagraphStyle("abs_h", parent=h2, spaceBefore=0)),
    P("Company insiders have to report their trades in their own stock to the SEC within two business days. This "
      "project takes every open-market buy and sale in an S&P 500 company from 2006 to mid 2026, 179,579 insider "
      "events, of which 168,013 in 774 companies have a price history, and measures the stock on two clocks: from "
      "the insider's own trade, and from the first day after the filing, when anyone can act on it. Holding "
      "periods run from one day to two years, with three months as the main one.", abstract),
    P("Hold every stock for three months after an insider buys it, and roll that from 2006 to 2026. Bought at the "
      "close on the insider's own trade day it returns 13.1% a year. Bought at the close after the filing it "
      "returns 10.4%. SPY returned 11.1%. So at three months the trade is worth 1.8% a year to the insider and "
      "nothing to anyone else, and neither number is different from zero. Per trade he is 1.4% ahead of the "
      "market after three months (t = 2.1), an outsider 0.5% (t = 0.8). What the insider has, he has in the "
      "first days. He buys after the stock has lost 3.4% to the market in a month, and a week later it is 0.75% "
      "ahead (t = 6.3), half of that before the filing and the other half on the day the market reads it. Held "
      "for a week only, the same two portfolios return 41.6% and 11.5% a year.", abstract),
    P("After the filing, in 2017 to 2026 on their own, the sign is the reverse of the textbook. Held for three "
      "months from the filing, the stocks insiders bought returned 8.7% a year and the ones they sold 14.1%, "
      "against 12.4% for the average S&P 500 stock. Neither gap is significant at three months (t = -1.4 and "
      "1.6), so there is no signal to count on in either direction. The clearest numbers are on the sell side: "
      "in the week after the filing, and after sales by directors, the stock beats the average stock (t = 3.3 "
      "and 3.0). The insider's own first week is still there, and what an outsider could earn in 2006-2016 is "
      "gone.", abstract),
]))

# 1
s.append(P("1. The idea", h1))
s.append(P("An insider knows his company better than the market does, and when he buys its stock with his own money "
           "the trade is public two days later. Following insider buys is one of the oldest ideas in investing, and "
           "the research mostly agrees that buys carry information and sales do not. Most of that evidence is from "
           "small companies and from before filings were on the internet within days."))
s.append(P("The question here is narrower, and has two parts. How well do insiders in the largest US companies "
           "invest in their own stock? And how much of that is left for someone who can only read the filing? The "
           "difference between the two is what an insider trade is worth to the insider and not to anyone else."))

# 2
s.append(P("2. Method", h1))
s.append(P("2.1 Data", h2))
s.append(P("The trades are from the SEC's insider transactions data sets, the quarterly files with every Form 3, 4 "
           "and 5 since 2006: 4.46 million filings in 20,760 companies. Kept are open-market buys and sales of "
           "common stock (transaction codes P and S on Form 4), so no option exercises, grants, gifts or tax "
           "withholding. Membership of the S&P 500 is taken day by day from a public history of the index "
           "(fja05680/sp500 on GitHub), and a trade counts if the company was in the index on the day of the "
           "filing. That gives 628,198 trades, which is 31,948 buys and 596,250 sales."))
s.append(P("Trades by the same insider in the same company, reported on the same day and in the same direction, are "
           "one event: 179,579 events, 15,923 buys and 163,656 sales. Prices are daily closes from Yahoo Finance, "
           "and from Tiingo for companies that are no longer listed. A ticker can belong to another company today "
           "than it did in 2008, so a price series is only used for a company in the years where the prices "
           "insiders wrote on their filings follow its close. 168,013 events (94%) have such a series."))
s.append(table(sample_table(), [3.4 * cm, 2.0 * cm, 2.6 * cm, 2.2 * cm, 2.6 * cm],
               "Table 1. Events with a price, by the most senior role of the insider. Roles are read from the "
               "relationship boxes and the title on the filing. Heads of a division who call themselves CEO are "
               "counted as other officers. Insiders sell ten times as often as they buy, since most of their pay "
               "is stock."))
s.append(P("2.2 Two clocks", h2))
s += bullets([
    "The insider's clock starts at the close on the day of his trade (the last one, if the filing has several). "
    "It measures how well he invests.",
    "The public clock starts at the close on the first trading day after the filing. The SEC data has the date of "
    "a filing and not the time, and many arrive after the market has closed, so this is the first price an "
    "outsider can be sure to get. It measures what the information is worth once everybody has it. When several "
    "insiders of a company report the same day it is one signal, so here the unit is company and day."])
s.append(P("For 60% of the events the public clock starts two trading days or less after the trade, for 97% within "
           "three."))
s.append(P("2.3 The measures", h2))
s.append(P("Per trade: abnormal return is the total return of the stock minus the total return of SPY over the "
           "same days, for 1 day, 1 week, 1, 3, 6, 12 and 24 months. It is given in basis points, 100 to a "
           "percent. Events in the same month share the same market, so t-values are clustered by calendar month. "
           "At 12 and 24 months the holding periods of different months overlap as well, the clustering does not "
           "cover that, and the t-values there are too high."))
s.append(P("As a portfolio: every day, hold all stocks with an insider buy (or sale) in the last week, month, three "
           "months and so on, equal weight, and compound the daily returns from 2006 to 2026. This gives one "
           "number per clock, a return per year, and it has no overlapping events, so it is the measure to trust "
           "at the long horizons. The average of single trades is not compounded, on purpose: one stock moves 20% "
           "against the market in three months, and a geometric average per trade mostly measures that. The daily "
           "returns are also regressed on SPY for an alpha and a beta."))
s.append(P("A second benchmark is the average S&P 500 stock: the equal-weighted return of the index members, day "
           "by day. Over the whole sample it returned the same as SPY, 11.1% a year, but not in every part of it: "
           "10.0% against 7.5% in 2006-2016, and 12.4% against 15.3% in 2017-2026, when a few very large "
           "companies carried SPY. For the question whether an insider trade picks a better stock than the next "
           "one in the index, the average stock is the fair comparison, and section 3.6 uses it."))
s.append(P("A stock that stops trading inside the holding period is followed to the end. Of the 773 price series "
           "148 end early. Three of those are fragments that fit no company and are not used. The other 145 each "
           "have a line in a hand-made table: a cash buyout pays the deal price (48 series), a deal paid in "
           "shares is worth the last traded price (94), and when the shares are cancelled in a bankruptcy the "
           "stock ends at zero (3: Chesapeake Energy, Denbury and Alpha Natural Resources). After that the money "
           "sits in SPY for the rest of the period."))
s.append(P("Insiders often buy and sell at the same time. 10% of the days with a public buy also have a public sale "
           "in the same company, and a third have had one in the 30 days before. Those days are flagged and kept: "
           "the main results use days with one direction only, and section 3.5 shows the others."))

# 3
s.append(P("3. Results", h1))
s.append(P("3.1 In one number", h2))
s.append(figure(FIG / "growth.png", "Figure 1. $1 in a portfolio that holds every stock for three months (left) or "
                "a week (right) after an insider buy, equal weight, from the two starting points, and $1 in SPY. "
                "End value and compounded return per year. Log scale. On the 3 to 4% of days when the one-week "
                "portfolio has nothing to hold the money is in SPY. No trading costs.", width=TEXT_W * 0.86))
s.append(table(portfolio_table(), [1.3 * cm, 3.1 * cm, 2.2 * cm] + [1.7 * cm] * 6,
               "Table 2. The portfolios by holding period, 2006 to 1 October 2026. SPY returned 11.1% a year. "
               "Return per year and over SPY are compounded. Alpha is from the daily returns regressed on SPY, "
               "with t-values that allow for autocorrelation. The beta is between 1.12 and 1.16 for the bought "
               "portfolios and between 0.98 and 1.04 for the sold ones."))
s.append(P("A dollar that follows every insider buy for three months, starting at the close on the day of the "
           "trade, grows to $12.74 from 2006 to October 2026: 13.1% a year. Starting at the close after the filing "
           "it grows to $7.78, 10.4% a year, against $8.80 and 11.1% in SPY. So at three months the insider's own "
           "start is worth 1.8% a year over SPY and the outsider's is 0.6% under it, and neither alpha is "
           "different from zero (Table 2). From six months on both clocks end close to SPY."))
s.append(P("The gain sits in the first days, and the one-week portfolio shows how much it is. Held for a week from "
           "the insider's close the dollar grows to $1,347: 41.6% a year, or 27.5% a year over SPY when the two "
           "are compounded against each other as in Table 2. From the close after the filing it grows to $9.58. "
           "The first number is not a strategy, nobody outside can trade at the insider's close and the portfolio "
           "turns over every week before costs. It is the value of knowing about the trade when it is made. For "
           "sales the mirror image is smaller but clear: the stocks insiders sell lag SPY by 4.9% a year in the "
           "week after the sale, with a t-value above 4 on the alpha, and by nothing from the filing on."))

s.append(P("3.2 Per trade", h2))
s.append(figure(FIG / "horizons.png", "Figure 2. Average stock return minus SPY after insider buys and sales, on "
                "the two clocks. Bars are averages, lines 95% intervals. Same scale in both panels.",
                width=TEXT_W * 0.86))
s.append(table(horizon_table(), [1.3 * cm, 2.1 * cm] + [1.9 * cm] * 7,
               "Table 3. Average abnormal return in basis points, t-value in brackets. The insider: 14,562 buys and "
               "153,451 sales. An outsider: 9,747 company-days with buys only and 111,354 with sales only."))
s.append(P("After an insider buys, the stock beats the market by 29 basis points the next day and 75 within a week, "
           "with t-values above 6. It keeps the lead: 143 basis points after three months, still significant "
           "(t = 2.1), and about the same after six months and after two years, where the noise has taken over. "
           "On the public clock the picture is different. The first day gives 7 basis points (t = 2.2) and that "
           "is the last number that can be told from zero. Three months gives +54 with a t-value of 0.8. The "
           "averages stay positive out to two years, but with intervals two to three times their own size, and "
           "the portfolios in Table 2 show that this is the beta of the stocks and not the buys."))
s.append(P("In money: $1m invested with the insider on his trade day is $7,500 ahead of SPY a week later and "
           "$14,300 after three months. $1m invested by an outsider the day after the filing is $800 and $5,400 "
           "ahead, neither different from zero."))
s.append(P("After a sale the stock is 6 basis points behind the market a week later on the insider's clock "
           "(t = -2.7), the same small loss the weekly portfolio shows, and 7 behind after three months, which is "
           "nothing. Most of the 6 comes on the day after the filing (-5, t = -7.2), so the market reads sales too, "
           "and from the close of that day there is nothing at any horizon. Insiders sell for many reasons, they "
           "have to pay taxes and spread their wealth, and it shows."))

s.append(P("3.3 Where the gain is", h2))
s.append(figure(FIG / "timeline.png", "Figure 3. Insider buys, the same average stock in the stretches before and "
                "after the filing. Grey: before the trade. Blue: from the trade to the close on the day after the "
                "filing, where the outsider's clock starts. Light blue: the outsider's holding periods.",
                width=TEXT_W * 0.62))
s.append(P("Insiders buy on weakness. In the month before the trade the stock has lost 3.4% against the market "
           "(t = -8.6), and before a sale it has gained 2.5%. From the trade to the close on the filing day the "
           "stock recovers 34 basis points, before most of the market can have seen the filing. The next day, "
           "the first with the filing out, it gains another 38 (t = 11.8). So the market does react to insider "
           "buys, and it is done within a day."))
s.append(P("This sets the range for an outsider. Buying at the close on the filing day, which is only possible for "
           "the filings that arrive in trading hours, would catch the 38 basis points of the next day. Buying at "
           "the next close gives 7. The data has closing prices only, so it cannot say what buying at the open "
           "in between gives."))
s.append(figure(FIG / "paths.png", "Figure 4. Stock return minus SPY from 20 trading days before to 24 months "
                "after the public day 0, average over company-days. Left of day 0 the line shows where the stock "
                "was against its day-0 price. Dashed: the buys against the average S&P 500 stock instead of "
                "SPY.", width=TEXT_W * 0.7))
s.append(P("The slow rise of the buy line over the two years is not a late payoff. Stocks that insiders buy move "
           "more than the market, with a beta of about 1.15, and SPY has gone up in most years, so they drift "
           "ahead of it for that reason alone. Against the average S&P 500 stock the line is flat for three "
           "months (+7 basis points) and negative after that, and the alpha of the portfolios from the filing is "
           "between -0.5% and -1.2% a year for every holding period from three months up, with t-values around "
           "-0.5 (Table 2)."))

s.append(P("3.4 Roles", h2))
s.append(figure(FIG / "roles.png", "Figure 5. Stock return minus SPY three months after buys and sales, by role, on "
                "both clocks. Number of events under each role. The 84 buys and 1,647 sales by insiders with "
                "only the box Other ticked are left out of the chart.", width=TEXT_W * 0.86))
s.append(table(role_table(), [2.8 * cm, 1.6 * cm, 2.9 * cm, 2.9 * cm, 2.9 * cm, 2.9 * cm],
               "Table 4. Buys by role, average abnormal return in basis points, t-value in brackets. Per insider "
               "event on both clocks, days with sales in the same company included."))
s.append(P("Of the five roles in Table 4 the CEO does best on his own clock: 123 basis points in the first week "
           "(t = 7.1), against 59 to 100 for the others, and the difference to them is significant (t = 3.5). After three months he is "
           "257 ahead (t = 2.6), the only role still clearly above zero. CFOs, other officers and directors all "
           "earn something in the first weeks. So do owners of more than 10% in the first week, but they are "
           "mostly funds and other companies buying large blocks, and after a month nothing can be said about "
           "them."))
s.append(P("On the public clock none of the five has a three-month result that can be told from zero. The CEO is "
           "highest, "
           "+145 basis points with a t-value of 1.6. For sales every role in the chart is within 40 basis "
           "points of zero after three months, but they differ from each other: after sales by CEOs and by "
           "directors the stock does about 40 basis points better than after the rest, and after sales by other "
           "officers about 40 worse (t = 2.3, 2.2 and -3.4). Section 3.6 comes back to this."))
s.append(P("The group with only Other ticked looks like the exception, +525 basis points on the public clock three "
           "months after a buy, "
           "but it is 84 buys in about 30 companies, a fifth of them in 2008 and 2009. They are funds and "
           "foundations with a stake under 10% and heads of subsidiaries. The Gates foundation trust bought "
           "AutoNation nine times in the winter of 2008-2009 and the stock beat SPY by 40% over the next three "
           "months. Without AutoNation the group is at +129."))

s.append(P("3.5 Buying and selling at the same time", h2))
s.append(figure(FIG / "mixed.png", "Figure 6. Public clock, three months. Company-days with buys (left) and sales "
                "(right), split by what the other insiders of the company did. Number of company-days in brackets.",
                width=TEXT_W * 0.85))
s.append(P("Here the split does something. A buy with no insider sale in the company that day or the 30 days "
           "before is followed by +111 basis points over three months, a buy after recent sales by -72. Neither "
           "is different from zero, but the difference between them is, 184 basis points with a t-value of 2.3. "
           "On the 1,077 days when buys and sales in the same company became public together, nothing can be "
           "said. Sales that come after insider buys are followed by a stock that beats the market, +41 after "
           "three months and +109 after six (t = 2.3), so the earlier buy seems to count for more than the later "
           "sale."))
s.append(P("Several insiders buying on the same day looks better than one alone: +41 basis points in the first "
           "week against +3, and +225 against +30 after three months. The differences have t-values of 2.0 and "
           "1.9. These are the best public numbers among the splits of the buy days, and with this many splits "
           "tested a few t-values around 2 are what chance gives. Section 3.6 looks at them again after 2017."))

s.append(P("3.6 After the filing, 2017 to 2026", h2))
s.append(P("The last ten years on their own, and only the outsider's question: is anything left after the public "
           "day, in either direction? Two things are different from the sections above. The sample is the filings "
           "from 2017 on, 78,893 events with a price, 99% of all. And the comparison that matters is the average "
           "S&P 500 stock, since SPY beat it by 2.8% a year in this period."))
s.append(figure(FIG / "since.png", "Figure 7. 2017 to 1 October 2026, public clock. $1 in every stock insiders "
                "bought, and in every stock they sold, from the close after the filing and held for three months, "
                "next to the average S&P 500 stock and SPY.", width=TEXT_W * 0.6))
s.append(table(focus_trade_table(), [1.3 * cm, 4.1 * cm] + [1.9 * cm] * 6,
               "Table 5. 2017 to 2026, public clock, per trade: average abnormal return in basis points, t-value "
               "in brackets. 4,417 company-days with buys only and 52,841 with sales only. The t-values at 12 and "
               "24 months are too high (section 2.3)."))
s.append(table(focus_portfolio_table(), [2.9 * cm, 3.5 * cm] + [1.72 * cm] * 6,
               "Table 6. 2017 to 2026, the portfolios from the close after the filing, by holding period. The "
               "average stock returned 12.4% a year and SPY 15.3%. Bought minus sold: the daily difference "
               "between the two portfolios with the market taken out, per year."))
s.append(P("Buys first. The stocks insiders bought did worse than the average stock for every holding period: "
           "8.7% a year against 12.4% when held for three months, and per trade 59 basis points behind after "
           "three months and 246 after twelve. Against the average stock none of it is significant: the alpha is "
           "-3.9% a year with a t-value of -1.4, and between -0.5 and -1.5 for the other holding periods. But it "
           "is the wrong sign six times out of six."))
s.append(P("Sales are the mirror image. The stocks insiders sold returned 14.1% a year held for three months, "
           "1.5% more than the average stock (alpha +2.4%, t = 1.6), and held for the first week after the filing "
           "5.4% more (alpha +6.5%, t = 3.3). Against SPY the three-month portfolio is level, 1.0% a year behind. "
           "Bought minus sold is negative for every holding period, -4.7% a year at three months (t = -1.2) and "
           "-11.2% at one week (t = -2.0)."))
s.append(P("It is the senior sellers. After sales by CEOs, directors and owners of more than 10% the stock beats "
           "the average stock by 80, 114 and 158 basis points over three months (t = 2.0, 3.0 and 2.0). Sales by "
           "other officers, six out of ten sales, are followed by nothing (-4). For buys no role stands out: "
           "CEOs +74 (t = 0.9), directors -80 (t = -1.6). The splits of section 3.5 point the same way as in the "
           "whole sample and are weaker at three months: several insiders buying together is +70 against -75 "
           "for one alone (difference t = 1.4), and a buy after recent sales is -137 against -25 for a buy "
           "alone (t = -1.0). In the first week several insiders are still 53 basis points ahead of one alone "
           "(t = 2.0)."))
s.append(P("So after the public day, since 2017, a buy is no reason to expect a better stock and a sale no reason "
           "to expect a worse one. If anything it is the other way round. The likely reason is not the insiders "
           "but momentum: they buy stocks that have fallen and sell stocks that have risen (section 3.3), and "
           "what follows may be what follows any stock that has just fallen or risen. This test does not "
           "separate the two. The careful reading is: at three months no usable signal in either direction, and "
           "the sign to expect is the reverse of the textbook."))
s.append(table(period_table(), [1.9 * cm, 1.6 * cm, 2.3 * cm, 2.3 * cm, 2.5 * cm, 3.0 * cm, 3.1 * cm],
               "Table 7. Insider buys before and after 2017. Per trade in basis points with t-values, against "
               "SPY unless it says average stock. Bought minus sold: the three-month portfolios from the filing, "
               "as in Table 6."))
s.append(P("Before 2017 it was different. In 2006-2016 the week after the filing gave 30 basis points (t = 2.7), "
           "and three months gave 178 against SPY and 62 against the average stock. Almost all of the "
           "three-month number is one year: stocks that insiders bought during 2009 beat SPY by 16% over the "
           "next three months, and without 2009 it is 24 basis points. The insider's own first week is still "
           "there, 88 basis points before 2017 and 58 after, both with t-values above 4. Figure 1 shows the "
           "same from the other side: the outsider's line runs ahead of SPY until 2016 and has lost that lead "
           "since."))

# 4
s.append(P("4. What it says", h1))
s.append(P("An insider buy in an S&P 500 company is worth something to the insider, and most of it in the first "
           "days. Per trade he is 1.4% ahead of the market after three months and 0.75% after one week, and the CEO "
           "does better than the rest. As a portfolio the first week is 27.5% a year over SPY, before and after "
           "2017, and three months is 1.8%. To anyone else it is worth close to nothing: the market takes the news "
           "in on the day after the filing, an outsider who buys at that close and holds for three months ends "
           "0.6% a year behind SPY, and over one and two years the stocks insiders buy earn what their beta says "
           "they should."))
s.append(P("Since 2017 there is not even that. Held for three months from the filing, the stocks insiders bought "
           "have trailed the average S&P 500 stock by 3.3% a year and the ones they sold have led it by 1.5%, "
           "neither significantly. A model that takes insider trades as an input should not expect buys to be good news "
           "at a horizon of three months, and should check them against plain momentum first."))
s.append(P("The limits:"))
s += bullets([
    "Coverage. 6% of the events (9% of the buys) are in companies with no free price history, 11% in 2006-2016 "
    "and 1% since 2017. They are companies that were bought or went under before the free sources start to "
    "keep them, among them Wachovia, Bear Stearns, Washington Mutual, Lehman Brothers and CIT. Leaving out the "
    "failures flatters the buys of the early years. The results for 2017-2026 are not touched by this.",
    "Closing prices. The outsider acts at the close of the day after the filing. A faster outsider gets part of "
    "the 38 basis points of that day, and the data cannot say how much.",
    "No trading costs. The one-week portfolios turn over about 50 times a year.",
    "Momentum is not controlled for. Insiders trade against the recent move of the stock, and what follows a "
    "buy or a sale is partly what follows any stock that has just fallen or risen.",
    "First Republic and Signature Bank are missing altogether. A bank without a holding company files with the "
    "FDIC and not with the SEC, so their insider trades are not in the data.",
    "Large companies only. In small companies, where fewer people read the filings, the result may be "
    "different.",
])
s.append(P("Left out on purpose and natural next steps: a control for momentum, trades under 10b5-1 plans (marked "
           "on the form since 2023), option exercises, opening prices for the day after the filing, and companies "
           "outside the S&P 500. The repository has a script that freezes these numbers, so the same test can be "
           "run on the filings of the coming quarters."))

# appendix
s.append(P("Appendix. Reproducing it", h1))
s.append(P("Python 3 with pandas, numpy, matplotlib, pyarrow, requests, yfinance and reportlab. A name and email "
           "for the SEC and a free Tiingo key go in .env. In order:"))
for line in ["python src/ingest.py      # SEC data sets + S&P 500 history -> data/trades.parquet",
             "python src/prices.py      # Yahoo, then Tiingo -> data/prices.parquet (takes hours)",
             "python src/events.py      # events, both clocks, all horizons -> data/events.csv",
             "python src/analysis.py    # data/results.csv + figures/",
             "python report/build_report.py"]:
    s.append(P(line.replace(" ", "&nbsp;"), mono))
s.append(Spacer(1, 6))
s.append(P("Data: SEC insider transactions data sets (sec.gov), S&P 500 history from github.com/fja05680/sp500, "
           "prices from Yahoo Finance and Tiingo. Nothing in the repository is licensed data. The hand-made tables "
           "are in data/manual: 21 tickers the history file names differently from the filings, and what happened "
           "to the 145 companies whose prices end early.", caption))


# headings stick to whatever comes right after them
def flat(*fs):
    out = []
    for f in fs:
        out += f._content if isinstance(f, KeepTogether) else [f]
    return out


story, i = [], 0
while i < len(s):
    f = s[i]
    if isinstance(f, Paragraph) and f.style.name in ("h1", "h2") and i + 1 < len(s):
        nxt = s[i + 1]
        if isinstance(nxt, Paragraph) and nxt.style.name == "h2" and i + 2 < len(s):
            story.append(KeepTogether(flat(f, nxt, s[i + 2])))
            i += 3
            continue
        story.append(KeepTogether(flat(f, nxt)))
        i += 2
        continue
    story.append(f)
    i += 1

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=2.0 * cm,
                        bottomMargin=2.0 * cm, title=TITLE, author="Viktor Sadowski")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"report -> {OUT}")
