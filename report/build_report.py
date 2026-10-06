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
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

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
                      spaceAfter=6, allowWidows=0, allowOrphans=0)      # no single line alone on a page
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


def P(t, s=body):
    # "t = 3.5" stays on one line
    return Paragraph(t.replace("S&P", "S&amp;P").replace("t = ", "t&nbsp;=&nbsp;"), s)


def bullets(items, lead=None):
    b = [Paragraph(t, bullet, bulletText="•") for t in items]
    # the line that introduces a list stays on the page of its first point
    return [KeepTogether([P(lead), b[0]])] + b[1:] if lead else b


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


HORIZONS = [1, 5, 21, 63, 126, 252, 504]
HLAB = ["1 day", "1 week", "1 month", "3 months", "6 months", "12 months", "24 months"]


def tt(t):
    # a t-value with one decimal, and no -0.0
    return f"{0.0 if round(t, 1) == 0 else t:.1f}"


def bt(mean, t):
    # +64 (0.6). Round first, so a tiny negative number is not printed as -0
    m = round(mean)
    return f"{m:+d} ({t:.1f})" if m else f"0 ({tt(t)})"


def horizon_table():
    rows = [["", ""] + HLAB]
    h = res("horizons")
    for side, sname in (("buy", "Buys"), ("sell", "Sales")):
        for clock, cname in (("insider", "the insider"), ("public", "an outsider")):
            x = h[(h.side == side) & (h.clock == clock)].set_index("horizon").loc[HORIZONS]
            rows.append([sname, cname] + [bt(m, t) for m, t in zip(x.mean_bps, x.t)])
    return rows


# ── content ───────────────────────────────────────────────────────────────────

s = []
s += [Spacer(1, 1.2 * cm),
      P(TITLE, title),
      P("Buys and sales by insiders in S&P 500 companies, 2006 to 2026: what the insider earns, and what "
        "is left when the filing is public", subtitle),
      P("Viktor Sadowski · October 2026 · github.com/Viktorsadowski/insider-trade-worth", byline),
      Spacer(1, 0.6 * cm)]

# the summary keeps the two questions apart: what the insider earns on his own trade, and what someone gets who
# trades on the public filing. One label for each, so nobody has to guess whose number it is
abs_h = ParagraphStyle("abs_h", parent=h2, spaceBefore=0)
abs_lab = ParagraphStyle("abs_lab", parent=h2, fontSize=10, leading=13, spaceBefore=5, spaceAfter=2)
s.append(box([
    P("Summary", abs_h),
    P("Company insiders have to report trades in their own stock to the SEC within two business days. This project "
      "takes every open-market buy and sale by insiders in S&P 500 companies from 2006 to mid 2026, 179,579 events, "
      "and asks two separate questions. What does the insider earn on his own trade, counted from the close on the "
      "day he trades? And what can someone earn who only has the public filing and buys at the close on the trading "
      "day after it? Returns are measured against the market (SPY), and three months is the main holding period.",
      abstract),
    P("The insider's own trade", abs_lab),
    P("The average insider buy comes after a fall: in the month before the trade the stock has lost 3.4% "
      "against the market. Then it turns. From the trade to the filing day it gains 0.34% on the market, "
      "before most of the market can have seen the filing. On the day after the filing, when the market "
      "reads it, it gains another 0.38%. After that little is added: the gain is 0.75% a week after the "
      "trade (t = 6.3) and 0.87% after a month. A portfolio that buys every stock insiders buy, at the "
      "close on the insider's trade day, and holds it for a week returns 41.6% a year, against 11.1% for "
      "SPY, but the trade is normally not public at that point. Held for three months it returns 13.1%, "
      "which cannot be told from SPY, and that gain is all from before 2017.", abstract),
    P("A trade on the public filing", abs_lab),
    P("Someone who only has the filing can buy at the close on the trading day after it, and by then the "
      "move is over. The stock gains 0.07% the next day (t = 2.2) and nothing measurable after that. A "
      "portfolio that buys the same stocks at that close and holds them for three months returns 10.4% a "
      "year, less than SPY. Since 2017 the direction is the opposite of the textbook, where buys are "
      "good news: the stocks insiders bought returned 8.7% a year and the ones they sold 14.1%, against "
      "12.4% for the average S&P 500 stock. At three months neither gap is significant (t = -1.4 and "
      "1.6), so there is no signal to count on in either direction. It is not momentum either: against "
      "stocks with the same past return the result is the same.", abstract),
]))

# 1
s.append(P("1. The question", h1))
s.append(P("An insider knows his company better than the market does, and when he buys its stock with his own "
           "money the trade is public two days later. Following insider buys is one of the oldest ideas in "
           "investing, and the research mostly agrees that buys carry information and sales do not. Most of that "
           "evidence is from small companies and from before filings were on the internet within days. The "
           "question here has two parts. How well do insiders in the largest US companies invest in their own "
           "stock? And how much of that is left for someone who can only read the filing?"))

# 2
s.append(P("2. Data and method", h1))
s.append(P("The trades are from the SEC's insider transactions data sets, the quarterly files with every Form 3, 4 "
           "and 5 since 2006. Only open-market buys and sales of common stock are kept (codes P and S on Form 4), "
           "so no option exercises, grants, gifts or tax withholding, and only in companies that were in the S&P "
           "500 on the day of the filing (point-in-time members from fja05680/sp500 on GitHub). Trades by the same "
           "insider in the same company, reported on the same day and in the same direction, count as one event. "
           "That gives 179,579 events: 15,923 buys and 163,656 sales. Filings run to June 2026 and prices to 1 "
           "October 2026."))
s.append(P("Prices are daily closes from Yahoo Finance, and from Tiingo for companies that are no longer listed. A "
           "ticker can belong to a different company today than it did in 2008, so a price series is only used for "
           "a company in the years where the prices on the insiders' filings match its closing prices. 168,013 "
           "events (94%) in 774 companies have such a series: 14,562 buys with a median size of $99k and 153,451 "
           "sales with a median of $742k. Insiders sell ten times as often as they buy, since most of their pay is "
           "stock. When they buy, they buy in bursts: November 2008, August 2011 and March 2020 are the three "
           "biggest months (Figure 1)."))
s.append(figure(FIG / "sample.png", "Figure 1. Events with a price history. The days are counted from the insider's "
                "last trade on the filing.",
                width=TEXT_W))
s += bullets(lead="Every trade is measured from two starting points, called clocks here. Day 0 is the day a "
                  "clock starts:", items=[
    "The insider's clock is his own trade. It starts at the close on the day he trades (the last day, if the filing "
    "has several) and measures what the insider himself earns. The trade is normally not public at that point.",
    "The public clock is the trade someone else can make with public information. It starts at the close on the "
    "first trading day after the filing, called the public day here. The SEC data has the date of a filing and not "
    "the time, and many arrive after the market has closed, so this is the first price an outsider can be sure to "
    "get. Several insiders of a company reporting on the same day count as one signal, so the unit here is one "
    "company on one day. For 97% of the events the public day is within three trading days of the trade."])
s += bullets(lead="And it is measured in two ways:", items=[
    "Per trade: the total return of the stock minus the total return of SPY over the same days, from 1 day to 24 "
    "months, in basis points (100 basis points are 1%). t-values are clustered by calendar month, since events in "
    "the same month share the same market, and from three months up they also allow for the overlap between the "
    "holding periods of neighbouring months.",
    "As a portfolio: every day, hold all stocks with an insider buy (or sale) in the last week, month, three "
    "months and so on, equal weight, and compound the daily returns. This gives one number per clock, a return per "
    "year, with no overlapping events, so it is the measure to trust at the long horizons. The per-trade number is "
    "a plain average of the single returns and not a compounded one, on purpose: a single stock typically moves "
    "20% against the market in three months, and compounding swings of that size pulls the result down whatever "
    "the insider did. The daily returns of the portfolio are also regressed on SPY for an alpha and a beta."])
s.append(P("A stock that stops trading inside the holding period is followed to the end. 145 price series end "
           "early, and each has a line in a hand-made table: a cash buyout pays the deal price (48 series), a deal "
           "paid in shares is worth the last traded price (94), and a bankruptcy that cancels the shares ends at "
           "zero (3). After that the money sits in SPY."))

# 3
s.append(P("3. Results", h1))
s.append(P("3.1 In one number", h2))
s.append(P("Take a dollar that follows every insider buy and holds each stock for three months. On the insider's "
           "clock, bought at the close on the day of his trade, it grows to $12.74 from 2006 to October 2026, "
           "13.1% a year. On the public clock, bought at the close on the public day, it grows to $7.78, 10.4% a "
           "year. In SPY it grows to $8.80, 11.1% a year (Figure 2). Neither portfolio is significantly different "
           "from SPY once the beta of the stocks is taken out, and with holding periods of six months or more both "
           "end close to SPY."))
s.append(figure(FIG / "growth.png", "Figure 2. At the end of each line: the end value and the compounded return per "
                "year. Each panel has its own scale.",
                width=TEXT_W))
s.append(P("The insider's gain is in the first days. Held for a week from the insider's close the dollar grows to "
           "$1,347, 41.6% a year. Held for a week from the public day it grows to $9.58, 11.5% a year. The trade "
           "is normally not public at the insider's close, and no trading costs are counted although the portfolio "
           "turns over weekly. So the 41.6% is not a return an outsider can earn. It is what knowing about the "
           "trade on the day it is made would be worth. Sales show the same in the other direction, only smaller: "
           "held for the week after the sale, the stocks insiders sell return 5.6% a year, and from the public day "
           "the same as SPY. Figure 3 shows every holding period: the insider's edge fades within three months on "
           "both sides, and an outsider's two portfolios stay at SPY from the first week."))
s.append(figure(FIG / "holds.png", "Figure 3. Compounded returns, 2006 to 1 October 2026. The beta is between 1.12 "
                "and 1.16 for the bought portfolios and between 0.98 and 1.04 for the sold ones.",
                width=TEXT_W * 0.78))

s.append(P("3.2 Per trade", h2))
s.append(P("Counted from the insider's own trade, the stock beats the market by 29 basis points the next day, 75 "
           "within a week and 87 within a month. The average keeps growing, to 143 after three months, but the "
           "noise grows faster: the t-value is 1.6 there and under 1 from six months on. In dollars, $1m invested "
           "alongside the insider on his trade day is $7,500 ahead of SPY a week later."))
s.append(table(horizon_table(), [1.3 * cm, 2.1 * cm] + [1.9 * cm] * 7,
               "Table 1. Average stock return minus SPY in basis points, t-value in brackets. The insider: from "
               "the close on his trade day, 14,562 buys and 153,451 sales. An outsider: from the close on the "
               "public day, 9,747 company-days with buys only and 111,354 with sales only. From six months up "
               "there are fewer events, up to 11% fewer at 24 months, since the latest filings do not have a full "
               "holding period yet."))
s.append(P("Counted from the public day, which is all an outsider can trade on, the first day gives 7 basis points "
           "(t = 2.2), and that is the only number in the outsider's rows of Table 1 that can be told from zero. "
           "Sales come to almost nothing on both clocks. On the insider's clock the stock is 6 basis points behind "
           "after a week, and most of that falls on the day after the filing (-5, t = -7.2), so the market reads "
           "sales too. Insiders sell for many reasons, such as paying taxes and spreading their wealth, so a sale "
           "says less than a buy."))
s.append(P("Figure 4 shows where the insider's gain comes from. Insiders buy on weakness: in the month before the "
           "trade the stock has lost 3.4% against the market, and before a sale it has gained 2.5%. After a buy "
           "the stock recovers 34 basis points from the trade to the close on the filing day, before most of the "
           "market can have seen the filing. The next day, the first full day with the filing public, it gains "
           "another 38 (t = 11.8). So the market does react to insider buys, and the reaction is over within a "
           "day. An outsider who buys at the close of that day, the public day, gets 7 the day after. A faster "
           "outsider, buying at the close on the filing day, would catch the 38, but that is only possible for the "
           "filings that arrive in trading hours."))
s.append(figure(FIG / "timeline.png", "Figure 4. The grey bar and the two dark blue bars follow each other in time, "
                "up to the close on the public day. The three light blue bars are an outsider's: they all start at "
                "that close, so they overlap.",
                width=TEXT_W * 0.81))
s.append(P("Over two years the buy lines in Figure 5 rise slowly, and that comes from the benchmark. Stocks that "
           "insiders buy move more than the market, with a beta of about 1.15, and SPY has gone up in most years. "
           "Against the average S&P 500 stock (the equal-weighted index, section 3.4) the outsider's line is flat "
           "for three months and negative after that, and the portfolios from the public day return no more than "
           "SPY at any holding period from three months up (Figure 3). After sales the three lines in the right "
           "panel stay within half a percent of zero for two years."))
s.append(figure(FIG / "clocks.png", "Figure 5. The two clocks day by day, to 24 months after day 0.",
                width=TEXT_W))

s.append(P("3.3 Who trades", h2))
s.append(P("The CEO does best on his own buys: 123 basis points in the first week against 59 to 100 for the other "
           "roles, and the difference from the others is significant (t = 3.5). After three months he is 257 "
           "ahead, the only role with a t-value above 2 (2.2). For an outsider, no role has a three-month result "
           "that can be told from zero (Figure 6). Owners of more than 10% are mostly funds and companies buying "
           "large blocks, and beyond one month their results are too noisy to say anything."))
s.append(figure(FIG / "roles.png", "Figure 6. Roles are read from the relationship boxes and the title on the "
                "filing. The most senior role counts, and heads of a division who call themselves CEO are counted "
                "as other officers. Not in the chart: 84 buys by insiders with only the box Other ticked.",
                width=TEXT_W))
s.append(P("Insiders of the same company often trade in both directions: 10% of the public days with a buy also "
           "have a sale. The public clock in Table 1 uses days with one direction only. Figure 7 splits the days "
           "further, all on the public clock. A buy with no insider sale in the company that day or the 30 days "
           "before is followed by +111 basis points over three months, a buy after recent sales by -72, and the "
           "difference has a t-value of 1.8. After a sale that follows recent insider buys the stock beats the "
           "market by 41 basis points over three months, so the earlier buy may count for more than the later "
           "sale. When several insiders report buys on the same day, the stock gains 41 basis points in the week "
           "after the public day, against 3 when one reports alone (t = 2.0). That is the clearest result for an "
           "outsider among the splits, and with this many splits tested, one t-value of 2 is what chance alone "
           "would give."))
s.append(figure(FIG / "mixed.png", "Figure 7. The public clock. The two same-day groups are the same company-days "
                "in both panels.",
                width=TEXT_W * 0.92))

s.append(P("3.4 After the filing, 2017 to 2026", h2))
s.append(P("This section takes the years from 2017 on their own, and first the outsider's question: is anything "
           "left after the public day, in either direction? The sample is the 78,899 events from 2017 on. The "
           "benchmark is the average S&P 500 stock, the equal-weighted return of the index members. Over the whole "
           "sample it returned the same as SPY, 11.1% a year. The two halves differ: it returned 10.0% against "
           "7.5% for SPY in 2006-2016, and 12.4% against 15.3% since 2017, when a few very large companies carried "
           "SPY. To see whether an insider picks a better stock than the next one in the index, the average stock "
           "is the fair comparison."))
s.append(P("The stocks insiders bought did worse than the average stock at every holding period (Figure 8): 8.7% a "
           "year against 12.4% when held for three months (t = -1.4). None of these six gaps is significant, but "
           "the sign is wrong at every holding period, from one week to 24 months. Sales go the other way. The "
           "stocks insiders sold returned 14.1% a year held for three months (t = 1.6), and 18.5% held for the "
           "first week after the public day (t = 3.3). The sales that stand out are those by directors: after them "
           "the stock beats the average stock by 114 basis points over three months (t = 2.4). Sales by other "
           "officers, six in ten of all sales, are followed by nothing. At every holding period the bought stocks "
           "return less than the sold ones, 8.7% against 14.1% a year at three months (t = -1.2 for the "
           "difference)."))
s.append(P("So since 2017 a buy is no reason to expect a better stock after the public day, and a sale no reason "
           "to expect a worse one. If anything it is the other way round. The obvious suspect is momentum: "
           "insiders buy stocks that have fallen and sell stocks that have risen, and the returns that follow may "
           "be those of any stock that has just fallen or risen. Section 3.5 tests that."))
s.append(figure(FIG / "since.png", "Figure 8. 2017 to 1 October 2026, the public clock.",
                width=TEXT_W))
s.append(P("Before 2017 it was different (Figure 9). The week after the public day gave an outsider 30 basis "
           "points (t = 2.7), and three months gave 178, or 62 against the average stock. Almost all of the 178 is "
           "one year: counted from the public day, the stocks insiders bought during 2009 beat SPY by 16% over "
           "three months, and without 2009 the 178 becomes 24. The insider's own first-week gain is in both "
           "halves: 88 basis points before 2017 and 58 after. His three-month gain is only in the first: 248 basis "
           "points before and 11 after. As a portfolio on the insider's clock that is 14.1% a year against 7.5% "
           "for SPY before 2017, and 11.9% against 15.3% after."))
s.append(figure(FIG / "periods.png", "Figure 9. The two halves of the sample, 2006 to 2016 and 2017 to 1 October "
                "2026.",
                width=TEXT_W * 0.82))

s.append(P("3.5 Is it momentum?", h2))
s.append(P("Stocks that have fallen or risen have their own pattern afterwards, whoever trades them, and insiders "
           "trade exactly those stocks. Sort the S&P 500 members every day into fifths by their return over the "
           "past year (without its last month), and again by their return over the last month. Of the insider "
           "buys, 32% are in the lowest fifth on the first sort and 35% on the second, where an even spread would "
           "be 20%. Of the sales, 27% and 29% are in the highest fifth (Figure 10)."))
s.append(figure(FIG / "past.png", "Figure 10. The stocks as they stood on the public day, days with both buys and "
                "sales left out.",
                width=TEXT_W * 0.84))
s.append(P("So every trade is measured once more, against the stocks that looked the same: the S&P 500 members in "
           "the same fifth on both sorts on day 0, about 17 stocks, with the stock itself left out. If past return "
           "were the reason for a result, the result would be zero against these."))
s.append(P("The first week changes little (Figure 11). On the insider's clock it is 68 basis points against the "
           "matched stocks, where it was 75 against SPY, and the t-value goes up to 7.7, since stocks that looked "
           "the same take out more noise than the index does. The insider's three-month gain is 72 basis points (t "
           "= 2.0), where it was 143 against SPY, and the CEO's is 207 (t = 3.5). An outsider gets 2 basis points "
           "in the first week and -5 after three months. As a portfolio, the week after an insider buy returns "
           "40.9% a year on the insider's clock for the buys that have matched stocks, against 12.6% for the "
           "matched stocks themselves."))
s.append(figure(FIG / "benchmarks.png", "Figure 11. The main results per trade, once for each benchmark.",
                width=TEXT_W * 0.92))
s.append(P("Since 2017, on the public clock, the stocks insiders bought are 50 basis points behind their matched "
           "stocks after three months, compared with 59 behind the average stock, and the stocks they sold are 40 "
           "ahead (t = 1.6). As portfolios the bought stocks returned 8.7% a year and their matched stocks 12.3%, "
           "the sold stocks 14.1% and theirs 12.0% (Figure 12). The matched stocks did what the average stock did, "
           "12.4%, so in these years it made little difference to the next three months whether a large stock had "
           "fallen or risen. At three months neither gap is significant. For buys the gap grows with the holding "
           "period: after twelve months the bought stocks are 269 basis points behind their matched stocks (t = "
           "-2.9). Momentum is not the reason for it."))
s.append(figure(FIG / "lookalikes.png", "Figure 12. 2017 to 1 October 2026, the public clock. Company-days that "
                "have matched stocks.",
                width=TEXT_W * 0.68))

# 4
s.append(P("4. What it says", h1))
s.append(P("An insider buy in an S&P 500 company is worth something to the insider himself, and the value is in "
           "the first days: 0.75% over the market in a week, and more for the CEO. That first week is there both "
           "before and after 2017. The average insider's three-month gain is weak (t = 1.6 against SPY, 2.0 "
           "against the matched stocks), and against SPY all of it is from before 2017. To anyone trading on the "
           "public filing the buy is worth close to nothing: the market absorbs the news on the day after the "
           "filing, and an outsider who buys at that close and holds for three months earns 10.4% a year, against "
           "11.1% for SPY."))
s.append(P("Since 2017 not even the sign holds for an outsider: bought on the public day and held for three "
           "months, the stocks insiders bought have returned 3.7 points a year less than the average S&P 500 stock "
           "and the ones they sold 1.7 points more, neither significantly, and past return does not explain it. A "
           "model that takes insider trades as an input should not expect buys to be good news at three months."))
s += bullets(lead="The limits:", items=[
    "Coverage. 6% of the events (9% of the buys) are in companies with no free price history, 11% in 2006-2016 and "
    "1% since 2017, among them Wachovia, Bear Stearns, Washington Mutual, Lehman Brothers and CIT. Leaving out the "
    "failures flatters the buys of the early years.",
    "Closing prices and no trading costs. The outsider acts at the close of the day after the filing, and a faster "
    "one gets part of the 38 basis points of that day. The insider's clock starts at the close of his trade day, "
    "and for buys the price on his filing is on average about 0.1% under that close. The one-week portfolios turn "
    "over about 50 times a year.",
    "Past return is the only thing the matched stocks share with the stock. Size, valuation and industry are not "
    "controlled for.",
    "First Republic and Signature Bank are missing. A bank without a holding company files with the FDIC and not "
    "with the SEC.",
    "Large companies only. In small companies, where fewer people read the filings, the result may be different.",
])
s.append(P("Next steps: a match on size, valuation and industry, 10b5-1 plans (trades scheduled in advance), "
           "option exercises, opening prices and smaller companies. The numbers are frozen in the repository, so "
           "the test can be rerun on later filings."))

# where the code and the data are
s.append(Spacer(1, 4))
s.append(P("Code: github.com/Viktorsadowski/insider-trade-worth, five scripts that rebuild everything "
           "from public sources (run order in the README). Data: SEC insider transactions data sets, S&P "
           "500 history from github.com/fja05680/sp500, prices from Yahoo Finance and Tiingo, nothing "
           "licensed. data/results.csv has every split, also the ones not shown here.", caption))


# headings stick to whatever comes right after them. Before a chart or a table they are kept together with it.
# Before a paragraph the heading only asks for room for itself and a few lines, so the paragraph can still run
# over to the next page and no page ends with a hole because a whole paragraph did not fit
def flat(*fs):
    out = []
    for f in fs:
        out += f._content if isinstance(f, KeepTogether) else [f]
    return out


def is_head(f):
    return isinstance(f, Paragraph) and f.style.name in ("h1", "h2")


story, i = [], 0
while i < len(s):
    f = s[i]
    if is_head(f) and i + 1 < len(s):
        heads = [f, s[i + 1]] if is_head(s[i + 1]) and i + 2 < len(s) else [f]
        nxt = s[i + len(heads)]
        if isinstance(nxt, Paragraph) and nxt.style.name == "body":
            story += [CondPageBreak((1.3 * len(heads) + 1.7) * cm)] + heads
            i += len(heads)
            continue
        story.append(KeepTogether(flat(*heads, nxt)))
        i += len(heads) + 1
        continue
    story.append(f)
    i += 1

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=2.0 * cm,
                        bottomMargin=2.0 * cm, title=TITLE, author="Viktor Sadowski")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"report -> {OUT}")
