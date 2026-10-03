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


def P(t, s=body):
    return Paragraph(t.replace("S&P", "S&amp;P"), s)


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
      P("Buys and sales by insiders in S&P 500 companies, 2006 to 2026: what the insider earns, and what is left "
        "when the filing is public", subtitle),
      P("Viktor Sadowski · October 2026 · github.com/Viktorsadowski/insider-trade-worth", byline),
      Spacer(1, 0.6 * cm)]

s.append(box([
    P("Summary", ParagraphStyle("abs_h", parent=h2, spaceBefore=0)),
    P("Company insiders have to report trades in their own stock to the SEC within two business days. This project "
      "takes every open-market buy and sale by insiders in S&P 500 companies from 2006 to mid 2026, 179,579 "
      "events, and measures the stock against the market on two clocks: from the insider's own trade, and from "
      "the first close after the filing, when anyone can act on it. Three months is the main holding period.",
      abstract),
    P("What the insider has, he has in the first days. He buys after the stock has lost 3.4% to the market in a "
      "month, and a week later it is 0.75% ahead (t = 6.3), half of it before the filing and half on the day the "
      "market reads it. A portfolio that holds every stock for the week after an insider buys it returns 41.6% a "
      "year, against 11.1% for SPY. Held for three months it returns 13.1%, which is no longer different from "
      "SPY, and all of that edge is from before 2017.", abstract),
    P("For someone who can only read the filing, nothing is left. Bought at the close after the filing and held "
      "for three months, the same stocks return 10.4% a year. Since 2017 the sign is the reverse of the textbook: "
      "the stocks insiders bought returned 8.7% a year and the ones they sold 14.1%, against 12.4% for the average "
      "S&P 500 stock. Neither gap is significant (t = -1.4 and 1.6), so there is no signal to count on in either "
      "direction. The likely reason is momentum: insiders buy what has fallen and sell what has risen.", abstract),
]))

# 1
s.append(P("1. The question", h1))
s.append(P("An insider knows his company better than the market does, and when he buys its stock with his own money "
           "the trade is public two days later. Following insider buys is one of the oldest ideas in investing, and "
           "the research mostly agrees that buys carry information and sales do not. Most of that evidence is from "
           "small companies and from before filings were on the internet within days. The question here has two "
           "parts. How well do insiders in the largest US companies invest in their own stock? And how much of "
           "that is left for someone who can only read the filing?"))

# 2
s.append(P("2. Data and method", h1))
s.append(P("The trades are from the SEC's insider transactions data sets, the quarterly files with every Form 3, 4 "
           "and 5 since 2006. Kept are open-market buys and sales of common stock (codes P and S on Form 4), so no "
           "option exercises, grants, gifts or tax withholding, in companies that were in the S&P 500 on the day "
           "of the filing (point-in-time members from fja05680/sp500 on GitHub). Trades by the same insider in the "
           "same company, reported on the same day and in the same direction, are one event: 179,579 events, "
           "15,923 buys and 163,656 sales."))
s.append(P("Prices are daily closes from Yahoo Finance, and from Tiingo for companies that are no longer listed. A "
           "ticker can belong to another company today than it did in 2008, so a price series is only used for a "
           "company in the years where the prices insiders wrote on their filings follow its close. 168,013 events "
           "(94%) in 774 companies have such a series: 14,562 buys with a median size of $99k and 153,451 sales "
           "with a median of $742k. Insiders sell ten times as often as they buy, since most of their pay is "
           "stock. When they buy, they buy in bursts: November 2008, August 2011 and March 2020 are the three "
           "biggest months (Figure 1)."))
s.append(figure(FIG / "sample.png", "Figure 1. Left: insider buys per month, events with a price. Right: trading "
                "days from the insider's last trade to the first close an outsider can act on.", width=TEXT_W))
s += bullets(lead="The stock is followed on two clocks:", items=[
    "The insider's clock starts at the close on the day of his trade (the last one, if the filing has several). "
    "It measures how well he invests.",
    "The public clock starts at the close on the first trading day after the filing. The SEC data has the date of "
    "a filing and not the time, and many arrive after the market has closed, so this is the first price an "
    "outsider can be sure to get. Several insiders of a company reporting the same day are one signal, so here "
    "the unit is company and day. For 97% of the events it starts within three trading days of the trade."])
s += bullets(lead="And with two measures:", items=[
    "Per trade: the total return of the stock minus the total return of SPY over the same days, from 1 day to 24 "
    "months, in basis points (100 to a percent). t-values are clustered by calendar month, since events in the "
    "same month share the same market, and from three months up they also allow for the overlap between the "
    "holding periods of neighbouring months.",
    "As a portfolio: every day, hold all stocks with an insider buy (or sale) in the last week, month, three "
    "months and so on, equal weight, and compound the daily returns. This gives one number per clock, a return "
    "per year, with no overlapping events, so it is the measure to trust at the long horizons. The average of "
    "single trades is not compounded, on purpose: one stock moves 20% against the market in three months, and a "
    "geometric average per trade mostly measures that. The daily returns of the portfolio are also regressed "
    "on SPY for an alpha and a beta."])
s.append(P("A stock that stops trading inside the holding period is followed to the end. 145 price series end "
           "early, and each has a line in a hand-made table: a cash buyout pays the deal price (48 series), a deal "
           "paid in shares is worth the last traded price (94), and a bankruptcy that cancels the shares ends at "
           "zero (3). After that the money sits in SPY."))

# 3
s.append(P("3. Results", h1))
s.append(P("3.1 In one number", h2))
s.append(P("A dollar that follows every insider buy for three months, starting at the close on the day of the "
           "trade, grows to $12.74 from 2006 to October 2026, 13.1% a year. Starting at the close after the filing "
           "it grows to $7.78, 10.4% a year, against $8.80 and 11.1% in SPY. Neither is different from SPY once "
           "the beta of the stocks is taken out, and from six months on both clocks end close to it."))
s.append(figure(FIG / "growth.png", "Figure 2. $1 in a portfolio that holds every stock for three months (left) or "
                "a week (right) after an insider buy, equal weight, from the two starting points, and $1 in SPY. "
                "End value and compounded return per year. Log scale, each panel has its own. No trading costs.",
                width=TEXT_W * 0.92))
s.append(P("The gain sits in the first days. Held for a week from the insider's close the dollar grows to $1,347, "
           "41.6% a year. From the close after the filing it grows to $9.58. Nobody outside can trade at the "
           "insider's close, and the portfolio turns over every week before costs, so the 41.6% is what it is "
           "worth to know about the trade when it is made. Sales show the same in the other direction, smaller: "
           "the stocks insiders sell return 5.6% a year in the week after the sale, and the same as SPY from the "
           "filing on. Figure 3 has every holding period: the insider's edge fades within three months on both "
           "sides, and the two portfolios an outsider can hold stay at SPY from the first week."))
s.append(figure(FIG / "holds.png", "Figure 3. Compounded return per year of the four portfolios by holding period, "
                "2006 to 1 October 2026. A filled point: the alpha against SPY has a t-value above 2. The beta is "
                "between 1.12 and 1.16 for the bought portfolios and between 0.98 and 1.04 for the sold ones.",
                width=TEXT_W * 0.74))

s.append(P("3.2 Per trade", h2))
s.append(table(horizon_table(), [1.3 * cm, 2.1 * cm] + [1.9 * cm] * 7,
               "Table 1. Average stock return minus SPY in basis points, t-value in brackets. The insider: 14,562 "
               "buys and 153,451 sales. An outsider: 9,747 company-days with buys only and 111,354 with sales "
               "only. From six months up there are fewer, up to 11% at 24 months, since the latest filings have "
               "no full holding period yet."))
s.append(P("After an insider buys, the stock beats the market by 29 basis points the next day, 75 within a week and "
           "87 within a month. The average keeps growing, to 143 after three months, but the noise grows faster: "
           "the t-value is 1.6 there and under 1 from six months on. In money, $1m invested with the insider on "
           "his trade day is $7,500 ahead of SPY a week later."))
s.append(P("On the public clock the first day gives 7 basis points (t = 2.2), and that is the last number that can "
           "be told from zero. Sales are close to nothing on both clocks: 6 basis points behind after a week on "
           "the insider's clock, most of it on the day after the filing (-5, t = -7.2), so the market reads sales "
           "too. Insiders sell for many reasons, they have to pay taxes and spread their wealth, so a sale says "
           "less than a buy."))
s.append(figure(FIG / "timeline.png", "Figure 4. Insider buys, before and after the filing. Grey: the month before "
                "the trade. Blue: from the trade to the close on the day after the filing, where the outsider's "
                "clock starts. These three follow each other. Light blue: the outsider's first day, week and "
                "month, all counted from his close, so they overlap. Lines are 95% intervals.",
                width=TEXT_W * 0.66))
s.append(P("Figure 4 shows where the gain is. Insiders buy on weakness: in the month before the trade the stock "
           "has lost 3.4% against the market, and before a sale it has gained 2.5%. From the trade to the close on "
           "the filing day the stock recovers 34 basis points, before most of the market can have seen the "
           "filing. The next day, the first with the filing out, it gains another 38 (t = 11.8). So the market "
           "does react to insider buys, and it is done within a day. An outsider who buys at that close gets 7 the "
           "day after. A faster one, buying at the close on the filing day, would catch the 38, which is only "
           "possible for the filings that arrive in trading hours."))
s.append(figure(FIG / "clocks.png", "Figure 5. Average stock return minus SPY day by day, from day 0 of each "
                "clock to 24 months after. Dashed: the outsider's line against the average S&P 500 stock instead "
                "of SPY.", width=TEXT_W * 0.92))
s.append(P("Over two years the buy lines rise slowly, and that comes from the benchmark. Stocks that insiders buy "
           "move more than the market, with a beta of about 1.15, and SPY has gone up in most years. Against the "
           "average S&P 500 stock (the equal-weighted index, section 3.4) the outsider's line is flat for three "
           "months and negative after that, and the portfolios from the filing return no more than SPY at any "
           "holding period from three months up (Figure 3). After sales all three lines stay within half a "
           "percent of zero for two years."))

s.append(P("3.3 Who trades", h2))
s.append(figure(FIG / "roles.png", "Figure 6. Buys by role: average stock return minus SPY after one week and "
                "after three months, the insider's clock next to the public one. Lines are 95% intervals, and the "
                "two panels have different scales. Roles are read from the relationship boxes and the title on "
                "the filing, the most senior one counts, and heads of a division who call themselves CEO are "
                "other officers.", width=TEXT_W * 0.92))
s.append(P("The CEO does best on his own clock: 123 basis points in the first week against 59 to 100 for the "
           "others, and the difference to them is significant (t = 3.5). After three months he is 257 ahead, the "
           "only role with a t-value above 2 (2.2). On the public clock no role has a three-month result that can be "
           "told from zero. Owners of more than 10% are mostly funds and companies buying large blocks, and "
           "after a month nothing can be said about them."))
s.append(P("Insiders with only the box Other ticked are not in the chart. They look like an exception at +525 "
           "basis points three months after the filing, but it is 84 buys, nine of them the Gates foundation trust "
           "buying AutoNation in the winter of 2008-2009. Without AutoNation the group is at +129."))
s.append(P("Insiders of the same company often trade in both directions: 10% of the days with a public buy also "
           "have a public sale. The public clock in Table 1 uses days with one direction only. Figure 7 splits "
           "further. A buy with no insider sale in the company that day or the 30 days before is followed by +111 "
           "basis points over three months, a buy after recent sales by -72, and the difference has a t-value of "
           "1.8. Sales that come after insider buys are followed by a stock that beats the market, +41 after "
           "three months, so the earlier buy may count for more than the later sale. "
           "Several insiders buying on the same day are followed by +41 basis points in the first week against +3 "
           "for one alone (t = 2.0). That is the best public number among the splits, and with this many splits "
           "tested one t-value of 2 is what chance gives."))
s.append(figure(FIG / "mixed.png", "Figure 7. Public clock, three months: company-days with buys (left) and sales "
                "(right), split by what the other insiders of the company did. Number of days in brackets, bigger "
                "is in dollars, lines are 95% intervals.", width=TEXT_W * 0.78))

s.append(P("3.4 After the filing, 2017 to 2026", h2))
s.append(P("The last ten years on their own, and only the outsider's question: is anything left after the public "
           "day, in either direction? The sample is the 78,899 events from 2017 on. The benchmark is the average "
           "S&P 500 stock, the equal-weighted return of the index members. Over the whole sample it returned the "
           "same as SPY, 11.1% a year, but 10.0% against 7.5% in 2006-2016 and 12.4% against 15.3% since, when a "
           "few very large companies carried SPY. To see whether an insider picks a better stock than the next "
           "one in the index, the average stock is the fair comparison."))
s.append(figure(FIG / "since.png", "Figure 8. 2017 to 1 October 2026, public clock. Left: $1 in every stock "
                "insiders bought, and in every stock they sold, from the close after the filing and held for three "
                "months, next to the average S&P 500 stock and SPY. Right: the compounded return per year of the "
                "same two portfolios for every holding period. A filled point: the alpha against the average "
                "stock has a t-value above 2.", width=TEXT_W))
s.append(P("The stocks insiders bought did worse than the average stock for every holding period: 8.7% a year "
           "against 12.4% when held for three months. None of it is significant, but it is the wrong sign six "
           "times out of six. Sales go the other way. The stocks insiders sold returned 14.1% a year held for "
           "three months, and 18.5% held for the first week after the filing (t = 3.3). The sales that stand out "
           "are the directors': after them the stock beats the average stock by 114 basis points over three "
           "months (t = 2.4), while sales by other officers, six in ten, are followed by nothing. Bought "
           "minus sold is negative for every holding period, -4.7% a year at three months (t = -1.2)."))
s.append(P("So since 2017 a buy is no reason to expect a better stock after the public day, and a sale no reason "
           "to expect a worse one. If anything it is the other way round. The likely reason is momentum: insiders "
           "buy stocks that have fallen and sell stocks that have risen, and what follows may be what follows any "
           "stock that has just fallen or risen. This test does not separate the two."))
s.append(figure(FIG / "periods.png", "Figure 9. Insider buys before and after 2017: average stock return minus "
                "SPY after one week and after three months, the insider's clock next to the public one. Lines are "
                "95% intervals, and the two panels have different scales.", width=TEXT_W * 0.72))
s.append(P("Before 2017 it was different. The week after the filing gave an outsider 30 basis points (t = 2.7), "
           "and three months gave 178, or 62 against the average stock. Almost all of the three-month number "
           "is one year: stocks that insiders bought during 2009 beat SPY by 16% over the next three months, and "
           "without 2009 it is 24 basis points. The insider's own first week is still there, 88 basis points "
           "before 2017 and 58 after. His three months are gone: 248 basis points before and 11 after, and as a "
           "portfolio 14.1% a year against 7.5% for SPY before 2017 and 11.9% against 15.3% after."))

# 4
s.append(P("4. What it says", h1))
s.append(P("An insider buy in an S&P 500 company is worth something to the insider, and it is in the first days: "
           "0.75% over the market in a week, more for the CEO, before and after 2017. The three-month gain is "
           "smaller than its noise and is from before 2017. To anyone else the trade is worth close to nothing. "
           "The market takes the news in on the day after the filing, and an outsider who buys at that close and "
           "holds for three months ends 0.6% a year behind SPY."))
s.append(P("Since 2017 not even the sign holds. Held for three months from the filing, the stocks insiders bought "
           "have trailed the average S&P 500 stock by 3.3% a year and the ones they sold have led it by 1.5%, "
           "neither significantly. A model that takes insider trades as an input should not expect buys to be "
           "good news at a horizon of three months, and should check them against plain momentum first."))
s += bullets(lead="The limits:", items=[
    "Coverage. 6% of the events (9% of the buys) are in companies with no free price history, 11% in 2006-2016 "
    "and 1% since 2017, among them Wachovia, Bear Stearns, Washington Mutual, Lehman Brothers and CIT. Leaving "
    "out the failures flatters the buys of the early years.",
    "Closing prices and no trading costs. The outsider acts at the close of the day after the filing. A faster "
    "one gets part of the 38 basis points of that day, and the data cannot say how much. The insider's clock "
    "starts at the close of his trade day, and for buys the price on his filing is on average about 0.1% under "
    "that close. The one-week portfolios turn over about 50 times a year.",
    "Momentum is not controlled for. What follows a buy or a sale is partly what follows any stock that has just "
    "fallen or risen.",
    "First Republic and Signature Bank are missing. A bank without a holding company files with the FDIC and "
    "not with the SEC.",
    "Large companies only. In small companies, where fewer people read the filings, the result may be different.",
])
s.append(P("Next steps: a control for momentum, trades under 10b5-1 plans (marked on the form since 2023), option "
           "exercises, opening prices and companies outside the S&P 500. The numbers are frozen in the repository, "
           "so the same test can be run on later filings."))

# where the code and the data are
s.append(Spacer(1, 4))
s.append(P("Code: github.com/Viktorsadowski/insider-trade-worth, five scripts that rebuild everything from public "
           "sources (the run order is in the README). Data: SEC insider transactions data sets (sec.gov), S&P 500 "
           "history from github.com/fja05680/sp500, prices from Yahoo Finance and Tiingo. Nothing in the "
           "repository is licensed data. The hand-made tables are in data/manual, and data/results.csv has every "
           "split, also the ones not shown here.", caption))


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
