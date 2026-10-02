#!/usr/bin/env python3
"""
SEC insider data sets + S&P 500 history -> data/trades.parquet, data/universe.csv, data/sp500_members.csv

  trades.parquet     one row per open-market buy (P) or sale (S) on a Form 4, for companies that were in the
                     S&P 500 at some point since 2006. in_sp500 says if the company was a member on the filing date
  universe.csv       which SEC company (cik) belongs to which S&P 500 ticker, and for which dates
  sp500_members.csv  ticker, start, end: the membership spells read from the fja05680 history file
  sp500_gaps.csv     S&P ticker-years with no Form 4 at all. Should be close to empty, otherwise a company is
                     missing a cik (fix in data/manual/cik_overrides.csv)

  python src/ingest.py                 # downloads what is missing into data/raw first (SEC: about 1 GB, 82 zips)
  python src/ingest.py --no-download   # only rebuild from data/raw
"""

import argparse
import io
import re
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import requests

from config import (DATA_DIR, MANUAL_DIR, RAW_DIR, ROLES, SEC_DIR, SEC_URLS, SEC_USER_AGENT, SP500_URL, quarters)

SP_FILE = RAW_DIR / "sp500_history.csv"
FIRST_DAY = pd.Timestamp("2006-01-01")

# a filing counts as a real accession number, everything else is a broken line in the tsv
RE_ACC = r"^\d{10}-\d{2}-\d{6}$"


# ── download ───────────────────────────────────────────────────────────────────

def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SEC_DIR.mkdir(parents=True, exist_ok=True)
    # the S&P file changes every few weeks, so always take a fresh one
    print("download S&P 500 history (fja05680/sp500)")
    r = requests.get(SP500_URL, timeout=60)
    r.raise_for_status()
    SP_FILE.write_bytes(r.content)

    todo = [(y, q) for y, q in quarters() if not (SEC_DIR / f"{y}q{q}_form345.zip").exists()]
    if todo and not SEC_USER_AGENT:
        sys.exit("SEC_USER_AGENT is not set. Copy .env.example to .env and put your name and email in it, "
                 "the SEC blocks requests without it.")
    for y, q in todo:
        for url in SEC_URLS:
            r = requests.get(url.format(y=y, q=q), timeout=180,
                             headers={"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"})
            if r.status_code != 404:
                break
            time.sleep(0.3)
        if r.status_code == 404:
            print(f"  {y} Q{q}: not published yet")
            continue
        r.raise_for_status()
        # a blocked request comes back as a small html page, not a zip
        if not r.content.startswith(b"PK"):
            sys.exit(f"  {y} Q{q}: the SEC did not send a zip (blocked or rate limited?). Check SEC_USER_AGENT.")
        (SEC_DIR / f"{y}q{q}_form345.zip").write_bytes(r.content)
        print(f"  {y} Q{q}: {len(r.content) / 1e6:.1f} MB")
        time.sleep(0.3)     # far below the 10 requests a second the SEC allows


# ── S&P 500 membership ─────────────────────────────────────────────────────────

def norm_ticker(t: str) -> str:
    """BF.B, BF-B and BFB are the same thing. Also drops the -YYYYMM suffix some delisted names carry"""
    t = re.sub(r"-\d{6}$", "", str(t).strip().upper())
    return re.sub(r"[^A-Z0-9]", "", t)


def sp500_spells() -> pd.DataFrame:
    """ticker, start, end. end is the first day the ticker is no longer in the list (empty = still in)"""
    h = pd.read_csv(SP_FILE)
    h["date"] = pd.to_datetime(h["date"])
    h = h.sort_values("date")
    open_since, rows = {}, []
    for date, tickers in zip(h["date"], h["tickers"]):
        now = {t.strip() for t in tickers.split(",") if t.strip()}
        for t in list(open_since):
            if t not in now:
                rows.append({"ticker": t, "start": open_since.pop(t), "end": date})
        for t in now:
            open_since.setdefault(t, date)
    rows += [{"ticker": t, "start": s, "end": pd.NaT} for t, s in open_since.items()]
    sp = pd.DataFrame(rows).sort_values(["ticker", "start"]).reset_index(drop=True)
    sp["key"] = sp["ticker"].map(norm_ticker)
    # only spells that reach into the sample
    sp = sp[sp["end"].isna() | (sp["end"] > FIRST_DAY)].reset_index(drop=True)
    return sp


# ── reading the SEC zips ───────────────────────────────────────────────────────

def read_tsv(zf: zipfile.ZipFile, name: str, cols) -> pd.DataFrame:
    member = next((n for n in zf.namelist() if n.upper().endswith(name)), None)
    if member is None:
        raise FileNotFoundError(f"{name} not in zip, has {zf.namelist()}")
    # no quoting: free text fields have stray quotes. A line break inside a text field then gives a broken
    # row, those are dropped on the accession number check
    df = pd.read_csv(io.TextIOWrapper(zf.open(member), encoding="utf-8", errors="replace"), sep="\t", dtype=str,
                     usecols=lambda c: c in cols, quoting=3, on_bad_lines="skip", low_memory=False)
    missing = set(cols) - set(df.columns)
    if missing:
        raise KeyError(f"{name}: columns missing {missing}")
    df = df[df["ACCESSION_NUMBER"].fillna("").str.match(RE_ACC)]
    return df


def sec_date(s: pd.Series) -> pd.Series:
    # 31-MAR-2006. A few are typed differently, those go through the slow parser
    d = pd.to_datetime(s, format="%d-%b-%Y", errors="coerce")
    bad = d.isna() & s.notna()
    if bad.any():
        d[bad] = pd.to_datetime(s[bad], errors="coerce")
    return d


NOT_A_TICKER = {"NYSE", "NASDAQ", "NASD", "AMEX", "OTC", "OTCBB", "OTCQB", "OTCQX", "NONE", "NA", "NYSEMKT", "NMS",
                "NYSEAMERICAN", "COMMON", "STOCK", "NULL"}


def symbol_keys(raw: str) -> list:
    """the ticker field is typed by hand: 'BRK.A, BRK.B', 'NYSE: IBM', 'ibm', 'N O G', 'N/A', 'SEE REMARKS'.
    Returns normalised tickers. Spaces inside a part are dropped, so 'N O G' is NOG and 'MSF A' is not A"""
    s = str(raw).upper().strip(" []\"'")
    if s in ("", "NAN", "N/A", "NA", "NONE", "N.A.", "NOT APPLICABLE"):
        return []
    s = re.sub(r"\([^)]*\)", " ", s)                  # "(NYSE)"
    s = re.sub(r"\b[A-Z ]{2,12}:", " ", s)            # "NYSE:"
    out = []
    for part in re.split(r"[,;/&]|\bAND\b", s):
        tok = re.sub(r"\s+", "", part).strip("\"'.-[]")
        if re.fullmatch(r"[A-Z]{1,6}([.\-][A-Z]{1,2})?", tok) and tok not in NOT_A_TICKER:
            out.append(norm_ticker(tok))
    return out


# words that can stand next to "CEO" in the title of the company's own chief executive. Anything else left
# over ("ceo amazon web services", "president & ceo, southern nuclear") means he runs a division
CEO_FILLER = re.compile(
    r"\b(chairman|chairwoman|chairperson|chaiman|chair|cob|chmn|chrmn|chm|president|pres|director|dir|of|the|"
    r"board|bd|and|co|vice|exec|executive|officer|off|chief|ceo|c\.e\.o|interim|acting|founder|member|managing|"
    r"secretary|treasurer|cfo|coo|operating|financial|principal|company|corp|corporation|inc|group|global|"
    r"worldwide|sr|senior|evp|svp|vp|elect|designate|trustee|lead|cto|cio)\b")


def classify_role(rel, title, issuer="") -> str:
    """CEO, CFO, Other officer, Director, 10% owner, Other. Officer titles are free text"""
    rel = str(rel).upper()
    t = str(title).lower()
    if "OFFICER" in rel:
        if re.search(r"\bc\.?e\.?o\b|chief exec|principal executive", t):
            # drop the company's own name ("ceo gap, inc."), then see if a division name is left
            for w in re.findall(r"[a-z]{3,}", str(issuer).lower()):
                t = re.sub(rf"\b{w}\b", " ", t)
            left = re.sub(r"[^a-z]+", " ", CEO_FILLER.sub(" ", t)).strip()
            return "CEO" if left == "" else "Other officer"
        if re.search(r"\bc\.?f\.?o\b|chief fin|principal financial", t) and \
                not re.search(r"deputy|assistant|asst", t):
            return "CFO"
        return "Other officer"
    if "DIRECTOR" in rel:
        return "Director"
    if "TENPERCENT" in rel:
        return "10% owner"
    return "Other"


def read_submissions() -> pd.DataFrame:
    """pass 1: every Form 3/4/5 filing, only to learn which company used which ticker when"""
    out = []
    for y, q in quarters():
        p = SEC_DIR / f"{y}q{q}_form345.zip"
        if not p.exists():
            print(f"  {y} Q{q}: zip missing, skipped")
            continue
        with zipfile.ZipFile(p) as zf:
            s = read_tsv(zf, "SUBMISSION.TSV", ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK",
                                                "ISSUERNAME", "ISSUERTRADINGSYMBOL"])
        s["q"] = f"{y}q{q}"
        out.append(s)
    s = pd.concat(out, ignore_index=True)
    s["cik"] = pd.to_numeric(s["ISSUERCIK"], errors="coerce")
    s["filing_date"] = sec_date(s["FILING_DATE"])
    s = s.dropna(subset=["cik", "filing_date"])
    s["cik"] = s["cik"].astype("int64")
    s = s.rename(columns={"ACCESSION_NUMBER": "accession", "DOCUMENT_TYPE": "form", "ISSUERNAME": "issuer",
                          "ISSUERTRADINGSYMBOL": "symbol"})
    return s[["accession", "q", "cik", "filing_date", "form", "issuer", "symbol"]]


# ── which company is which S&P ticker ──────────────────────────────────────────

def match_universe(sub: pd.DataFrame, sp: pd.DataFrame) -> pd.DataFrame:
    """
    cik, ticker, start, end, how: which SEC company was the S&P 500 member behind a ticker, and when.
    The ticker field on the filings is the only link, and it is messier than a lookup:
      - other companies use the same letters (share class "A", "SEE remarks", a small bank with ticker T),
        so only one company at a time can own a ticker: the one filing under it at the end of the spell
      - the history file uses the latest name for old years (AABA for Yahoo, BKNG for Priceline). The owner
        at the end of the spell is then the member for the whole spell, also while it filed as YHOO
      - some companies got a new cik and kept the ticker (Alphabet, Disney, Medtronic). If another company
        filed under the ticker right until the owner started, it is the same business under its old cik
    What this cannot see (old cik under an old ticker, like Walgreen WAG -> WBA) shows up in sp500_gaps.csv
    and is fixed by hand in data/manual/cik_overrides.csv.
    """
    pairs = sub[["cik", "symbol"]].drop_duplicates()
    pairs["key"] = pairs["symbol"].map(symbol_keys)
    pairs = pairs.explode("key").dropna(subset=["key"])
    pairs = pairs[pairs["key"].isin(set(sp["key"]))]
    f = sub[["cik", "symbol", "filing_date"]].merge(pairs, on=["cik", "symbol"])
    by_key = {k: g for k, g in f.groupby("key")}
    last_day = sub["filing_date"].max()
    pad = pd.Timedelta(days=30)

    def use(w, cik):
        """when did this company file under the ticker: 5th filing, median, 5th from the end (a few
        stragglers come years early or late), number"""
        d = np.sort(w.loc[w.cik == cik, "filing_date"].values)
        return (pd.Timestamp(d[min(4, len(d) - 1)]), pd.Timestamp(d[len(d) // 2]),
                pd.Timestamp(d[max(len(d) - 5, 0)]), len(d))

    rows = []
    for r in sp.itertuples():
        start = max(r.start, FIRST_DAY)
        end = r.end if pd.notna(r.end) else last_day + pd.Timedelta(days=1)
        g = by_key.get(r.key)
        found = []
        if g is not None:
            w = g[(g.filing_date >= start - pad) & (g.filing_date < end + pad)]
            n_tail = w[w.filing_date >= min(end, last_day) - pd.Timedelta(days=730)].groupby("cik").size()
            n_all = w.groupby("cik").size()
            owner = None
            if len(n_tail) and n_tail.max() >= 3:
                owner = n_tail.idxmax()
            elif len(n_all) and n_all.max() >= 3:
                owner = n_all.idxmax()
            if owner is not None:
                chain = [owner]
                cuts = []
                for _ in range(3):
                    first, _, _, _ = use(w, chain[-1])
                    best = None
                    for c in n_all[n_all >= 10].index:
                        if c in chain:
                            continue
                        _, med, late, n = use(w, c)
                        # it used the ticker before the owner did, and stopped about when the owner started
                        if med < first and pd.Timedelta(days=-365) <= first - late <= pd.Timedelta(days=180):
                            if best is None or n > best[1]:
                                best = (c, n)
                    if best is None:
                        break
                    chain.append(best[0])
                    cuts.append(first)
                # the chain runs backwards in time: owner, the one before it, ...
                bounds = [r.end] + cuts + [r.start]
                for i, c in enumerate(chain):
                    s0, e0 = bounds[i + 1], bounds[i]
                    if i > 0 and i == len(chain) - 1:
                        s0 = r.start
                    if pd.isna(e0) or s0 < e0:
                        found.append((c, s0, e0, "ticker on filings" if i == 0 else "same ticker, earlier cik"))
            else:
                # nobody filed under this ticker during the spell: take whoever used it in the 3 years after
                n_after = g[(g.filing_date >= end) & (g.filing_date < end + pd.Timedelta(days=3 * 365))] \
                    .groupby("cik").size()
                if len(n_after) and n_after.max() >= 3:
                    found = [(n_after.idxmax(), r.start, r.end, "ticker used after the spell")]
        for c, s0, e0, how in found:
            rows.append({"cik": c, "ticker": r.ticker, "start": s0, "end": e0, "how": how})
        if not found:
            rows.append({"cik": np.nan, "ticker": r.ticker, "start": r.start, "end": r.end, "how": "no match"})

    # hand-made fixes: ticker, cik, start, end (start/end empty = the whole spell)
    ov_path = MANUAL_DIR / "cik_overrides.csv"
    if ov_path.exists():
        ov = pd.read_csv(ov_path, comment="#")
        for o in ov.itertuples():
            for r in sp[sp.ticker == o.ticker].itertuples():
                s0 = max(pd.Timestamp(o.start), r.start) if pd.notna(o.start) else r.start
                e0 = pd.Timestamp(o.end) if pd.notna(o.end) else r.end
                if pd.notna(r.end) and (pd.isna(e0) or e0 > r.end):
                    e0 = r.end
                if pd.isna(e0) or s0 < e0:
                    rows.append({"cik": int(o.cik), "ticker": o.ticker, "start": s0, "end": e0, "how": "manual"})
    u = pd.DataFrame(rows)
    # a ticker that is fixed by hand is no longer unmatched
    fixed = set(u.loc[u.how == "manual", "ticker"])
    u = u[~((u.how == "no match") & u.ticker.isin(fixed))]

    names = sub.sort_values("filing_date").groupby("cik")["issuer"].last()
    u["issuer"] = u["cik"].map(names)
    return u.sort_values(["ticker", "start"]).reset_index(drop=True)


def gaps(sub: pd.DataFrame, u: pd.DataFrame, sp: pd.DataFrame) -> pd.DataFrame:
    """S&P ticker-years without a single Form 4. An S&P 500 company has insider filings every year"""
    f4 = sub[sub.form.isin(["4", "4/A"])]
    last_day = sub.filing_date.max()
    f4 = f4[["cik", "filing_date"]].merge(u.dropna(subset=["cik"])[["cik", "ticker", "start", "end"]], on="cik")
    f4 = f4[(f4.filing_date >= f4.start) & (f4.filing_date < f4.end.fillna(pd.Timestamp("2100-01-01")))]
    n = f4.groupby(["ticker", f4.filing_date.dt.year]).size()
    rows = []
    for r in sp.itertuples():
        end = r.end if pd.notna(r.end) else last_day
        for y in range(max(r.start.year, 2006), min(end.year, last_day.year) + 1):
            # skip a first or last year the ticker was only in for a few months
            a = max(r.start, pd.Timestamp(y, 1, 1))
            b = min(end, last_day, pd.Timestamp(y, 12, 31))
            if (b - a).days >= 120 and n.get((r.ticker, y), 0) == 0:
                rows.append({"ticker": r.ticker, "year": y})
    return pd.DataFrame(rows, columns=["ticker", "year"])


# ── the trades ─────────────────────────────────────────────────────────────────

def read_trades(sub: pd.DataFrame, ciks: set) -> pd.DataFrame:
    """pass 2: open-market buys and sales on Form 4 and 4/A for the S&P companies, with the insider's role"""
    keep = sub[sub.cik.isin(ciks) & sub.form.isin(["4", "4/A"])]
    out = []
    for qname, sq in keep.groupby("q"):
        acc = set(sq.accession)
        with zipfile.ZipFile(SEC_DIR / f"{qname}_form345.zip") as zf:
            tr = read_tsv(zf, "NONDERIV_TRANS.TSV", ["ACCESSION_NUMBER", "NONDERIV_TRANS_SK", "SECURITY_TITLE",
                                                     "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
                                                     "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD",
                                                     "SHRS_OWND_FOLWNG_TRANS", "DIRECT_INDIRECT_OWNERSHIP"])
            tr = tr[tr.TRANS_CODE.isin(["P", "S"]) & tr.ACCESSION_NUMBER.isin(acc)]
            ro = read_tsv(zf, "REPORTINGOWNER.TSV", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME",
                                                     "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"])
            ro = ro[ro.ACCESSION_NUMBER.isin(set(tr.ACCESSION_NUMBER))].copy()
        if tr.empty:
            continue
        # a filing can have several owners (a fund and its partners). The trade is counted once, under the
        # most senior of them
        ro = ro.merge(sq[["accession", "issuer"]], left_on="ACCESSION_NUMBER", right_on="accession", how="left")
        ro["role"] = [classify_role(a, b, c) for a, b, c in
                      zip(ro.RPTOWNER_RELATIONSHIP, ro.RPTOWNER_TITLE, ro.issuer)]
        ro["rank"] = ro.role.map({r: i for i, r in enumerate(ROLES)})
        ro["n_owners"] = ro.groupby("ACCESSION_NUMBER").RPTOWNERCIK.transform("nunique")
        lead = ro.sort_values("rank", kind="stable").drop_duplicates("ACCESSION_NUMBER")
        m = tr.merge(lead[["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME", "RPTOWNER_TITLE", "role",
                           "n_owners"]], on="ACCESSION_NUMBER", how="left")
        m = m.merge(sq[["accession", "cik", "filing_date", "form", "issuer", "symbol"]],
                    left_on="ACCESSION_NUMBER", right_on="accession", how="inner")
        out.append(m)
        print(f"  {qname}: {len(m):>6} buys and sales")
    t = pd.concat(out, ignore_index=True)

    t["trade_date"] = sec_date(t.TRANS_DATE)
    t["shares"] = pd.to_numeric(t.TRANS_SHARES, errors="coerce")
    t["price"] = pd.to_numeric(t.TRANS_PRICEPERSHARE, errors="coerce")
    t["owned_after"] = pd.to_numeric(t.SHRS_OWND_FOLWNG_TRANS, errors="coerce")
    t["owner_cik"] = pd.to_numeric(t.RPTOWNERCIK, errors="coerce")
    t["role"] = t.role.fillna("Other")
    t = t.rename(columns={"TRANS_CODE": "code", "RPTOWNERNAME": "owner", "RPTOWNER_TITLE": "title",
                          "SECURITY_TITLE": "security", "DIRECT_INDIRECT_OWNERSHIP": "direct",
                          "TRANS_ACQUIRED_DISP_CD": "acq_disp", "NONDERIV_TRANS_SK": "sk"})
    for c in ("owner", "title", "security", "issuer", "symbol"):
        t[c] = t[c].fillna("").str.replace('"', "", regex=False).str.replace(r"\s+", " ", regex=True).str.strip()
    return t


def clean(t: pd.DataFrame) -> pd.DataFrame:
    """the filters, each one counted so the report can say what was dropped"""
    n0 = len(t)
    log = [("buys and sales read", n0)]

    def step(name, mask):
        nonlocal t
        log.append((f"dropped: {name}", int((~mask).sum())))
        t = t[mask]

    step("no trade date, shares or owner", t.trade_date.notna() & (t.shares > 0) & t.owner_cik.notna())
    # a buy must add shares and a sale remove them
    step("buy/sale code and acquired/disposed flag disagree",
         ((t.code == "P") & (t.acq_disp != "D")) | ((t.code == "S") & (t.acq_disp != "A")))
    step("no price or price of 0", t.price > 0)
    # preferred shares, notes and warrants are on the same form but are not the stock
    other = t.security.str.contains(r"prefer|\bpfd|\bpref\b|pref\.|\bnotes?\b|debenture|\bbonds?\b|warrant|"
                                    r"capital securities", case=False, regex=True)
    step("not the common stock (preferred, notes, warrants)", ~other)
    lag = (t.filing_date - t.trade_date).dt.days
    step("trade dated after the filing or more than 2 years before", (lag >= 0) & (lag <= 730))

    # amendments (4/A) mostly repeat the original filing. Keep only trades that were not reported before:
    # same company, insider, day and direction
    t = t.sort_values(["filing_date", "accession", "sk"], kind="stable")
    key = ["cik", "owner_cik", "trade_date", "code"]
    first_acc = t[t.form == "4"].groupby(key).accession.first().rename("first_acc")
    t = t.join(first_acc, on=key)
    step("amendment repeating a trade already filed", ~((t.form == "4/A") & t.first_acc.notna()))
    t = t.drop(columns="first_acc")

    # the same trade filed twice under different filings: keep the rows of the earliest filing
    key2 = ["cik", "owner_cik", "trade_date", "code", "shares", "price"]
    first = t.groupby(key2).accession.transform("first")
    step("same trade in a second filing", t.accession == first)

    t["usd"] = t.shares * t.price
    t["lag_days"] = (t.filing_date - t.trade_date).dt.days
    log.append(("kept", len(t)))
    return t, pd.DataFrame(log, columns=["step", "rows"])


def flag_members(t: pd.DataFrame, u: pd.DataFrame) -> pd.DataFrame:
    """in_sp500: was the company in the index on the day the filing became public. sp_ticker: under which name"""
    um = u.dropna(subset=["cik"]).copy()
    um["cik"] = um.cik.astype("int64")
    um["end_f"] = um.end.fillna(pd.Timestamp("2100-01-01"))
    x = t[["cik", "filing_date"]].drop_duplicates().merge(um[["cik", "ticker", "start", "end_f"]], on="cik")
    x = x[(x.filing_date >= x.start) & (x.filing_date < x.end_f)]
    tick = x.groupby(["cik", "filing_date"]).ticker.agg(lambda s: "|".join(sorted(set(s)))).rename("sp_ticker")
    t = t.join(tick, on=["cik", "filing_date"])
    t["in_sp500"] = t.sp_ticker.notna()
    # tickers to look for a price under: every ticker the company had in the index, plus the tickers of the
    # companies it shares one with (Baker Hughes Inc is only known as BHGE here, its prices are under BKR)
    by_cik = um.groupby("cik").ticker.agg(set)
    by_ticker = um.groupby("ticker").cik.agg(set)
    allt = {}
    for cik, own in by_cik.items():
        seen = set(own)
        for tk in own:
            for other in by_ticker[tk]:
                seen |= by_cik[other]
        allt[cik] = "|".join(sorted(seen))
    t["tickers"] = t.cik.map(allt)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()
    if not a.no_download:
        download()

    sp = sp500_spells()
    sp[["ticker", "start", "end"]].to_csv(DATA_DIR / "sp500_members.csv", index=False)
    print(f"S&P 500: {sp.ticker.nunique()} tickers since 2006, {len(sp)} membership spells")

    print("pass 1: who filed under which ticker")
    sub = read_submissions()
    print(f"  {len(sub):,} filings, {sub.cik.nunique():,} companies, "
          f"{sub.filing_date.min().date()} to {sub.filing_date.max().date()}")

    u = match_universe(sub, sp)
    u.to_csv(DATA_DIR / "universe.csv", index=False)
    miss = u[u.cik.isna()]
    print(f"universe.csv: {u.cik.nunique()} companies for {u.ticker.nunique()} tickers, "
          f"{miss.ticker.nunique()} tickers without a company: {' '.join(sorted(miss.ticker.unique()))}")
    g = gaps(sub, u, sp)
    g.to_csv(DATA_DIR / "sp500_gaps.csv", index=False)
    print(f"sp500_gaps.csv: {len(g)} ticker-years without a Form 4 ({g.ticker.nunique()} tickers)")

    print("pass 2: the trades")
    t = read_trades(sub, set(u.cik.dropna().astype("int64")))
    t, log = clean(t)
    t = flag_members(t, u)
    cols = ["accession", "form", "cik", "issuer", "symbol", "sp_ticker", "tickers", "in_sp500", "filing_date",
            "trade_date", "lag_days", "code", "shares", "price", "usd", "owned_after", "direct", "security",
            "owner_cik", "owner", "role", "title", "n_owners"]
    t = t.sort_values(["filing_date", "cik", "owner_cik", "trade_date"])[cols]
    t.to_parquet(DATA_DIR / "trades.parquet", index=False)
    log.to_csv(DATA_DIR / "ingest_log.csv", index=False)

    print(log.to_string(index=False))
    m = t[t.in_sp500]
    print(f"trades.parquet: {len(t):,} rows, {len(m):,} filed while in the S&P 500 "
          f"({(m.code == 'P').sum():,} buys, {(m.code == 'S').sum():,} sales)")
    print(m.groupby(["role", "code"]).size().unstack(fill_value=0).reindex(ROLES).fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
