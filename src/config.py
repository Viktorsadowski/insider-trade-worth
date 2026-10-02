"""
Paths, sample range and settings shared by all scripts.

Keys and the SEC contact line are read from a .env file in the project root (not in git), or from normal
env vars. See .env.example.
Set INSIDER_DATA_DIR to keep the data folder somewhere else (e.g. outside OneDrive).
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_env():
    # tiny .env reader so there is no extra dependency. KEY=value per line, real env vars win
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()

DATA_DIR = Path(os.environ.get("INSIDER_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
SEC_DIR = RAW_DIR / "sec"
PRICE_DIR = DATA_DIR / "prices"
MANUAL_DIR = ROOT / "data" / "manual"      # small hand-made tables, these are in git
FIG_DIR = ROOT / "figures"

# SEC wants a name and an email in the User-Agent, and max 10 requests a second
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "")
TIINGO_API_KEY = os.environ.get("TIINGO_API_KEY", "")

# SEC insider data sets: first and last quarter (year, quarter). 2006 Q1 is the first one they publish
FIRST_Q, LAST_Q = (2006, 1), (2026, 2)
# the SEC moved the folder in 2026, older quarters are still under the first one
SEC_URLS = ["https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{y}q{q}_form345.zip",
            "https://www.sec.gov/files/datastandardsinnovation/data/insider-transactions-data-sets/{y}q{q}_form345.zip"]

# point-in-time S&P 500 members since 1996, kept up to date by fja05680
SP500_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
             "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")

# prices from a bit before the first filing, so the run-up before a trade can be measured
PRICE_START = "2005-06-01"
BENCHMARK = "SPY"

# holding periods in trading days: 1 day, 1 week, 1, 3, 6, 12 and 24 months
HORIZONS = [1, 5, 21, 63, 126, 252, 504]
MAIN_H = 63          # 3 months: the horizon used when a chart or a sentence can only show one

# roles, most senior first. An insider with several roles gets the first one that fits
ROLES = ["CEO", "CFO", "Other officer", "Director", "10% owner", "Other"]

# chart colours, same as nba-moneyball
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, LIGHT_GREY = "#2a78d6", "#eb6834", "#c9c8c3"


def quarters():
    y, q = FIRST_Q
    while (y, q) <= LAST_Q:
        yield y, q
        y, q = (y, q + 1) if q < 4 else (y + 1, 1)
