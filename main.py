import asyncio
import gc
import bisect
import traceback
import hashlib
import html as html_lib
import hmac
import json
import os
import pickle
import re
import secrets
import smtplib
import sqlite3
import time
import urllib.parse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

import bs4
import numpy as np
import pandas as pd
import requests
import uvicorn
import yfinance as yf
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, PlainTextResponse, Response
from openai import OpenAI, RateLimitError

load_dotenv()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DB_FILE = os.getenv("DB_FILE", "users.db")
AI_BASE_URL = os.getenv("AI_BASE_URL", "http://localhost:11434/v1")
AI_API_KEY = os.getenv("AI_API_KEY", "ollama")
AI_MODEL = os.getenv("AI_MODEL", "llama3")
SESSION_TTL = 30 * 24 * 3600

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "")
SENDER_NAME = os.getenv("SENDER_NAME", "QUANTIFY")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
BREVO_API_KEY = os.getenv("BREVO_API_KEY", "")
CONTACT_NOTIFY_EMAIL = os.getenv("CONTACT_NOTIFY_EMAIL", "quantify.app.official@gmail.com")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
LEMONSQUEEZY_CHECKOUT_URL = os.getenv("LEMONSQUEEZY_CHECKOUT_URL", "")
LEMONSQUEEZY_WEBHOOK_SECRET = os.getenv("LEMONSQUEEZY_WEBHOOK_SECRET", "")
GUMROAD_CHECKOUT_URL = os.getenv("GUMROAD_CHECKOUT_URL", "")
GUMROAD_ACCESS_TOKEN = os.getenv("GUMROAD_ACCESS_TOKEN", "")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
POLYGON_BASE_URL = "https://api.polygon.io"
SITE_URL = os.getenv("SITE_URL", "https://quantify.trading")
TRIAL_DAYS = 7
TRIAL_REMINDER_HOURS_BEFORE = 48

# Constituent refresh is deliberately infrequent.
UNIVERSE_TTL = 24 * 3600
POLYGON_CONCURRENCY = int(os.getenv("POLYGON_CONCURRENCY", "20"))
HISTORICAL_TTL = 600
HISTORICAL_FALLBACK_RETENTION = 24 * 3600  # how long a stale entry stays servable as a
                                            # last-known-good fallback if a live fetch fails
                                            # (separate from HISTORICAL_TTL, which governs
                                            # normal cache-hit freshness)
NEWS_TTL = 900
SHORT_INTEREST_TTL = 24 * 3600
CONSTITUENT_HTTP_TIMEOUT = 12

DATA_DIR = Path(os.getenv("DATA_DIR", "."))
UNIVERSE_FILE = DATA_DIR / "universe_cache.json"
BACKTEST_FILE = DATA_DIR / "backtest_cache.json"
HISTORICAL_SNAPSHOT_FILE = DATA_DIR / "historical_snapshot.pkl"
SECTOR_FILE = DATA_DIR / "sector_cache.json"
SECTOR_TTL = 7 * 24 * 3600  # sector/market-cap barely changes -- weekly refresh is plenty
BACKTEST_SAMPLE_SIZE = 200  # Reverted from a brief full-universe (600) experiment: that roughly
                             # doubled our total yfinance request volume (regular scan + backtest
                             # both hitting ~518 tickers), which is the likely trigger for yfinance
                             # rate-limiting us wholesale and the resulting memory/uptime incident.
                             # The dynamic universe-note copy already handles this sample size
                             # honestly, so reverting is copy-safe.
BACKTEST_REFRESH_SECONDS = 7 * 24 * 3600
# Round-trip cost applied to every backtested signal: commission is zero at retail
# brokers now, so this is spread plus slippage on large-cap S&P/Nasdaq names. 10bp is
# deliberately on the pessimistic side for names this liquid -- a backtest should not
# flatter itself on the one assumption nobody can check.
BACKTEST_ROUND_TRIP_COST_PCT = 0.10
# Bump when the shape or meaning of the results changes. A cached result from an older
# version is discarded and recomputed rather than rendered, so the site never shows
# numbers whose methodology no longer matches what the page says it did.
BACKTEST_SCHEMA_VERSION = 2
BACKTEST_CACHE = {"computed_at": None, "results": None, "error": None}
MARKET_AI_SUMMARY_FILE = DATA_DIR / "market_ai_summary_cache.json"
MARKET_AI_SUMMARY_CACHE = {"scan_date": None, "generated_at": None, "headline": None, "summary": None, "error": None}
HIGH_SCORE_ALERT_THRESHOLD = 90
HIGH_SCORE_DIGEST_FILE = DATA_DIR / "high_score_digest_state.json"
HIGH_SCORE_DIGEST_STATE = {"sent_date": None}

SNP500_SOURCE = os.getenv(
    "SNP500_SOURCE",
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
)
SNP500_FALLBACK_SOURCE = os.getenv(
    "SNP500_FALLBACK_SOURCE",
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
)
NASDAQ100_SOURCE = os.getenv(
    "NASDAQ100_SOURCE",
    "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies",
)
NASDAQ100_FALLBACK_SOURCE = os.getenv(
    "NASDAQ100_FALLBACK_SOURCE",
    "https://www.nasdaq.com/products/global-indexes/nasdaq-100/companies",
)

app = FastAPI(title="QUANTIFY.")

# Without this, the browser can serve a stale, fully-rendered copy of a logged-in page
# straight from its back-forward cache after logout -- hitting Back looks like you're
# still logged in (the DOM is just a frozen snapshot; no request ever reaches the
# server, so the session being deleted server-side doesn't matter until the user
# actually clicks something). Cache-Control: no-store is what disables bfcache for a
# page in every major browser, not just disk/memory caching.
_NO_STORE_PREFIXES = ("/terminal", "/market", "/watchlist", "/backtest", "/portfolio", "/settings", "/subscription", "/contact", "/api/")


@app.middleware("http")
async def no_store_authenticated_pages(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith(_NO_STORE_PREFIXES):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
        response.headers["Pragma"] = "no-cache"
    return response


# Health checks (Render's own + our own deploy-verification curls) hit "/" on every
# request just like a real visitor, and would otherwise permanently inflate traffic
# counts -- filter anything that doesn't look like an actual browser.
_PAGE_VIEW_BOT_UA_MARKERS = (
    # generic self-identifying crawlers ("bot" alone covers googlebot, bingbot, gptbot,
    # claudebot, petalbot, semrushbot, ahrefsbot, discordbot, twitterbot, slackbot, ...)
    "bot", "spider", "crawl", "slurp", "scrapy", "archiver", "feedfetcher",
    # HTTP clients and scripts
    "curl", "wget", "python-requests", "python-httpx", "python-urllib", "go-http-client",
    "okhttp", "java/", "libwww", "httpclient", "axios", "node-fetch", "guzzle", "postman",
    # uptime / infrastructure probes
    "render", "uptimerobot", "pingdom", "monitor", "statuscake", "betteruptime",
    # headless browsers and auditing tools
    "headlesschrome", "phantomjs", "puppeteer", "playwright", "selenium", "lighthouse",
    "google-inspectiontool", "chrome-privacy-preserving-prefetch",
    # link-preview fetchers: these fire once per link paste on X/Discord/Slack/iMessage,
    # so without them every post inflates the visitor count before a human clicks.
    "facebookexternalhit", "whatsapp", "telegram", "skypeuripreview", "embedly",
    "quora link preview", "vkshare", "redditbot", "discordbot", "linkedinbot",
)


def _looks_like_bot(user_agent: str) -> bool:
    ua = user_agent.lower()
    if not ua:
        return True
    return any(marker in ua for marker in _PAGE_VIEW_BOT_UA_MARKERS)


# Vulnerability scanners (probing for /wp-admin, /.git/config, etc.) fake a real
# browser User-Agent, so the UA check above doesn't catch them -- an allowlist of the
# site's actual page routes is the only reliable filter, since a scanner's made-up path
# can never be one of these.
_PAGE_VIEW_ALLOWED_PATHS = {
    "/", "/pricing", "/faq", "/about", "/demo", "/stocks", "/record", "/terms", "/privacy", "/accept-disclaimer",
    "/login", "/signup", "/check-email", "/verify-email", "/forgot-password",
    "/reset-password", "/terminal", "/market", "/watchlist", "/backtest",
    "/portfolio", "/subscription", "/contact", "/settings",
}


# Channel attribution. utm_source values in the wild are inconsistent (x / twitter /
# tw all mean the same channel), so everything is folded onto one canonical name before
# it is stored -- otherwise the dashboard splits one channel across three rows.
_CHANNEL_ALIASES = {
    "x": "x", "twitter": "x", "tw": "x", "t.co": "x", "x.com": "x",
    "reddit": "reddit", "rd": "reddit", "r": "reddit",
    "ig": "ig", "instagram": "ig", "insta": "ig",
}
# Referrer fallback, so links already posted without utm params are still attributed.
# Matched against the referring host, longest suffix first.
_CHANNEL_REFERRER_HOSTS = (
    ("t.co", "x"), ("twitter.com", "x"), ("x.com", "x"),
    ("reddit.com", "reddit"), ("redd.it", "reddit"), ("com.reddit.frontpage", "reddit"),
    ("instagram.com", "ig"), ("com.instagram.android", "ig"),
)
PRIMARY_CHANNELS = ("x", "reddit", "ig")


def _normalize_channel(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    cleaned = re.sub(r"[^a-z0-9_.-]", "", raw.strip().lower())[:32]
    if not cleaned:
        return None
    return _CHANNEL_ALIASES.get(cleaned, cleaned)


def _channel_from_referrer(referrer: str) -> Optional[str]:
    if not referrer:
        return None
    host = (urllib.parse.urlparse(referrer).netloc or referrer).lower()
    if "quantify.trading" in host:
        return None
    for needle, channel in _CHANNEL_REFERRER_HOSTS:
        if needle in host:
            return channel
    return None


def _resolve_channel(request: Request, referrer: str) -> Optional[str]:
    """First-touch attribution: an explicit ?utm_source wins, then the channel already
    stored for this browser, then whatever the referring host implies. Deliberately
    first-touch -- the channel that first brought someone here is the one that earned
    the signup, not whichever page they happened to be on the day they converted."""
    explicit = _normalize_channel(request.query_params.get("utm_source"))
    if explicit:
        return explicit
    stored = _normalize_channel(request.cookies.get("qtfy_src"))
    if stored:
        return stored
    return _channel_from_referrer(referrer)


def _log_page_view(path: str, visitor_id: str, referrer: str, utm_source: Optional[str]):
    try:
        conn = db()
        conn.execute(
            "INSERT INTO page_views(path,visitor_id,referrer,created_at,utm_source) VALUES(?,?,?,?,?)",
            (path, visitor_id, referrer[:300], time.time(), utm_source),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Error: {type(e).__name__}] page view log failed: {e}")


# Named so a typo can never silently create a fourth event stream that nothing reads.
EVENT_DEMO_VIEW = "demo_view"
EVENT_TERMINAL_VIEW = "terminal_view"
EVENT_SIGNUP = "signup_completed"
EVENT_PAYMENT = "payment_completed"
TRACKED_EVENTS = (EVENT_DEMO_VIEW, EVENT_TERMINAL_VIEW, EVENT_SIGNUP, EVENT_PAYMENT)
_EVENT_PATHS = {"/demo": EVENT_DEMO_VIEW, "/terminal": EVENT_TERMINAL_VIEW}


def _log_event(name: str, visitor_id: Optional[str] = None,
               utm_source: Optional[str] = None, email: Optional[str] = None):
    try:
        conn = db()
        conn.execute(
            "INSERT INTO events(name,visitor_id,utm_source,email,created_at) VALUES(?,?,?,?,?)",
            (name, visitor_id, utm_source, email, time.time()),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Error: {type(e).__name__}] event log failed ({name}): {e}")


def _log_payment_event(email: str):
    """A payment arrives by webhook with no browser cookie, so the only way to attribute
    it is the channel frozen on the account at signup. Runs entirely in a worker thread --
    both the lookup and the insert -- so it never blocks the event loop."""
    channel = None
    try:
        conn = db()
        row = conn.execute("SELECT signup_utm_source FROM users WHERE email=?", (email,)).fetchone()
        conn.close()
        if row:
            channel = row["signup_utm_source"]
    except Exception as e:
        print(f"[Error: {type(e).__name__}] payment attribution lookup failed for {email}: {e}")
    _log_event(EVENT_PAYMENT, None, channel, email)


@app.middleware("http")
async def track_page_views(request: Request, call_next):
    path = request.url.path
    # Visit any page once with ?notrack=1 (e.g. https://quantify.trading/?notrack=1) to
    # opt this browser out of analytics permanently -- for recording demo videos etc.
    # without polluting real visitor stats. Sets a long-lived cookie; to undo it, clear
    # cookies for the site.
    opted_out = request.cookies.get("qtfy_notrack") == "1" or request.query_params.get("notrack") == "1"
    should_track = (
        request.method == "GET"
        # /stock/<TICKER> is one real page per ticker, so it can't be an exact-match
        # entry -- without the prefix check every visit from search would go uncounted.
        and (path in _PAGE_VIEW_ALLOWED_PATHS or path.startswith("/stock/"))
        and not opted_out
        and not _looks_like_bot(request.headers.get("user-agent", ""))
    )
    existing_visitor_id = request.cookies.get("qtfy_vid")
    visitor_id = existing_visitor_id or secrets.token_hex(16)
    referrer = request.headers.get("referer", "")
    channel = _resolve_channel(request, referrer)
    request.state.qtfy_visitor_id = visitor_id
    request.state.qtfy_channel = channel
    response = await call_next(request)
    # A 4xx/5xx is not a visit. Vulnerability scanners probing made-up paths are already
    # filtered by the allowlist, but /stock/<TICKER> is prefix-matched and 404s for a
    # ticker that does not exist, and that should not count either. Redirects (303 off a
    # gated page) are real visits and stay counted.
    delivered = response.status_code < 400
    if should_track and delivered:
        asyncio.create_task(asyncio.to_thread(_log_page_view, path, visitor_id, referrer, channel))
        # Two of the four tracked events are page views, so they are recorded here rather
        # than inside the route handlers -- same bot, notrack and status filtering, one
        # place to change. Only a 200 counts: /terminal 303-redirects logged-out visitors
        # and that is not a terminal view.
        event_name = _EVENT_PATHS.get(path)
        if event_name and response.status_code == 200:
            asyncio.create_task(asyncio.to_thread(_log_event, event_name, visitor_id, channel, None))
    if not existing_visitor_id:
        response.set_cookie("qtfy_vid", visitor_id, max_age=365 * 86400, httponly=True, samesite="lax")
    # First-touch: only written when the browser has no channel yet, so a visitor who
    # arrives from Reddit and later clicks an X link still counts as Reddit.
    if channel and not _normalize_channel(request.cookies.get("qtfy_src")):
        response.set_cookie("qtfy_src", channel, max_age=365 * 86400, httponly=True, samesite="lax")
    if request.query_params.get("notrack") == "1" and request.cookies.get("qtfy_notrack") != "1":
        response.set_cookie("qtfy_notrack", "1", max_age=5 * 365 * 86400, httponly=True, samesite="lax")
    return response


ai_client = None
try:
    ai_client = OpenAI(base_url=AI_BASE_URL, api_key=AI_API_KEY)
except Exception as e:
    print(f"[Error: {type(e).__name__}] OpenAI client initialization failed: {e}")
    ai_client = None

CACHE = {
    "historical": {},
    "news": {},
    "earnings": {},
}
UNIVERSE = []
UNIVERSE_META = {"sp500": [], "nasdaq100": []}
UNIVERSE_STATUS = {"ready": False, "source": None, "updated_at": None, "error": None}
UNIVERSE_LOCK = asyncio.Lock()
BATCH_LOCK = asyncio.Lock()
BATCH_STATUS = {"running": False, "processed": 0, "total": 0, "saved": 0, "started_at": None, "finished_at": None, "error": None}
AI_STATUS = {"running": False, "processed": 0, "total": 0, "ready": 0, "started_at": None, "finished_at": None, "error": None}
AI_TASK = None
AI_QUOTA_EXHAUSTED_DATE = None  # date_str() of the last day the AI provider reported a tokens-per-day cap hit
AI_CONCURRENCY = max(1, int(os.getenv("AI_CONCURRENCY", "4")))
QUANT_PASS_THRESHOLD = float(os.getenv("QUANT_PASS_THRESHOLD", "83"))
# Each day's highest-scoring tickers that did NOT qualify are recorded to a SEPARATE
# watch list on /record -- tracked out of curiosity, never mixed into the strategy's
# actual numbers. Defined as a top-N rather than a score threshold on purpose: the score
# distribution is bimodal with a wide empty gap (non-passing tops out around 60, passing
# starts at 83), so any fixed "within X points of the cutoff" band would never match.
NEAR_MISS_COUNT = int(os.getenv("NEAR_MISS_COUNT", "5"))
# Validated overnight against real 2yr history for the full ~518-ticker universe, across
# three separate sweeps (500 -> 3000 -> ~10000 distinct entry rules, results identical
# across all three -- this is a converged, not a lucky, result): a long-term uptrend
# pullback beats the prior weighted RSI/MACD/momentum formula out-of-sample at the 60d
# and 90d horizons, on a LARGER validated sample than the formula it replaced (60d win
# rate 62.1% vs 54.5%, 90d 74.6% vs 62.6%, n=905/639 vs n=574/348).
PULLBACK_MIN = 0.10
PULLBACK_MAX = 0.25
PULLBACK_CENTER = (PULLBACK_MIN + PULLBACK_MAX) / 2
PULLBACK_HALF_WIDTH = (PULLBACK_MAX - PULLBACK_MIN) / 2
OVERALL_SCORE_THRESHOLD = float(os.getenv("OVERALL_SCORE_THRESHOLD", "50"))
AI_PROMPT_VERSION = 4
# Scans run multiple times a day now, but the AI pass only used to re-run once per
# ticker per day (matched to the old once-daily scan cadence) -- alpha_score could
# drift over the day while timing_score/the AI's whole risk review stayed frozen
# from whichever scan first ran, so overall_score silently blended a fresh number
# with a stale one. Re-score only when alpha_score has actually moved enough to matter,
# not on every scan -- re-running AI for every already-scored ticker every scan would
# burn through the AI provider's daily token quota far faster (see
# AI_QUOTA_EXHAUSTED_DATE) for very little real benefit on tickers that haven't changed.
AI_RESCORE_ALPHA_DRIFT = float(os.getenv("AI_RESCORE_ALPHA_DRIFT", "4.0"))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
# Full-universe recompute at four fixed checkpoints ET, weekdays, instead of hourly --
# the strategy's own signal (20d-high pullback + 200d SMA) is daily-bar-based, so hourly
# never bought anything except a bigger slice of the shared AI token quota (the automatic
# per-ticker AI review re-runs after every scan). Chosen to land on the moments that
# actually matter for this setup rather than an arbitrary hourly tick: final pre-market
# positioning, the open's immediate reaction, where things settle an hour in, and the
# day's final result after the close.
SCAN_TIMES_ET = [(9, 0), (9, 40), (10, 30), (16, 0)]
SCAN_RETRY_INTERVAL_SECONDS = 15 * 60
SCAN_MAX_ATTEMPTS_PER_SLOT = 3
# The AI market summary used to regenerate on every hourly scan (up to 11x/day), which
# was a meaningful slice of the shared daily AI token quota for a feature that doesn't
# need to be that fresh. Decided against paying for a bigger quota -- instead this only
# runs once, on the first scan at or after the regular market close, and the same-day
# cache guard in generate_market_summary_ai() keeps it from re-running on the 5pm/6pm
# scans that follow.
MARKET_SUMMARY_AFTER_HOUR_ET = 16

LOGIN_ATTEMPTS = {}
LOGIN_MAX_ATTEMPTS = 5
LOGIN_LOCKOUT_SECONDS = 15 * 60

RESET_ATTEMPTS = {}
RESET_MAX_ATTEMPTS = 5

SEND_CODE_ATTEMPTS = {}
SEND_CODE_MAX_ATTEMPTS = 5

SIGNUP_ATTEMPTS = {}
SIGNUP_MAX_ATTEMPTS = 5

CONTACT_ATTEMPTS = {}
CONTACT_MAX_ATTEMPTS = 5


def _register_failed_attempt(store: dict, key: str) -> int:
    now = time.time()
    attempts = [t for t in store.get(key, []) if now - t < LOGIN_LOCKOUT_SECONDS]
    attempts.append(now)
    store[key] = attempts
    return len(attempts)


def _is_locked_out(store: dict, key: str, max_attempts: int) -> bool:
    now = time.time()
    attempts = [t for t in store.get(key, []) if now - t < LOGIN_LOCKOUT_SECONDS]
    store[key] = attempts
    return len(attempts) >= max_attempts


def _clear_attempts(store: dict, key: str):
    store.pop(key, None)
CACHE_STATUS = {
    "warming": False,
    "ready": False,
    "today_rows": 0,
    "started_at": None,
    "finished_at": None,
    "error": None,
}
WARMUP_TASK = None

# -----------------------------------------------------------------------------
# Database
# -----------------------------------------------------------------------------
def db():
    conn = sqlite3.connect(DB_FILE, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db():
    try:
        conn = db()
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1,
            reset_code_hash TEXT,
            reset_expires REAL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            expires_at REAL NOT NULL,
            FOREIGN KEY(email) REFERENCES users(email)
        );
        CREATE TABLE IF NOT EXISTS daily_scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            universe TEXT,
            price REAL NOT NULL,
            change_pct REAL NOT NULL,
            alpha_score REAL NOT NULL,
            rsi REAL NOT NULL,
            macd REAL NOT NULL,
            entry REAL,
            tp REAL,
            sl REAL,
            ai_report TEXT,
            short_percent REAL,
            created_at REAL NOT NULL,
            UNIQUE(scan_date, ticker)
        );
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            captured_at REAL NOT NULL,
            price REAL,
            alpha_score REAL
        );
        CREATE INDEX IF NOT EXISTS idx_scan_history_ticker_date ON scan_history(ticker, scan_date);
        -- Public track record. APPEND-ONLY BY DESIGN: the scanner INSERTs here and
        -- nothing else ever touches it. There must never be an UPDATE or DELETE
        -- against this table anywhere in the codebase, and no endpoint that lets a
        -- human add, edit or remove a row -- that is the entire point of the page it
        -- feeds. UNIQUE(record_date,ticker,list_type) + INSERT OR IGNORE means the
        -- first scan of the day that flags a ticker fixes its entry price, and the
        -- three later scans that day cannot change it.
        CREATE TABLE IF NOT EXISTS track_record (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            list_type TEXT NOT NULL,
            entry_price REAL NOT NULL,
            alpha_score REAL,
            overall_score REAL,
            timing_verdict TEXT,
            benchmark_price REAL,
            recorded_at REAL NOT NULL,
            prev_hash TEXT,
            row_hash TEXT NOT NULL,
            UNIQUE(record_date, ticker, list_type)
        );
        CREATE INDEX IF NOT EXISTS idx_track_record_list ON track_record(list_type, id);
        CREATE TABLE IF NOT EXISTS user_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            ticker TEXT NOT NULL,
            target_price REAL NOT NULL,
            is_sent INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            FOREIGN KEY(email) REFERENCES users(email)
        );
        CREATE TABLE IF NOT EXISTS portfolio_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            ticker TEXT NOT NULL,
            scan_date TEXT NOT NULL,
            price REAL,
            change_pct REAL,
            alpha_score REAL,
            rsi REAL,
            macd REAL,
            timing_score REAL,
            timing_verdict TEXT,
            ai_report TEXT,
            note TEXT,
            saved_at REAL NOT NULL,
            FOREIGN KEY(email) REFERENCES users(email)
        );
        CREATE TABLE IF NOT EXISTS watchlist_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            ticker TEXT NOT NULL,
            added_at REAL NOT NULL,
            UNIQUE(email, ticker),
            FOREIGN KEY(email) REFERENCES users(email)
        );
        CREATE TABLE IF NOT EXISTS page_views (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            visitor_id TEXT NOT NULL,
            referrer TEXT,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_page_views_created_at ON page_views(created_at);
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            visitor_id TEXT,
            utm_source TEXT,
            email TEXT,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at);
        -- Survives account deletion on purpose: without it, "delete account, sign up
        -- again" hands out an unlimited number of free trials. Stores no readable
        -- address -- only a peppered one-way hash that can be compared but not reversed.
        -- Company fundamentals, refreshed once a day. Separate from daily_scans because
        -- these barely move intraday and are fetched from a different source on a
        -- different schedule; mixing them would mean refetching balance sheets four
        -- times a day for numbers that change quarterly.
        CREATE TABLE IF NOT EXISTS fundamentals (
            ticker TEXT PRIMARY KEY,
            sector TEXT,
            market_cap REAL,
            trailing_pe REAL,
            price_to_book REAL,
            ev_ebitda REAL,
            peg REAL,
            return_on_equity REAL,
            profit_margin REAL,
            debt_to_equity REAL,
            current_ratio REAL,
            revenue_growth REAL,
            earnings_growth REAL,
            dividend_yield REAL,
            payout_ratio REAL,
            fetched_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS trial_ledger (
            email_hash TEXT PRIMARY KEY,
            first_trial_at REAL NOT NULL
        );
        """)
        # Channel attribution. page_views.utm_source is per-visit; users.signup_utm_source
        # is the frozen first-touch channel of the account, which is what lets a payment
        # (which arrives by webhook with no browser cookie) be traced back to a channel.
        pv_cols = {r[1] for r in conn.execute("PRAGMA table_info(page_views)").fetchall()}
        if "utm_source" not in pv_cols:
            conn.execute("ALTER TABLE page_views ADD COLUMN utm_source TEXT")
        user_cols = {r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
        if "password_hash" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
        if "salt" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN salt TEXT")

        user_info = conn.execute("PRAGMA table_info(users)").fetchall()
        legacy_password_col = next((r for r in user_info if r[1] == "password"), None)
        if legacy_password_col is not None and int(legacy_password_col[3]) == 1:
            # Legacy schema had a NOT NULL "password" column that current code never
            # populates, which made every new signup fail with an IntegrityError.
            conn.execute("PRAGMA foreign_keys=OFF")
            conn.execute("DROP TABLE IF EXISTS users_migrated")
            conn.execute("""
                CREATE TABLE users_migrated (
                    email TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    reset_code_hash TEXT,
                    reset_expires REAL
                )
            """)
            conn.execute("""
                INSERT INTO users_migrated (email,password_hash,salt,is_active,reset_code_hash,reset_expires)
                SELECT email,COALESCE(password_hash,''),COALESCE(salt,''),COALESCE(is_active,1),reset_code_hash,reset_expires
                FROM users
                WHERE password_hash IS NOT NULL AND salt IS NOT NULL
            """)
            conn.execute("DROP TABLE users")
            conn.execute("ALTER TABLE users_migrated RENAME TO users")
            conn.execute("PRAGMA foreign_keys=ON")

        user_cols = {r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
        if "pref_theme" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN pref_theme TEXT NOT NULL DEFAULT 'dark'")
        if "pref_theme_light_default" not in user_cols:
            # One-time: the column above still defaults new rows to 'dark' (SQLite can't
            # ALTER COLUMN...SET DEFAULT without a full table rebuild), and the app's own
            # default flipped to light. Flip every row still sitting on the old invisible
            # default so existing accounts see the new theme too, gated by this marker
            # column so it only ever runs once -- a later explicit dark-mode choice in
            # Settings is never touched by this again.
            conn.execute("ALTER TABLE users ADD COLUMN pref_theme_light_default INTEGER NOT NULL DEFAULT 0")
            conn.execute("UPDATE users SET pref_theme='light' WHERE pref_theme='dark'")
        if "pref_language" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN pref_language TEXT NOT NULL DEFAULT 'en'")
        if "pref_default_mode" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN pref_default_mode TEXT NOT NULL DEFAULT 'Long-Term Momentum Pullback'")
        if "pref_default_sort" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN pref_default_sort TEXT NOT NULL DEFAULT 'overall_score'")
        if "pref_default_view" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN pref_default_view TEXT NOT NULL DEFAULT 'list'")
        if "pref_high_score_alerts" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN pref_high_score_alerts INTEGER NOT NULL DEFAULT 0")
        if "verify_token_hash" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN verify_token_hash TEXT")
        if "verify_expires" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN verify_expires REAL")
        if "disclaimer_accepted_at" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN disclaimer_accepted_at REAL")
        if "created_at" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN created_at REAL")
        if "trial_ends_at" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN trial_ends_at REAL")
        if "subscription_status" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN subscription_status TEXT NOT NULL DEFAULT 'trial'")
        if "ls_customer_id" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN ls_customer_id TEXT")
        if "ls_subscription_id" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN ls_subscription_id TEXT")
        if "gumroad_subscription_id" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN gumroad_subscription_id TEXT")
        if "trial_reminder_sent_at" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN trial_reminder_sent_at REAL")
        if "trial_ended_email_sent_at" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN trial_ended_email_sent_at REAL")
        # Retention tracking. login_count/last_login_at only move on a real
        # re-authentication, which with a 30-day session cookie badly undercounts
        # return visits -- last_seen_at (written once per ET day on any authenticated
        # request) is the signal that actually answers "did they come back".
        if "last_login_at" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN last_login_at REAL")
        if "login_count" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN login_count INTEGER NOT NULL DEFAULT 0")
        if "last_seen_at" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN last_seen_at REAL")
        # Trial onboarding drip: one timestamp per email so a send can never repeat.
        if "welcome_sent_at" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN welcome_sent_at REAL")
        if "week1_sent_at" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN week1_sent_at REAL")
        if "trial_end_sent_at" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN trial_end_sent_at REAL")
        if "pref_marketing_emails" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN pref_marketing_emails INTEGER NOT NULL DEFAULT 1")
        if "unsub_token" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN unsub_token TEXT")
        if "signup_utm_source" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN signup_utm_source TEXT")
        # Google accounts get a random password the user never sees, so anything that
        # asks them to confirm with their password is impossible for them to satisfy.
        if "auth_provider" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN auth_provider TEXT")
        now_ts = time.time()
        conn.execute("UPDATE users SET created_at=? WHERE created_at IS NULL", (now_ts,))
        conn.execute("UPDATE users SET trial_ends_at=? WHERE trial_ends_at IS NULL", (now_ts + TRIAL_DAYS * 86400,))

        scan_info = conn.execute("PRAGMA table_info(daily_scans)").fetchall()
        scan_cols = {r[1] for r in scan_info}

        if "universe" not in scan_cols:
            conn.execute("ALTER TABLE daily_scans ADD COLUMN universe TEXT")
        if "ai_status" not in scan_cols:
            conn.execute("ALTER TABLE daily_scans ADD COLUMN ai_status TEXT NOT NULL DEFAULT 'PENDING'")
        if "ai_mode" not in scan_cols:
            conn.execute("ALTER TABLE daily_scans ADD COLUMN ai_mode TEXT")
        if "ai_updated_at" not in scan_cols:
            conn.execute("ALTER TABLE daily_scans ADD COLUMN ai_updated_at REAL")
        if "ai_error" not in scan_cols:
            conn.execute("ALTER TABLE daily_scans ADD COLUMN ai_error TEXT")
        scan_cols = {r[1] for r in conn.execute("PRAGMA table_info(daily_scans)").fetchall()}
        if "quant_pass" not in scan_cols:
            conn.execute("ALTER TABLE daily_scans ADD COLUMN quant_pass INTEGER NOT NULL DEFAULT 0")

        scan_cols = {r[1] for r in conn.execute("PRAGMA table_info(daily_scans)").fetchall()}
        if "pct_from_52w_high" not in scan_cols:
            conn.execute("ALTER TABLE daily_scans ADD COLUMN pct_from_52w_high REAL")
        if "pct_from_52w_low" not in scan_cols:
            conn.execute("ALTER TABLE daily_scans ADD COLUMN pct_from_52w_low REAL")
        if "above_200d_sma" not in scan_cols:
            conn.execute("ALTER TABLE daily_scans ADD COLUMN above_200d_sma INTEGER")
        if "timing_score" not in scan_cols:
            conn.execute("ALTER TABLE daily_scans ADD COLUMN timing_score REAL")
        if "timing_verdict" not in scan_cols:
            conn.execute("ALTER TABLE daily_scans ADD COLUMN timing_verdict TEXT")
        if "ai_language" not in scan_cols:
            conn.execute("ALTER TABLE daily_scans ADD COLUMN ai_language TEXT NOT NULL DEFAULT 'en'")
        if "volume_ratio" not in scan_cols:
            conn.execute("ALTER TABLE daily_scans ADD COLUMN volume_ratio REAL")
        if "ai_prompt_version" not in scan_cols:
            conn.execute("ALTER TABLE daily_scans ADD COLUMN ai_prompt_version INTEGER")
        # The two quantities calculate_alpha_score() decides on. They were computed and
        # thrown away every scan; the score radar needs them as numbers, not re-derived
        # guesses, or the chart would be showing something other than what scored.
        if "pct_off_20d_high" not in scan_cols:
            conn.execute("ALTER TABLE daily_scans ADD COLUMN pct_off_20d_high REAL")
        if "pct_above_200d_sma" not in scan_cols:
            conn.execute("ALTER TABLE daily_scans ADD COLUMN pct_above_200d_sma REAL")
        if "ai_scored_alpha_score" not in scan_cols:
            conn.execute("ALTER TABLE daily_scans ADD COLUMN ai_scored_alpha_score REAL")

        alert_cols = {r[1] for r in conn.execute("PRAGMA table_info(user_alerts)").fetchall()}
        if "direction" not in alert_cols:
            # Existing alerts predate this column -- they were always "price rose to at
            # least X", so backfilling the default ('above') keeps their behavior
            # unchanged instead of silently flipping them to a falling-price alert.
            conn.execute("ALTER TABLE user_alerts ADD COLUMN direction TEXT NOT NULL DEFAULT 'above'")

        scan_info = conn.execute("PRAGMA table_info(daily_scans)").fetchall()
        short_col = next((r for r in scan_info if r[1] == "short_percent"), None)
        if short_col is not None and int(short_col[3]) == 1:
            conn.execute("PRAGMA foreign_keys=OFF")
            conn.execute("DROP TABLE IF EXISTS daily_scans_migrated")
            conn.execute("""
                CREATE TABLE daily_scans_migrated (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    universe TEXT,
                    price REAL NOT NULL,
                    change_pct REAL NOT NULL,
                    alpha_score REAL NOT NULL,
                    rsi REAL NOT NULL,
                    macd REAL NOT NULL,
                    entry REAL,
                    tp REAL,
                    sl REAL,
                    ai_report TEXT,
                    short_percent REAL,
                    ai_status TEXT NOT NULL DEFAULT 'PENDING',
                    ai_mode TEXT,
                    ai_updated_at REAL,
                    ai_error TEXT,
                    created_at REAL NOT NULL,
                    UNIQUE(scan_date, ticker)
                )
            """)
            conn.execute("""
                INSERT INTO daily_scans_migrated
                (id,scan_date,ticker,universe,price,change_pct,alpha_score,rsi,macd,
                 entry,tp,sl,ai_report,short_percent,ai_status,ai_mode,ai_updated_at,ai_error,created_at)
                SELECT id,scan_date,ticker,universe,price,change_pct,alpha_score,rsi,macd,
                       entry,tp,sl,ai_report,short_percent,
                       COALESCE(ai_status,'PENDING'),ai_mode,ai_updated_at,ai_error,created_at
                FROM daily_scans
            """)
            conn.execute("DROP TABLE daily_scans")
            conn.execute("ALTER TABLE daily_scans_migrated RENAME TO daily_scans")
            conn.execute("PRAGMA foreign_keys=ON")

        portfolio_cols = {r[1] for r in conn.execute("PRAGMA table_info(portfolio_items)").fetchall()}
        if "shares" not in portfolio_cols:
            conn.execute("ALTER TABLE portfolio_items ADD COLUMN shares REAL")

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Error: {type(e).__name__}] Database initialization error: {e}")


init_db()


def today_str():
    return datetime.now().strftime("%Y-%m-%d")


def display_date():
    return datetime.now().strftime("%B %d, %Y")


# -----------------------------------------------------------------------------
# Authentication
# -----------------------------------------------------------------------------
def hash_password(password: str, salt_hex: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt_hex), 310_000
    ).hex()


def make_password_hash(password: str):
    salt_hex = secrets.token_hex(16)
    return hash_password(password, salt_hex), salt_hex


def verify_password(password: str, stored_hash: str, salt_hex: str) -> bool:
    return hmac.compare_digest(hash_password(password, salt_hex), stored_hash)


def validate_password_policy(password: str):
    if len(password) < 10:
        return False, "Password must be at least 10 characters long."
    if not re.search(r"[A-Za-z]", password):
        return False, "Password must include at least 1 letter."
    if not re.search(r"[0-9]", password):
        return False, "Password must include at least 1 number."
    return True, ""


def validate_email(email: str):
    return bool(re.fullmatch(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", email))


# The pepper turns the ledger from a rainbow-table lookup into something that cannot be
# reversed even with the database file: email addresses are guessable, a bare sha256 of
# one is not private. Persisted on the Render disk so it survives restarts -- if it were
# regenerated, every existing ledger row would silently stop matching.
TRIAL_PEPPER_FILE = DATA_DIR / "trial_pepper.txt"


def _load_trial_pepper() -> str:
    env = os.getenv("TRIAL_LEDGER_SECRET", "").strip()
    if env:
        return env
    try:
        if TRIAL_PEPPER_FILE.exists():
            existing = TRIAL_PEPPER_FILE.read_text().strip()
            if existing:
                return existing
        generated = secrets.token_hex(32)
        TRIAL_PEPPER_FILE.write_text(generated)
        print(f"[trial] generated a new trial-ledger pepper at {TRIAL_PEPPER_FILE}", flush=True)
        return generated
    except Exception as e:
        # Losing the pepper only weakens the hash, so a read-only disk must not take the
        # signup path down with it.
        print(f"[Error: {type(e).__name__}] trial pepper unavailable ({e}) — using a weaker fallback", flush=True)
        return "quantify-trial-ledger-fallback"


TRIAL_PEPPER = _load_trial_pepper()
_GMAIL_DOMAINS = {"gmail.com", "googlemail.com"}


def canonical_email(email: str) -> str:
    """Collapse the aliases that all land in one real inbox, so one person cannot take
    one trial per alias. Plus-addressing (kim+1@) is routed to kim@ by essentially every
    provider; dots are ignored only by Gmail, so that part is Gmail-only."""
    email = (email or "").strip().lower()
    if "@" not in email:
        return email
    local, _, domain = email.rpartition("@")
    local = local.split("+", 1)[0]
    if domain in _GMAIL_DOMAINS:
        local = local.replace(".", "")
        domain = "gmail.com"
    return f"{local}@{domain}"


def trial_email_hash(email: str) -> str:
    return hashlib.sha256(f"{TRIAL_PEPPER}:{canonical_email(email)}".encode()).hexdigest()


def claim_trial(email: str) -> float:
    """Returns the trial_ends_at to give this signup. A first-time address gets the full
    TRIAL_DAYS and is written into the ledger; an address that has already had one gets
    its original (long expired) end date back, so the account is created normally but
    lands straight on the subscribe page. Deliberately not a signup refusal -- someone
    who wants to pay must still be able to."""
    now = time.time()
    full_trial = now + TRIAL_DAYS * 86400
    try:
        h = trial_email_hash(email)
        conn = db()
        row = conn.execute("SELECT first_trial_at FROM trial_ledger WHERE email_hash=?", (h,)).fetchone()
        if row:
            conn.close()
            return float(row["first_trial_at"]) + TRIAL_DAYS * 86400
        conn.execute("INSERT OR IGNORE INTO trial_ledger(email_hash,first_trial_at) VALUES(?,?)", (h, now))
        conn.commit()
        conn.close()
    except Exception as e:
        # Never block a signup on the ledger -- a broken anti-abuse check must not cost
        # a real customer their account.
        print(f"[Error: {type(e).__name__}] trial ledger check failed for signup: {e}", flush=True)
    return full_trial


def create_session(email: str):
    raw = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw.encode()).hexdigest()
    try:
        conn = db()
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))
        conn.execute(
            "INSERT INTO sessions(token_hash,email,expires_at) VALUES(?,?,?)",
            (token_hash, email, time.time() + SESSION_TTL),
        )
        # Both login paths (password + Google OAuth) funnel through here, and only
        # after authentication has already succeeded -- the one place a real login
        # can be recorded without touching any auth decision.
        now = time.time()
        conn.execute(
            "UPDATE users SET last_login_at=?, login_count=COALESCE(login_count,0)+1, last_seen_at=? WHERE email=?",
            (now, now, email),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Error: {type(e).__name__}] Session creation error: {e}")
    return raw


def get_logged_in_user(request: Request) -> Optional[str]:
    raw = request.cookies.get("session_user")
    if not raw:
        return None
    token_hash = hashlib.sha256(raw.encode()).hexdigest()
    try:
        conn = db()
        row = conn.execute(
            "SELECT email,expires_at FROM sessions WHERE token_hash=?", (token_hash,)
        ).fetchone()
        if row and row["expires_at"] > time.time():
            # Retention signal: record that this account was active today. The
            # last_seen_at guard makes every repeat page load the same day a zero-row
            # no-op, so this stays cheap. Purely additive -- the returned value and
            # every other branch below are unchanged.
            try:
                now_et = datetime.now(ZoneInfo("America/New_York"))
                day_start = now_et.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
                conn.execute(
                    "UPDATE users SET last_seen_at=? WHERE email=? AND (last_seen_at IS NULL OR last_seen_at < ?)",
                    (time.time(), row["email"], day_start),
                )
                conn.commit()
            except Exception as exc:
                print(f"[Error: {type(exc).__name__}] last_seen_at update failed: {exc}")
            conn.close()
            return row["email"]
        if row:
            conn.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash,))
            conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Error: {type(e).__name__}] Logged-in user check error: {e}")
    return None


def disclaimer_accepted(email: str) -> bool:
    conn = db()
    row = conn.execute("SELECT disclaimer_accepted_at FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    return bool(row and row["disclaimer_accepted_at"])


def has_active_access(email: str) -> bool:
    conn = db()
    row = conn.execute("SELECT trial_ends_at,subscription_status FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    if not row:
        return False
    if row["subscription_status"] == "active":
        return True
    trial_ends_at = row["trial_ends_at"]
    return bool(trial_ends_at and time.time() < trial_ends_at)

# -----------------------------------------------------------------------------
# Universe loading
# -----------------------------------------------------------------------------
def normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper().replace("/", "-").replace(".", "-")


def load_universe_cache():
    if not UNIVERSE_FILE.exists():
        return False
    try:
        payload = json.loads(UNIVERSE_FILE.read_text(encoding="utf-8"))
        sp = [normalize_ticker(x) for x in payload.get("sp500", [])]
        ndx = [normalize_ticker(x) for x in payload.get("nasdaq100", [])]
        if len(sp) >= 450 and len(ndx) >= 90:
            UNIVERSE_META["sp500"] = sp
            UNIVERSE_META["nasdaq100"] = ndx
            UNIVERSE[:] = list(dict.fromkeys(sp + ndx))
            UNIVERSE_STATUS.update({
                "ready": True,
                "source": payload.get("source"),
                "updated_at": payload.get("updated_at"),
                "error": None,
            })
            return True
    except Exception as exc:
        print(f"[Error: {type(exc).__name__}] Universe cache load error: {exc}")
        UNIVERSE_STATUS["error"] = f"universe cache error: {exc}"
    return False


def _fetch_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/151.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8",
        "Cache-Control": "no-cache",
    }
    response = requests.get(
        url,
        headers=headers,
        timeout=CONSTITUENT_HTTP_TIMEOUT,
        allow_redirects=True,
    )
    response.raise_for_status()
    return response.text


def _extract_table_symbols(html: str, candidates: tuple[str, ...], min_count: int, max_count: int) -> list[str]:
    soup = bs4.BeautifulSoup(html, "html.parser")
    found = []

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue

        headers = [
            cell.get_text(" ", strip=True).lower()
            for cell in rows[0].find_all(["th", "td"])
        ]
        col_idx = None
        for i, header in enumerate(headers):
            normalized = re.sub(r"[^a-z0-9 ]+", " ", header).strip()
            if normalized in candidates:
                col_idx = i
                break

        if col_idx is None:
            continue

        symbols = []
        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if col_idx >= len(cells):
                continue
            text = cells[col_idx].get_text(" ", strip=True)
            match = re.search(r"([A-Z][A-Z0-9.-]{0,9})", text.upper())
            if match:
                symbols.append(normalize_ticker(match.group(1)))

        symbols = list(dict.fromkeys(symbols))
        if min_count <= len(symbols) <= max_count:
            found.extend(symbols)

    found = list(dict.fromkeys(found))
    if min_count <= len(found) <= max_count:
        return found
    raise ValueError(f"Could not parse constituent table from {len(found)} symbols")


def _fetch_csv_symbols(url: str, symbol_columns: tuple[str, ...], min_count: int, max_count: int) -> list[str]:
    response = requests.get(
        url,
        headers={
            "User-Agent": "QUANTIFY/1.0 (market-universe metadata)",
            "Accept": "text/csv,text/plain,*/*",
        },
        timeout=CONSTITUENT_HTTP_TIMEOUT,
    )
    response.raise_for_status()

    from io import StringIO
    df = pd.read_csv(StringIO(response.text))
    normalized = {re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_"): c for c in df.columns}

    col = None
    for candidate in symbol_columns:
        if candidate in normalized:
            col = normalized[candidate]
            break
    if col is None:
        raise ValueError(f"No ticker column in CSV: {list(df.columns)}")

    symbols = [
        normalize_ticker(x)
        for x in df[col].dropna().astype(str)
        if re.fullmatch(r"[A-Za-z0-9./^-]{1,15}", str(x).strip())
    ]
    symbols = list(dict.fromkeys(symbols))
    if not (min_count <= len(symbols) <= max_count):
        raise ValueError(f"Unexpected constituent count: {len(symbols)}")
    return symbols


def fetch_sp500_sync():
    errors = []
    try:
        return _fetch_csv_symbols(
            SNP500_SOURCE,
            ("symbol", "ticker", "ticker_symbol"),
            450,
            550,
        )
    except Exception as exc:
        errors.append(f"primary={type(exc).__name__}: {exc}")

    try:
        html = _fetch_html(SNP500_FALLBACK_SOURCE)
        return _extract_table_symbols(
            html,
            candidates=("symbol", "ticker"),
            min_count=450,
            max_count=550,
        )
    except Exception as exc:
        errors.append(f"fallback={type(exc).__name__}: {exc}")

    print(f"[Error: RuntimeError] Failed to load S&P 500 source: {' | '.join(errors)}")
    raise RuntimeError("S&P 500 sources failed; " + " | ".join(errors))


def fetch_ndx100_sync():
    errors = []
    try:
        html = _fetch_html(NASDAQ100_SOURCE)
        symbols = _extract_table_symbols(
            html,
            candidates=("symbol", "ticker", "ticker symbol"),
            min_count=95,
            max_count=110,
        )
        return symbols
    except Exception as exc:
        errors.append(f"official={type(exc).__name__}: {exc}")

    try:
        html = _fetch_html(NASDAQ100_FALLBACK_SOURCE)
        return _extract_table_symbols(
            html,
            candidates=("symbol", "ticker", "ticker symbol"),
            min_count=95,
            max_count=110,
        )
    except Exception as exc:
        errors.append(f"fallback={type(exc).__name__}: {exc}")

    print(f"[Error: RuntimeError] Failed to load Nasdaq-100 source: {' | '.join(errors)}")
    raise RuntimeError("Nasdaq-100 sources failed; " + " | ".join(errors))


async def refresh_universe(force=False):
    async with UNIVERSE_LOCK:
        if not force and UNIVERSE_STATUS["ready"] and UNIVERSE_STATUS["updated_at"]:
            try:
                if time.time() - float(UNIVERSE_STATUS["updated_at"]) < UNIVERSE_TTL:
                    return True
            except Exception:
                pass
        try:
            sp, ndx = await asyncio.gather(
                asyncio.to_thread(fetch_sp500_sync),
                asyncio.to_thread(fetch_ndx100_sync),
            )
            payload = {
                "sp500": sp, "nasdaq100": ndx,
                "updated_at": time.time(),
                "source": {"sp500": SNP500_SOURCE, "nasdaq100": NASDAQ100_SOURCE},
            }
            UNIVERSE_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = UNIVERSE_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(UNIVERSE_FILE)
            UNIVERSE_META["sp500"] = sp
            UNIVERSE_META["nasdaq100"] = ndx
            UNIVERSE[:] = list(dict.fromkeys(sp + ndx))
            UNIVERSE_STATUS.update({"ready": True, "source": payload["source"],
                                    "updated_at": payload["updated_at"], "error": None})
            return True
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            UNIVERSE_STATUS["error"] = message
            print(f"[Error: {type(exc).__name__}] universe refresh failed: {message}", flush=True)
            if UNIVERSE:
                return True
            return False

# -----------------------------------------------------------------------------
# Market data and indicators
# -----------------------------------------------------------------------------
def normalize_series(df, column):
    value = df[column]
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    return pd.to_numeric(value, errors="coerce")


def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def calculate_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line, macd - signal_line


def calculate_alpha_score(close, high):
    """Pullback-in-uptrend: price 10-25% below its own prior 20-day high (using the
    high through yesterday, not today, to avoid a stock's own gap-up day counting as
    its own pullback reference point), and price above its 200-day SMA. Score is 83-100
    inside the validated zone (peaking at the 17.5% mid-point), and always below the
    83 pass threshold outside it or below the 200d trend -- so QUANT_PASS_THRESHOLD
    reproduces the exact AND-gate that was backtested, with the in-zone gradient used
    only to order results, not as a separately validated claim."""
    if len(close) < 200 or len(high) < 21:
        return None
    try:
        hh20_prior = float(high.iloc[-21:-1].max())
        if hh20_prior <= 0:
            return None
        price = float(close.iloc[-1])
        sma200 = close.rolling(200).mean().iloc[-1]
        if pd.isna(sma200):
            return None
        if price <= float(sma200):
            return 40.0
        pct_off_high = 1.0 - price / hh20_prior
        if pct_off_high < PULLBACK_MIN or pct_off_high > PULLBACK_MAX:
            dist = max(PULLBACK_MIN - pct_off_high, pct_off_high - PULLBACK_MAX, 0.0)
            return round(max(0.0, 60.0 - dist * 100), 1)
        centering = max(0.0, 1.0 - abs(pct_off_high - PULLBACK_CENTER) / PULLBACK_HALF_WIDTH)
        return round(83.0 + centering * 17.0, 1)
    except Exception as e:
        print(f"[Error: {type(e).__name__}] Alpha score calculation error: {e}")
        return None


def calculate_alpha_score_series(close, high):
    """Vectorized replay of calculate_alpha_score() across an entire price history,
    used only for the backtest — must stay numerically identical to the per-row version."""
    hh20_prior = high.rolling(20).max().shift(1)
    sma200 = close.rolling(200).mean()
    above_trend = close > sma200
    pct_off_high = 1.0 - close / hh20_prior
    in_zone = (pct_off_high >= PULLBACK_MIN) & (pct_off_high <= PULLBACK_MAX)

    centering = (1.0 - (pct_off_high - PULLBACK_CENTER).abs() / PULLBACK_HALF_WIDTH).clip(lower=0.0)
    in_zone_score = 83.0 + centering * 17.0

    below_min = (PULLBACK_MIN - pct_off_high).clip(lower=0.0)
    above_max = (pct_off_high - PULLBACK_MAX).clip(lower=0.0)
    dist_outside = below_min.where(below_min > 0, above_max)
    out_of_zone_score = (60.0 - dist_outside * 100.0).clip(lower=0.0)

    score = in_zone_score.where(in_zone, out_of_zone_score)
    score = score.where(above_trend, 40.0)
    return score.clip(0.0, 100.0).round(1)


# Index tickers use Yahoo-style symbols (^GSPC/^NDX) throughout the codebase, but
# Massive/Polygon's Stocks plan doesn't include its Indices product (confirmed: I:SPX
# 403s as "NOT_AUTHORIZED" on the Stocks Starter key). SPY/QQQ are near-exact tracking
# proxies and are ordinary equities, so this alias keeps every existing call site working
# without a separate Indices subscription.
POLYGON_INDEX_PROXY = {"^GSPC": "SPY", "^NDX": "QQQ"}
POLYGON_TIMESPAN = {"1h": (1, "hour"), "1d": (1, "day"), "1wk": (1, "week"), "1mo": (1, "month")}
# Stocks Starter includes 5 years of history -- "1mo" would ask for more under the old
# yfinance period="max", but 5y of monthly bars is still 60 points, plenty for that chart.
POLYGON_LOOKBACK_DAYS = {"1h": 730, "1d": 730, "1wk": 1825, "1mo": 1825}


class _PolygonRateLimitError(Exception):
    pass


_SHARE_CLASS_RE = re.compile(r"^([A-Z]+)-([A-Z])$")


def _polygon_fetch_sync(ticker: str, interval: str):
    ticker = POLYGON_INDEX_PROXY.get(ticker, ticker)
    # Yahoo-style share-class tickers (BRK-B, BF-B) use a hyphen; Polygon uses a dot
    # (BRK.B, BF.B) -- confirmed by testing both forms against the live API.
    m = _SHARE_CLASS_RE.match(ticker)
    if m:
        ticker = f"{m.group(1)}.{m.group(2)}"
    multiplier, timespan = POLYGON_TIMESPAN.get(interval, (1, "day"))
    lookback = POLYGON_LOOKBACK_DAYS.get(interval, 730)
    to_date = datetime.utcnow().date()
    from_date = to_date - timedelta(days=lookback)
    url = (f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/"
           f"{from_date.isoformat()}/{to_date.isoformat()}")
    headers = {"Authorization": f"Bearer {POLYGON_API_KEY}"}
    params = {"adjusted": "true", "sort": "asc", "limit": 50000}
    rows = []
    next_url = url
    use_params = True
    while next_url:
        resp = requests.get(next_url, headers=headers, params=params if use_params else None, timeout=15)
        use_params = False
        if resp.status_code == 429:
            raise _PolygonRateLimitError(f"Polygon rate limit for {ticker}")
        resp.raise_for_status()
        payload = resp.json()
        rows.extend(payload.get("results") or [])
        next_url = payload.get("next_url")
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    df = df[["Open", "High", "Low", "Close", "Volume"]].sort_index()
    return df[~df.index.duplicated(keep="last")]


def _polygon_fetch_reference_sync(ticker: str):
    ticker = POLYGON_INDEX_PROXY.get(ticker, ticker)
    m = _SHARE_CLASS_RE.match(ticker)
    if m:
        ticker = f"{m.group(1)}.{m.group(2)}"
    url = f"{POLYGON_BASE_URL}/v3/reference/tickers/{ticker}"
    headers = {"Authorization": f"Bearer {POLYGON_API_KEY}"}
    resp = requests.get(url, headers=headers, timeout=15)
    if resp.status_code == 429:
        raise _PolygonRateLimitError(f"Polygon rate limit for {ticker}")
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    results = resp.json().get("results") or {}
    sector = results.get("sic_description")
    market_cap = results.get("market_cap")
    if not sector and market_cap is None:
        return None
    return {"sector": sector, "market_cap": market_cap}


# Sector/market-cap barely changes, so this mirrors the historical-snapshot pattern
# (in-memory + disk-persisted, main.py:920-944) rather than a DB table -- refreshed
# roughly weekly, not on every scan.
SECTOR_CACHE: dict = {}


def _load_sector_cache():
    global SECTOR_CACHE
    try:
        if SECTOR_FILE.exists():
            SECTOR_CACHE = json.loads(SECTOR_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[Error: {type(exc).__name__}] Sector cache load failed: {exc}", flush=True)


def _save_sector_cache():
    try:
        SECTOR_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = SECTOR_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(SECTOR_CACHE, ensure_ascii=False), encoding="utf-8")
        tmp.replace(SECTOR_FILE)
    except Exception as exc:
        print(f"[Error: {type(exc).__name__}] Sector cache save failed: {exc}", flush=True)


async def refresh_sector_data(tickers=None):
    tickers = list(tickers or UNIVERSE)
    if not tickers:
        return
    semaphore = asyncio.Semaphore(POLYGON_CONCURRENCY)
    missing = []

    async def fetch_one(ticker):
        async with semaphore:
            try:
                info = await asyncio.to_thread(_polygon_fetch_reference_sync, ticker)
            except Exception as exc:
                print(f"[Error: {type(exc).__name__}] Sector fetch failed ({ticker}): {exc}")
                info = None
            if info is None:
                missing.append(ticker)
            else:
                SECTOR_CACHE[ticker] = {**info, "ts": time.time()}

    await asyncio.gather(*(fetch_one(t) for t in tickers))
    _save_sector_cache()
    print(f"[Sector] Refreshed {len(tickers) - len(missing)}/{len(tickers)} tickers"
          + (f" — no data for: {missing[:20]}{'...' if len(missing) > 20 else ''}" if missing else ""), flush=True)


RATE_LIMIT_BACKOFF_SECONDS = 10.0


async def _download_single_with_backoff(ticker, interval):
    for attempt in range(2):
        try:
            return await asyncio.to_thread(_polygon_fetch_sync, ticker, interval)
        except _PolygonRateLimitError:
            if attempt == 0:
                print(f"[RateLimit] Polygon rate limit hit on {ticker} — backing off {RATE_LIMIT_BACKOFF_SECONDS:.0f}s")
                await asyncio.sleep(RATE_LIMIT_BACKOFF_SECONDS)
                continue
            print(f"[RateLimit] Polygon still rate-limited on {ticker} after backoff — giving up")
            return None
        except Exception as exc:
            print(f"[Error: {type(exc).__name__}] Polygon download failed ({ticker}): {exc}")
            return None
    return None

# The in-memory historical cache doesn't survive a restart, which left the fallback with
# nothing to serve during tonight's yfinance outage after a deploy. This mirrors the same
# {ticker: {"data":.., "ts":..}} shape but on disk, written once per successful daily
# scan (not per ticker -- 518 individual writes would be needlessly slow) so there's a
# real chart to fall back to even hours after a restart.
_PERSISTED_SNAPSHOT_CACHE = {"data": None, "mtime": None}


def save_historical_snapshot(raw: dict):
    try:
        snapshot = {ticker: {"data": df, "ts": time.time()} for ticker, df in raw.items()}
        HISTORICAL_SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = HISTORICAL_SNAPSHOT_FILE.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(snapshot, f)
        tmp.replace(HISTORICAL_SNAPSHOT_FILE)
    except Exception as exc:
        print(f"[Error: {type(exc).__name__}] Historical snapshot save failed: {exc}", flush=True)


def _get_persisted_snapshot() -> dict:
    try:
        if not HISTORICAL_SNAPSHOT_FILE.exists():
            return {}
        mtime = HISTORICAL_SNAPSHOT_FILE.stat().st_mtime
        if _PERSISTED_SNAPSHOT_CACHE["data"] is None or _PERSISTED_SNAPSHOT_CACHE["mtime"] != mtime:
            with open(HISTORICAL_SNAPSHOT_FILE, "rb") as f:
                _PERSISTED_SNAPSHOT_CACHE["data"] = pickle.load(f)
            _PERSISTED_SNAPSHOT_CACHE["mtime"] = mtime
        return _PERSISTED_SNAPSHOT_CACHE["data"] or {}
    except Exception as exc:
        print(f"[Error: {type(exc).__name__}] Historical snapshot load failed: {exc}", flush=True)
        return {}


async def download_stock(ticker: str, interval="1d", cache=True):
    key = f"single:{ticker}:{interval}"
    cached = CACHE["historical"].get(key) if cache else None
    if cached and time.time() - cached["ts"] < HISTORICAL_TTL:
        return cached["data"]
    data = await _download_single_with_backoff(ticker, interval)
    if data is None or data.empty:
        if cached is not None:
            age_min = (time.time() - cached["ts"]) / 60
            print(f"[Fallback] Live fetch failed for {ticker} — serving in-memory cached data from {age_min:.0f}m ago", flush=True)
            stale = cached["data"].copy()
            stale.attrs["stale_as_of"] = cached["ts"]
            return stale
        if interval == "1d":
            persisted = _get_persisted_snapshot().get(ticker)
            if persisted is not None:
                age_hr = (time.time() - persisted["ts"]) / 3600
                print(f"[Fallback] Live fetch failed for {ticker} — serving disk snapshot from {age_hr:.1f}h ago", flush=True)
                stale = persisted["data"].copy()
                stale.attrs["stale_as_of"] = persisted["ts"]
                return stale
        return None
    data = data.dropna(how="all")
    if cache and len(data) >= 30:
        CACHE["historical"][key] = {"data": data, "ts": time.time()}
    return data


async def batch_download_stocks(tickers):
    results = {}
    unique = list(dict.fromkeys(tickers))
    BATCH_STATUS.update({"processed": 0, "total": len(unique)})

    now = time.time()
    to_fetch = []
    for ticker in unique:
        cached = CACHE["historical"].get(f"single:{ticker}:1d")
        if cached and now - cached["ts"] < HISTORICAL_TTL:
            results[ticker] = cached["data"]
        else:
            to_fetch.append(ticker)
    if len(to_fetch) < len(unique):
        print(f"[Cache reuse] {len(unique) - len(to_fetch)} tickers served from recent cache, downloading {len(to_fetch)} fresh")
    BATCH_STATUS["processed"] = len(unique) - len(to_fetch)

    # Massive/Polygon Starter has no multi-ticker aggregates endpoint (unlike yfinance's
    # batch download), but its unlimited-call quota means per-ticker concurrent fetches
    # are the right shape here instead of yfinance's old chunk-then-pause-then-retry dance.
    semaphore = asyncio.Semaphore(POLYGON_CONCURRENCY)
    failed = []

    async def fetch_one(ticker):
        async with semaphore:
            df = await _download_single_with_backoff(ticker, "1d")
            if df is not None and len(df) >= 30:
                CACHE["historical"][f"single:{ticker}:1d"] = {"data": df, "ts": time.time()}
                results[ticker] = df
            else:
                failed.append(ticker)
            BATCH_STATUS["processed"] += 1

    await asyncio.gather(*(fetch_one(t) for t in to_fetch))
    if failed:
        preview = failed[:20]
        print(f"[Batch] {len(failed)}/{len(to_fetch)} tickers had no usable data: {preview}"
              f"{'...' if len(failed) > 20 else ''}")
    return results

# -----------------------------------------------------------------------------
# News / short interest / AI
# -----------------------------------------------------------------------------
def fetch_news_sync(ticker: str, attempt: int = 0):
    try:
        items = yf.Ticker(ticker).news or []
        result = []
        for item in items[:5]:
            content = item.get("content", item)
            title = content.get("title")
            url = ((content.get("canonicalUrl") or {}).get("url")
                   or (content.get("clickThroughUrl") or {}).get("url"))
            if title and url:
                result.append({"title": title, "url": url, "published": content.get("pubDate")})
        return result
    except Exception as exc:
        if attempt == 0:
            time.sleep(1.5)
            return fetch_news_sync(ticker, attempt=1)
        print(f"[Error: {type(exc).__name__}] News fetch error ({ticker}): {exc}")
        return None


async def fetch_stock_news(ticker):
    cached = CACHE["news"].get(ticker)
    if cached and time.time() - cached["ts"] < NEWS_TTL:
        return cached["data"]
    data = await asyncio.to_thread(fetch_news_sync, ticker)
    if data is not None:
        CACHE["news"][ticker] = {"data": data, "ts": time.time()}
    return data




def fetch_earnings_sync(ticker, attempt: int = 0):
    try:
        df = yf.Ticker(ticker).get_earnings_dates(limit=8)
        if df is None or df.empty:
            return {"last": None, "next": None}
        last = None
        next_earn = None
        for idx, row in df.iterrows():
            reported = row.get("Reported EPS")
            if pd.isna(reported):
                if next_earn is None:
                    next_earn = {"date": idx.strftime("%Y-%m-%d")}
            elif last is None:
                est = row.get("EPS Estimate")
                surprise = row.get("Surprise(%)")
                last = {
                    "date": idx.strftime("%Y-%m-%d"),
                    "eps_estimate": None if pd.isna(est) else round(float(est), 2),
                    "eps_actual": round(float(reported), 2),
                    "surprise_pct": None if pd.isna(surprise) else round(float(surprise), 2),
                    "beat": bool(surprise > 0) if not pd.isna(surprise) else None,
                }
            if last is not None:
                break
        return {"last": last, "next": next_earn}
    except Exception as exc:
        if attempt == 0:
            time.sleep(1.5)
            return fetch_earnings_sync(ticker, attempt=1)
        print(f"[Error: {type(exc).__name__}] Earnings fetch error ({ticker}): {exc}")
        return None


async def get_earnings(ticker):
    cached = CACHE["earnings"].get(ticker)
    if cached and time.time() - cached["ts"] < SHORT_INTEREST_TTL:
        return cached["data"]
    data = await asyncio.to_thread(fetch_earnings_sync, ticker)
    if data is not None:
        CACHE["earnings"][ticker] = {"data": data, "ts": time.time()}
    return data


LANGUAGE_NAMES = {"en": "English", "ko": "Korean"}

# Lightweight UI translation for the logged-in app only (terminal, market, watchlist,
# backtest, portfolio, subscription, contact, settings) -- the marketing/landing pages
# and legal pages deliberately stay English-only (see this session's own scoping
# decision: legal text needs a real translation, not a casual one, and the landing
# page's copy is hand-tuned marketing that doesn't map 1:1). Keyed by short string IDs
# rather than the English text itself so a later English copy edit doesn't silently
# orphan its Korean counterpart.
UI_STRINGS = {
    "nav_scanner": {"en": "Scanner", "ko": "스캐너"},
    "nav_market": {"en": "Market", "ko": "마켓"},
    "nav_watchlist": {"en": "Watchlist", "ko": "관심종목"},
    "nav_backtest": {"en": "Backtest", "ko": "백테스트"},
    "nav_portfolio": {"en": "Portfolio", "ko": "포트폴리오"},
    "nav_settings": {"en": "Settings", "ko": "설정"},
    "nav_subscription": {"en": "Subscription", "ko": "구독"},
    "nav_contact": {"en": "Contact", "ko": "문의"},
    "nav_logout": {"en": "Log out", "ko": "로그아웃"},
    "loading": {"en": "Loading...", "ko": "불러오는 중..."},
    "informational_only": {"en": "informational only, not investment advice", "ko": "정보 제공 목적일 뿐, 투자 조언이 아닙니다"},
    "ai_market_summary": {"en": "AI Market Summary", "ko": "AI 마켓 서머리"},
    "market_summary": {"en": "Market Summary", "ko": "마켓 서머리"},
    "by_universe": {"en": "By Universe", "ko": "지수별 현황"},
    "heatmap": {"en": "Heatmap", "ko": "히트맵"},
    "heatmap_hint": {"en": "click any tile to open its chart — bigger tiles are larger-cap", "ko": "타일을 클릭하면 차트가 열립니다 — 큰 타일일수록 시가총액이 큰 종목입니다"},
    "group_by": {"en": "Group by", "ko": "그룹 기준"},
    "opt_index": {"en": "Index", "ko": "지수"},
    "opt_sector": {"en": "Sector", "ko": "섹터"},
    "watchlist_title": {"en": "Watchlist", "ko": "관심종목"},
    "add_ticker_placeholder": {"en": "Add ticker (e.g. NVDA)", "ko": "종목 추가 (예: NVDA)"},
    "add_btn": {"en": "Add", "ko": "추가"},
    "strategy_performance": {"en": "Strategy Performance", "ko": "전략 성과"},
    "strategy_performance_hint": {"en": "(real historical replay, not a guarantee of future results)", "ko": "(실제 과거 데이터 재현, 미래 수익을 보장하지 않습니다)"},
    "methodology": {"en": "Methodology", "ko": "방법론"},
    "backtest_methodology_body": {
        "en": ("QUANTIFY looks for stocks in a long-term uptrend (price above its 200-day moving average) that have "
               "pulled back 10-25% from their own recent 20-day high. This exact rule was chosen by backtesting "
               "thousands of alternative entry rules against two years of real price history, ranking them on the "
               "first 70% of that window only, then validating the leaders on the untouched final 30% — the "
               "out-of-sample numbers below are that validation, not the numbers used to pick the rule. Returns are "
               "net of a 0.10% round-trip cost, and the S&amp;P comparison is measured over the same windows the "
               "strategy traded, not over every day in the period."),
        "ko": ("QUANTIFY는 장기 상승 추세(200일 이동평균선 위)에 있으면서 최근 20일 고점 대비 10~25% 눌린 종목을 찾습니다. "
               "이 규칙은 2년치 실제 가격 데이터로 수천 개의 대안 진입 규칙을 백테스트해서, 앞쪽 70% 구간에서만 순위를 매긴 뒤 "
               "손대지 않은 마지막 30% 구간에서 상위 규칙들을 검증해 선택했습니다. 아래 out-of-sample 수치가 그 검증 결과이며, "
               "규칙을 고르는 데 쓴 수치가 아닙니다. 수익률은 왕복 0.10% 비용을 뺀 값이고, S&amp;P 비교는 전체 기간 평균이 아니라 "
               "전략이 실제로 진입한 것과 같은 구간에서 측정했습니다."),
    },
    "sp500_avg_matched": {"en": "S&P 500, same windows", "ko": "S&P 500 (동일 구간)"},
    # Terminal action bar + score card. Added in this session's earlier work and shipped
    # English-only; caught by the Korean sweep.
    "act_price_alert": {"en": "Price alert", "ko": "가격 알림"},
    "act_add_portfolio": {"en": "Add to portfolio", "ko": "포트폴리오에 추가"},
    "act_alert_hint": {"en": "Manage alerts", "ko": "알림 관리"},
    "act_shares_ph": {"en": "Shares", "ko": "수량"},
    "act_entry_ph": {"en": "Entry price $", "ko": "매수가 $"},
    "act_target_ph": {"en": "Target price $", "ko": "목표가 $"},
    "act_set_alert": {"en": "Set Alert", "ko": "알림 설정"},
    "act_save_portfolio": {"en": "Save to Portfolio", "ko": "포트폴리오에 저장"},
    "act_portfolio_hint": {"en": "Leave entry price blank to use today's scan price. Not sure how many shares?",
                            "ko": "매수가를 비워두면 오늘 스캔 가격이 쓰입니다. 몇 주를 살지 모르겠다면?"},
    "act_sizing_link": {"en": "Position sizing calculator", "ko": "포지션 사이즈 계산기"},
    "act_toggle": {"en": "Set an alert or add to portfolio", "ko": "알림 설정 또는 포트폴리오에 추가"},
    "score_radar_alt": {"en": "Score breakdown radar", "ko": "점수 구성 레이더 차트"},
    "sf_value": {"en": "VALUE", "ko": "저평가"},
    "sf_growth": {"en": "GROWTH", "ko": "성장"},
    "sf_profit": {"en": "PROFIT", "ko": "수익성"},
    "sf_health": {"en": "HEALTH", "ko": "건전성"},
    "sf_dividend": {"en": "DIVIDEND", "ko": "배당"},
    "sf_ai": {"en": "AI CHECK", "ko": "AI 평가"},
    "sf_value_help": {"en": "How cheap the price is against earnings and book value, ranked against other companies in the same sector. Higher means cheaper than its peers.",
                       "ko": "이익과 순자산 대비 주가가 얼마나 싼지를 같은 섹터의 다른 기업들과 비교한 순위입니다. 높을수록 동종 업계보다 쌉니다."},
    "sf_growth_help": {"en": "Revenue and earnings growth, ranked against sector peers. Higher means growing faster than the companies it competes with.",
                        "ko": "매출과 이익 성장률을 같은 섹터 기업들과 비교한 순위입니다. 높을수록 경쟁사보다 빠르게 성장하고 있습니다."},
    "sf_profit_help": {"en": "Return on equity and profit margin, ranked against sector peers. Higher means it converts revenue into profit better than its peers.",
                        "ko": "자기자본이익률(ROE)과 순이익률을 같은 섹터 기업들과 비교한 순위입니다. 높을수록 매출을 이익으로 잘 바꿉니다."},
    "sf_health_help": {"en": "Debt relative to equity and short-term liquidity, ranked against sector peers. Higher means a stronger balance sheet.",
                        "ko": "자기자본 대비 부채와 단기 유동성을 같은 섹터 기업들과 비교한 순위입니다. 높을수록 재무구조가 튼튼합니다."},
    "sf_dividend_help": {"en": "Dividend yield ranked against sector peers. Blank means the company pays no dividend, which is not the same as a bad one.",
                          "ko": "배당수익률을 같은 섹터 기업들과 비교한 순위입니다. 비어 있으면 배당을 지급하지 않는 것이며, 배당이 나쁘다는 뜻이 아닙니다."},
    "sf_ai_help": {"en": "The AI's entry-timing review of the current setup — blow-off-top and dead-cat-bounce risk. This is the only axis about timing rather than the company.",
                    "ko": "현재 진입 시점에 대한 AI의 검토입니다 — 급등 후 고점과 데드캣 바운스 위험. 여섯 축 중 유일하게 기업이 아니라 타이밍에 관한 항목입니다."},
    "sf_title": {"en": "Company profile", "ko": "기업 프로필"},
    "sf_vs_sector": {"en": "ranked against", "ko": "섹터 비교:"},
    "sf_no_data": {"en": "Fundamentals not loaded for this ticker yet.", "ko": "이 종목의 재무 데이터가 아직 준비되지 않았습니다."},
    "sf_nodata": {"en": "no data", "ko": "데이터 없음"},
    "sf_nodiv": {"en": "pays no dividend", "ko": "배당 없음"},
    "sf_unscored": {"en": "not reviewed", "ko": "미평가"},
    "sf_rev": {"en": "rev", "ko": "매출"},
    "sf_earn": {"en": "earnings", "ko": "이익"},
    "sf_margin": {"en": "margin", "ko": "마진"},
    "sf_debt": {"en": "debt/equity", "ko": "부채비율"},
    "sf_current": {"en": "current ratio", "ko": "유동비율"},
    "sf_yield": {"en": "yield", "ko": "배당수익률"},
    "sf_payout": {"en": "payout", "ko": "성향"},
    "sf_pe": {"en": "P/E", "ko": "PER"},
    "sf_pb": {"en": "P/B", "ko": "PBR"},
    "sf_footnote": {"en": "Each axis is a percentile against other companies in the same sector — wider is better. A blank axis means the figure does not exist for this company, not that it scores zero.",
                     "ko": "각 축은 같은 섹터 기업들과 비교한 백분위입니다 — 넓을수록 좋습니다. 빈 축은 해당 수치가 이 기업에 존재하지 않는다는 뜻이며, 0점이라는 의미가 아닙니다."},
    "score_detail_show": {"en": "Breakdown", "ko": "구성 보기"},
    "score_detail_hide": {"en": "Hide breakdown", "ko": "구성 접기"},
    "score_note": {"en": "<b>Trend</b>, <b>Pullback</b> and <b>AI check</b> are what produce the score. <b>Room</b> and <b>Momentum</b> are shown for context and do not affect it.",
                    "ko": "<b>추세</b>, <b>눌림목</b>, <b>AI 점검</b>이 점수를 만듭니다. <b>여력</b>과 <b>모멘텀</b>은 참고용이며 점수에 영향을 주지 않습니다."},
    "score_passed": {"en": "This one cleared today's scan.", "ko": "이 종목은 오늘 스캔을 통과했습니다."},
    "score_not_passed": {"en": "This one did <b>not</b> clear today's scan.", "ko": "이 종목은 오늘 스캔을 통과하지 <b>못했습니다</b>."},
    "score_split": {"en": "averaged into the total", "ko": "를 평균내어 총점 산출"},
    "score_quant_only": {"en": "Quant score only. The AI check runs on tickers that clear the quant bar, and this one has not, so there is no combined score.",
                          "ko": "퀀트 점수만 있습니다. AI 점검은 퀀트 기준을 통과한 종목에만 실행되는데 이 종목은 통과하지 못해 종합 점수가 없습니다."},
    "axis_trend": {"en": "TREND", "ko": "추세"},
    "axis_pullback": {"en": "PULLBACK", "ko": "눌림목"},
    "axis_ai": {"en": "AI CHECK", "ko": "AI 점검"},
    "axis_room": {"en": "ROOM", "ko": "여력"},
    "axis_momentum": {"en": "MOMENTUM", "ko": "모멘텀"},
    # Pre-existing gaps found in the same sweep.
    "no_scan_data": {"en": "No scan data yet.", "ko": "아직 스캔 데이터가 없습니다."},
    "scan_previous_day": {"en": "Showing the last completed scan ({date}) — today's is still being reviewed.",
                           "ko": "마지막으로 완료된 스캔({date})을 보여주는 중입니다 — 오늘 스캔은 아직 검토 중입니다."},
    "no_scan_data_check": {"en": "No scan data yet — check back after the next scan.", "ko": "아직 스캔 데이터가 없습니다 — 다음 스캔 후 다시 확인해주세요."},
    "heatmap_failed": {"en": "Could not load the heatmap.", "ko": "히트맵을 불러오지 못했습니다."},
    "ai_summary_pending": {"en": "AI summary is being prepared for today's scan — check back after the next cycle.",
                            "ko": "오늘 스캔에 대한 AI 요약을 준비하는 중입니다 — 다음 주기 후 다시 확인해주세요."},
    "not_yet_reviewed": {"en": "Not yet reviewed", "ko": "아직 검토되지 않음"},
    "cleared_quant_bar": {"en": "Cleared Quant Bar", "ko": "퀀트 기준 통과"},
    # Tooltips: never translated since they were added, so a Korean reader got English
    # help text on hover for every control in the action bar.
    "tip_track_ticker": {"en": "Track this ticker without deciding anything right now", "ko": "지금 아무 결정도 하지 않고 이 종목만 추적합니다"},
    "tip_alert_dir": {"en": "Alert when price rises to/above, or falls to/below, the target", "ko": "가격이 목표가 이상으로 오르거나 이하로 내릴 때 알림"},
    "tip_target": {"en": "Get an email when the price reaches this value", "ko": "가격이 이 값에 도달하면 이메일로 알려줍니다"},
    "tip_set_alert": {"en": "Email me when the price hits my target", "ko": "목표가에 도달하면 이메일 보내기"},
    "tip_manage_alerts": {"en": "View or cancel your existing price alerts", "ko": "기존 가격 알림 확인 또는 취소"},
    "tip_shares": {"en": "How many shares you own or are tracking", "ko": "보유 중이거나 추적 중인 주식 수"},
    "tip_entry": {"en": "What you paid — defaults to today's scan price if left blank", "ko": "매수 단가 — 비워두면 오늘 스캔 가격이 사용됩니다"},
    "tip_save_portfolio": {"en": "Add this ticker to My Portfolio", "ko": "이 종목을 내 포트폴리오에 추가"},
    "tip_ai_refresh": {"en": "Price/RSI/trend above refresh at each scan; this AI risk review only re-runs when the quant score has moved enough to matter",
                        "ko": "위의 가격·RSI·추세는 스캔할 때마다 갱신되지만, 이 AI 리스크 리뷰는 퀀트 점수가 의미 있게 움직였을 때만 다시 실행됩니다"},
    "faq_more": {"en": "See the <a href=\"/faq\" style=\"color:var(--head);text-decoration:underline\">FAQ</a> for more.",
                  "ko": "자세한 내용은 <a href=\"/faq\" style=\"color:var(--head);text-decoration:underline\">자주 묻는 질문</a>을 참고하세요."},
    "nothing_watched": {"en": "Nothing watched yet — add any ticker above, regardless of whether it clears the quant bar.", "ko": "아직 관심종목이 없습니다 — 위에서 종목을 추가하세요, 퀀트 기준 통과 여부와 상관없이 추가 가능합니다."},
    "remove_btn": {"en": "Remove", "ko": "삭제"},
    "backtest_computing": {"en": "Backtest is still computing on the server — check back soon.", "ko": "백테스트를 서버에서 계산하는 중입니다 — 잠시 후 다시 확인해주세요."},
    "day_forward_return": {"en": "-Day Forward Return", "ko": "일 후 수익률"},
    "strategy_avg": {"en": "Strategy avg", "ko": "전략 평균"},
    "strategy_win_rate": {"en": "Strategy win rate", "ko": "전략 승률"},
    "when_right_wrong": {"en": "When right / wrong", "ko": "적중 / 실패 시"},
    "worst_case": {"en": "Worst case", "ko": "최악의 경우"},
    "sp500_avg": {"en": "S&amp;P 500 avg (same period)", "ko": "S&amp;P 500 평균 (동일 기간)"},
    "page_portfolio": {"en": "Portfolio", "ko": "포트폴리오"},
    "page_subscription": {"en": "Subscription", "ko": "구독"},
    "page_contact": {"en": "Contact", "ko": "문의"},
    "page_settings": {"en": "Settings", "ko": "설정"},
    "back_to_terminal": {"en": "&larr; Back to Terminal", "ko": "&larr; 터미널로 돌아가기"},
    "sector_concentration": {"en": "Sector Concentration", "ko": "섹터 집중도"},
    "position_sizing": {"en": "Position Sizing Calculator", "ko": "포지션 사이즈 계산기"},
    "account_size": {"en": "Account size $", "ko": "계좌 자금 $"},
    "risk_per_trade": {"en": "Risk per trade %", "ko": "거래당 리스크 %"},
    "entry_price": {"en": "Entry price $", "ko": "진입 가격 $"},
    "stop_distance": {"en": "Stop distance %", "ko": "손절 거리 %"},
    "calculate_btn": {"en": "Calculate", "ko": "계산"},
    "sizing_disclaimer": {"en": "Sizing arithmetic only — not a recommendation, and it doesn't account for correlation between your positions.", "ko": "단순 계산일 뿐 추천이 아니며, 보유 종목 간 상관관계는 반영하지 않습니다."},
    "portfolio_disclaimer": {"en": "QUANTIFY is informational and educational only, not investment advice. Nothing here is a recommendation to buy or sell any security. All investment decisions are solely your own responsibility.", "ko": "QUANTIFY는 정보 제공 및 교육 목적으로만 제공되며, 투자 조언이 아닙니다. 여기 있는 어떤 내용도 특정 증권의 매수·매도 추천이 아닙니다. 모든 투자 결정의 책임은 전적으로 본인에게 있습니다."},
    "add_share_count_hint": {"en": "Add a share count to your holdings (edit when saving from the terminal) to see sector concentration here.", "ko": "보유 종목에 수량을 입력하면(터미널에서 저장할 때 입력 가능) 여기서 섹터 집중도를 볼 수 있습니다."},
    "sector_rule_of_thumb": {"en": "Rule of thumb only, not a risk model — flags over 30% in one sector.", "ko": "단순 참고용 기준일 뿐 리스크 모델이 아닙니다 — 한 섹터가 30%를 넘으면 표시합니다."},
    "fill_four_fields": {"en": "Fill in all four fields with positive numbers.", "ko": "네 개 항목 모두 양수로 입력해주세요."},
    "nothing_saved_open_ticker": {"en": "Nothing saved yet. Open a ticker in the terminal and click ☆ Save to Portfolio.", "ko": "아직 저장된 종목이 없습니다. 터미널에서 종목을 열고 ☆ 포트폴리오에 저장을 클릭하세요."},
    "nothing_saved": {"en": "Nothing saved yet.", "ko": "아직 저장된 게 없습니다."},
    "current_plan": {"en": "Current Plan", "ko": "현재 플랜"},
    "active_subscription": {"en": "Active Subscription", "ko": "구독 중"},
    "active_sub_thanks": {"en": "Your subscription is active. Thanks for supporting QUANTIFY.", "ko": "구독이 활성화되어 있습니다. QUANTIFY를 이용해주셔서 감사합니다."},
    "trial_ended_badge": {"en": "Trial Ended", "ko": "체험 기간 종료"},
    "subscribe_btn": {"en": "Subscribe", "ko": "구독하기"},
    "paid_plans_soon": {"en": "Paid plans coming soon", "ko": "유료 플랜 준비 중"},
    "contact_us": {"en": "Contact Us", "ko": "문의하기"},
    "your_message": {"en": "Your message", "ko": "메시지"},
    "message_placeholder": {"en": "Bug report, feedback, question — anything.", "ko": "버그 제보, 피드백, 질문 등 무엇이든 남겨주세요."},
    "send_message_btn": {"en": "Send Message", "ko": "보내기"},
    "write_message_first": {"en": "Write a message first.", "ko": "먼저 메시지를 입력해주세요."},
    "account": {"en": "Account", "ko": "계정"},
    "email_label": {"en": "Email", "ko": "이메일"},
    "member_since": {"en": "Member since", "ko": "가입일"},
    "plan_status": {"en": "Plan status", "ko": "플랜 상태"},
    "manage_subscription": {"en": "Manage Subscription &rarr;", "ko": "구독 관리 &rarr;"},
    "display": {"en": "Display", "ko": "화면"},
    "theme_label": {"en": "Theme", "ko": "테마"},
    "opt_light": {"en": "Light", "ko": "라이트"},
    "opt_dark": {"en": "Dark", "ko": "다크"},
    "ai_report_language": {"en": "Language", "ko": "언어"},
    "default_scanner_sort": {"en": "Default scanner sort", "ko": "기본 정렬 기준"},
    "opt_score": {"en": "Score", "ko": "점수"},
    "opt_change_pct": {"en": "Change %", "ko": "변동률 %"},
    "opt_ticker_az": {"en": "Ticker A-Z", "ko": "티커 A-Z"},
    "default_scanner_view": {"en": "Default scanner view", "ko": "기본 보기 방식"},
    "opt_list": {"en": "List", "ko": "목록"},
    "opt_heatmap": {"en": "Heatmap", "ko": "히트맵"},
    "email_high_score": {"en": "Email me when a stock scores 90+", "ko": "90점 이상 종목 발견 시 이메일 알림"},
    "opt_off": {"en": "Off", "ko": "끄기"},
    "opt_on": {"en": "On", "ko": "켜기"},
    "save_settings_btn": {"en": "Save Settings", "ko": "설정 저장"},
    "price_alerts": {"en": "Price Alerts", "ko": "가격 알림"},
    "change_password": {"en": "Change Password", "ko": "비밀번호 변경"},
    "current_password": {"en": "Current password", "ko": "현재 비밀번호"},
    "new_password": {"en": "New password", "ko": "새 비밀번호"},
    "password_hint": {"en": "10+ characters, with at least 1 letter and 1 number", "ko": "10자 이상, 영문 1자와 숫자 1자 이상 포함"},
    "change_password_btn": {"en": "Change Password", "ko": "비밀번호 변경"},
    "danger_zone": {"en": "Danger Zone", "ko": "위험 구역"},
    "delete_account_warning": {"en": "Permanently delete your account and all associated data (portfolio, watchlist, alerts). This cannot be undone.", "ko": "계정과 관련된 모든 데이터(포트폴리오, 관심종목, 알림)를 영구적으로 삭제합니다. 되돌릴 수 없습니다."},
    "confirm_password": {"en": "Confirm password", "ko": "비밀번호 확인"},
    "delete_account_btn": {"en": "Delete My Account", "ko": "계정 삭제"},
    "manage_alerts_link": {"en": "Manage alerts", "ko": "알림 관리"},
    "loading_empty_hint": {"en": "No alerts set. Open a ticker in the terminal and click Set Alert.", "ko": "설정된 알림이 없습니다. 터미널에서 종목을 열고 알림 설정을 클릭하세요."},
    "language_toggle_hint": {"en": "Changes the app interface and the AI-written quant/risk reviews. Ticker symbols, company names and price data stay as they are. AI reviews share a daily usage limit, so the first report in a new language can take a minute to generate.", "ko": "앱 화면과 AI가 작성한 퀀트/리스크 리뷰가 함께 바뀝니다. 티커 심볼, 회사명, 가격 데이터는 그대로 유지됩니다. AI 리뷰는 하루 사용량 한도를 공유하기 때문에, 새 언어로 첫 리포트를 만들 때는 잠시 걸릴 수 있습니다."},
    "high_score_alert_hint": {"en": "One email a day, after market close, listing every ticker that reached a 90+ score that day — not one email per ticker.", "ko": "종목당 이메일이 아니라, 하루 한 번 장 마감 후 그날 90점 이상을 기록한 모든 종목을 모아 이메일로 보내드립니다."},
    "password_char_hint": {"en": "10+ characters, with at least 1 letter and 1 number", "ko": "10자 이상, 영문 1자와 숫자 1자 이상 포함"},
    "status_active": {"en": "Active Subscription", "ko": "구독 중"},
    "status_trial": {"en": "Free Trial", "ko": "무료 체험"},
    "status_expired": {"en": "Trial Ended", "ko": "체험 종료"},
    "status_cancelled": {"en": "Cancelled", "ko": "해지됨"},
    "status_paused": {"en": "Paused", "ko": "일시 정지"},
    "sent_suffix": {"en": "(sent)", "ko": "(발송됨)"},
    "dir_below": {"en": "&#8595; at/below", "ko": "&#8595; 이하"},
    "dir_above": {"en": "&#8593; at/above", "ko": "&#8593; 이상"},
    "confirm_delete_account": {"en": "Are you sure? This permanently deletes your account and cannot be undone.", "ko": "정말 삭제하시겠습니까? 계정이 영구적으로 삭제되며 되돌릴 수 없습니다."},
    "confirm_email_to_delete": {"en": "Type your account email to confirm", "ko": "확인을 위해 계정 이메일을 입력하세요"},
    "my_subscription": {"en": "My Subscription", "ko": "내 구독"},
    "onboarding_title": {"en": "Quick guide to QUANTIFY", "ko": "QUANTIFY 빠른 안내"},
    "onboarding_badges_p": {"en": "<b>Badges</b> are the AI's read on entry timing: <b>Favorable</b> (setup looks clean), <b>Caution</b> (some risk worth knowing about), or <b>Risk</b> (skip or wait). Never a buy/sell order.", "ko": "<b>배지</b>는 AI가 판단한 진입 타이밍입니다: <b>Favorable</b>(깨끗한 셋업), <b>Caution</b>(알아둘 리스크 있음), <b>Risk</b>(건너뛰거나 기다리는 게 나음). 매수/매도 지시가 아닙니다."},
    "onboarding_score_p": {"en": "<b>Score (0-100)</b> combines the quant scan (is this a long-term uptrend that's pulled back to a good entry zone?) with the AI's risk check. Only names that clear the bar show up at all.", "ko": "<b>점수(0-100)</b>는 퀀트 스캔(장기 상승 추세 중 좋은 진입 구간까지 눌림목이 왔는가)과 AI 리스크 체크를 결합한 값입니다. 기준을 통과한 종목만 목록에 나타납니다."},
    "onboarding_scanner_p": {"en": "<b>The scanner list</b> updates a few times a day — tap any ticker to load its chart, technicals, and full AI report.", "ko": "<b>스캐너 목록</b>은 하루 몇 차례 갱신됩니다 — 종목을 탭하면 차트, 기술적 지표, AI 리포트 전체가 표시됩니다."},
    "onboarding_help_p": {"en": "Little <b>?</b> icons next to unfamiliar terms (RSI, MACD, Trend...) explain what they mean — tap or hover any of them anytime.", "ko": "낯선 용어(RSI, MACD, 추세 등) 옆의 작은 <b>?</b> 아이콘을 누르거나 마우스를 올리면 뜻을 볼 수 있습니다."},
    "got_it": {"en": "Got it", "ko": "확인"},
    "market_scanner": {"en": "Market Scanner", "ko": "마켓 스캐너"},
    "jump_to_ticker": {"en": "Jump to ticker (e.g. TSLA)", "ko": "종목으로 이동 (예: TSLA)"},
    "sort_score": {"en": "Sort: Score", "ko": "정렬: 점수"},
    "sort_change": {"en": "Sort: Change %", "ko": "정렬: 변동률 %"},
    "sort_ticker": {"en": "Sort: Ticker A-Z", "ko": "정렬: 티커 A-Z"},
    "all_badges": {"en": "All Badges", "ko": "전체 배지"},
    "all_markets": {"en": "All Markets", "ko": "전체 시장"},
    "all_sectors": {"en": "All Sectors", "ko": "전체 섹터"},
    "preparing_list": {"en": "Preparing constituent list...", "ko": "종목 목록 준비 중..."},
    "target_price_placeholder": {"en": "Target price $", "ko": "목표가 $"},
    "set_alert_btn": {"en": "Set Alert", "ko": "알림 설정"},
    "shares_placeholder": {"en": "Shares", "ko": "수량"},
    "entry_price_placeholder": {"en": "Entry price $", "ko": "매수가 $"},
    "save_to_portfolio_btn": {"en": "Save to Portfolio", "ko": "포트폴리오에 저장"},
    "earnings_label": {"en": "Earnings:", "ko": "실적:"},
    "trend_label": {"en": "Trend", "ko": "추세"},
    "score_trend_label": {"en": "Score Trend (Today)", "ko": "점수 추세 (오늘)"},
    "high52_label": {"en": "52W High", "ko": "52주 최고"},
    "low52_label": {"en": "52W Low", "ko": "52주 최저"},
    "rsi_macd_tip": {"en": "RSI: below 30 usually means oversold, above 70 usually means overbought. MACD: positive means upward momentum, negative means downward.", "ko": "RSI: 30 이하는 보통 과매도, 70 이상은 보통 과매수를 의미합니다. MACD: 양수는 상승 모멘텀, 음수는 하락 모멘텀을 의미합니다."},
    "high52_tip": {"en": "How far the price is below its highest point in the last 52 weeks. Closer to 0% means near the high.", "ko": "최근 52주 최고가 대비 현재가가 얼마나 낮은지를 나타냅니다. 0%에 가까울수록 최고가에 근접한 것입니다."},
    "low52_tip": {"en": "How far the price is above its lowest point in the last 52 weeks.", "ko": "최근 52주 최저가 대비 현재가가 얼마나 높은지를 나타냅니다."},
    "trend_tip": {"en": "Whether the price is above (Uptrend) or below (Downtrend) its 200-day moving average — a common gauge of the long-term direction.", "ko": "현재가가 200일 이동평균선 위(상승 추세)인지 아래(하락 추세)인지를 나타내며, 장기 방향성을 가늠하는 일반적인 지표입니다."},
    "score_trend_tip": {"en": "How this ticker's quant score has moved since today's first scan — rising or falling.", "ko": "오늘 첫 스캔 이후 이 종목의 퀀트 점수가 오르고 있는지 내리고 있는지를 나타냅니다."},
    "ai_quant_report": {"en": "AI Quant Report", "ko": "AI 퀀트 리포트"},
    "loading_ai_analysis": {"en": "Loading AI analysis based on real data...", "ko": "실제 데이터를 기반으로 AI 분석을 불러오는 중..."},
    "usage_tip": {"en": "This flags entry timing on a single ticker, not a full plan. Many investors cap any one pick at a small slice of their total portfolio and spread bets across several signals rather than one — sizing and diversification are on you, not this tool.", "ko": "이 도구는 개별 종목의 진입 타이밍을 알려줄 뿐, 전체 투자 계획이 아닙니다. 많은 투자자는 종목 하나에 전체 자금의 일부만 배분하고 여러 신호에 분산합니다 — 비중과 분산은 이 도구가 아니라 본인의 판단입니다."},
    "news_label": {"en": "News", "ko": "뉴스"},
    "waiting_for_news": {"en": "Waiting for news...", "ko": "뉴스를 불러오는 중..."},
    "sma20": {"en": "SMA 20", "ko": "SMA 20"},
    "sma50": {"en": "SMA 50", "ko": "SMA 50"},
    "sma200": {"en": "SMA 200", "ko": "SMA 200"},
    "bollinger": {"en": "Bollinger Bands", "ko": "볼린저 밴드"},
    "volume_label": {"en": "Volume", "ko": "거래량"},
}


def t(key: str, lang: str) -> str:
    entry = UI_STRINGS.get(key)
    if not entry:
        return key
    return entry.get(lang) or entry.get("en") or key


# Dark palette, kept identical to the terminal's so the two never drift. Every page
# below already declares the same variable names against a light :root, so appending
# this override plus a data-theme stamp is all a page needs to follow the setting --
# which until now only /terminal did, leaving the other seven light-only.
DARK_THEME_VARS = (
    'html[data-theme="dark"]{--bg:#000000;--panel:#000000;--panel2:#0a0a0a;--border:#222222;'
    '--border2:#181818;--text:#a8a8a8;--head:#ffffff;--dim:#787878;--green:#26a69a;'
    '--red:#ef5350;--orange:#ff9800;--grid-line:#161616;'
    '--sb-bg:#0a0a0a;--sb-border:#222222;--sb-text:#8a8a8a;--sb-text-active:#ffffff;'
    '--sb-hover:#161616;--sb-danger:#ef5350}'
    'html[data-theme="dark"] .badge-ok{background:rgba(38,166,154,.15)}'
    'html[data-theme="dark"] .badge-warn{background:rgba(255,152,0,.15)}'
    'html[data-theme="dark"] .badge-danger{background:rgba(239,83,80,.15)}'
)


def adopt_browser_language(request: Request, email: str):
    """Carry an explicit language choice made before signing in onto the account.

    Without this, someone who switches to Korean on the landing page and then signs up
    lands in an English app, because the account was created with the 'en' default and
    the account preference outranks nothing -- the cookie would still win for display,
    but Settings would show English and the choice would not follow them to another
    browser. Only a real cookie counts here; a browser-detected language is a guess and
    is never written to the account.
    """
    chosen = (request.cookies.get(LANG_COOKIE) or "").strip().lower()
    if chosen not in LANGUAGE_NAMES:
        return
    try:
        conn = db()
        conn.execute("UPDATE users SET pref_language=? WHERE email=? AND pref_language<>?",
                     (chosen, email, chosen))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Error: {type(e).__name__}] language adopt failed for {email}: {e}")


def get_user_theme(email: str) -> str:
    conn = db()
    row = conn.execute("SELECT pref_theme FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    theme = row["pref_theme"] if row else "light"
    return theme if theme in ("dark", "light") else "light"


LANG_COOKIE = "qtfy_lang"
LANG_COOKIE_MAX_AGE = 365 * 86400


def _parse_accept_language(header: str) -> Optional[str]:
    """Pick the highest-q language we actually support out of an Accept-Language header
    ("ko-KR,ko;q=0.9,en-US;q=0.8" -> "ko"). Region subtags are dropped: ko-KR and ko are
    the same product to us."""
    best, best_q = None, -1.0
    for part in (header or "").split(","):
        part = part.strip()
        if not part:
            continue
        tag, _, params = part.partition(";")
        q = 1.0
        params = params.strip()
        if params.startswith("q="):
            try:
                q = float(params[2:])
            except ValueError:
                q = 1.0
        base = tag.strip().lower().split("-")[0]
        if base in LANGUAGE_NAMES and q > best_q:
            best, best_q = base, q
    return best


def resolve_lang(request: Request, user: Optional[str] = None) -> str:
    """One language decision for every page, public or not.

    Order matters and is the whole point: anything the reader chose outranks anything
    we guessed. The ?lang= parameter is a click on the header toggle; the cookie is that
    same choice remembered; the account preference is the choice made in Settings; only
    then do we fall back to sniffing the browser. Auto-detection can therefore never
    overwrite a decision someone actually made -- it only fills the gap before they make
    one.
    """
    q = (request.query_params.get("lang") or "").strip().lower()
    if q in LANGUAGE_NAMES:
        return q
    cookie = (request.cookies.get(LANG_COOKIE) or "").strip().lower()
    if cookie in LANGUAGE_NAMES:
        return cookie
    if user:
        saved = get_user_lang(user)
        if saved in LANGUAGE_NAMES:
            return saved
    detected = _parse_accept_language(request.headers.get("accept-language", ""))
    return detected or "en"


def get_user_lang(email: str) -> str:
    conn = db()
    row = conn.execute("SELECT pref_language FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    lang = row["pref_language"] if row else "en"
    return lang if lang in LANGUAGE_NAMES else "en"


def translate_body(body: str, lang: str, replacements: list) -> str:
    # Plain find-and-replace on the already-built English HTML/JS string rather than
    # making the template itself an f-string -- these page bodies are dense with JS
    # template literals (`${...}`), and turning the whole block into an f-string would
    # mean every one of those braces needs doubling to survive, which is exactly the
    # kind of mechanical, error-prone edit that already caused a real bug this session
    # (an unescaped apostrophe breaking a page's script). Each needle should be scoped
    # tightly enough (tag-bounded, not a bare word) that it can't match unintended text.
    if lang == "en":
        return body
    for needle, key in replacements:
        body = body.replace(needle, t(key, lang))
    return body


STRATEGY_MODE = "Long-Term Momentum Pullback"

MODE_CRITERIA = {
    "Long-Term Momentum Pullback": (
        "This mode looks for stocks in a confirmed long-term uptrend (price above its 200-day moving average) "
        "that have pulled back somewhat from recent highs — a potential entry within an established trend, not a "
        "reversal bet. Weight the 200-day trend and the pullback-from-high distance heavily; a stock far below its "
        "200-day MA does not fit this mode even if other numbers look attractive."
    ),
}


def ai_report_sync(ticker, price, change, mode, rsi, macd,
                    pct_from_high=None, pct_from_low=None, above_trend=None, language="en",
                    volume_ratio=None, news=None):
    if not ai_client:
        return None
    global AI_QUOTA_EXHAUSTED_DATE
    if AI_QUOTA_EXHAUSTED_DATE == today_str():
        return None
    language_name = LANGUAGE_NAMES.get(language, "English")
    near_high = pct_from_high is not None and pct_from_high >= -5
    near_low = pct_from_low is not None and pct_from_low <= 5
    overbought = rsi >= 70
    oversold = rsi <= 30
    high_text = (f"{pct_from_high:+.1f}% vs 52-week high — this counts as {'NEAR' if near_high else 'NOT near'} the high"
                 if pct_from_high is not None else "52-week high data unavailable")
    low_text = (f"{pct_from_low:+.1f}% vs 52-week low — this counts as {'NEAR' if near_low else 'NOT near'} the low"
                if pct_from_low is not None else "52-week low data unavailable")
    rsi_text = f"RSI {rsi:.2f} — this counts as {'OVERBOUGHT' if overbought else 'OVERSOLD' if oversold else 'neutral'}"
    trend_text = ("Above the 200-day moving average" if above_trend is True else
                  "Below the 200-day moving average" if above_trend is False else "200-day moving average data unavailable")
    volume_text = (f"Volume is {volume_ratio:.2f}x its 20-day average" if volume_ratio is not None
                   else "Volume data unavailable")
    mode_criteria = MODE_CRITERIA.get(mode, "")
    news_items = (news or [])[:5]
    if news_items:
        news_text = "Recent headlines:\n" + "\n".join(f"- {n.get('title','')}" for n in news_items if n.get("title"))
    else:
        news_text = "No recent news headlines were available."
    prompt = (
        f"Date: {display_date()}\nTicker: {ticker}\nPrice: ${price:.2f}\n"
        f"Change: {change:.2f}%\nMACD histogram: {macd:.4f}\n"
        f"{rsi_text}\n{high_text}\n{low_text}\n{trend_text}\n{volume_text}\n"
        f"Strategy mode: {mode}\n{mode_criteria}\n\n{news_text}\n\n"
        "This ticker was flagged automatically by a quant algorithm based purely on numeric thresholds, not "
        "hand-picked by a person. You must check for these two failure modes: "
        "(1) Has it already spiked short-term, sitting near its 52-week high with RSI in overbought territory, "
        "risking a blow-off top if someone chases it now? "
        "(2) Is this a bounce inside a long-term downtrend (below the 200-day MA) or near its 52-week low, "
        "risking a dead-cat bounce in a financially weak stock?\n"
        "The NEAR/NOT near and OVERBOUGHT/OVERSOLD/neutral labels above are the authoritative classification — "
        "every section you write must agree with them exactly. Do not independently judge from the raw percentage "
        "or RSI number whether it counts as 'near' or 'overbought'; use the label given. If you write a specific "
        "distance (e.g. '0.7% below the high'), the label next to it must match your own wording — check this "
        "before answering.\n"
        "Base your analysis only on the numbers and headlines actually provided. Do not estimate or invent missing "
        "data, and do not assume a headline's content beyond its title. This is informational analysis, not "
        "investment advice. Do not state a specific buy price, price target, or stop-loss price in any form. Do not "
        "just restate the numbers back — say what they mean for someone deciding whether to act now, later, or not "
        "at all, and commit to that view instead of hedging every sentence.\n\n"
        f"The values of quant_review, supply_demand, risk_review, news_analysis, and timing_reason must be written "
        f"in {language_name} only. The timing_verdict value itself must still be exactly one of the three English "
        "words below, never translated.\n"
        "Respond with ONLY the JSON object below, no other text:\n"
        '{"quant_review":"2-3 sentences on RSI/MACD/trend, and whether the setup actually fits the stated strategy mode\'s criteria",'
        '"supply_demand":"2-3 sentences on what volume-vs-average implies about buying or selling pressure",'
        '"risk_review":"2-3 sentences on the two failure modes above",'
        '"news_analysis":"2-3 sentences on what the headlines suggest, and whether they support or contradict the technical setup — or state plainly that no useful headlines were available",'
        '"timing_verdict":"one of: Favorable, Caution, or Risk",'
        '"timing_score":an integer from 0 to 100,'
        '"timing_reason":"2-3 sentence explanation for the score that references at least one specific number given above"}'
    )
    max_retries = 5
    extra_args = {"reasoning_effort": "low"} if "gpt-oss" in AI_MODEL else {}
    for attempt in range(max_retries):
        try:
            response = ai_client.chat.completions.create(
                model=AI_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.2,
                response_format={"type": "json_object"}, **extra_args,
            )
            content = response.choices[0].message.content.strip()
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                print(f"[Error: JSONDecodeError] Failed to parse AI response ({ticker}): {exc} | raw={content[:200]}")
                return None
            score = parsed.get("timing_score")
            try:
                score = max(0.0, min(100.0, float(score))) if score is not None else None
            except (TypeError, ValueError):
                score = None
            verdict = parsed.get("timing_verdict")
            if verdict not in ("Favorable", "Caution", "Risk"):
                verdict = None
            return {"report_json": content, "timing_score": score, "timing_verdict": verdict}
        except RateLimitError as exc:
            if "tokens per day" in str(exc).lower() or "TPD" in str(exc):
                # This is a hard daily cap, not a transient per-minute limit — every
                # remaining ticker today will fail identically, so stop burning
                # retries (and wall-clock time) on requests that cannot succeed.
                AI_QUOTA_EXHAUSTED_DATE = today_str()
                print(f"[Error: RateLimitError] AI daily token quota exhausted — skipping retries for the rest of today ({ticker}): {exc}")
                return None
            wait = 3.0 * (attempt + 1)
            match = re.search(r"try again in ([\d.]+)s", str(exc))
            if match:
                wait = float(match.group(1)) + 1.0
            if attempt < max_retries - 1:
                print(f"[Retry {attempt + 1}/{max_retries}] AI report rate limited ({ticker}): waiting {wait:.1f}s")
                time.sleep(wait)
                continue
            print(f"[Error: RateLimitError] AI report generation error ({ticker}): exceeded {max_retries} retries: {exc}")
            return None
        except Exception as exc:
            print(f"[Error: {type(exc).__name__}] AI report generation error ({ticker}): {exc}")
            return None


async def generate_ai_report(*args):
    return await asyncio.to_thread(ai_report_sync, *args)


def market_summary_ai_sync(stats):
    if not ai_client:
        return None
    global AI_QUOTA_EXHAUSTED_DATE
    if AI_QUOTA_EXHAUSTED_DATE == today_str():
        return None
    fmt_pct = lambda v: f"{v:+.2f}%" if v is not None else "unavailable"
    top_tickers = stats.get("top_tickers") or []
    top_lines = "\n".join(
        f"- {t['ticker']}: {fmt_pct(t['change_pct'])}, quant score {t['alpha_score']:.0f}, AI verdict so far: {t['timing_verdict'] or 'not yet reviewed'}"
        for t in top_tickers
    ) or "No tickers cleared the quant bar in this scan."
    vb = stats.get("verdict_breakdown") or {}
    prompt = (
        f"Date: {display_date()}\n"
        f"Universe scanned: {stats.get('universe_count')} stocks (S&P 500 + Nasdaq-100)\n"
        f"Advancers: {stats.get('advancers')}  Decliners: {stats.get('decliners')}  "
        f"Average change across the universe: {fmt_pct(stats.get('avg_change_pct'))}\n"
        f"S&P 500 index change today: {fmt_pct(stats.get('sp500_change_pct'))}\n"
        f"Nasdaq-100 index change today: {fmt_pct(stats.get('nasdaq_change_pct'))}\n"
        f"Cleared the quant bar (pullback-in-uptrend setup): {stats.get('detected_count')} of {stats.get('universe_count')}\n"
        f"AI risk verdicts issued so far today across all reviewed tickers -- "
        f"Favorable: {vb.get('Favorable', 0)}, Caution: {vb.get('Caution', 0)}, Risk: {vb.get('Risk', 0)}\n"
        f"Top tickers that cleared the quant bar today, ranked by quant score:\n{top_lines}\n\n"
        "Write a short, factual summary of today's market for a retail investor, based only on the numbers above. "
        "Do not predict what happens next, do not suggest buying or selling anything, and do not editorialize beyond "
        "what these specific numbers show. Cover: overall breadth (how many stocks rose vs fell), how the two major "
        "indices moved, and what today's scan actually turned up. This is informational only, never investment advice.\n"
        "Respond with ONLY the JSON object below, no other text:\n"
        '{"headline":"one short sentence, under 15 words, stating today\'s overall tone in plain terms",'
        '"summary":"3-4 sentences covering breadth, index moves, and what the scan found today"}'
    )
    max_retries = 5
    extra_args = {"reasoning_effort": "low"} if "gpt-oss" in AI_MODEL else {}
    for attempt in range(max_retries):
        try:
            response = ai_client.chat.completions.create(
                model=AI_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.2,
                response_format={"type": "json_object"}, **extra_args,
            )
            content = response.choices[0].message.content.strip()
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                print(f"[Error: JSONDecodeError] Failed to parse market AI summary response: {exc} | raw={content[:200]}")
                return None
            headline = parsed.get("headline")
            summary = parsed.get("summary")
            if not headline or not summary:
                return None
            return {"headline": headline, "summary": summary}
        except RateLimitError as exc:
            if "tokens per day" in str(exc).lower() or "TPD" in str(exc):
                AI_QUOTA_EXHAUSTED_DATE = today_str()
                print(f"[Error: RateLimitError] AI daily token quota exhausted — skipping market summary retries for today: {exc}")
                return None
            wait = 3.0 * (attempt + 1)
            match = re.search(r"try again in ([\d.]+)s", str(exc))
            if match:
                wait = float(match.group(1)) + 1.0
            if attempt < max_retries - 1:
                print(f"[Retry {attempt + 1}/{max_retries}] Market AI summary rate limited: waiting {wait:.1f}s")
                time.sleep(wait)
                continue
            print(f"[Error: RateLimitError] Market AI summary generation error: exceeded {max_retries} retries: {exc}")
            return None
        except Exception as exc:
            print(f"[Error: {type(exc).__name__}] Market AI summary generation error: {exc}")
            return None


def analyze_dataframe(ticker, df):
    if df is None or len(df) < 70:
        return None
    try:
        close = normalize_series(df, "Close").dropna()
        if len(close) < 70:
            return None
        high = normalize_series(df, "High").reindex(close.index)
        rsi_series = calculate_rsi(close)
        _, _, macd_hist = calculate_macd(close)
        price = float(close.iloc[-1]); prev = float(close.iloc[-2])
        change = (price / prev - 1) * 100 if prev else 0.0
        score = calculate_alpha_score(close, high)
        if score is None:
            return None

        window = close.tail(252)
        high_52w = float(window.max()); low_52w = float(window.min())
        pct_from_high = round((price / high_52w - 1) * 100, 2) if high_52w else None
        pct_from_low = round((price / low_52w - 1) * 100, 2) if low_52w else None
        above_trend = None
        pct_above_sma200 = None
        if len(close) >= 200:
            sma200 = close.rolling(200).mean().iloc[-1]
            if pd.notna(sma200) and float(sma200) > 0:
                above_trend = bool(price > float(sma200))
                pct_above_sma200 = round((price / float(sma200) - 1) * 100, 2)

        # Same 20-day prior high calculate_alpha_score() uses -- through yesterday, so a
        # gap-up day cannot serve as its own pullback reference.
        pct_off_20d_high = None
        if len(high) >= 21:
            hh20_prior = float(high.iloc[-21:-1].max())
            if hh20_prior > 0:
                pct_off_20d_high = round((1.0 - price / hh20_prior) * 100, 2)

        volume_ratio = None
        try:
            volume = normalize_series(df, "Volume").dropna()
            if len(volume) >= 20:
                avg_volume_20d = float(volume.tail(20).mean())
                if avg_volume_20d > 0:
                    volume_ratio = round(float(volume.iloc[-1]) / avg_volume_20d, 2)
        except Exception:
            volume_ratio = None

        return {"ticker": ticker, "price": round(price, 2), "change": round(change, 2),
                "alpha_score": score, "rsi": round(float(rsi_series.iloc[-1]), 2),
                "macd": round(float(macd_hist.iloc[-1]), 4),
                "pct_from_52w_high": pct_from_high, "pct_from_52w_low": pct_from_low,
                "above_200d_sma": above_trend, "volume_ratio": volume_ratio,
                "pct_off_20d_high": pct_off_20d_high, "pct_above_200d_sma": pct_above_sma200}
    except Exception as e:
        print(f"[Error: {type(e).__name__}] Dataframe analysis error ({ticker}): {e}")
        return None



# -----------------------------------------------------------------------------
# Score radar
# -----------------------------------------------------------------------------
# A radar chart only tells the truth if a bigger shape always means "better for this
# strategy", so every axis below is normalised in that direction -- including RSI, where
# for a pullback entry LOW is the good end, not high.
#
# The harder honesty problem: only three of these five actually move the score.
# calculate_alpha_score() decides on exactly two things (is price above the 200-day SMA,
# and how centred the pullback is in the 10-25% band), and the AI's timing_score is the
# other half of the overall. Room and Momentum are context the reader may want but they
# do not score anything. Each axis carries scored=True/False and the UI shows the
# difference, because a five-axis chart that implies five inputs would be the exact kind
# of overclaim this product cannot afford.
SNOWFLAKE_AXES = ("TREND", "PULLBACK", "AI CHECK", "ROOM", "MOMENTUM")


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)



# --- Snowflake scoring -------------------------------------------------------------
# "Undervalued" is meaningless in the abstract: a 30x P/E is dear for a utility and
# cheap for software. Every valuation and quality axis is therefore a percentile against
# the ticker's OWN SECTOR, which is what makes the shape comparable across the board.
# Sectors with too few peers to rank fairly fall back to the whole universe.
MIN_SECTOR_PEERS = 8
_PEER_CACHE: dict = {"at": 0.0, "by_sector": {}, "all": {}}
_PEER_CACHE_TTL = 900.0

# (column, higher_is_better)
_PEER_METRICS = (
    ("trailing_pe", False), ("price_to_book", False), ("ev_ebitda", False),
    ("return_on_equity", True), ("profit_margin", True),
    ("debt_to_equity", False), ("current_ratio", True),
    ("revenue_growth", True), ("earnings_growth", True),
    ("dividend_yield", True),
)


def _load_peer_groups() -> tuple:
    now = time.time()
    if now - _PEER_CACHE["at"] < _PEER_CACHE_TTL and _PEER_CACHE["by_sector"]:
        return _PEER_CACHE["by_sector"], _PEER_CACHE["all"]
    by_sector: dict = {}
    all_vals: dict = {c: [] for c, _ in _PEER_METRICS}
    try:
        conn = db()
        cols = ",".join(c for c, _ in _PEER_METRICS)
        for r in conn.execute(f"SELECT sector,{cols} FROM fundamentals"):
            sec = r["sector"] or "(unknown)"
            bucket = by_sector.setdefault(sec, {c: [] for c, _ in _PEER_METRICS})
            for c, _ in _PEER_METRICS:
                v = r[c]
                # A negative P/E means the company lost money; it is not "cheap", so it
                # must not rank as the best value in its sector. Drop it from the ladder.
                if v is None:
                    continue
                if c in ("trailing_pe", "price_to_book", "ev_ebitda") and v <= 0:
                    continue
                bucket[c].append(float(v))
                all_vals[c].append(float(v))
        conn.close()
    except Exception as e:
        print(f"[Error: {type(e).__name__}] peer group load failed: {e}", flush=True)
        return {}, {c: [] for c, _ in _PEER_METRICS}
    for bucket in by_sector.values():
        for c in bucket:
            bucket[c].sort()
    for c in all_vals:
        all_vals[c].sort()
    _PEER_CACHE.update({"at": now, "by_sector": by_sector, "all": all_vals})
    return by_sector, all_vals


def _percentile(sorted_vals: list, value: float, higher_is_better: bool) -> Optional[float]:
    n = len(sorted_vals)
    if n < 3 or value is None:
        return None
    lo = bisect.bisect_left(sorted_vals, value)
    hi = bisect.bisect_right(sorted_vals, value)
    rank = (lo + hi) / 2.0
    pct = rank / n * 100.0
    return pct if higher_is_better else 100.0 - pct


def _axis_from(metrics: list, row, by_sector: dict, all_vals: dict) -> Optional[float]:
    """Average the available percentiles for one axis. Missing inputs are skipped rather
    than scored as zero -- a company with no dividend is not a company with a terrible
    dividend, and a missing field should not dent the shape as if it were bad news."""
    sec = (row["sector"] or "(unknown)") if "sector" in row.keys() else "(unknown)"
    bucket = by_sector.get(sec) or {}
    scores = []
    for col, higher in metrics:
        v = row[col] if col in row.keys() else None
        if v is None:
            continue
        if col in ("trailing_pe", "price_to_book", "ev_ebitda") and v <= 0:
            continue
        peers = bucket.get(col) or []
        if len(peers) < MIN_SECTOR_PEERS:
            peers = all_vals.get(col) or []
        pct = _percentile(peers, float(v), higher)
        if pct is not None:
            scores.append(pct)
    if not scores:
        return None
    return sum(scores) / len(scores)


def _fmt_ratio(v, suffix="", digits=1):
    if v is None:
        return None
    try:
        return f"{float(v):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return None


def _fmt_pct(v, digits=1):
    if v is None:
        return None
    try:
        return f"{float(v) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return None


SNOWFLAKE_AXES = ("VALUE", "GROWTH", "PROFIT", "HEALTH", "DIVIDEND", "AI CHECK")


def _snowflake_axes(row, ticker: Optional[str] = None, lang: str = "en") -> Optional[dict]:
    """Six axes over the company's own fundamentals, ranked against its sector peers.

    Every axis is a 0-100 percentile where bigger is better, so the shape reads at a
    glance the way a snowflake is supposed to: a wide shape is a cheap, growing,
    profitable, solvent company. An axis with no data is returned as None and drawn as a
    gap rather than a zero -- a company that pays no dividend has no dividend rank, and
    denting the shape for it would be inventing bad news.
    """
    if row is None:
        return None
    d = dict(row)
    tk = ticker or d.get("ticker")

    fund = None
    if tk:
        try:
            conn = db()
            fund = conn.execute("SELECT * FROM fundamentals WHERE ticker=?", (tk,)).fetchone()
            conn.close()
        except Exception as e:
            print(f"[Error: {type(e).__name__}] fundamentals lookup failed for {tk}: {e}")

    by_sector, all_vals = _load_peer_groups() if fund is not None else ({}, {})

    def axis(metrics):
        return _axis_from(metrics, fund, by_sector, all_vals) if fund is not None else None

    value_v = axis([("trailing_pe", False), ("price_to_book", False), ("ev_ebitda", False)])
    growth_v = axis([("revenue_growth", True), ("earnings_growth", True)])
    profit_v = axis([("return_on_equity", True), ("profit_margin", True)])
    health_v = axis([("debt_to_equity", False), ("current_ratio", True)])
    div_v = axis([("dividend_yield", True)])

    timing = d.get("timing_score")
    ai_v = float(timing) if timing is not None else None

    g = (lambda c: (fund[c] if fund is not None and c in fund.keys() else None))

    def pos(c):
        """Ratios that are only meaningful when positive. A negative P/E means losses and
        a negative P/B means negative book value; both are excluded from the ranking, so
        printing them next to a high score would contradict the shape."""
        v = g(c)
        return v if (v is not None and v > 0) else None

    def div(c):
        """A company paying nothing has no yield and no payout to report -- showing
        "0.0%" reads as a measured figure rather than an absence."""
        v = g(c)
        return v if (v is not None and v > 0) else None

    sector = g("sector") or "-"

    L = (lambda k: t(k, lang))

    def raw(*parts):
        got = [x for x in parts if x]
        return " · ".join(got) if got else L("sf_nodata")

    axes = [
        {"key": "VALUE", "label_key": "sf_value", "value": value_v, "scored": True,
         "raw": raw(_fmt_ratio(pos("trailing_pe"), "x " + L("sf_pe")), _fmt_ratio(pos("price_to_book"), "x " + L("sf_pb"))),
         "help_key": "sf_value_help"},
        {"key": "GROWTH", "label_key": "sf_growth", "value": growth_v, "scored": True,
         "raw": raw(_fmt_pct(g("revenue_growth")) and f'{L("sf_rev")} {_fmt_pct(g("revenue_growth"))}',
                    _fmt_pct(g("earnings_growth")) and f'{L("sf_earn")} {_fmt_pct(g("earnings_growth"))}'),
         "help_key": "sf_growth_help"},
        {"key": "PROFIT", "label_key": "sf_profit", "value": profit_v, "scored": True,
         "raw": raw(_fmt_pct(g("return_on_equity")) and f'ROE {_fmt_pct(g("return_on_equity"))}',
                    _fmt_pct(g("profit_margin")) and f'{L("sf_margin")} {_fmt_pct(g("profit_margin"))}'),
         "help_key": "sf_profit_help"},
        {"key": "HEALTH", "label_key": "sf_health", "value": health_v, "scored": True,
         "raw": raw(_fmt_ratio(g("debt_to_equity"), "% " + L("sf_debt"), 0),
                    _fmt_ratio(pos("current_ratio"), "x " + L("sf_current"), 2)),
         "help_key": "sf_health_help"},
        {"key": "DIVIDEND", "label_key": "sf_dividend", "value": div_v, "scored": True,
         "raw": (raw(_fmt_ratio(div("dividend_yield"), "% " + L("sf_yield"), 2),
                     _fmt_pct(div("payout_ratio")) and f'{L("sf_payout")} {_fmt_pct(div("payout_ratio"))}')
                 if div("dividend_yield") else L("sf_nodiv")),
         "help_key": "sf_dividend_help"},
        {"key": "AI CHECK", "label_key": "sf_ai", "value": ai_v, "scored": True,
         "raw": (f"{timing:.0f}/100" if timing is not None else None) or L("sf_unscored"),
         "help_key": "sf_ai_help"},
    ]

    known = [a["value"] for a in axes if a["value"] is not None]
    return {
        "axes": axes,
        "sector": sector,
        "have_fundamentals": fund is not None,
        "covered": len(known),
        "quant_score": d.get("alpha_score"),
        "ai_score": timing,
        "overall_score": (round((float(d["alpha_score"]) + float(timing)) / 2.0, 1)
                          if d.get("alpha_score") is not None and timing is not None else None),
        "verdict": d.get("timing_verdict"),
        "passed": bool(d.get("quant_pass")),
    }


async def build_scan_row(ticker: str, mode: str, df=None, make_ai=False):
    if df is None:
        df = await download_stock(ticker, "1d")
    row = analyze_dataframe(ticker, df)
    if row is None:
        return None
    if make_ai:
        ai_result = await generate_ai_report(
            ticker, row["price"], row["change"], mode, row["rsi"], row["macd"],
            row.get("pct_from_52w_high"), row.get("pct_from_52w_low"), row.get("above_200d_sma"),
        )
        row["ai_report"] = ai_result["report_json"] if ai_result else None
        row["timing_score"] = ai_result["timing_score"] if ai_result else None
        row["timing_verdict"] = ai_result["timing_verdict"] if ai_result else None
    else:
        row["short_percent"] = None
        row["ai_report"] = None
        row["timing_score"] = None
        row["timing_verdict"] = None
    return row

# -----------------------------------------------------------------------------
# Public track record (append-only)
#
# Read /record's page copy before changing anything here: the value of that page rests
# entirely on these rows being written automatically and never edited afterwards. Adding
# an UPDATE, a DELETE, or any human-facing way to add a row would silently destroy it.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Fundamentals
# -----------------------------------------------------------------------------
# The score radar used to plot technical inputs only, because that was all the app
# stored. These are the numbers a snowflake is actually supposed to answer -- is this
# company cheap, growing, profitable, solvent, paying you -- so they are fetched and
# kept for the whole universe, once a day.
# Balance sheets are quarterly. A 20-hour TTL meant re-fetching all ~518 tickers every
# single day for numbers that had not moved, which is the request volume that got the
# provider to start refusing us -- and worse, it would have expired the 491 good rows we
# already hold and blanked the snowflake for everyone. Two weeks is still far more often
# than the underlying figures change.
FUNDAMENTALS_TTL = 14 * 24 * 3600
FUNDAMENTALS_FETCH_GAP = 0.15         # polite spacing; a full universe pass takes ~7 min

_FUNDAMENTAL_FIELDS = (
    ("sector", "sector"),
    ("market_cap", "marketCap"),
    ("trailing_pe", "trailingPE"),
    ("price_to_book", "priceToBook"),
    ("ev_ebitda", "enterpriseToEbitda"),
    ("peg", "trailingPegRatio"),
    ("return_on_equity", "returnOnEquity"),
    ("profit_margin", "profitMargins"),
    ("debt_to_equity", "debtToEquity"),
    ("current_ratio", "currentRatio"),
    ("revenue_growth", "revenueGrowth"),
    ("earnings_growth", "earningsGrowth"),
    ("dividend_yield", "dividendYield"),
    ("payout_ratio", "payoutRatio"),
)


def _fetch_one_fundamental(ticker: str, attempts: int = 4) -> Optional[dict]:
    """The provider answers the first few requests of a fresh process with a body that
    parses fine but carries no fields, while it negotiates a session. That is why the
    tickers at the very start of the universe were the ones left without fundamentals --
    they were spent on the handshake. Back off and try again rather than writing them
    off for the day."""
    info, last_detail = {}, "no attempt made"
    for attempt in range(attempts):
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception as e:
            last_detail = f"{type(e).__name__}: {str(e)[:120]}"
            info = {}
        else:
            if not info.get("marketCap"):
                last_detail = f"response had {len(info)} fields, no marketCap"
        if info.get("marketCap"):
            break
        if attempt + 1 < attempts:
            time.sleep(1.5 * (2 ** attempt))   # 1.5s, 3s, 6s
    if not info.get("marketCap"):
        FUNDAMENTALS_STATUS["last_fetch_failure"] = f"{ticker}: {last_detail}"
        print(f"[fundamentals] {ticker} gave up after {attempts} attempts — {last_detail}", flush=True)
        return None
    row = {"ticker": ticker, "fetched_at": time.time()}
    for col, key in _FUNDAMENTAL_FIELDS:
        v = info.get(key)
        if col == "sector":
            row[col] = str(v) if v else None
        else:
            try:
                row[col] = float(v) if v is not None else None
            except (TypeError, ValueError):
                row[col] = None
    return row


def _save_fundamentals(rows: list):
    if not rows:
        return
    cols = ["ticker", "fetched_at"] + [c for c, _ in _FUNDAMENTAL_FIELDS]
    placeholders = ",".join("?" for _ in cols)
    updates = ",".join(f"{c}=excluded.{c}" for c in cols if c != "ticker")
    conn = db()
    try:
        conn.executemany(
            f"INSERT INTO fundamentals({','.join(cols)}) VALUES({placeholders}) "
            f"ON CONFLICT(ticker) DO UPDATE SET {updates}",
            [tuple(r.get(c) for c in cols) for r in rows],
        )
        conn.commit()
    except Exception as e:
        print(f"[Error: {type(e).__name__}] fundamentals save failed: {e}", flush=True)
    finally:
        conn.close()


async def _warm_provider_session():
    """Spend the handshake on a throwaway request instead of on real tickers. Without
    this the first names in the list absorb the empty responses and are the ones that
    end up missing."""
    for attempt in range(4):
        row = await asyncio.to_thread(_fetch_one_fundamental, "AAPL", 1)
        if row:
            FUNDAMENTALS_STATUS["provider_warm"] = True
            return True
        await asyncio.sleep(2 * (attempt + 1))
    FUNDAMENTALS_STATUS["provider_warm"] = False
    print("[fundamentals] provider session would not warm up — proceeding anyway", flush=True)
    return False


async def refresh_fundamentals(force: bool = False, tickers: Optional[list] = None,
                               label: str = "universe"):
    """Sequential on purpose: this is the same provider the price scan uses, and running
    both wide at once is what triggered rate-limiting before. Nothing waits on it."""
    tickers = list(tickers if tickers is not None else UNIVERSE)
    if not tickers:
        return
    conn = db()
    fresh = {r["ticker"] for r in conn.execute(
        "SELECT ticker FROM fundamentals WHERE fetched_at >= ?", (time.time() - FUNDAMENTALS_TTL,)
    )} if not force else set()
    conn.close()
    todo = [t for t in tickers if t not in fresh]
    if not todo:
        # "ran, nothing was stale" has to be distinguishable from "never ran" -- they
        # look identical from the outside, which is the whole reason this is here.
        FUNDAMENTALS_STATUS["last_fetch_attempted"] = 0
        FUNDAMENTALS_STATUS["last_fetch_saved"] = 0
        return
    # Fetch what people actually open first. A pass takes ~7 minutes and a restart can
    # cut it short, and in universe order the tickers left behind were arbitrary -- which
    # is how a ticker sitting at the top of today's scanner ended up with an empty
    # snowflake. Today's detected names come first, then everything else.
    print(f"[fundamentals:{label}] refreshing {len(todo)} of {len(tickers)}", flush=True)
    FUNDAMENTALS_STATUS["last_fetch_attempted"] = len(todo)
    # If a known-good ticker comes back empty the provider is refusing us, and grinding
    # through hundreds of doomed requests only digs the hole deeper. Stop the pass and
    # let the caller wait it out.
    if not await _warm_provider_session():
        FUNDAMENTALS_STATUS.update({"phase": "provider refusing requests",
                                    "blocked_since": FUNDAMENTALS_STATUS.get("blocked_since") or time.time()})
        FUNDAMENTALS_STATUS["last_fetch_saved"] = 0
        print(f"[fundamentals:{label}] provider is refusing requests — skipping this pass", flush=True)
        return
    FUNDAMENTALS_STATUS["blocked_since"] = None
    batch, saved = [], 0
    for i, tk in enumerate(todo):
        row = await asyncio.to_thread(_fetch_one_fundamental, tk)
        if row:
            batch.append(row)
        if len(batch) >= 25:
            await asyncio.to_thread(_save_fundamentals, batch)
            saved += len(batch)
            batch = []
        await asyncio.sleep(FUNDAMENTALS_FETCH_GAP)
    if batch:
        await asyncio.to_thread(_save_fundamentals, batch)
        saved += len(batch)
    print(f"[fundamentals:{label}] saved {saved}/{len(todo)}", flush=True)
    FUNDAMENTALS_STATUS["last_fetch_saved"] = saved
    _PEER_CACHE["at"] = 0.0   # ladders are stale the moment new rows land


# Two jobs with different reasons to exist:
#
#   detected  -- the ~50 tickers whose snowflake anyone actually opens. ~40 seconds, so
#                it runs after every scan and a restart mid-pass costs nothing.
#   universe  -- all ~518, and its only job is the sector peer ladders the percentiles
#                are measured against. Ranking 50 detected names against each other
#                would be meaningless: they land 4-5 per sector, below the threshold for
#                a fair comparison, so the whole "cheap for its sector" claim collapses.
#                Fundamentals move quarterly, so weekly is plenty for a ladder.
FUNDAMENTALS_HOUR_ET = 18            # quiet window: 2h after the last scan of the day
FUNDAMENTALS_UNIVERSE_DOW = 6        # Sunday -- the full pass, for the peer ladders
FUNDAMENTALS_MIN_COVERAGE = 0.60     # ladders need breadth, not completeness

# A create_task'd scheduler that dies takes its traceback with it, and an empty
# snowflake looks identical whether the job is running, waiting on a lock, or dead.
# This is the only way to tell those apart from outside.
FUNDAMENTALS_STATUS: dict = {
    "started": False, "loops": 0, "last_loop_at": None, "last_error": None,
    "last_detected_count": None, "last_fetch_attempted": None, "last_fetch_saved": None,
    "waiting_on_lock": False, "phase": "not started",
    "last_fetch_failure": None, "provider_warm": None, "blocked_since": None,
}


def detected_tickers_today() -> list:
    try:
        conn = db()
        rows = [r["ticker"] for r in conn.execute(
            "SELECT ticker FROM daily_scans WHERE scan_date=? AND quant_pass=1", (today_str(),))]
        conn.close()
        return rows
    except Exception as e:
        print(f"[Error: {type(e).__name__}] detected ticker lookup failed: {e}", flush=True)
        return []


async def refresh_detected_fundamentals():
    """The only tickers whose snowflake anyone actually opens. ~50 names, about 40
    seconds, so this can run after every scan instead of once a day -- and a restart
    mid-pass costs almost nothing."""
    detected = detected_tickers_today()
    FUNDAMENTALS_STATUS["last_detected_count"] = len(detected)
    if detected:
        await refresh_fundamentals(tickers=detected, label="detected")


async def fundamentals_scheduler():
    FUNDAMENTALS_STATUS.update({"started": True, "phase": "warming up"})
    await asyncio.sleep(60)
    while True:
        try:
            FUNDAMENTALS_STATUS.update({
                "loops": FUNDAMENTALS_STATUS["loops"] + 1,
                "last_loop_at": datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S"),
                "phase": "waiting for batch lock", "waiting_on_lock": True,
            })
            # Whatever is on the scanner right now gets covered first and cheaply.
            async with BATCH_LOCK:
                FUNDAMENTALS_STATUS.update({"waiting_on_lock": False, "phase": "detected refresh"})
                await refresh_detected_fundamentals()

            # The ladder: rebuild weekly, or straight away if it is too thin to rank
            # against. A brand-new deploy has no ladder at all, and without one every
            # percentile would be computed against a handful of rows.
            conn = db()
            have = conn.execute("SELECT COUNT(*) FROM fundamentals").fetchone()[0]
            conn.close()
            now_et = datetime.now(ZoneInfo("America/New_York"))
            thin = bool(UNIVERSE) and have < len(UNIVERSE) * FUNDAMENTALS_MIN_COVERAGE
            due = now_et.weekday() == FUNDAMENTALS_UNIVERSE_DOW and now_et.hour == FUNDAMENTALS_HOUR_ET
            if thin or due:
                print(f"[fundamentals] ladder {have}/{len(UNIVERSE)} "
                      f"({'too thin' if thin else 'weekly rebuild'})", flush=True)
                FUNDAMENTALS_STATUS["phase"] = "universe ladder rebuild"
                async with BATCH_LOCK:
                    await refresh_fundamentals()
            FUNDAMENTALS_STATUS.update({"phase": "idle", "last_error": None})
        except Exception as e:
            FUNDAMENTALS_STATUS.update({
                "phase": "error",
                "last_error": f"{type(e).__name__}: {e}",
                "waiting_on_lock": False,
            })
            print(f"[Error: {type(e).__name__}] fundamentals scheduler: {e}", flush=True)
            traceback.print_exc()
        # Retrying a refusing provider every hour is what keeps it refusing.
        blocked = FUNDAMENTALS_STATUS.get("blocked_since")
        await asyncio.sleep(6 * 3600 if blocked else 3600)


def _track_row_hash(prev_hash, record_date, ticker, list_type, entry_price, alpha_score, recorded_at):
    payload = f"{prev_hash or ''}|{record_date}|{ticker}|{list_type}|{entry_price}|{alpha_score}|{recorded_at}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _benchmark_close():
    """S&P proxy close from the cache the scan already populated -- no extra API call."""
    try:
        cached = CACHE["historical"].get("single:^GSPC:1d")
        if cached is None:
            return None
        closes = normalize_series(cached["data"], "Close").dropna()
        return round(float(closes.iloc[-1]), 2) if not closes.empty else None
    except Exception:
        return None


def write_track_record(conn, date, results, now, benchmark=None):
    """Append today's qualifying tickers. Called only from the scanner.

    INSERT OR IGNORE against UNIQUE(record_date,ticker,list_type) means the first scan of
    the day that flags a ticker sets its entry price, and the later scans that day are
    no-ops -- entries can never be re-priced after the fact.
    """
    if benchmark is None:
        benchmark = _benchmark_close()
    # The day's closest-to-qualifying names, by score, among those that did not pass.
    near_miss = {
        r["ticker"] for r in sorted(
            (r for r in results
             if not r.get("quant_pass") and r.get("alpha_score") is not None and r.get("price") is not None),
            key=lambda r: r["alpha_score"], reverse=True,
        )[:NEAR_MISS_COUNT]
    }
    heads = {}
    for lt in ("passed", "near_miss"):
        row = conn.execute(
            "SELECT row_hash FROM track_record WHERE list_type=? ORDER BY id DESC LIMIT 1", (lt,)
        ).fetchone()
        heads[lt] = row["row_hash"] if row else None
    for r in results:
        alpha = r.get("alpha_score")
        price = r.get("price")
        if price is None or alpha is None:
            continue
        if r.get("quant_pass"):
            list_type = "passed"
        elif r["ticker"] in near_miss:
            list_type = "near_miss"
        else:
            continue
        ticker = r["ticker"]
        # Skip before hashing so a same-day duplicate can't advance the chain head.
        if conn.execute(
            "SELECT 1 FROM track_record WHERE record_date=? AND ticker=? AND list_type=?",
            (date, ticker, list_type),
        ).fetchone():
            continue
        row_hash = _track_row_hash(heads[list_type], date, ticker, list_type, price, alpha, now)
        try:
            conn.execute("""
                INSERT OR IGNORE INTO track_record
                (record_date,ticker,list_type,entry_price,alpha_score,overall_score,
                 timing_verdict,benchmark_price,recorded_at,prev_hash,row_hash)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (date, ticker, list_type, price, alpha, None, None, benchmark, now,
                  heads[list_type], row_hash))
            heads[list_type] = row_hash
        except Exception as exc:
            print(f"[Error: {type(exc).__name__}] Track record save error ({ticker}): {exc}", flush=True)


def verify_track_chain(rows):
    """Recompute the chain; returns (ok, first_broken_index). Rows must be id-ascending."""
    prev = None
    for i, r in enumerate(rows):
        expected = _track_row_hash(prev, r["record_date"], r["ticker"], r["list_type"],
                                   r["entry_price"], r["alpha_score"], r["recorded_at"])
        if expected != r["row_hash"] or (r["prev_hash"] or None) != prev:
            return False, i
        prev = r["row_hash"]
    return True, None


# -----------------------------------------------------------------------------
# Batch scanner
# -----------------------------------------------------------------------------
async def run_eod_batch_process(mode="Long-Term Momentum Pullback"):
    async with BATCH_LOCK:
        BATCH_STATUS.update({"running": True, "processed": 0, "total": len(UNIVERSE), "saved": 0, "started_at": time.time(), "finished_at": None, "error": None})
        try:
            if not UNIVERSE:
                await refresh_universe()
            if not UNIVERSE:
                BATCH_STATUS.update({"running": False, "saved": 0, "finished_at": time.time(), "error": "Universe is unavailable."})
                return []
            raw = await batch_download_stocks(UNIVERSE)
            results = []
            for ticker, df in raw.items():
                try:
                    row = await build_scan_row(ticker, mode, df=df, make_ai=False)
                    if row:
                        universe = "S&P 500" if ticker in UNIVERSE_META["sp500"] else "Nasdaq-100"
                        row["universe"] = universe
                        results.append(row)
                except Exception as exc:
                    print(f"[Error: {type(exc).__name__}] Scan row error ({ticker}): {exc}")
            # Only overwrite the on-disk snapshot when this scan actually got most of the
            # universe -- a mostly-empty scan (e.g. rate-limited) would otherwise stomp a
            # good snapshot from an earlier run with a near-empty one, destroying the
            # exact fallback data this is meant to protect.
            if len(raw) >= len(UNIVERSE) * 0.5:
                await asyncio.to_thread(save_historical_snapshot, raw)
            del raw  # 518 tickers' worth of price history -- free it now, not at function exit

            for row in results:
                row["short_percent"] = None
                row["ai_report"] = None
                row["quant_pass"] = 1 if float(row.get("alpha_score") or 0) >= QUANT_PASS_THRESHOLD else 0

            conn = db(); now = time.time(); date = today_str()
            for r in results:
                try:
                    conn.execute("""
                        INSERT INTO daily_scans
                        (scan_date,ticker,universe,price,change_pct,alpha_score,rsi,macd,
                         pct_from_52w_high,pct_from_52w_low,above_200d_sma,volume_ratio,
                         pct_off_20d_high,pct_above_200d_sma,
                         ai_report,short_percent,ai_status,ai_mode,ai_updated_at,ai_error,quant_pass,created_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(scan_date,ticker) DO UPDATE SET
                            universe=excluded.universe, price=excluded.price, change_pct=excluded.change_pct,
                            alpha_score=excluded.alpha_score, rsi=excluded.rsi, macd=excluded.macd,
                            pct_from_52w_high=excluded.pct_from_52w_high, pct_from_52w_low=excluded.pct_from_52w_low,
                            above_200d_sma=excluded.above_200d_sma, volume_ratio=excluded.volume_ratio,
                            pct_off_20d_high=excluded.pct_off_20d_high,
                            pct_above_200d_sma=excluded.pct_above_200d_sma,
                            ai_report=COALESCE(excluded.ai_report,daily_scans.ai_report),
                            short_percent=excluded.short_percent,
                            ai_status=CASE WHEN daily_scans.ai_report IS NOT NULL THEN daily_scans.ai_status ELSE 'PENDING' END,
                            ai_mode=CASE WHEN daily_scans.ai_report IS NOT NULL THEN daily_scans.ai_mode ELSE excluded.ai_mode END,
                            ai_updated_at=daily_scans.ai_updated_at,
                            ai_error=NULL,
                            quant_pass=excluded.quant_pass,
                            created_at=excluded.created_at
                    """, (
                        date, r["ticker"], r["universe"], r["price"], r["change"], r["alpha_score"],
                        r["rsi"], r["macd"],
                        r.get("pct_from_52w_high"), r.get("pct_from_52w_low"),
                        (1 if r.get("above_200d_sma") else (0 if r.get("above_200d_sma") is False else None)),
                        r.get("volume_ratio"),
                        r.get("pct_off_20d_high"), r.get("pct_above_200d_sma"),
                        r.get("ai_report"),
                        r.get("short_percent"), "PENDING", mode, None, None, r.get("quant_pass", 0), now
                    ))
                    BATCH_STATUS["saved"] += 1
                except Exception as exc:
                    print(f"[Error: {type(exc).__name__}] Batch DB save error ({r.get('ticker', '?')}): {exc}", flush=True)

            conn.commit()
            for r in results:
                try:
                    conn.execute(
                        "INSERT INTO scan_history(scan_date,ticker,captured_at,price,alpha_score) VALUES(?,?,?,?,?)",
                        (date, r["ticker"], now, r["price"], r.get("alpha_score")),
                    )
                except Exception as exc:
                    print(f"[Error: {type(exc).__name__}] Scan history save error ({r.get('ticker', '?')}): {exc}", flush=True)
            try:
                await download_stock("^GSPC", "1d")
            except Exception as exc:
                print(f"[Error: {type(exc).__name__}] Benchmark fetch for track record failed: {exc}", flush=True)
            write_track_record(conn, date, results, now, _benchmark_close())
            # Fundamentals for whatever this scan just detected. Scheduled rather than
            # awaited: the scan should not wait ~40s on a company-data provider, and a
            # newly detected ticker having an empty snowflake for a minute is fine --
            # having one for a day, which is what the daily-only job produced, was not.
            asyncio.create_task(refresh_detected_fundamentals())
            cutoff_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
            conn.execute("DELETE FROM scan_history WHERE scan_date < ?", (cutoff_date,))
            conn.commit()
            alerts = conn.execute("SELECT id,email,ticker,target_price,direction FROM user_alerts WHERE is_sent=0").fetchall()
            prices = {r["ticker"]: r["price"] for r in results}
            # An alert on a ticker outside the S&P 500 + Nasdaq-100 scan universe (a
            # watchlisted small-cap, an ETF, anything not in `results`) used to just
            # never fire -- silently, forever, with no error shown anywhere. Fetch those
            # directly instead of assuming every alerted ticker was in the batch.
            missing = list(dict.fromkeys(a["ticker"] for a in alerts if a["ticker"] not in prices))
            if missing:
                fetched = await asyncio.gather(*(get_current_price(t) for t in missing))
                prices.update(dict(zip(missing, fetched)))
            for alert in alerts:
                price = prices.get(alert["ticker"])
                if price is None:
                    continue
                direction = alert["direction"] or "above"
                hit = price >= alert["target_price"] if direction == "above" else price <= alert["target_price"]
                if hit:
                    verb = "risen to" if direction == "above" else "fallen to"
                    if await asyncio.to_thread(send_email_notification, alert["email"],
                        f"[QUANTIFY Alert] {alert['ticker']} hit your target price",
                        f"{alert['ticker']} has {verb} ${price:.2f}.\nYour target: ${alert['target_price']:.2f} or {direction}."):
                        conn.execute("UPDATE user_alerts SET is_sent=1 WHERE id=?", (alert["id"],))
            conn.commit(); conn.close()
            start_ai_prefetch(mode)
            start_market_summary_ai()
            await asyncio.to_thread(send_high_score_digest)
            return results
        except Exception as exc:
            print(f"[Error: {type(exc).__name__}] run_eod_batch_process fatal error: {exc}")
            BATCH_STATUS.update({"error": str(exc)})
            raise
        finally:
            BATCH_STATUS.update({"running": False, "finished_at": time.time()})
            gc.collect()

# -----------------------------------------------------------------------------
# Strategy backtest (real historical replay, no invented numbers)
# -----------------------------------------------------------------------------
def load_backtest_cache():
    if not BACKTEST_FILE.exists():
        return False
    try:
        payload = json.loads(BACKTEST_FILE.read_text(encoding="utf-8"))
        BACKTEST_CACHE.update(payload)
        return True
    except Exception as exc:
        print(f"[Error: {type(exc).__name__}] Backtest cache load error: {exc}")
        return False


def load_market_ai_summary_cache():
    if not MARKET_AI_SUMMARY_FILE.exists():
        return False
    try:
        payload = json.loads(MARKET_AI_SUMMARY_FILE.read_text(encoding="utf-8"))
        MARKET_AI_SUMMARY_CACHE.update(payload)
        return True
    except Exception as exc:
        print(f"[Error: {type(exc).__name__}] Market AI summary cache load error: {exc}")
        return False


def load_high_score_digest_state():
    if not HIGH_SCORE_DIGEST_FILE.exists():
        return False
    try:
        payload = json.loads(HIGH_SCORE_DIGEST_FILE.read_text(encoding="utf-8"))
        HIGH_SCORE_DIGEST_STATE.update(payload)
        return True
    except Exception as exc:
        print(f"[Error: {type(exc).__name__}] High score digest state load error: {exc}")
        return False


def _summarize_returns(returns, cost_pct: float = 0.0):
    """cost_pct is a round-trip cost subtracted from every observation before anything
    is averaged, so the win rate moves too -- a trade that made less than the spread was
    not a win. Passing 0 gives the raw gross numbers."""
    if not returns:
        return None
    net = [x - cost_pct for x in returns] if cost_pct else list(returns)
    wins = [x for x in net if x > 0]
    losses = [x for x in net if x <= 0]
    win_rate = len(wins) / len(net) * 100
    return {
        "avg_return_pct": round(sum(net) / len(net), 2),
        "win_rate_pct": round(win_rate, 1),
        "n": len(net),
        "avg_win_pct": round(sum(wins) / len(wins), 2) if wins else None,
        "avg_loss_pct": round(sum(losses) / len(losses), 2) if losses else None,
        "worst_pct": round(min(net), 2),
        "cost_pct_applied": cost_pct or 0.0,
    }


async def run_backtest():
    # Share BATCH_LOCK with the regular market scan so the two never run at the same
    # time — both do a large batch of downloads, and holding two full price-history
    # sets in memory at once is what triggered the Render OOM restarts earlier tonight.
    async with BATCH_LOCK:
        await _run_backtest_locked()


async def _run_backtest_locked():
    import random
    tickers = list(UNIVERSE)
    if not tickers:
        return
    # BACKTEST_SAMPLE_SIZE is set above the real universe size, so this covers every
    # ticker (no sampling) -- the seeded Random() only matters if BACKTEST_SAMPLE_SIZE
    # is ever lowered again, in which case it keeps whichever subset gets picked
    # deterministic instead of reshuffling on every redeploy.
    # NOTE: BACKTEST_SAMPLE_SIZE (200) is below the real universe (~518), so this DOES
    # sample -- a seeded random 200, stable across redeploys. An earlier comment here
    # claimed full coverage; that was left behind when the size was reverted from 600.
    sample = tickers if len(tickers) <= BACKTEST_SAMPLE_SIZE else random.Random(42).sample(tickers, BACKTEST_SAMPLE_SIZE)
    horizons = [30, 60, 90]
    forward_returns = {h: [] for h in horizons}
    in_sample = {h: [] for h in horizons}
    out_sample = {h: [] for h in horizons}
    bench_returns = {h: [] for h in horizons}
    # The S&P measured over the *same* windows the strategy actually traded, not over
    # every day in the period. This is the comparison that means something: a
    # pullback-in-uptrend rule fires after dips, and the market's own return following a
    # dip is not the market's average return, so the unconditional benchmark flatters
    # the strategy. Both are reported.
    bench_matched = {h: [] for h in horizons}
    signal_dates = []
    signal_count = 0
    for i_ticker, ticker in enumerate(sample):
        try:
            # cache=False: this loop touches every ticker in the universe exactly once
            # and never revisits it, so caching would just dump ~518 large DataFrames
            # into a cache meant for short-lived chart/scan lookups -- over the several
            # minutes this sequential loop takes, that's a plausible driver of the
            # gradual memory growth Render flagged after this went from 200 to full
            # universe tonight.
            df = await download_stock(ticker, "1d", cache=False)
            if df is None:
                continue
            close = normalize_series(df, "Close").dropna()
            if len(close) < 300:
                continue
            high = normalize_series(df, "High").reindex(close.index)
            scores = calculate_alpha_score_series(close, high)
            passed = (scores >= QUANT_PASS_THRESHOLD).to_numpy()
            n = len(close)
            split_idx = int(n * 0.7)
            # Only count a fresh crossing above the threshold as one signal, not every
            # consecutive day a stock stays above it — otherwise one long trending stock
            # dominates the sample with autocorrelated near-duplicate observations.
            prev = np.concatenate(([False], passed[:-1]))
            entries = passed & ~prev
            for i in range(70, n):
                if not entries[i]:
                    continue
                signal_count += 1
                entry = float(close.iloc[i])
                entry_date = close.index[i]
                for h in horizons:
                    if i + h < n and entry:
                        ret = float(close.iloc[i + h] / entry - 1) * 100
                        forward_returns[h].append(ret)
                        (in_sample[h] if i < split_idx else out_sample[h]).append(ret)
                        signal_dates.append((h, entry_date))
        except Exception as exc:
            print(f"[Error: {type(exc).__name__}] Backtest ticker error ({ticker}): {exc}")
            continue
        if (i_ticker + 1) % 100 == 0:
            gc.collect()
    try:
        spx = await download_stock("^GSPC", "1d", cache=False)
        spx_close = normalize_series(spx, "Close").dropna()
        n = len(spx_close)
        for i in range(0, n):
            entry = float(spx_close.iloc[i])
            for h in horizons:
                if i + h < n and entry:
                    bench_returns[h].append(float(spx_close.iloc[i + h] / entry - 1) * 100)

        # Same-window benchmark. searchsorted gives the S&P bar on or after the signal
        # date, which handles a ticker trading on a day the index series lacks.
        spx_index = spx_close.index
        for h, entry_date in signal_dates:
            try:
                pos = int(spx_index.searchsorted(entry_date))
            except Exception:
                continue
            if pos >= n or pos + h >= n:
                continue
            base = float(spx_close.iloc[pos])
            if base:
                bench_matched[h].append(float(spx_close.iloc[pos + h] / base - 1) * 100)
    except Exception as exc:
        print(f"[Error: {type(exc).__name__}] Backtest benchmark error: {exc}")

    cost = BACKTEST_ROUND_TRIP_COST_PCT
    results = {
        "schema_version": BACKTEST_SCHEMA_VERSION,
        "signal_count": signal_count,
        "tickers_sampled": len(sample),
        "horizons": {
            str(h): {
                # Headline numbers are net of cost -- the gross ones are kept alongside
                # so the difference is inspectable rather than hidden.
                "strategy": _summarize_returns(forward_returns[h], cost),
                "strategy_gross": _summarize_returns(forward_returns[h]),
                "benchmark": _summarize_returns(bench_returns[h]),
                "benchmark_matched": _summarize_returns(bench_matched[h]),
            }
            for h in horizons
        },
        "validation": {
            **{f"in_sample_{h}d": _summarize_returns(in_sample[h], cost) for h in horizons},
            **{f"out_of_sample_{h}d": _summarize_returns(out_sample[h], cost) for h in horizons},
        },
        "assumptions": {
            "round_trip_cost_pct": cost,
            "universe": "current S&P 500 + Nasdaq-100 constituents",
            "survivorship_bias": ("Present and not removable with this data. The universe is today's "
                                  "index membership, so companies removed from the index during the "
                                  "test period are absent. This biases the results upward by an unknown "
                                  "amount."),
            "overlapping_windows": ("Signals overlap, so the observation counts are not independent "
                                    "samples and the averages carry more uncertainty than n suggests."),
            "sampling": f"{len(sample)} of {len(tickers)} tickers, seeded so the subset is stable.",
        },
    }
    BACKTEST_CACHE.update({"computed_at": time.time(), "results": results, "error": None})
    try:
        BACKTEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = BACKTEST_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(BACKTEST_CACHE, ensure_ascii=False), encoding="utf-8")
        tmp.replace(BACKTEST_FILE)
    except Exception as exc:
        print(f"[Error: {type(exc).__name__}] Backtest cache save error: {exc}")
    gc.collect()


async def backtest_scheduler():
    loaded = load_backtest_cache()
    computed_at = BACKTEST_CACHE.get("computed_at")
    age_hr = round((time.time() - computed_at) / 3600, 1) if computed_at else None
    print(f"[backtest] cache file loaded={loaded} computed_at_age_hours={age_hr} "
          f"(file={BACKTEST_FILE}, exists={BACKTEST_FILE.exists()})", flush=True)
    while True:
        cached_results = BACKTEST_CACHE.get("results") or {}
        needs_refresh = (not BACKTEST_CACHE.get("computed_at")
                          or cached_results.get("schema_version") != BACKTEST_SCHEMA_VERSION
                          or time.time() - BACKTEST_CACHE["computed_at"] > BACKTEST_REFRESH_SECONDS)
        if not needs_refresh:
            await asyncio.sleep(3600)
            continue
        if not UNIVERSE:
            # Universe hasn't loaded yet (e.g. right after a fresh deploy with no
            # persistent disk) — retry soon instead of waiting a full hour.
            await asyncio.sleep(30)
            continue
        try:
            await run_backtest()
            n = BACKTEST_CACHE.get("results", {}).get("signal_count", 0)
            print(f"[backtest] Refreshed — {n} historical signals sampled", flush=True)
        except Exception as exc:
            BACKTEST_CACHE["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[Error: {type(exc).__name__}] Backtest scheduler error: {exc}", flush=True)
        await asyncio.sleep(3600)

# -----------------------------------------------------------------------------
# Automatic server cache warm-up
# -----------------------------------------------------------------------------
async def warm_server_cache():
    global WARMUP_TASK
    if CACHE_STATUS["warming"]:
        return

    CACHE_STATUS.update({
        "warming": True,
        "ready": False,
        "today_rows": 0,
        "started_at": time.time(),
        "finished_at": None,
        "error": None,
    })

    try:
        await refresh_universe()
        if not UNIVERSE:
            raise RuntimeError("Could not fetch S&P 500 / Nasdaq-100 constituents.")

        conn = db()
        count = conn.execute(
            "SELECT COUNT(*) FROM daily_scans WHERE scan_date=?",
            (today_str(),)
        ).fetchone()[0]
        conn.close()

        CACHE_STATUS["today_rows"] = int(count)

        if count == 0:
            print(f"[warmup] No data for today -> starting a {len(UNIVERSE)}-ticker market scan on the server", flush=True)
            await run_eod_batch_process()
        else:
            print(f"[warmup] Found {count} rows for today -> skipping market scan", flush=True)
            start_ai_prefetch("Long-Term Momentum Pullback")

        conn = db()
        count = conn.execute(
            "SELECT COUNT(*) FROM daily_scans WHERE scan_date=?",
            (today_str(),)
        ).fetchone()[0]
        conn.close()

        CACHE_STATUS["today_rows"] = int(count)
        CACHE_STATUS["ready"] = count > 0
        print(f"[warmup] Server cache ready: {count} rows", flush=True)

    except Exception as exc:
        CACHE_STATUS["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[Error: {type(exc).__name__}] Warmup error: {CACHE_STATUS['error']}", flush=True)
    finally:
        CACHE_STATUS["warming"] = False
        CACHE_STATUS["finished_at"] = time.time()


def start_server_warmup():
    global WARMUP_TASK
    if WARMUP_TASK and not WARMUP_TASK.done():
        return False
    WARMUP_TASK = asyncio.create_task(warm_server_cache())

    def _done(task):
        try:
            task.result()
        except Exception as exc:
            print(f"[Error: {type(exc).__name__}] Warmup task error: {exc}", flush=True)

    WARMUP_TASK.add_done_callback(_done)
    return True

# -----------------------------------------------------------------------------
# Background jobs
# -----------------------------------------------------------------------------
async def scheduler():
    await asyncio.sleep(2)
    while True:
        try:
            await refresh_universe()
        except Exception as exc:
            print(f"[Error: {type(exc).__name__}] Scheduler error: {exc}")
        await asyncio.sleep(3600)


async def sector_scheduler():
    _load_sector_cache()
    await asyncio.sleep(10)  # let the universe load first
    while True:
        try:
            if not UNIVERSE:
                await refresh_universe()
            oldest_ts = min((v.get("ts", 0) for v in SECTOR_CACHE.values()), default=0)
            stale = not SECTOR_CACHE or time.time() - oldest_ts > SECTOR_TTL
            missing = [t for t in UNIVERSE if t not in SECTOR_CACHE]
            if stale:
                await refresh_sector_data(UNIVERSE)
            elif missing:
                await refresh_sector_data(missing)
        except Exception as exc:
            print(f"[Error: {type(exc).__name__}] Sector scheduler error: {exc}", flush=True)
        await asyncio.sleep(3600)


def _next_scan_time_et(now_et: datetime) -> datetime:
    todays = [now_et.replace(hour=h, minute=m, second=0, microsecond=0) for h, m in SCAN_TIMES_ET]
    upcoming_today = [t for t in todays if t > now_et]
    if upcoming_today and now_et.weekday() < 5:
        return min(upcoming_today)
    day = now_et + timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    first_hour, first_minute = SCAN_TIMES_ET[0]
    return day.replace(hour=first_hour, minute=first_minute, second=0, microsecond=0)


async def market_scan_scheduler():
    while True:
        now_et = datetime.now(ZoneInfo("America/New_York"))
        target = _next_scan_time_et(now_et)
        wait = max(1.0, (target - now_et).total_seconds())
        print(f"[scheduler] Next scan at {target.strftime('%Y-%m-%d %H:%M %Z')} (in {wait/60:.0f}m).", flush=True)
        await asyncio.sleep(wait)
        for attempt in range(1, SCAN_MAX_ATTEMPTS_PER_SLOT + 1):
            if BATCH_LOCK.locked():
                print("[scheduler] Skipping scheduled market scan — a scan is already running.", flush=True)
                break
            try:
                print(f"[scheduler] Starting scheduled market scan (attempt {attempt}/{SCAN_MAX_ATTEMPTS_PER_SLOT}).", flush=True)
                results = await run_eod_batch_process()
                universe_size = len(UNIVERSE) or 1
                if len(results) >= universe_size * 0.5:
                    break
                print(f"[scheduler] Scan only covered {len(results)}/{universe_size} tickers — likely still "
                      f"rate-limited.", flush=True)
            except Exception as exc:
                print(f"[Error: {type(exc).__name__}] Scheduled market scan failed: {exc}", flush=True)
            if attempt < SCAN_MAX_ATTEMPTS_PER_SLOT:
                # A few short retries within this slot rather than hammering yfinance on
                # a fixed cadence -- if it's still rate-limited after these, just wait
                # for the next scheduled slot instead of retrying for hours.
                await asyncio.sleep(SCAN_RETRY_INTERVAL_SECONDS)


async def warm_market_indices():
    try:
        await asyncio.gather(download_stock("^GSPC", "1d"), download_stock("^NDX", "1d"))
    except Exception as exc:
        print(f"[Error: {type(exc).__name__}] Index warm-up failed: {exc}", flush=True)


async def index_warm_scheduler():
    while True:
        await warm_market_indices()
        await asyncio.sleep(480)


def _prune_timed_cache(cache_dict: dict, max_age_seconds: float) -> int:
    now = time.time()
    stale_keys = [k for k, v in cache_dict.items() if now - v.get("ts", 0) > max_age_seconds]
    for k in stale_keys:
        del cache_dict[k]
    return len(stale_keys)


def _prune_attempt_dict(attempt_dict: dict, window_seconds: float) -> int:
    now = time.time()
    stale_keys = [k for k, times in attempt_dict.items() if not any(now - t < window_seconds for t in times)]
    for k in stale_keys:
        del attempt_dict[k]
    return len(stale_keys)


async def cache_prune_scheduler():
    while True:
        await asyncio.sleep(300)
        try:
            removed_hist = _prune_timed_cache(CACHE["historical"], HISTORICAL_FALLBACK_RETENTION)
            removed_news = _prune_timed_cache(CACHE["news"], NEWS_TTL * 3)
            removed_earnings = _prune_timed_cache(CACHE["earnings"], SHORT_INTEREST_TTL * 2)
            removed_attempts = sum(
                _prune_attempt_dict(d, LOGIN_LOCKOUT_SECONDS)
                for d in (LOGIN_ATTEMPTS, RESET_ATTEMPTS, SEND_CODE_ATTEMPTS, SIGNUP_ATTEMPTS, CONTACT_ATTEMPTS)
            )
            if removed_hist or removed_news or removed_earnings or removed_attempts:
                print(f"[cache] Pruned stale entries — historical:{removed_hist} news:{removed_news} "
                      f"earnings:{removed_earnings} rate_limit_keys:{removed_attempts}", flush=True)
            if removed_hist:
                gc.collect()
        except Exception as exc:
            print(f"[Error: {type(exc).__name__}] Cache prune error: {exc}", flush=True)


def _trial_subscribe_link() -> str:
    return f"{SITE_URL}/subscription"


def _unsub_token_for(conn, email: str) -> str:
    """Stable per-user unsubscribe token, generated on first use."""
    row = conn.execute("SELECT unsub_token FROM users WHERE email=?", (email,)).fetchone()
    token = row["unsub_token"] if row else None
    if not token:
        token = secrets.token_urlsafe(24)
        conn.execute("UPDATE users SET unsub_token=? WHERE email=?", (token, email))
        conn.commit()
    return token


def _unsub_footer(token: str) -> str:
    return (f"\n\n---\nDon't want these? Unsubscribe: {SITE_URL}/unsubscribe?token={token}\n"
            f"(You'll still get account emails like password resets and price alerts you set.)")


def _unsub_headers(token: str) -> dict:
    # Lets Gmail/Apple Mail show a one-click unsubscribe control, which meaningfully
    # reduces the odds of being marked as spam instead.
    return {"List-Unsubscribe": f"<{SITE_URL}/unsubscribe?token={token}>",
            "List-Unsubscribe-Post": "List-Unsubscribe=One-Click"}


def send_welcome_email(email: str, token: str) -> bool:
    body = (
        "Thanks for signing up. One thing to try first: open today's scan and read the "
        "plain-English note under any FAVORABLE ticker. That note is the whole point of "
        "this thing. If it doesn't make sense to you, tell me and I'll fix how it's written.\n\n"
        f"Today's scan: {SITE_URL}/terminal"
        + _unsub_footer(token)
    )
    return send_email_notification(email, "One thing to try first", body,
                                   headers=_unsub_headers(token))


def _week1_picks(conn, days=7, limit_win=2, limit_loss=1):
    """Two recorded picks that went up and one that went down, from the public record.

    Returns (winners, losers) or (None, None) when the record is too young to show
    both sides -- the email's whole point is showing a miss, so it is skipped rather
    than sent one-sided.
    """
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    rows = conn.execute(
        "SELECT ticker,record_date,entry_price FROM track_record "
        "WHERE list_type='passed' AND record_date >= ?", (cutoff,)
    ).fetchall()
    if not rows:
        return None, None
    latest = conn.execute("SELECT MAX(scan_date) FROM daily_scans").fetchone()[0]
    prices = {r["ticker"]: r["price"] for r in conn.execute(
        "SELECT ticker,price FROM daily_scans WHERE scan_date=?", (latest,)
    ).fetchall()} if latest else {}
    scored = []
    for r in rows:
        cur = prices.get(r["ticker"])
        if not cur or not r["entry_price"]:
            continue
        scored.append({"ticker": r["ticker"], "date": r["record_date"], "entry": r["entry_price"],
                       "now": cur, "chg": round((cur - r["entry_price"]) / r["entry_price"] * 100, 2)})
    winners = sorted([x for x in scored if x["chg"] > 0], key=lambda x: -x["chg"])[:limit_win]
    losers = sorted([x for x in scored if x["chg"] < 0], key=lambda x: x["chg"])[:limit_loss]
    if len(winners) < limit_win or len(losers) < limit_loss:
        return None, None
    return winners, losers


def send_week1_email(email: str, token: str, winners, losers) -> bool:
    def line(x):
        return f"  {x['ticker']} — flagged {x['date']} at ${x['entry']}, now ${x['now']} ({x['chg']:+}%)"
    body = (
        "What the scanner caught this week:\n\n"
        + "\n".join(line(w) for w in winners)
        + "\n\nAnd one it missed:\n\n"
        + "\n".join(line(l) for l in losers)
        + f"\n\nEvery pick the scanner has ever made, winners and losers, is public here:\n"
        f"{SITE_URL}/record\n\n"
        "I send this because a screener that only shows you its wins isn't worth paying for."
        + _unsub_footer(token)
    )
    return send_email_notification(email, "What the scanner caught this week (and one it missed)",
                                   body, headers=_unsub_headers(token))


def send_trial_ending_email(email: str, token: str) -> bool:
    body = (
        "Your trial ends tomorrow. If you're keeping it, nothing to do, it just continues "
        "at $9.99/month. If you're not, I'd really like to know why. One line is enough. "
        "That answer is worth more to me than the $9.99.\n\n"
        f"Manage your subscription: {_trial_subscribe_link()}"
        + _unsub_footer(token)
    )
    return send_email_notification(email, "Your trial ends tomorrow", body,
                                   headers=_unsub_headers(token))


def send_trial_ended_email(email: str) -> bool:
    body = (
        f"Your {TRIAL_DAYS}-day QUANTIFY free trial has ended.\n\n"
        f"Subscribe to keep using the scanner and AI reports: {_trial_subscribe_link()}\n\n"
        f"Questions? Just reply to this email."
    )
    return send_email_notification(email, "[QUANTIFY.] Your free trial has ended", body)


async def trial_lifecycle_scheduler():
    while True:
        await asyncio.sleep(3600)
        try:
            now = time.time()
            conn = db()
            # --- Trial onboarding drip. Each email has its own *_sent_at column, so a
            # send can never repeat, and all three respect pref_marketing_emails.
            sent_counts = {"welcome": 0, "week1": 0, "ending": 0, "week1_skipped": 0}

            welcome = conn.execute(
                "SELECT email FROM users WHERE is_active=1 AND welcome_sent_at IS NULL "
                "AND pref_marketing_emails=1"
            ).fetchall()
            for row in welcome:
                try:
                    token = _unsub_token_for(conn, row["email"])
                    if await asyncio.to_thread(send_welcome_email, row["email"], token):
                        conn.execute("UPDATE users SET welcome_sent_at=? WHERE email=?", (now, row["email"]))
                        conn.commit()
                        sent_counts["welcome"] += 1
                except Exception as exc:
                    print(f"[Error: {type(exc).__name__}] Welcome email failed for {row['email']}: {exc}", flush=True)

            week1 = conn.execute(
                "SELECT email FROM users WHERE is_active=1 AND week1_sent_at IS NULL "
                "AND pref_marketing_emails=1 AND created_at IS NOT NULL AND created_at <= ?",
                (now - 3 * 86400,),
            ).fetchall()
            if week1:
                winners, losers = _week1_picks(conn)
                if not winners:
                    # No recorded loser to show yet -- this email exists to show a miss,
                    # so it waits rather than going out one-sided.
                    sent_counts["week1_skipped"] = len(week1)
                else:
                    for row in week1:
                        try:
                            token = _unsub_token_for(conn, row["email"])
                            if await asyncio.to_thread(send_week1_email, row["email"], token, winners, losers):
                                conn.execute("UPDATE users SET week1_sent_at=? WHERE email=?", (now, row["email"]))
                                conn.commit()
                                sent_counts["week1"] += 1
                        except Exception as exc:
                            print(f"[Error: {type(exc).__name__}] Week-1 email failed for {row['email']}: {exc}", flush=True)

            ending = conn.execute(
                "SELECT email FROM users WHERE subscription_status='trial' "
                "AND trial_end_sent_at IS NULL AND pref_marketing_emails=1 "
                "AND trial_ends_at IS NOT NULL AND trial_ends_at > ? AND trial_ends_at <= ?",
                (now, now + 24 * 3600),
            ).fetchall()
            for row in ending:
                try:
                    token = _unsub_token_for(conn, row["email"])
                    if await asyncio.to_thread(send_trial_ending_email, row["email"], token):
                        conn.execute("UPDATE users SET trial_end_sent_at=? WHERE email=?", (now, row["email"]))
                        conn.commit()
                        sent_counts["ending"] += 1
                except Exception as exc:
                    print(f"[Error: {type(exc).__name__}] Trial-ending email failed for {row['email']}: {exc}", flush=True)

            ended = conn.execute(
                "SELECT email FROM users WHERE subscription_status='trial' "
                "AND trial_ended_email_sent_at IS NULL AND trial_ends_at IS NOT NULL AND trial_ends_at <= ?",
                (now,),
            ).fetchall()
            for row in ended:
                try:
                    if await asyncio.to_thread(send_trial_ended_email, row["email"]):
                        conn.execute("UPDATE users SET trial_ended_email_sent_at=? WHERE email=?", (now, row["email"]))
                        conn.commit()
                except Exception as exc:
                    print(f"[Error: {type(exc).__name__}] Trial-ended email failed for {row['email']}: {exc}", flush=True)
            conn.close()
            if any(sent_counts.values()) or ended:
                print(f"[trial] drip sent: {sent_counts['welcome']} welcome, "
                      f"{sent_counts['week1']} week-1, {sent_counts['ending']} trial-ending"
                      + (f" ({sent_counts['week1_skipped']} week-1 deferred: record has no loser yet)"
                         if sent_counts["week1_skipped"] else "")
                      + f"; {len(ended)} trial-ended", flush=True)
        except Exception as exc:
            print(f"[Error: {type(exc).__name__}] Trial lifecycle scheduler error: {exc}", flush=True)


def _gumroad_api_get(path: str, params: dict) -> Optional[dict]:
    if not GUMROAD_ACCESS_TOKEN:
        return None
    try:
        resp = requests.get(f"https://api.gumroad.com/v2{path}",
                             params={**params, "access_token": GUMROAD_ACCESS_TOKEN}, timeout=10)
        data = resp.json()
        if resp.status_code == 200 and data.get("success"):
            return data
        print(f"[gumroad] API call {path} failed ({resp.status_code}): {data}", flush=True)
        return None
    except Exception as exc:
        print(f"[Error: {type(exc).__name__}] Gumroad API call {path} failed: {exc}", flush=True)
        return None


async def gumroad_reconcile_scheduler():
    # Gumroad's Ping webhook has no "subscription cancelled" event at all (only
    # sale/refund) — a user who cancels future billing without a refund never
    # generates a ping, so we'd otherwise keep their access active forever.
    # Gumroad's own docs recommend reconciling periodically via the API instead
    # of treating pings as authoritative state.
    while True:
        await asyncio.sleep(24 * 3600)
        if not GUMROAD_ACCESS_TOKEN:
            continue
        try:
            conn = db()
            rows = conn.execute(
                "SELECT email,gumroad_subscription_id FROM users "
                "WHERE subscription_status='active' AND gumroad_subscription_id IS NOT NULL AND gumroad_subscription_id<>''"
            ).fetchall()
            downgraded = 0
            for row in rows:
                data = await asyncio.to_thread(_gumroad_api_get, f"/subscribers/{row['gumroad_subscription_id']}", {})
                sub = (data or {}).get("subscribers") or {}
                if sub and (sub.get("status") != "alive" or sub.get("ended_at")):
                    conn.execute("UPDATE users SET subscription_status='expired' WHERE email=?", (row["email"],))
                    conn.commit()
                    downgraded += 1
            conn.close()
            if downgraded:
                print(f"[gumroad] Reconciliation downgraded {downgraded} account(s) with an ended subscription", flush=True)
        except Exception as exc:
            print(f"[Error: {type(exc).__name__}] Gumroad reconcile scheduler error: {exc}", flush=True)


@app.on_event("startup")
async def startup():
    init_db()
    load_universe_cache()
    load_market_ai_summary_cache()
    load_high_score_digest_state()
    asyncio.create_task(refresh_universe())
    asyncio.create_task(scheduler())
    asyncio.create_task(sector_scheduler())
    asyncio.create_task(market_scan_scheduler())
    asyncio.create_task(index_warm_scheduler())
    asyncio.create_task(cache_prune_scheduler())
    # run_backtest() now shares BATCH_LOCK with the regular scan (see run_backtest),
    # so this can no longer run concurrently with it and double up on memory — that
    # was the actual cause of tonight's OOM restarts, not the scheduler itself.
    asyncio.create_task(backtest_scheduler())
    asyncio.create_task(trial_lifecycle_scheduler())
    asyncio.create_task(gumroad_reconcile_scheduler())
    asyncio.create_task(fundamentals_scheduler())
    asyncio.create_task(asyncio.to_thread(check_email_config))
    asyncio.get_running_loop().call_later(3, start_server_warmup)


def _send_via_brevo(to_email, subject, body, max_retries=3, reply_to=None, headers=None):
    payload = {
        "sender": {"email": SENDER_EMAIL, "name": SENDER_NAME},
        "to": [{"email": to_email}],
        "subject": subject,
        "textContent": body,
    }
    if reply_to:
        payload["replyTo"] = {"email": reply_to}
    if headers:
        payload["headers"] = headers
    headers = {"api-key": BREVO_API_KEY, "Content-Type": "application/json", "Accept": "application/json"}
    for attempt in range(max_retries):
        try:
            resp = requests.post("https://api.brevo.com/v3/smtp/email", json=payload, headers=headers, timeout=10)
            if resp.status_code in (200, 201, 202):
                return True
            if resp.status_code in (401, 403):
                print(f"[Error: BrevoAuthError] Brevo rejected the request ({resp.status_code}): {resp.text[:300]}")
                return False
            print(f"[Error: BrevoError] Unexpected response {resp.status_code}: {resp.text[:300]}")
        except Exception as exc:
            print(f"[Retry {attempt + 1}/{max_retries}] Brevo send failed ({type(exc).__name__}): {exc}")
        if attempt < max_retries - 1:
            time.sleep(2.0 * (attempt + 1))
    print(f"[Error: BrevoError] Email send error after {max_retries} attempts (to {to_email})")
    return False


def _send_via_sendgrid(to_email, subject, body, max_retries=3, reply_to=None, headers=None):
    payload = {
        "personalizations": [{"to": [{"email": to_email}]}],
        "from": {"email": SENDER_EMAIL, "name": SENDER_NAME},
        "subject": subject,
        "content": [{"type": "text/plain", "value": body}],
    }
    if reply_to:
        payload["reply_to"] = {"email": reply_to}
    if headers:
        payload["headers"] = headers
    headers = {"Authorization": f"Bearer {SENDGRID_API_KEY}", "Content-Type": "application/json"}
    for attempt in range(max_retries):
        try:
            resp = requests.post("https://api.sendgrid.com/v3/mail/send", json=payload, headers=headers, timeout=10)
            if resp.status_code in (200, 201, 202):
                return True
            if resp.status_code in (401, 403):
                print(f"[Error: SendGridAuthError] SendGrid rejected the request ({resp.status_code}): {resp.text[:300]}")
                return False
            print(f"[Error: SendGridError] Unexpected response {resp.status_code}: {resp.text[:300]}")
        except Exception as exc:
            print(f"[Retry {attempt + 1}/{max_retries}] SendGrid send failed ({type(exc).__name__}): {exc}")
        if attempt < max_retries - 1:
            time.sleep(2.0 * (attempt + 1))
    print(f"[Error: SendGridError] Email send error after {max_retries} attempts (to {to_email})")
    return False


def _send_via_smtp(to_email, subject, body, max_retries=3, reply_to=None, headers=None):
    msg = MIMEMultipart(); msg["From"] = f"{SENDER_NAME} <{SENDER_EMAIL}>"; msg["To"] = to_email; msg["Subject"] = subject
    if reply_to:
        msg["Reply-To"] = reply_to
    for k, v in (headers or {}).items():
        msg[k] = v
    msg.attach(MIMEText(body, "plain", "utf-8"))
    for attempt in range(max_retries):
        try:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as server:
                server.starttls(); server.login(SENDER_EMAIL, SENDER_PASSWORD); server.sendmail(SENDER_EMAIL, to_email, msg.as_string())
            return True
        except smtplib.SMTPAuthenticationError as exc:
            print(f"[Error: SMTPAuthenticationError] SMTP login rejected — check SENDER_EMAIL/SENDER_PASSWORD (Gmail needs an App Password): {exc}")
            return False
        except Exception as exc:
            if attempt < max_retries - 1:
                wait = 2.0 * (attempt + 1)
                print(f"[Retry {attempt + 1}/{max_retries}] Email send failed ({type(exc).__name__}): {exc} — retrying in {wait:.0f}s")
                time.sleep(wait)
                continue
            print(f"[Error: {type(exc).__name__}] Email send error after {max_retries} attempts (to {to_email}): {exc}")
            return False
    return False


def send_email_notification(to_email, subject, body, max_retries=3, reply_to=None, headers=None):
    if BREVO_API_KEY and SENDER_EMAIL:
        return _send_via_brevo(to_email, subject, body, max_retries, reply_to=reply_to, headers=headers)
    if SENDGRID_API_KEY and SENDER_EMAIL:
        return _send_via_sendgrid(to_email, subject, body, max_retries, reply_to=reply_to, headers=headers)
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print(f"[Error: EmailNotConfigured] No email backend configured — could not send '{subject}' to {to_email}")
        return False
    return _send_via_smtp(to_email, subject, body, max_retries, reply_to=reply_to, headers=headers)


def check_email_config():
    if BREVO_API_KEY and SENDER_EMAIL:
        try:
            resp = requests.get("https://api.brevo.com/v3/senders", headers={"api-key": BREVO_API_KEY, "Accept": "application/json"}, timeout=10)
            if resp.status_code == 200:
                senders = [s.get("email") for s in resp.json().get("senders", [])]
                if SENDER_EMAIL in senders:
                    print(f"[email] Brevo configured — sending as verified sender {SENDER_EMAIL}.")
                else:
                    print(f"[email] Brevo API key is valid but {SENDER_EMAIL} is not a verified sender yet — emails will be rejected until you verify it in Brevo.")
            else:
                print(f"[email] Brevo API key check failed ({resp.status_code}): {resp.text[:200]}")
        except Exception as exc:
            print(f"[email] Brevo connectivity check failed ({type(exc).__name__}): {exc}")
        return
    if SENDGRID_API_KEY and SENDER_EMAIL:
        try:
            resp = requests.get("https://api.sendgrid.com/v3/verified_senders", headers={"Authorization": f"Bearer {SENDGRID_API_KEY}"}, timeout=10)
            if resp.status_code == 200:
                senders = [s.get("from_email") for s in resp.json().get("results", [])]
                if SENDER_EMAIL in senders:
                    print(f"[email] SendGrid configured — sending as verified sender {SENDER_EMAIL}.")
                else:
                    print(f"[email] SendGrid API key is valid but {SENDER_EMAIL} is not a verified sender yet — emails will be rejected until you verify it in SendGrid.")
            else:
                print(f"[email] SendGrid API key check failed ({resp.status_code}): {resp.text[:200]}")
        except Exception as exc:
            print(f"[email] SendGrid connectivity check failed ({type(exc).__name__}): {exc}")
        return
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("[email] No email backend configured (BREVO_API_KEY, SENDGRID_API_KEY, or SENDER_EMAIL/SENDER_PASSWORD) — verification, password reset, and alert emails will not be sent.")
        return
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
        print(f"[email] SMTP login OK for {SENDER_EMAIL} — outgoing email is configured correctly.")
    except smtplib.SMTPAuthenticationError as exc:
        print(f"[email] SMTP login FAILED for {SENDER_EMAIL} — emails will not send. Check SENDER_PASSWORD (Gmail requires an App Password, not your normal password): {exc}")
    except Exception as exc:
        print(f"[email] SMTP connectivity check failed ({type(exc).__name__}): {exc}")

# -----------------------------------------------------------------------------
# Server-side AI prefetch/cache
# -----------------------------------------------------------------------------
async def run_ai_prefetch(mode="Long-Term Momentum Pullback"):
    global AI_STATUS
    AI_STATUS.update({"running": True, "processed": 0, "total": 0, "ready": 0,
                      "started_at": time.time(), "finished_at": None, "error": None})
    conn=db()
    rows=conn.execute("""
        SELECT ticker,price,change_pct,rsi,macd,pct_from_52w_high,pct_from_52w_low,above_200d_sma,volume_ratio,alpha_score
        FROM daily_scans
        WHERE scan_date=? AND quant_pass=1
          AND (
            ai_report IS NULL OR ai_mode<>? OR ai_prompt_version IS NULL OR ai_prompt_version<>?
            OR (ai_scored_alpha_score IS NOT NULL AND ABS(alpha_score - ai_scored_alpha_score) >= ?)
          )
        ORDER BY alpha_score DESC
    """,(today_str(),mode,AI_PROMPT_VERSION,AI_RESCORE_ALPHA_DRIFT)).fetchall()
    conn.close()
    AI_STATUS["total"]=len(rows)
    if not rows:
        AI_STATUS.update({"running": False, "processed": 0, "ready": 0,
                          "finished_at": time.time()})
        return
    sem=asyncio.Semaphore(AI_CONCURRENCY)

    async def one(row):
        async with sem:
            ticker=row["ticker"]
            try:
                conn=db()
                conn.execute("UPDATE daily_scans SET ai_status='RUNNING',ai_mode=?,ai_error=NULL WHERE scan_date=? AND ticker=?",
                             (mode,today_str(),ticker))
                conn.commit(); conn.close()

                above_trend = bool(row["above_200d_sma"]) if row["above_200d_sma"] is not None else None
                news = await fetch_stock_news(ticker)
                ai_result=await generate_ai_report(
                    ticker,row["price"],row["change_pct"],mode,row["rsi"],row["macd"],
                    row["pct_from_52w_high"],row["pct_from_52w_low"],above_trend,"en",
                    row["volume_ratio"],news,
                )

                conn=db()
                if ai_result:
                    conn.execute("""UPDATE daily_scans
                                    SET ai_report=?,ai_status='READY',
                                        ai_mode=?,ai_updated_at=?,ai_error=NULL,
                                        timing_score=?,timing_verdict=?,ai_prompt_version=?,
                                        ai_scored_alpha_score=?
                                    WHERE scan_date=? AND ticker=?""",
                                 (ai_result["report_json"],mode,time.time(),
                                  ai_result["timing_score"],ai_result["timing_verdict"],AI_PROMPT_VERSION,
                                  row["alpha_score"],
                                  today_str(),ticker))
                    AI_STATUS["ready"]+=1
                else:
                    conn.execute("""UPDATE daily_scans
                                    SET ai_status='ERROR',ai_mode=?,
                                        ai_updated_at=?,ai_error=?
                                    WHERE scan_date=? AND ticker=?""",
                                 (mode,time.time(),"AI returned no report",today_str(),ticker))
                conn.commit(); conn.close()
            except Exception as exc:
                msg=f"{type(exc).__name__}: {exc}"
                print(f"[Error: {type(exc).__name__}] AI prefetch worker error ({ticker}): {msg}",flush=True)
                try:
                    conn=db()
                    conn.execute("""UPDATE daily_scans SET ai_status='ERROR',ai_mode=?,
                                    ai_updated_at=?,ai_error=? WHERE scan_date=? AND ticker=?""",
                                 (mode,time.time(),msg,today_str(),ticker))
                    conn.commit(); conn.close()
                except Exception:
                    pass
            finally:
                AI_STATUS["processed"]+=1

    try:
        await asyncio.gather(*(one(row) for row in rows))
    except Exception as exc:
        AI_STATUS["error"]=f"{type(exc).__name__}: {exc}"
        print(f"[Error: {type(exc).__name__}] AI prefetch gather error: {AI_STATUS['error']}",flush=True)
    finally:
        AI_STATUS["running"]=False
        AI_STATUS["finished_at"]=time.time()

def start_ai_prefetch(mode="Long-Term Momentum Pullback"):
    global AI_TASK
    if AI_STATUS.get("running") or (AI_TASK and not AI_TASK.done()):
        return False
    AI_TASK=asyncio.create_task(run_ai_prefetch(mode))
    def done(task):
        try:
            task.result()
        except Exception as exc:
            print(f"[Error: {type(exc).__name__}] AI prefetch task error: {exc}",flush=True)
    AI_TASK.add_done_callback(done)
    return True


def _index_day_change(df):
    try:
        close = normalize_series(df, "Close").dropna()
        return round((float(close.iloc[-1]) / float(close.iloc[-2]) - 1) * 100, 2)
    except Exception:
        return None


async def generate_market_summary_ai():
    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.hour < MARKET_SUMMARY_AFTER_HOUR_ET:
        return
    if MARKET_AI_SUMMARY_CACHE.get("scan_date") == today_str():
        return
    conn = db()
    rows = conn.execute("SELECT universe,change_pct,timing_verdict,quant_pass FROM daily_scans WHERE scan_date=?", (today_str(),)).fetchall()
    top_rows = conn.execute(
        "SELECT ticker,change_pct,alpha_score,timing_verdict FROM daily_scans "
        "WHERE scan_date=? AND quant_pass=1 ORDER BY alpha_score DESC LIMIT 5",
        (today_str(),),
    ).fetchall()
    conn.close()
    total = len(rows)
    if not total:
        return
    advancers = sum(1 for r in rows if (r["change_pct"] or 0) > 0)
    decliners = sum(1 for r in rows if (r["change_pct"] or 0) < 0)
    avg_change = round(sum(r["change_pct"] or 0 for r in rows) / total, 2)
    detected = sum(1 for r in rows if r["quant_pass"])
    verdict_counts = {"Favorable": 0, "Caution": 0, "Risk": 0}
    for r in rows:
        if r["timing_verdict"] in verdict_counts:
            verdict_counts[r["timing_verdict"]] += 1
    sp500_df, ndx_df = await asyncio.gather(download_stock("^GSPC", "1d"), download_stock("^NDX", "1d"))
    stats = {
        "universe_count": total, "advancers": advancers, "decliners": decliners, "avg_change_pct": avg_change,
        "detected_count": detected, "verdict_breakdown": verdict_counts,
        "sp500_change_pct": _index_day_change(sp500_df) if sp500_df is not None else None,
        "nasdaq_change_pct": _index_day_change(ndx_df) if ndx_df is not None else None,
        "top_tickers": [dict(r) for r in top_rows],
    }
    result = await asyncio.to_thread(market_summary_ai_sync, stats)
    if result:
        MARKET_AI_SUMMARY_CACHE.update({"scan_date": today_str(), "generated_at": time.time(),
                                         "headline": result["headline"], "summary": result["summary"], "error": None})
    else:
        MARKET_AI_SUMMARY_CACHE["error"] = "Market AI summary generation failed"
    try:
        MARKET_AI_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = MARKET_AI_SUMMARY_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(MARKET_AI_SUMMARY_CACHE, ensure_ascii=False), encoding="utf-8")
        tmp.replace(MARKET_AI_SUMMARY_FILE)
    except Exception as exc:
        print(f"[Error: {type(exc).__name__}] Market AI summary cache write failed: {exc}")


def start_market_summary_ai():
    task = asyncio.create_task(generate_market_summary_ai())
    def done(task):
        try:
            task.result()
        except Exception as exc:
            print(f"[Error: {type(exc).__name__}] Market AI summary task error: {exc}", flush=True)
    task.add_done_callback(done)


def send_high_score_digest():
    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.hour < MARKET_SUMMARY_AFTER_HOUR_ET:
        return
    if HIGH_SCORE_DIGEST_STATE.get("sent_date") == today_str():
        return

    def mark_sent():
        HIGH_SCORE_DIGEST_STATE["sent_date"] = today_str()
        try:
            HIGH_SCORE_DIGEST_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = HIGH_SCORE_DIGEST_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(HIGH_SCORE_DIGEST_STATE, ensure_ascii=False), encoding="utf-8")
            tmp.replace(HIGH_SCORE_DIGEST_FILE)
        except Exception as exc:
            print(f"[Error: {type(exc).__name__}] High score digest state write failed: {exc}")

    try:
        conn = db()
        rows = conn.execute(
            "SELECT ticker,alpha_score,timing_score,timing_verdict,price,change_pct FROM daily_scans "
            "WHERE scan_date=? AND quant_pass=1 AND timing_score IS NOT NULL",
            (today_str(),),
        ).fetchall()
        hits = [dict(r) for r in rows if (r["alpha_score"] + r["timing_score"]) / 2 >= HIGH_SCORE_ALERT_THRESHOLD]
        if not hits:
            conn.close()
            mark_sent()
            return
        hits.sort(key=lambda h: (h["alpha_score"] + h["timing_score"]) / 2, reverse=True)
        recipients = [r["email"] for r in conn.execute(
            "SELECT email FROM users WHERE pref_high_score_alerts=1"
        ).fetchall()]
        conn.close()
        recipients = [e for e in recipients if has_active_access(e)]
        if not recipients:
            mark_sent()
            return
        lines = [
            f"{h['ticker']}: score {round((h['alpha_score'] + h['timing_score']) / 2, 1)}/100, "
            f"{h['timing_verdict'] or '-'}, ${h['price']:.2f} ({h['change_pct']:+.2f}%)"
            for h in hits
        ]
        plural = "s" if len(hits) != 1 else ""
        subject = f"[QUANTIFY] {len(hits)} stock{plural} scored {HIGH_SCORE_ALERT_THRESHOLD}+ today"
        body = (
            f"{len(hits)} stock{plural} reached a score of {HIGH_SCORE_ALERT_THRESHOLD}+ in today's scan ({display_date()}):\n\n"
            + "\n".join(lines)
            + f"\n\nSee the full scan and AI review at {SITE_URL}/terminal\n\n"
            "QUANTIFY is an informational and educational tool. This is not investment advice, "
            "a recommendation, or a solicitation to buy or sell any security. Scores reflect a "
            "model's output on the data available and can be wrong. Do your own research."
        )
        for email in recipients:
            send_email_notification(email, subject, body)
        mark_sent()
    except Exception as exc:
        print(f"[Error: {type(exc).__name__}] High score digest error: {exc}", flush=True)


# -----------------------------------------------------------------------------
# API endpoints
# -----------------------------------------------------------------------------
@app.get("/api/public-preview")
async def public_preview():
    # Unauthenticated, intentionally limited (top 3 only) -- the landing page's
    # terminal mockup used to be static/fake copy. Real, live proof of what the
    # product actually outputs is a stronger trust signal than any amount of
    # marketing copy, and it's honest since it's the same data a subscriber sees.
    conn = db()
    # Same rule as the scanner: a day that is still being scored would otherwise show
    # here as a handful of tickers, and this is the page a first-time visitor lands on.
    latest_date = display_scan_date(conn)
    if not latest_date:
        conn.close()
        return {"tickers": [], "scan_date": None, "detected_count": 0, "universe_count": len(UNIVERSE)}
    detected_count = conn.execute(
        "SELECT COUNT(*) FROM daily_scans WHERE scan_date=? AND quant_pass=1 AND timing_score IS NOT NULL",
        (latest_date,),
    ).fetchone()[0]
    rows = conn.execute("""
        SELECT ticker,price,change_pct,rsi,macd,pct_from_52w_high,pct_from_52w_low,timing_verdict,ai_report,
               ROUND((alpha_score+timing_score)/2.0,1) AS overall_score
        FROM daily_scans WHERE scan_date=? AND quant_pass=1 AND timing_score IS NOT NULL
        ORDER BY overall_score DESC LIMIT 3
    """, (latest_date,)).fetchall()
    conn.close()
    tickers = []
    for r in rows:
        d = dict(r)
        ai = {}
        if d.get("ai_report"):
            try:
                ai = json.loads(d["ai_report"])
            except (json.JSONDecodeError, TypeError):
                ai = {}
        cached = CACHE["historical"].get(f"single:{d['ticker']}:1d")
        sparkline = []
        if cached is not None:
            try:
                closes = normalize_series(cached["data"], "Close").dropna().tail(20)
                sparkline = [round(float(v), 2) for v in closes]
            except Exception:
                sparkline = []
        tickers.append({
            "ticker": d["ticker"], "price": d["price"], "change_pct": d["change_pct"],
            "rsi": d["rsi"], "macd": d["macd"],
            "pct_from_52w_high": d["pct_from_52w_high"], "pct_from_52w_low": d["pct_from_52w_low"],
            "timing_verdict": d["timing_verdict"], "overall_score": d["overall_score"],
            "quant_review": ai.get("quant_review"), "risk_review": ai.get("risk_review"),
            "sparkline": sparkline,
        })
    return {"tickers": tickers, "scan_date": latest_date, "detected_count": detected_count,
            "universe_count": len(UNIVERSE)}


# A scan writes its price rows in about a minute, but the AI review that gives each
# ticker a timing score lands one ticker at a time over the following several minutes.
# The scanner only lists tickers that have that score, so reading today unconditionally
# meant the list emptied out and refilled in front of whoever was looking. Most visibly
# after a restart, which re-runs the scan outside the normal schedule -- but it happened
# every morning at 09:00 too.
SCAN_READY_RATIO = 0.9


def display_scan_date(conn) -> Optional[str]:
    """The most recent scan day complete enough to show.

    Showing yesterday's finished list until today's is scored is both more useful and
    more honest than showing a fifth of today's: it is a whole picture of a stated date
    rather than a partial one of this one.
    """
    rows = conn.execute("""
        SELECT scan_date,
               SUM(CASE WHEN quant_pass=1 THEN 1 ELSE 0 END) AS passed,
               SUM(CASE WHEN quant_pass=1 AND timing_score IS NOT NULL THEN 1 ELSE 0 END) AS scored
        FROM daily_scans GROUP BY scan_date ORDER BY scan_date DESC LIMIT 5
    """).fetchall()
    partial = None
    for r in rows:
        passed, scored = r["passed"] or 0, r["scored"] or 0
        if passed and scored >= passed * SCAN_READY_RATIO:
            return r["scan_date"]
        if scored and partial is None:
            partial = r["scan_date"]
    # Nothing is complete yet (a genuinely first run): a partial day beats nothing.
    return partial or (rows[0]["scan_date"] if rows else None)


@app.get("/api/scan")
async def api_scan(request: Request):
    user = get_logged_in_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not disclaimer_accepted(user):
        return JSONResponse({"error": "Please accept the disclaimer before using QUANTIFY."}, status_code=403)
    if not has_active_access(user):
        return JSONResponse({"error": "Your free trial has ended. Subscribe to keep scanning."}, status_code=402)
    conn = db()
    # Every figure below describes the same day, so the count in the header can never
    # belong to a different scan than the list under it.
    scan_date = display_scan_date(conn) or today_str()
    scanned_count = conn.execute(
        "SELECT COUNT(*) FROM daily_scans WHERE scan_date=?", (scan_date,)
    ).fetchone()[0]
    last_updated = conn.execute(
        "SELECT MAX(created_at) FROM daily_scans WHERE scan_date=?", (scan_date,)
    ).fetchone()[0]
    rows = conn.execute("""
        SELECT ticker,universe,price,change_pct,alpha_score,quant_pass,timing_score,timing_verdict,
               ROUND((alpha_score+timing_score)/2.0,1) AS overall_score
        FROM daily_scans
        WHERE scan_date=? AND quant_pass=1 AND timing_score IS NOT NULL
          AND (alpha_score+timing_score)/2.0 >= ?
        ORDER BY overall_score DESC
    """, (scan_date, OVERALL_SCORE_THRESHOLD)).fetchall()
    conn.close()
    signals = []
    for r in rows:
        d = dict(r) | {"status": "ACTIVE DATA"}
        cached = CACHE["historical"].get(f"single:{d['ticker']}:1d")
        sparkline = []
        if cached is not None:
            try:
                closes = normalize_series(cached["data"], "Close").dropna().tail(15)
                sparkline = [round(float(v), 2) for v in closes]
            except Exception:
                sparkline = []
        d["sparkline"] = sparkline
        d["sector"] = (SECTOR_CACHE.get(d["ticker"]) or {}).get("sector")
        signals.append(d)
    return {"scanned_count": scanned_count, "universe_count": len(UNIVERSE),
            "quant_pass_count": len(rows), "last_updated": last_updated,
            "scan_date": scan_date,
            # True while a newer scan exists but is still being AI-scored. The list below
            # is the last finished day, and the page says so instead of implying the
            # numbers are from today.
            "showing_previous_day": scan_date != today_str(),
            "universe_status": UNIVERSE_STATUS, "cache": dict(CACHE_STATUS),
            "signals": signals}


@app.get("/api/heatmap")
async def api_heatmap(request: Request):
    user = get_logged_in_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not has_active_access(user):
        return JSONResponse({"error": "Your free trial has ended. Subscribe to keep scanning."}, status_code=402)
    conn = db()
    rows = conn.execute("""
        SELECT ticker,universe,change_pct,alpha_score,quant_pass,timing_verdict FROM daily_scans
        WHERE scan_date=? ORDER BY ticker
    """, (today_str(),)).fetchall()
    conn.close()
    tiles = []
    for r in rows:
        d = dict(r)
        info = SECTOR_CACHE.get(d["ticker"]) or {}
        d["sector"] = info.get("sector")
        cap = info.get("market_cap")
        # Rough visual size tier (mega/large/mid+), not a precise market-cap definition --
        # just enough to make the heatmap read as size-weighted rather than a flat grid.
        d["cap_tier"] = 1 if cap and cap >= 200e9 else 2 if cap and cap >= 10e9 else 3
        tiles.append(d)
    return {"tiles": tiles}


@app.get("/api/score-history")
async def api_score_history(request: Request, ticker: str = "AAPL"):
    user = get_logged_in_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not has_active_access(user):
        return JSONResponse({"error": "Your free trial has ended. Subscribe to keep scanning."}, status_code=402)
    ticker = normalize_ticker(ticker)
    if not re.fullmatch(r"[A-Z0-9.\-^=]{1,15}", ticker):
        return JSONResponse({"error": "Invalid ticker"}, status_code=400)
    conn = db()
    rows = conn.execute("""
        SELECT captured_at,price,alpha_score FROM scan_history
        WHERE ticker=? AND scan_date=? ORDER BY captured_at
    """, (ticker, today_str())).fetchall()
    conn.close()
    return {"ticker": ticker, "points": [dict(r) for r in rows]}


@app.get("/api/universe")
async def api_universe(request: Request, refresh: bool = False):
    if not get_logged_in_user(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if refresh:
        await refresh_universe(force=True)
    return {"status": UNIVERSE_STATUS, "counts": {"sp500": len(UNIVERSE_META["sp500"]),
            "nasdaq100": len(UNIVERSE_META["nasdaq100"]), "combined": len(UNIVERSE)}}


@app.get("/api/market-indices")
async def api_market_indices(request: Request):
    if not get_logged_in_user(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    async def series_for(symbol):
        df = await download_stock(symbol, "1d")
        if df is None or df.empty:
            return None
        close = normalize_series(df, "Close").dropna().tail(60)
        return [{"time": idx.strftime("%Y-%m-%d"), "close": round(float(v), 2)} for idx, v in close.items()]

    sp500, nasdaq100 = await asyncio.gather(series_for("^GSPC"), series_for("^NDX"))
    return {"sp500": sp500, "nasdaq100": nasdaq100}


@app.get("/api/market-summary")
async def market_summary(request: Request):
    user = get_logged_in_user(request)
    if not user: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not disclaimer_accepted(user):
        return JSONResponse({"error": "Please accept the disclaimer before using QUANTIFY."}, status_code=403)
    if not has_active_access(user):
        return JSONResponse({"error": "Your free trial has ended. Subscribe to keep using QUANTIFY."}, status_code=402)
    conn = db()
    rows = conn.execute("SELECT universe,change_pct,timing_verdict,quant_pass FROM daily_scans WHERE scan_date=?", (today_str(),)).fetchall()
    conn.close()
    total = len(rows)
    advancers = sum(1 for r in rows if (r["change_pct"] or 0) > 0)
    decliners = sum(1 for r in rows if (r["change_pct"] or 0) < 0)
    avg_change = round(sum(r["change_pct"] or 0 for r in rows) / total, 2) if total else None
    detected = sum(1 for r in rows if r["quant_pass"])
    verdict_counts = {"Favorable": 0, "Caution": 0, "Risk": 0}
    for r in rows:
        if r["timing_verdict"] in verdict_counts:
            verdict_counts[r["timing_verdict"]] += 1
    by_universe = {}
    for uni in ("S&P 500", "Nasdaq-100"):
        urows = [r for r in rows if r["universe"] == uni]
        ucount = len(urows)
        by_universe[uni] = {
            "count": ucount,
            "advancers": sum(1 for r in urows if (r["change_pct"] or 0) > 0),
            "decliners": sum(1 for r in urows if (r["change_pct"] or 0) < 0),
            "avg_change_pct": round(sum(r["change_pct"] or 0 for r in urows) / ucount, 2) if ucount else None,
        }

    sp500_df, ndx_df = await asyncio.gather(download_stock("^GSPC", "1d"), download_stock("^NDX", "1d"))
    ai_current = MARKET_AI_SUMMARY_CACHE.get("scan_date") == today_str()
    return {"universe_count": total, "advancers": advancers, "decliners": decliners,
            "avg_change_pct": avg_change, "detected_count": detected, "verdict_breakdown": verdict_counts,
            "by_universe": by_universe,
            "sp500_change_pct": _index_day_change(sp500_df) if sp500_df is not None else None,
            "nasdaq_change_pct": _index_day_change(ndx_df) if ndx_df is not None else None,
            "ai_headline": MARKET_AI_SUMMARY_CACHE.get("headline") if ai_current else None,
            "ai_summary": MARKET_AI_SUMMARY_CACHE.get("summary") if ai_current else None,
            "ai_generated_at": MARKET_AI_SUMMARY_CACHE.get("generated_at") if ai_current else None}


@app.get("/api/backtest-summary")
async def backtest_summary(request: Request):
    user = get_logged_in_user(request)
    if not user: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not disclaimer_accepted(user):
        return JSONResponse({"error": "Please accept the disclaimer before using QUANTIFY."}, status_code=403)
    if not has_active_access(user):
        return JSONResponse({"error": "Your free trial has ended. Subscribe to keep using QUANTIFY."}, status_code=402)
    return {"computed_at": BACKTEST_CACHE.get("computed_at"), "results": BACKTEST_CACHE.get("results"),
            "error": BACKTEST_CACHE.get("error")}


def _require_admin_token(token: Optional[str]):
    return bool(ADMIN_TOKEN) and bool(token) and hmac.compare_digest(token, ADMIN_TOKEN)


@app.get("/api/admin/run-batch")
async def api_run_batch(request: Request, token: Optional[str] = None):
    if not _require_admin_token(token):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not UNIVERSE:
        ready = await refresh_universe()
        if not ready or not UNIVERSE:
            return JSONResponse({"message": "Could not fetch S&P 500 / Nasdaq-100 constituents.",
                                 "running": False, "universe_status": UNIVERSE_STATUS}, status_code=503)
    if BATCH_LOCK.locked():
        return {"message": "A real market data batch is already running.", "running": True,
                "universe_count": len(UNIVERSE)}
    asyncio.create_task(run_eod_batch_process())
    return {"message": f"Started the real market data batch. Processing {len(UNIVERSE)} unique tickers in small batches.",
            "running": True, "universe_count": len(UNIVERSE)}


@app.get("/api/admin/ai-status")
async def api_ai_status(request: Request, token: Optional[str] = None):
    if not _require_admin_token(token):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return dict(AI_STATUS)


@app.get("/api/admin/run-backtest")
async def api_run_backtest(request: Request, token: Optional[str] = None):
    if not _require_admin_token(token):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not UNIVERSE:
        return JSONResponse({"error": "Universe not ready yet."}, status_code=503)
    asyncio.create_task(run_backtest())
    return {"message": "Backtest started."}


@app.get("/api/admin/stats")
async def api_admin_stats(request: Request, token: Optional[str] = None):
    if not _require_admin_token(token):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    conn = db()
    now = time.time()
    total_users = conn.execute("SELECT COUNT(*) c FROM users").fetchone()["c"]
    signups_24h = conn.execute("SELECT COUNT(*) c FROM users WHERE created_at >= ?", (now - 86400,)).fetchone()["c"]
    signups_7d = conn.execute("SELECT COUNT(*) c FROM users WHERE created_at >= ?", (now - 7 * 86400,)).fetchone()["c"]
    active_subs = conn.execute("SELECT COUNT(*) c FROM users WHERE subscription_status='active'").fetchone()["c"]
    rows = conn.execute(
        "SELECT email,created_at,subscription_status,last_login_at,login_count,last_seen_at "
        "FROM users WHERE created_at >= ? ORDER BY created_at DESC",
        (now - 14 * 86400,),
    ).fetchall()
    daily = {}
    for r in rows:
        day = datetime.fromtimestamp(r["created_at"], ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        daily[day] = daily.get(day, 0) + 1

    retention_rows = conn.execute(
        "SELECT email,created_at,last_login_at,login_count,last_seen_at FROM users"
    ).fetchall()

    views_24h = conn.execute("SELECT COUNT(*) c FROM page_views WHERE created_at >= ?", (now - 86400,)).fetchone()["c"]
    views_7d = conn.execute("SELECT COUNT(*) c FROM page_views WHERE created_at >= ?", (now - 7 * 86400,)).fetchone()["c"]
    visitors_24h = conn.execute(
        "SELECT COUNT(DISTINCT visitor_id) c FROM page_views WHERE created_at >= ?", (now - 86400,)
    ).fetchone()["c"]
    visitors_7d = conn.execute(
        "SELECT COUNT(DISTINCT visitor_id) c FROM page_views WHERE created_at >= ?", (now - 7 * 86400,)
    ).fetchone()["c"]
    view_rows = conn.execute(
        "SELECT path,referrer,created_at,visitor_id,utm_source FROM page_views WHERE created_at >= ? ORDER BY created_at DESC",
        (now - 7 * 86400,),
    ).fetchall()
    event_rows = conn.execute(
        "SELECT name,visitor_id,utm_source,email,created_at FROM events WHERE created_at >= ?",
        (now - 7 * 86400,),
    ).fetchall()
    # Fundamentals feed the snowflake; without a count here there is no way to tell a
    # batch still running from one that silently failed.
    fund_total = conn.execute("SELECT COUNT(*) FROM fundamentals").fetchone()[0]
    fund_fresh = conn.execute("SELECT COUNT(*) FROM fundamentals WHERE fetched_at >= ?",
                              (now - FUNDAMENTALS_TTL,)).fetchone()[0]
    fund_last = conn.execute("SELECT MAX(fetched_at) FROM fundamentals").fetchone()[0]
    fund_sectors = conn.execute("SELECT COUNT(DISTINCT sector) FROM fundamentals WHERE sector IS NOT NULL").fetchone()[0]
    have_f = {r["ticker"] for r in conn.execute("SELECT ticker FROM fundamentals")}
    missing_f = [t for t in UNIVERSE if t not in have_f]
    detected_today = [r["ticker"] for r in conn.execute(
        "SELECT ticker FROM daily_scans WHERE scan_date=? AND quant_pass=1", (today_str(),))]
    missing_detected = [t for t in detected_today if t not in have_f]
    conn.close()

    views_by_day = {}
    top_paths = {}
    top_referrers = {}
    for r in view_rows:
        day = datetime.fromtimestamp(r["created_at"], ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        views_by_day[day] = views_by_day.get(day, 0) + 1
        top_paths[r["path"]] = top_paths.get(r["path"], 0) + 1
        ref = r["referrer"]
        if ref:
            host = urllib.parse.urlparse(ref).netloc or ref
            if "quantify.trading" not in host:
                top_referrers[host] = top_referrers.get(host, 0) + 1
        else:
            top_referrers["(direct / no referrer)"] = top_referrers.get("(direct / no referrer)", 0) + 1

    # --- Channel attribution -------------------------------------------------------
    # Every visitor lands in exactly one bucket, so the columns add up to the visitor
    # total; x / reddit / ig are always present even at zero, because "no signups from
    # Reddit this week" is itself the answer to the question being asked.
    def _bucket(src):
        return src if src in PRIMARY_CHANNELS else ("(direct / unknown)" if not src else f"other:{src}")

    channel_visitors: dict = {}
    channel_views: dict = {}
    for r in view_rows:
        b = _bucket(r["utm_source"])
        channel_views[b] = channel_views.get(b, 0) + 1
        channel_visitors.setdefault(b, set()).add(r["visitor_id"])

    events_24h: dict = {name: 0 for name in TRACKED_EVENTS}
    events_7d: dict = {name: 0 for name in TRACKED_EVENTS}
    channel_events: dict = {}
    for r in event_rows:
        name = r["name"]
        events_7d[name] = events_7d.get(name, 0) + 1
        if r["created_at"] >= now - 86400:
            events_24h[name] = events_24h.get(name, 0) + 1
        b = _bucket(r["utm_source"])
        channel_events.setdefault(b, {n: 0 for n in TRACKED_EVENTS})
        channel_events[b][name] = channel_events[b].get(name, 0) + 1

    funnel = {}
    def _sort_key(k):
        # x / reddit / ig first, in that declared order, then everything else A-Z.
        return (PRIMARY_CHANNELS.index(k), "") if k in PRIMARY_CHANNELS else (len(PRIMARY_CHANNELS), k)

    for b in sorted(set(list(channel_visitors) + list(channel_events) + list(PRIMARY_CHANNELS)),
                    key=_sort_key):
        ev = channel_events.get(b, {})
        visitors = len(channel_visitors.get(b, ()))
        funnel[b] = {
            "unique_visitors": visitors,
            "page_views": channel_views.get(b, 0),
            "demo_views": ev.get(EVENT_DEMO_VIEW, 0),
            "terminal_views": ev.get(EVENT_TERMINAL_VIEW, 0),
            "signups": ev.get(EVENT_SIGNUP, 0),
            "payments": ev.get(EVENT_PAYMENT, 0),
            "signup_rate_pct": round(ev.get(EVENT_SIGNUP, 0) / visitors * 100, 1) if visitors else 0.0,
        }
    channels_block = {
        "note": ("First-touch attribution: ?utm_source wins, then the channel already stored on "
                 "the browser, then the referring host (t.co/x.com -> x, reddit.com -> reddit, "
                 "instagram.com -> ig). A payment has no browser cookie, so it is attributed to "
                 "the channel frozen on the account at signup. Data starts at the deploy that "
                 "added this; earlier visits show as (direct / unknown)."),
        "primary": list(PRIMARY_CHANNELS),
        "funnel_last_7d": funnel,
    }

    def _et_day(ts):
        return datetime.fromtimestamp(ts, ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

    returning_users = 0      # logged in again (new device / after cookie expiry)
    came_back_users = 0      # seen on a calendar day after signup -- the real return signal
    active_24h = 0
    active_7d = 0
    days_to_last_visit = []
    weeks = {}
    for r in retention_rows:
        created_at, last_seen = r["created_at"], r["last_seen_at"]
        if not created_at:
            continue
        week = datetime.fromtimestamp(created_at, ZoneInfo("America/New_York")).strftime("%G-W%V")
        w = weeks.setdefault(week, {"signups": 0, "returned": 0, "active_7d": 0})
        w["signups"] += 1
        if (r["login_count"] or 0) >= 2:
            returning_users += 1
        if last_seen:
            if last_seen >= now - 86400:
                active_24h += 1
            if last_seen >= now - 7 * 86400:
                active_7d += 1
                w["active_7d"] += 1
            # "Came back" only counts a visit on a later calendar day, so the signup
            # session itself can never make someone look retained.
            if _et_day(last_seen) > _et_day(created_at):
                came_back_users += 1
                w["returned"] += 1
                days_to_last_visit.append(round((last_seen - created_at) / 86400, 1))
    for w in weeks.values():
        w["returned_pct"] = round(w["returned"] / w["signups"] * 100, 1) if w["signups"] else 0.0
    median_days = None
    if days_to_last_visit:
        s = sorted(days_to_last_visit)
        median_days = s[len(s) // 2] if len(s) % 2 else round((s[len(s) // 2 - 1] + s[len(s) // 2]) / 2, 1)

    return {
        "total_users": total_users,
        "signups_last_24h": signups_24h,
        "signups_last_7d": signups_7d,
        "active_subscriptions": active_subs,
        "signups_by_day_et": daily,
        "recent_signups": [
            {
                "email": r["email"],
                "signed_up_at_et": datetime.fromtimestamp(r["created_at"], ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M"),
                "subscription_status": r["subscription_status"],
                "login_count": r["login_count"] or 0,
                "last_login_at_et": datetime.fromtimestamp(r["last_login_at"], ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M") if r["last_login_at"] else None,
                "last_seen_at_et": datetime.fromtimestamp(r["last_seen_at"], ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M") if r["last_seen_at"] else None,
                "days_to_last_visit": round((r["last_seen_at"] - r["created_at"]) / 86400, 1) if r["last_seen_at"] and r["created_at"] else None,
            }
            for r in rows[:30]
        ],
        "retention": {
            "note": "last_seen_at is written at most once per ET day on any authenticated page load; login_count only moves on a real re-login (session cookie lasts 30 days). Data starts from the deploy that added these columns, so existing accounts show no history for visits before then.",
            "returning_users_2plus_logins": returning_users,
            "returning_rate_pct": round(returning_users / total_users * 100, 1) if total_users else 0.0,
            "came_back_after_signup_day": came_back_users,
            "came_back_rate_pct": round(came_back_users / total_users * 100, 1) if total_users else 0.0,
            "never_came_back": total_users - came_back_users,
            "active_last_24h": active_24h,
            "active_last_7d": active_7d,
            "median_days_signup_to_last_visit": median_days,
            "by_signup_week_et": dict(sorted(weeks.items(), reverse=True)),
        },
        "page_views_last_24h": views_24h,
        "page_views_last_7d": views_7d,
        "unique_visitors_last_24h": visitors_24h,
        "unique_visitors_last_7d": visitors_7d,
        "page_views_by_day_et": views_by_day,
        "top_paths_last_7d": dict(sorted(top_paths.items(), key=lambda x: -x[1])[:15]),
        "top_referrers_last_7d": dict(sorted(top_referrers.items(), key=lambda x: -x[1])[:15]),
        "fundamentals": {
            "rows": fund_total,
            "fresh": fund_fresh,
            "universe": len(UNIVERSE),
            "sectors": fund_sectors,
            "last_fetch_et": (datetime.fromtimestamp(fund_last, ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M")
                              if fund_last else None),
            "missing": len(missing_f),
            "missing_sample": missing_f[:15],
            # The ones that matter: a detected ticker with no fundamentals is a visibly
            # empty snowflake on the page people actually open.
            "missing_detected": missing_detected,
            "scheduler": dict(FUNDAMENTALS_STATUS),
        },
        "channels": channels_block,
        "events_last_24h": events_24h,
        "events_last_7d": events_7d,
    }


@app.post("/api/admin/delete-user")
async def api_admin_delete_user(request: Request, token: Optional[str] = None, email: str = Form(...)):
    if not _require_admin_token(token):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    email = email.strip().lower()
    conn = db()
    row = conn.execute("SELECT email FROM users WHERE email=?", (email,)).fetchone()
    if not row:
        conn.close()
        return JSONResponse({"error": "User not found"}, status_code=404)
    for table in ("sessions", "portfolio_items", "watchlist_items", "user_alerts"):
        conn.execute(f"DELETE FROM {table} WHERE email=?", (email,))
    conn.execute("DELETE FROM users WHERE email=?", (email,))
    conn.commit()
    conn.close()
    return {"message": f"Deleted {email}."}


@app.post("/api/auto-scan")
async def api_auto_scan(request: Request):
    if not get_logged_in_user(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    conn = db()
    count = conn.execute(
        "SELECT COUNT(*) FROM daily_scans WHERE scan_date=?",
        (today_str(),)
    ).fetchone()[0]
    conn.close()

    if count > 0:
        return {
            "started": False,
            "cached": True,
            "count": int(count),
            "message": "Today's scan data is already ready."
        }

    if BATCH_STATUS["running"] or CACHE_STATUS["warming"]:
        return {
            "started": False,
            "cached": False,
            "running": True,
            "count": 0,
            "message": "The server is preparing today's scan."
        }

    start_server_warmup()
    return {
        "started": True,
        "cached": False,
        "running": True,
        "count": 0,
        "message": "Started the server scan."
    }


@app.get("/api/cache-status")
async def api_cache_status(request: Request):
    if not get_logged_in_user(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return {
        **CACHE_STATUS,
        "market_running": BATCH_STATUS["running"],
        "market_processed": BATCH_STATUS["processed"],
        "market_total": BATCH_STATUS["total"],
        "market_saved": BATCH_STATUS["saved"],
        "ai": dict(AI_STATUS),
    }


@app.get("/api/admin/batch-status")
async def api_batch_status(request: Request, token: Optional[str] = None):
    if not _require_admin_token(token):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return dict(BATCH_STATUS)


@app.get("/api/terminal-data-fast")
async def terminal_data_fast(request: Request, ticker: str = "AAPL", timeframe: str = "1d"):
    user = get_logged_in_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not disclaimer_accepted(user):
        return JSONResponse({"error": "Please accept the disclaimer before using QUANTIFY."}, status_code=403)
    if not has_active_access(user):
        return JSONResponse({"error": "Your free trial has ended. Subscribe to keep using QUANTIFY."}, status_code=402)
    ticker = normalize_ticker(ticker)
    if not re.fullmatch(r"[A-Z0-9.\-^=]{1,15}", ticker):
        return JSONResponse({"error": "Invalid ticker"}, status_code=400)
    if timeframe not in ("1h", "1d", "1wk", "1mo"):
        return JSONResponse({"error": "Invalid timeframe"}, status_code=400)
    df, earnings = await asyncio.gather(download_stock(ticker, timeframe), get_earnings(ticker))
    if df is None or df.empty:
        # The in-memory historical cache doesn't survive a restart, so if yfinance has
        # been down across a deploy there may be nothing left to fall back to there.
        # daily_scans is on the persistent disk and keeps whatever the last successful
        # scan saw, so it survives restarts -- no chart candles, but the indicator
        # panel (RSI/MACD/52w hi-lo/trend) can still show something instead of nothing.
        conn = db()
        row = conn.execute(
            "SELECT scan_date,price,change_pct,rsi,macd,pct_from_52w_high,pct_from_52w_low,above_200d_sma "
            "FROM daily_scans WHERE ticker=? ORDER BY scan_date DESC LIMIT 1",
            (ticker,),
        ).fetchone()
        conn.close()
        if row:
            return {"fast": {
                "ticker": ticker, "data_ok": True, "price": row["price"], "change": row["change_pct"],
                "pct_from_52w_high": row["pct_from_52w_high"], "pct_from_52w_low": row["pct_from_52w_low"],
                "above_200d_sma": bool(row["above_200d_sma"]) if row["above_200d_sma"] is not None else None,
                "rsi": row["rsi"], "macd": row["macd"], "earnings": None, "chart": [],
                "stale_db_date": row["scan_date"],
            }}
        return {"fast": {"ticker": ticker, "data_ok": False, "error": "Could not fetch real market data."}}
    try:
        close = normalize_series(df, "Close").dropna()
        if len(close) < 30:
            return {"fast": {"ticker": ticker, "data_ok": False, "error": "Not enough price history for analysis."}}
        rsi = calculate_rsi(close); _, _, macd_hist = calculate_macd(close)
        price = float(close.iloc[-1]); prev = float(close.iloc[-2]); change = (price / prev - 1) * 100 if prev else 0
        window = close.tail(252)
        high_52w = float(window.max()); low_52w = float(window.min())
        pct_from_high = round((price / high_52w - 1) * 100, 2) if high_52w else None
        pct_from_low = round((price / low_52w - 1) * 100, 2) if low_52w else None
        above_trend = None
        if len(close) >= 200:
            sma200_last = close.rolling(200).mean().iloc[-1]
            if pd.notna(sma200_last):
                above_trend = bool(price > float(sma200_last))
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        bb_std = close.rolling(20).std()
        bb_upper = sma20 + 2 * bb_std
        bb_lower = sma20 - 2 * bb_std
        chart = []
        for idx, row in df.tail(500).iterrows():
            try:
                vals = [float(row[k]) for k in ("Open", "High", "Low", "Close")]
                volume_value = int(row["Volume"])
                if any(pd.isna(v) for v in vals) or pd.isna(volume_value):
                    continue
                time_value = int(idx.timestamp()) if timeframe == "1h" else idx.strftime("%Y-%m-%d")
                point = {"time": time_value, "open": vals[0], "high": vals[1], "low": vals[2], "close": vals[3], "volume": volume_value}
                if idx in sma20.index and pd.notna(sma20.loc[idx]):
                    point["sma20"] = round(float(sma20.loc[idx]), 4)
                if idx in sma50.index and pd.notna(sma50.loc[idx]):
                    point["sma50"] = round(float(sma50.loc[idx]), 4)
                if idx in sma200.index and pd.notna(sma200.loc[idx]):
                    point["sma200"] = round(float(sma200.loc[idx]), 4)
                if idx in bb_upper.index and pd.notna(bb_upper.loc[idx]):
                    point["bb_upper"] = round(float(bb_upper.loc[idx]), 4)
                if idx in bb_lower.index and pd.notna(bb_lower.loc[idx]):
                    point["bb_lower"] = round(float(bb_lower.loc[idx]), 4)
                chart.append(point)
            except Exception:
                continue
        stale_as_of = df.attrs.get("stale_as_of")
        return {"fast": {"ticker": ticker, "data_ok": True, "price": round(price, 2), "change": round(change, 2),
            "pct_from_52w_high": pct_from_high, "pct_from_52w_low": pct_from_low, "above_200d_sma": above_trend,
            "rsi": round(float(rsi.iloc[-1]), 2), "macd": round(float(macd_hist.iloc[-1]), 4),
            "earnings": earnings, "chart": chart, "stale_as_of": stale_as_of}}
    except Exception as e:
        print(f"[Error: {type(e).__name__}] terminal_data_fast processing error ({ticker}): {e}")
        return {"fast": {"ticker": ticker, "data_ok": False, "error": f"Data processing error: {type(e).__name__}"}}


@app.get("/api/terminal-data-ai")
async def terminal_data_ai(request: Request, ticker: str = "AAPL", mode: str = "Long-Term Momentum Pullback", language: str = "en"):
    user = get_logged_in_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not disclaimer_accepted(user):
        return JSONResponse({"error": "Please accept the disclaimer before using QUANTIFY."}, status_code=403)
    if not has_active_access(user):
        return JSONResponse({"error": "Your free trial has ended. Subscribe to keep using QUANTIFY."}, status_code=402)
    ticker=normalize_ticker(ticker)
    if not re.fullmatch(r"[A-Z0-9.\-^=]{1,15}",ticker):
        return JSONResponse({"error":"Invalid ticker"},status_code=400)
    if language not in LANGUAGE_NAMES:
        language = "en"

    conn=db()
    row=conn.execute("""SELECT ai_report,short_percent,ai_status,ai_mode,ai_language,ai_updated_at,ai_error,timing_score,timing_verdict,
                                price,change_pct,rsi,macd,pct_from_52w_high,pct_from_52w_low,above_200d_sma,volume_ratio,ai_prompt_version,alpha_score,quant_pass,
                                pct_off_20d_high,pct_above_200d_sma
                        FROM daily_scans WHERE scan_date=? AND ticker=?
                        ORDER BY id DESC LIMIT 1""",(today_str(),ticker)).fetchone()

    if row is None:
        # Ticker outside today's S&P 500 / Nasdaq-100 scan (e.g. manually searched) has no
        # daily_scans row at all, so there is nothing to hang an AI report on — it would
        # otherwise sit on "Preparing AI analysis cache..." forever. Compute it live, the
        # same way the batch scanner does, so the AI path works for any valid ticker.
        df = await download_stock(ticker, "1d")
        analysis = analyze_dataframe(ticker, df)
        if analysis:
            universe = "S&P 500" if ticker in UNIVERSE_META["sp500"] else ("Nasdaq-100" if ticker in UNIVERSE_META["nasdaq100"] else "Other")
            quant_pass = 1 if float(analysis.get("alpha_score") or 0) >= QUANT_PASS_THRESHOLD else 0
            conn.execute("""
                INSERT INTO daily_scans
                (scan_date,ticker,universe,price,change_pct,alpha_score,rsi,macd,
                 pct_from_52w_high,pct_from_52w_low,above_200d_sma,volume_ratio,
                 ai_status,ai_mode,quant_pass,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(scan_date,ticker) DO NOTHING
            """, (
                today_str(), ticker, universe, analysis["price"], analysis["change"], analysis["alpha_score"],
                analysis["rsi"], analysis["macd"], analysis.get("pct_from_52w_high"), analysis.get("pct_from_52w_low"),
                (1 if analysis.get("above_200d_sma") else (0 if analysis.get("above_200d_sma") is False else None)),
                analysis.get("volume_ratio"),
                "PENDING", mode, quant_pass, time.time(),
            ))
            conn.commit()
            row = conn.execute("""SELECT ai_report,short_percent,ai_status,ai_mode,ai_language,ai_updated_at,ai_error,timing_score,timing_verdict,
                                          price,change_pct,rsi,macd,pct_from_52w_high,pct_from_52w_low,above_200d_sma,volume_ratio,ai_prompt_version,alpha_score,quant_pass,
                                pct_off_20d_high,pct_above_200d_sma
                                  FROM daily_scans WHERE scan_date=? AND ticker=?""", (today_str(), ticker)).fetchone()

    # A cached report only satisfies this request if it matches both the selected
    # strategy mode AND language, and was generated under the current prompt schema.
    # On a mismatch, generate a fresh one on demand (this is a shared cache slot, so
    # the next viewer with different mode/language will likewise regenerate it —
    # acceptable for this scale, not per-user cached).
    # AI reports are only ever generated for tickers that actually cleared today's quant
    # scan — never on-demand for an arbitrary lookup (watchlist ticker, manual search).
    # That keeps AI generation volume bounded to the small detected set instead of scaling
    # with how many tickers users happen to click on.
    needs_regen = row is not None and row["price"] is not None and row["quant_pass"] == 1 and (
        row["ai_report"] is None or row["ai_mode"] != mode or (row["ai_language"] or "en") != language
        or row["ai_prompt_version"] != AI_PROMPT_VERSION
    )
    news=await fetch_stock_news(ticker)
    if needs_regen:
        above_trend = bool(row["above_200d_sma"]) if row["above_200d_sma"] is not None else None
        ai_result = await generate_ai_report(
            ticker, row["price"], row["change_pct"], mode, row["rsi"], row["macd"],
            row["pct_from_52w_high"], row["pct_from_52w_low"], above_trend, language,
            row["volume_ratio"], news,
        )
        if ai_result:
            conn.execute("""UPDATE daily_scans
                            SET ai_report=?,ai_status='READY',ai_mode=?,ai_language=?,
                                ai_updated_at=?,ai_error=NULL,timing_score=?,timing_verdict=?,ai_prompt_version=?
                            WHERE scan_date=? AND ticker=?""",
                         (ai_result["report_json"], mode, language, time.time(),
                          ai_result["timing_score"], ai_result["timing_verdict"], AI_PROMPT_VERSION, today_str(), ticker))
            conn.commit()
            row = conn.execute("""SELECT ai_report,short_percent,ai_status,ai_mode,ai_language,ai_updated_at,ai_error,timing_score,timing_verdict,
                                          price,change_pct,rsi,macd,pct_from_52w_high,pct_from_52w_low,above_200d_sma,volume_ratio,alpha_score,quant_pass
                                  FROM daily_scans WHERE scan_date=? AND ticker=?""", (today_str(), ticker)).fetchone()
    conn.close()

    report_sections = None
    if row and row["ai_report"]:
        try:
            report_sections = json.loads(row["ai_report"])
        except json.JSONDecodeError:
            report_sections = {"quant_review": row["ai_report"]}

    alpha_score = row["alpha_score"] if row else None
    timing_score = row["timing_score"] if row else None
    overall_score = round((alpha_score + timing_score) / 2, 1) if alpha_score is not None and timing_score is not None else None
    # Reads the fundamentals table and the peer ladders, so it goes to a worker thread
    # rather than blocking the loop on this request path.
    snowflake = await asyncio.to_thread(_snowflake_axes, row, ticker, language)
    return {"ai":{
        "ai_report": row["ai_report"] if row else None,
        "report_sections": report_sections,
        "timing_score": timing_score,
        "alpha_score": alpha_score,
        "overall_score": overall_score,
        "timing_verdict": row["timing_verdict"] if row else None,
        "news": news,
        "market_summary": None,
        "quant_pass": bool(row["quant_pass"]) if row else False,
        "status": row["ai_status"] if row else "PENDING",
        "mode": row["ai_mode"] if row else mode,
        "updated_at": row["ai_updated_at"] if row else None,
        "error": row["ai_error"] if row else None,
        "language": row["ai_language"] if row and row["ai_report"] else None,
        "language_requested": language,
        "snowflake": snowflake,
        "quota_exhausted": AI_QUOTA_EXHAUSTED_DATE == today_str(),
        # The price the score/verdict were computed from -- this only updates at the
        # next scan cycle (SCAN_TIMES_ET), while the live price shown alongside it can
        # move the whole time in between. Without exposing this, a "Favorable" pullback
        # badge can keep showing next to a price that has already run well past the
        # entry it was actually scored on.
        "scan_price": row["price"] if row else None,
    }}


@app.post("/api/alerts/set")
async def set_alert(request: Request, ticker: str = Form(...), target_price: float = Form(...), direction: str = Form("above")):
    user = get_logged_in_user(request)
    if not user: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not has_active_access(user):
        return JSONResponse({"error": "Your free trial has ended. Subscribe to keep using price alerts."}, status_code=402)
    ticker = normalize_ticker(ticker)
    direction = direction if direction in ("above", "below") else "above"
    if not re.fullmatch(r"[A-Z0-9.\-^=]{1,15}", ticker) or target_price <= 0:
        return JSONResponse({"error": "Invalid alert"}, status_code=400)
    try:
        conn = db(); conn.execute("INSERT INTO user_alerts(email,ticker,target_price,direction,created_at) VALUES(?,?,?,?,?)", (user,ticker,target_price,direction,time.time())); conn.commit(); conn.close()
    except Exception as e:
        print(f"[Error: {type(e).__name__}] Alert setup error: {e}")
        return JSONResponse({"error": f"Database error: {type(e).__name__}"}, status_code=500)
    return {"message": f"Alert set: {ticker} {'≥' if direction == 'above' else '≤'} ${target_price:.2f}. Manage or cancel anytime from Settings → Price Alerts."}


@app.get("/api/alerts/list")
async def alerts_list(request: Request):
    user = get_logged_in_user(request)
    if not user: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not has_active_access(user):
        return JSONResponse({"error": "Your free trial has ended. Subscribe to keep using price alerts."}, status_code=402)
    conn = db()
    rows = conn.execute("SELECT id,ticker,target_price,direction,is_sent,created_at FROM user_alerts WHERE email=? ORDER BY created_at DESC", (user,)).fetchall()
    conn.close()
    return {"alerts": [dict(r) for r in rows]}


@app.post("/api/alerts/remove")
async def alerts_remove(request: Request, id: int = Form(...)):
    user = get_logged_in_user(request)
    if not user: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    conn = db()
    conn.execute("DELETE FROM user_alerts WHERE id=? AND email=?", (id, user))
    conn.commit(); conn.close()
    return {"message": "Alert removed."}


@app.post("/api/portfolio/save")
async def portfolio_save(request: Request, ticker: str = Form(...), note: str = Form(""), shares: str = Form(""), price: str = Form("")):
    user = get_logged_in_user(request)
    if not user: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not has_active_access(user):
        return JSONResponse({"error": "Your free trial has ended. Subscribe to keep using your portfolio."}, status_code=402)
    ticker = normalize_ticker(ticker)
    if not re.fullmatch(r"[A-Z0-9.\-^=]{1,15}", ticker):
        return JSONResponse({"error": "Invalid ticker"}, status_code=400)
    shares_val = None
    if shares.strip() != "":
        try:
            shares_val = float(shares)
            if not (shares_val > 0):
                raise ValueError
        except ValueError:
            return JSONResponse({"error": "Shares must be a positive number."}, status_code=400)
    price_val = None
    if price.strip() != "":
        try:
            price_val = float(price)
            if not (price_val > 0):
                raise ValueError
        except ValueError:
            return JSONResponse({"error": "Entry price must be a positive number."}, status_code=400)
    try:
        conn = db()
        row = conn.execute("""
            SELECT scan_date,price,change_pct,alpha_score,rsi,macd,timing_score,timing_verdict,ai_report
            FROM daily_scans WHERE ticker=? ORDER BY id DESC LIMIT 1
        """, (ticker,)).fetchone()
        if not row:
            conn.close()
            return JSONResponse({"error": "No scan data for this ticker yet. Run a scan first."}, status_code=400)
        entry_price = price_val if price_val is not None else row["price"]
        conn.execute("""
            INSERT INTO portfolio_items
            (email,ticker,scan_date,price,change_pct,alpha_score,rsi,macd,timing_score,timing_verdict,ai_report,note,saved_at,shares)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (user, ticker, row["scan_date"], entry_price, row["change_pct"], row["alpha_score"],
              row["rsi"], row["macd"], row["timing_score"], row["timing_verdict"], row["ai_report"],
              note.strip()[:500], time.time(), shares_val))
        conn.commit(); conn.close()
    except Exception as e:
        print(f"[Error: {type(e).__name__}] Portfolio save error: {e}")
        return JSONResponse({"error": f"Database error: {type(e).__name__}"}, status_code=500)
    return {"message": f"Saved {ticker} to your portfolio."}


async def get_current_price(ticker: str):
    try:
        df = await download_stock(ticker, "1d")
        if df is None or df.empty:
            return None
        closes = normalize_series(df, "Close").dropna()
        if closes.empty:
            return None
        return round(float(closes.iloc[-1]), 2)
    except Exception:
        return None


@app.get("/api/portfolio")
async def portfolio_list(request: Request):
    user = get_logged_in_user(request)
    if not user: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not has_active_access(user):
        return JSONResponse({"error": "Your free trial has ended. Subscribe to keep using your portfolio."}, status_code=402)
    conn = db()
    rows = conn.execute("""
        SELECT id,ticker,scan_date,price,change_pct,alpha_score,rsi,macd,timing_score,timing_verdict,ai_report,note,saved_at,shares
        FROM portfolio_items WHERE email=? ORDER BY saved_at DESC
    """, (user,)).fetchall()
    conn.close()
    tickers = list({r["ticker"] for r in rows})
    prices = await asyncio.gather(*(get_current_price(t) for t in tickers))
    current_price_by_ticker = dict(zip(tickers, prices))
    items = []
    for r in rows:
        d = dict(r)
        if d.get("ai_report"):
            try:
                d["ai_report"] = json.loads(d["ai_report"])
            except json.JSONDecodeError:
                pass
        d["overall_score"] = round((d["alpha_score"] + d["timing_score"]) / 2, 1) \
            if d.get("alpha_score") is not None and d.get("timing_score") is not None else None
        current_price = current_price_by_ticker.get(d["ticker"])
        d["current_price"] = current_price
        entry_price = d.get("price")
        if current_price is not None and entry_price:
            d["return_pct"] = round((current_price - entry_price) / entry_price * 100, 2)
            d["pl_dollar"] = round((current_price - entry_price) * d["shares"], 2) if d.get("shares") else None
        else:
            d["return_pct"] = None
            d["pl_dollar"] = None
        d["sector"] = (SECTOR_CACHE.get(d["ticker"]) or {}).get("sector")
        items.append(d)
    return {"items": items}


@app.post("/api/portfolio/remove")
async def portfolio_remove(request: Request, id: int = Form(...)):
    user = get_logged_in_user(request)
    if not user: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    conn = db()
    conn.execute("DELETE FROM portfolio_items WHERE id=? AND email=?", (id, user))
    conn.commit(); conn.close()
    return {"message": "Removed."}


@app.post("/api/watchlist/add")
async def watchlist_add(request: Request, ticker: str = Form(...)):
    user = get_logged_in_user(request)
    if not user: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not has_active_access(user):
        return JSONResponse({"error": "Your free trial has ended. Subscribe to keep using your watchlist."}, status_code=402)
    ticker = normalize_ticker(ticker)
    if not re.fullmatch(r"[A-Z0-9.\-^=]{1,15}", ticker):
        return JSONResponse({"error": "Invalid ticker"}, status_code=400)
    try:
        conn = db()
        conn.execute("INSERT OR IGNORE INTO watchlist_items(email,ticker,added_at) VALUES(?,?,?)", (user, ticker, time.time()))
        conn.commit(); conn.close()
    except Exception as e:
        print(f"[Error: {type(e).__name__}] Watchlist add error: {e}")
        return JSONResponse({"error": f"Database error: {type(e).__name__}"}, status_code=500)
    return {"message": f"Added {ticker} to your watchlist."}


@app.get("/api/watchlist")
async def watchlist_list(request: Request):
    user = get_logged_in_user(request)
    if not user: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not has_active_access(user):
        return JSONResponse({"error": "Your free trial has ended. Subscribe to keep using your watchlist."}, status_code=402)
    conn = db()
    rows = conn.execute("SELECT id,ticker,added_at FROM watchlist_items WHERE email=? ORDER BY added_at DESC", (user,)).fetchall()
    tickers = [r["ticker"] for r in rows]
    scan_by_ticker = {}
    if tickers:
        placeholders = ",".join("?" * len(tickers))
        scan_rows = conn.execute(
            f"SELECT ticker,change_pct,timing_verdict,ROUND((alpha_score+timing_score)/2.0,1) AS overall_score "
            f"FROM daily_scans WHERE scan_date=? AND ticker IN ({placeholders})",
            (today_str(), *tickers),
        ).fetchall()
        scan_by_ticker = {r["ticker"]: dict(r) for r in scan_rows}
    conn.close()
    prices = await asyncio.gather(*(get_current_price(t) for t in tickers))
    price_by_ticker = dict(zip(tickers, prices))
    items = []
    for r in rows:
        d = dict(r)
        d["price"] = price_by_ticker.get(d["ticker"])
        scan = scan_by_ticker.get(d["ticker"], {})
        d["change_pct"] = scan.get("change_pct")
        d["overall_score"] = scan.get("overall_score")
        d["timing_verdict"] = scan.get("timing_verdict")
        d["sector"] = (SECTOR_CACHE.get(d["ticker"]) or {}).get("sector")
        cached = CACHE["historical"].get(f"single:{d['ticker']}:1d")
        sparkline = []
        if cached is not None:
            try:
                closes = normalize_series(cached["data"], "Close").dropna().tail(15)
                sparkline = [round(float(v), 2) for v in closes]
            except Exception:
                sparkline = []
        d["sparkline"] = sparkline
        items.append(d)
    return {"items": items}


@app.post("/api/watchlist/remove")
async def watchlist_remove(request: Request, id: int = Form(...)):
    user = get_logged_in_user(request)
    if not user: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    conn = db()
    conn.execute("DELETE FROM watchlist_items WHERE id=? AND email=?", (id, user))
    conn.commit(); conn.close()
    return {"message": "Removed."}


@app.get("/api/settings")
async def get_settings(request: Request):
    user = get_logged_in_user(request)
    if not user: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    conn = db()
    row = conn.execute(
        "SELECT email,pref_theme,pref_language,pref_default_sort,pref_default_view,pref_high_score_alerts,created_at,trial_ends_at,subscription_status FROM users WHERE email=?",
        (user,)
    ).fetchone()
    conn.close()
    if not row:
        return JSONResponse({"error": "User not found"}, status_code=404)
    return dict(row)


@app.post("/api/account/delete")
async def delete_account(request: Request, password: str = Form(""), confirm_email: str = Form("")):
    user = get_logged_in_user(request)
    if not user: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    conn = db()
    row = conn.execute("SELECT password_hash,salt,auth_provider FROM users WHERE email=?", (user,)).fetchone()
    if not row:
        conn.close()
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if row["auth_provider"] == "google":
        # They signed in with Google and have no password of their own, so the account
        # email is the only thing they can be asked to confirm.
        if confirm_email.strip().lower() != user.lower():
            conn.close()
            return JSONResponse({"error": "Type your account email exactly to confirm."}, status_code=400)
    elif not row["password_hash"] or not await asyncio.to_thread(verify_password, password, row["password_hash"], row["salt"]):
        conn.close()
        return JSONResponse({"error": "Incorrect password."}, status_code=400)
    for table in ("sessions", "portfolio_items", "watchlist_items", "user_alerts"):
        conn.execute(f"DELETE FROM {table} WHERE email=?", (user,))
    conn.execute("DELETE FROM users WHERE email=?", (user,))
    conn.commit(); conn.close()
    response = JSONResponse({"message": "Account deleted."})
    response.delete_cookie("session_user")
    return response


@app.post("/api/settings")
async def update_settings(request: Request, theme: str = Form(...), language: str = Form(...),
                           default_sort: str = Form("overall_score"), default_view: str = Form("list"),
                           high_score_alerts: str = Form("0")):
    user = get_logged_in_user(request)
    if not user: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if theme not in ("dark", "light"):
        return JSONResponse({"error": "Invalid theme"}, status_code=400)
    if language not in LANGUAGE_NAMES:
        return JSONResponse({"error": "Invalid language"}, status_code=400)
    if default_sort not in ("overall_score", "change_pct", "ticker"):
        return JSONResponse({"error": "Invalid default sort"}, status_code=400)
    if default_view not in ("list", "heatmap"):
        return JSONResponse({"error": "Invalid default view"}, status_code=400)
    if high_score_alerts not in ("0", "1"):
        return JSONResponse({"error": "Invalid high score alerts value"}, status_code=400)
    conn = db()
    conn.execute("UPDATE users SET pref_theme=?,pref_language=?,pref_default_sort=?,pref_default_view=?,pref_high_score_alerts=? WHERE email=?",
                 (theme, language, default_sort, default_view, int(high_score_alerts), user))
    conn.commit(); conn.close()
    # Mirror the language onto the cookie so the header toggle, the browser and Settings
    # can never end up showing three different answers on the same machine.
    res = JSONResponse({"message": "Settings saved."})
    res.set_cookie(LANG_COOKIE, language, max_age=LANG_COOKIE_MAX_AGE, httponly=False, samesite="lax")
    return res


@app.post("/api/settings/password")
async def change_password(request: Request, current_password: str = Form(...), new_password: str = Form(...)):
    user = get_logged_in_user(request)
    if not user: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    conn = db()
    row = conn.execute("SELECT password_hash,salt FROM users WHERE email=?", (user,)).fetchone()
    if not row or not await asyncio.to_thread(verify_password, current_password, row["password_hash"], row["salt"]):
        conn.close()
        return JSONResponse({"error": "Current password is incorrect."}, status_code=400)
    ok, error = validate_password_policy(new_password)
    if not ok:
        conn.close()
        return JSONResponse({"error": error}, status_code=400)
    password_hash, salt = await asyncio.to_thread(make_password_hash, new_password)
    conn.execute("UPDATE users SET password_hash=?,salt=? WHERE email=?", (password_hash, salt, user))
    conn.commit(); conn.close()
    return {"message": "Password changed."}

# -----------------------------------------------------------------------------
# Landing page
# -----------------------------------------------------------------------------
LANDING_HTML = """<!doctype html><html lang="%%LANG%%"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="google-site-verification" content="ijDMR9nIE6oGHc1hocnPovy0BwrxRdGBL3DvWNy6OZI" />
<title>QUANTIFY. — Quant-Detected Stocks, AI Risk-Checked</title>
<meta name="description" content="A daily quant scan of the S&P 500 and Nasdaq-100, cross-checked by AI for blow-off-top and dead-cat-bounce risk. Informational only — never a buy or sell signal.">
<meta property="og:title" content="QUANTIFY. — Quant-Detected Stocks, AI Risk-Checked">
<meta property="og:description" content="A daily quant scan of the S&P 500 and Nasdaq-100, cross-checked by AI for blow-off-top and dead-cat-bounce risk. Informational only — never a buy or sell signal.">
<meta property="og:type" content="website">
<meta property="og:url" content="https://quantify.trading/">
<meta property="og:site_name" content="QUANTIFY.">
%%OG_IMAGE%%
%%HREFLANG%%
<link rel="canonical" href="https://quantify.trading/">
<script type="application/ld+json">
{
"@context": "https://schema.org",
"@type": "SoftwareApplication",
"name": "QUANTIFY",
"url": "https://quantify.trading/",
"description": "A daily quant scan of the S&P 500 and Nasdaq-100, cross-checked by AI for blow-off-top and dead-cat-bounce risk. Informational only, never a buy or sell signal.",
"applicationCategory": "FinanceApplication",
"operatingSystem": "Web",
"offers": {
"@type": "Offer",
"price": "9.99",
"priceCurrency": "USD",
"description": "7-day free trial, then $9.99/month, cancel anytime"
}
}
</script>
<style>
:root{--bg:#ffffff;--panel:#ffffff;--panel2:#f6f8f7;--border:#e1e6e3;--text:#3a4440;--head:#12201a;--dim:#7a8781;--dim2:#525f59;--green:#0e8a5f;--green-bright:#17a374;--green-soft:#e6f5ee;--red:#c8402c;--orange:#a8660a;--blue:#1f5fbf;--teal:#0a8f83}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font:18px/1.75 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}
a{color:var(--green);text-decoration:underline}
.wrap{max-width:1120px;margin:0 auto;padding:0 28px}
header{border-bottom:1px solid var(--border);position:sticky;top:0;background:rgba(255,255,255,.96);backdrop-filter:blur(6px);z-index:10}
.nav{display:flex;align-items:center;justify-content:space-between;padding:20px 28px;max-width:1120px;margin:0 auto}
.logo{color:var(--head);font-weight:800;font-size:22px;letter-spacing:-.3px;text-decoration:none}
.logo span{color:var(--green)}
.navlinks{display:flex;align-items:center;gap:28px;font-size:16px;flex-wrap:wrap;justify-content:flex-end}
.navlinks a.muted{color:var(--dim2);text-decoration:none;font-weight:500}
.navlinks a.muted:hover{color:var(--head)}
.btn{display:inline-block;padding:14px 26px;border:1px solid var(--green);color:#ffffff;background:var(--green);font:16px/1 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-weight:700;cursor:pointer;letter-spacing:0;border-radius:8px;text-decoration:none}
.btn:hover{background:var(--green-bright);border-color:var(--green-bright);color:#fff}
.btn-ghost{border-color:var(--border);color:var(--head);background:#ffffff}
.btn-ghost:hover{border-color:var(--dim2);background:var(--panel2)}
/* The hero used to offer three competing buttons; 435 home views produced 18 signup-page
   views and almost no demo clicks. One primary action only -- the lowest-friction one --
   with the trial as a plain text link underneath. */
.cta-link{color:var(--dim2);text-decoration:underline;font-weight:600}
.cta-link:hover{color:var(--green)}
/* Secondary to the filled demo button but still unmistakably a button. */
.btn-signup{border:2px solid var(--green);color:var(--green);background:#ffffff;font-weight:800;padding:16px 30px;font-size:17px}
.btn-signup:hover{background:var(--green);color:#ffffff}
.hero{padding:96px 24px 76px;text-align:center;background:radial-gradient(ellipse 900px 500px at 50% -10%,rgba(14,138,95,.07),transparent 65%)}
section:nth-of-type(even){background:var(--panel2)}
.eyebrow{display:inline-block;font-size:14px;font-weight:700;color:var(--orange);border:1px solid #ecdcb8;background:#fbf3e4;padding:7px 16px;border-radius:20px;letter-spacing:.3px;margin-bottom:26px}
h1{color:var(--head);font-size:58px;font-weight:800;line-height:1.15;letter-spacing:-1.5px;max-width:880px;margin:0 auto 24px;text-wrap:balance}
h1 .hl{color:var(--green)}
.sub{color:var(--dim2);font-size:21px;line-height:1.6;max-width:660px;margin:0 auto 38px}
.cta-row{display:flex;gap:16px;justify-content:center;flex-wrap:wrap;margin-bottom:16px}
.cta-note{font-size:14px;color:var(--dim)}
.mock{max-width:940px;margin:68px auto 0;border:1px solid var(--border);border-radius:16px;background:var(--panel);text-align:left;box-shadow:0 30px 70px -30px rgba(18,32,26,.25);overflow:hidden}
.mock-bar{display:flex;gap:6px;padding:12px 16px;border-bottom:1px solid var(--border);background:var(--panel2)}
.mock-dot{width:10px;height:10px;border-radius:50%;background:#d7ddd9}
.mock-grid{display:grid;grid-template-columns:1.1fr 1.6fr 1.3fr;gap:1px;background:var(--border)}
.mock-col{background:var(--panel);padding:18px}
.mock-h{font-size:12px;color:var(--dim);letter-spacing:.4px;margin-bottom:12px;border-bottom:1px solid var(--border);padding-bottom:8px;font-weight:700}
.mock-row{display:flex;justify-content:space-between;padding:8px 0;font-size:14.5px;border-bottom:1px solid var(--panel2)}
.mock-row b{color:var(--head)}
.badge{padding:3px 9px;border-radius:5px;font-weight:700;font-size:12px;display:inline-block}
.badge-ok{background:var(--green-soft);color:var(--green)}
.badge-warn{background:#fbf1e0;color:var(--orange)}
.badge-danger{background:#fbe6e2;color:var(--red)}
.mock-chartline{height:76px;border:1px solid var(--border);border-radius:6px;background:linear-gradient(180deg,transparent,rgba(14,138,95,.05));position:relative;margin:12px 0}
.mock-chartline svg{width:100%;height:100%;display:block}
.mock-ai{font-size:13.5px;color:var(--text);line-height:1.75}
.mock-ai b{color:var(--head);display:block;font-size:11.5px;letter-spacing:.4px;margin:12px 0 4px;text-transform:uppercase}
section{padding:80px 24px;border-top:1px solid var(--border)}
.section-head{text-align:center;max-width:680px;margin:0 auto 52px}
.section-head .kicker{font-size:13px;font-weight:700;color:var(--green);letter-spacing:1.2px;margin-bottom:12px}
.section-head h2{color:var(--head);font-size:36px;font-weight:800;letter-spacing:-.5px;margin-bottom:14px;text-wrap:balance}
.section-head p{color:var(--dim2);font-size:18px;line-height:1.65}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:26px;max-width:1040px;margin:0 auto}
.step{background:var(--panel);border:1px solid var(--border);border-left:3px solid var(--green);border-radius:10px;padding:30px}
.step:nth-child(2){border-left-color:var(--blue)}
.step:nth-child(3){border-left-color:var(--teal)}
.step .num{color:var(--green);font-size:13px;font-weight:800;letter-spacing:1px;margin-bottom:16px}
.step:nth-child(2) .num{color:var(--blue)}
.step:nth-child(3) .num{color:var(--teal)}
.step h3{color:var(--head);font-size:19px;font-weight:700;margin-bottom:12px}
.step p{color:var(--dim2);font-size:15.5px;line-height:1.7}
.features{display:grid;grid-template-columns:repeat(2,1fr);gap:1px;background:var(--border);max-width:800px;margin:0 auto;border:1px solid var(--border);border-radius:12px;overflow:hidden}
.feature{background:var(--panel);padding:30px}
.feature .icon{color:var(--green);font-size:22px;margin-bottom:14px}
.feature:nth-child(4n+2) .icon{color:var(--blue)}
.feature:nth-child(4n+3) .icon{color:var(--teal)}
.proof-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:26px;max-width:940px;margin:0 auto 24px}
.proof-card{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:30px;text-align:center}
.proof-card .horizon{color:var(--dim);font-size:12.5px;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:12px}
.proof-card .num{color:var(--green);font-size:38px;font-weight:800;margin-bottom:8px}
.proof-card .compare{color:var(--dim2);font-size:14.5px}
.proof-card .risk-row{color:var(--dim);font-size:13px;margin-top:12px;padding-top:12px;border-top:1px solid var(--border)}
.proof-card .worst-case{color:var(--red);font-size:14px;font-weight:700;margin-top:5px}
.proof-note{max-width:680px;margin:0 auto;text-align:center;color:var(--dim);font-size:14px;line-height:1.75}
.methodology{max-width:740px;margin:30px auto 0;background:var(--panel2);border:1px solid var(--border);padding:24px 28px;border-radius:12px}
.methodology h4{color:var(--head);font-size:14px;font-weight:700;text-transform:uppercase;letter-spacing:.4px;margin-bottom:14px}
.methodology ul{margin:0;padding-left:20px;color:var(--dim2);font-size:14px;line-height:1.85}
.methodology li{margin-bottom:5px}
.methodology b{color:var(--head)}
.feature:nth-child(4n+4) .icon{color:var(--orange)}
.feature h4{color:var(--head);font-size:16.5px;font-weight:700;margin-bottom:10px}
.feature p{color:var(--dim2);font-size:14.5px;line-height:1.7}
.final-wrap{text-align:center;max-width:840px;margin:0 auto 44px}
.final-wrap h2{color:var(--head);font-size:36px;font-weight:800;letter-spacing:-.5px;margin-bottom:16px;text-wrap:balance}
.final-wrap p{color:var(--dim2);font-size:18px;margin-bottom:32px}
.disclaimer{max-width:840px;margin:0 auto;background:#fbf9f2;border:1px solid #ecdcb8;border-radius:12px;padding:26px 30px}
.disclaimer .kicker{color:var(--orange);font-size:12.5px;letter-spacing:1px;margin-bottom:12px;font-weight:800}
.disclaimer p{color:var(--dim2);font-size:14px;line-height:1.75}
footer{border-top:1px solid var(--border);padding:38px 24px;text-align:center;color:var(--dim);font-size:14px}
footer a{color:var(--dim2);text-decoration:underline}
.btn{white-space:nowrap}
.lang-toggle{display:inline-block;padding:6px 12px;border:1px solid var(--border);border-radius:999px;color:var(--dim2);font-size:13.5px;font-weight:700;text-decoration:none;line-height:1;white-space:nowrap}
.lang-toggle:hover{border-color:var(--green);color:var(--green)}
[data-reveal]{opacity:0}
[data-reveal].is-visible{opacity:1}
[data-reveal].is-visible>*{opacity:0;transform:translateY(26px);animation:revealChild .7s cubic-bezier(.16,1,.3,1) forwards}
[data-reveal].is-visible>*:nth-child(1){animation-delay:.05s}
[data-reveal].is-visible>*:nth-child(2){animation-delay:.13s}
[data-reveal].is-visible>*:nth-child(3){animation-delay:.21s}
[data-reveal].is-visible>*:nth-child(4){animation-delay:.29s}
[data-reveal].is-visible>*:nth-child(5){animation-delay:.37s}
[data-reveal].is-visible>*:nth-child(n+6){animation-delay:.45s}
@keyframes revealChild{to{opacity:1;transform:translateY(0)}}
.hero-in{opacity:0;transform:translateY(20px);animation:heroIn .5s cubic-bezier(.16,1,.3,1) forwards}
/* Kept the stagger, cut the total: the primary CTA used to finish animating in at ~1.2s
   even though the HTML arrives in ~0.26s. Now it is fully on screen by ~0.7s. */
.hero-in-1{animation-delay:.02s}
.hero-in-2{animation-delay:.07s}
.hero-in-3{animation-delay:.14s}
.hero-in-4{animation-delay:.21s}
.hero-in-5{animation-delay:.3s}
@keyframes heroIn{to{opacity:1;transform:translateY(0)}}
@media(prefers-reduced-motion:reduce){[data-reveal],[data-reveal].is-visible>*,.hero-in{animation:none;transition:none;opacity:1;transform:none}}
.btn{transition:transform .15s ease,box-shadow .15s ease}
.btn:hover{transform:translateY(-1px);box-shadow:0 10px 24px -10px rgba(14,138,95,.45)}
.btn-hero{padding:17px 32px;font-size:17.5px;animation:heroPulse 2.6s ease-in-out infinite}
.btn-hero:hover{animation:none}
@keyframes heroPulse{0%,100%{box-shadow:0 0 0 0 rgba(14,138,95,.28)}50%{box-shadow:0 0 0 10px rgba(14,138,95,0)}}
@media(prefers-reduced-motion:reduce){.btn-hero{animation:none}}
.sticky-cta{position:fixed;left:0;right:0;bottom:0;z-index:20;background:var(--panel);border-top:1px solid var(--border);padding:14px 20px;display:none;align-items:center;justify-content:center;gap:18px;box-shadow:0 -10px 30px -15px rgba(18,32,26,.15)}
.sticky-cta.show{display:flex}
.sticky-cta span{color:var(--dim2);font-size:15px}
.sticky-cta span b{color:var(--head)}
@media(max-width:560px){.sticky-cta span{display:none}}
.btn-ghost:hover{box-shadow:none}
.step,.feature,.proof-card,.diff-col{transition:transform .2s ease,border-color .2s ease}
.step:hover,.feature:hover,.proof-card:hover{transform:translateY(-3px);border-color:var(--dim2)}
.live-dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);margin-right:7px;animation:pulse 2s ease-in-out infinite;vertical-align:middle}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
.mock-chartline svg polyline{stroke-dasharray:500;stroke-dashoffset:500;animation:draw 1.6s ease-out .2s forwards}
@keyframes draw{to{stroke-dashoffset:0}}
.diff-grid{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--border);max-width:940px;margin:0 auto;border:1px solid var(--border);border-radius:12px;overflow:hidden}
.diff-col{background:var(--panel);padding:34px}
.diff-col.against{background:#fafaf9}
.diff-col h4{font-size:15px;font-weight:700;margin-bottom:20px;text-transform:uppercase;letter-spacing:.5px;color:var(--dim)}
.diff-col.for h4{color:var(--green)}
.diff-item{display:flex;gap:12px;margin-bottom:17px;font-size:15px;color:var(--dim2);line-height:1.6}
.diff-item .mark{flex-shrink:0;font-weight:800;width:16px}
.diff-col.for .mark{color:var(--green)}
.diff-col.against .mark{color:var(--dim)}
.feature .icon svg{display:block}
@media(max-width:820px){
  h1{font-size:38px}
  .steps,.features,.proof-grid,.diff-grid{grid-template-columns:1fr}
  .mock-grid{grid-template-columns:1fr}
  .navlinks{gap:16px;font-size:14px}
  .section-head h2{font-size:28px}
}
@media(max-width:560px){
  .nav{padding:16px 18px}
  .logo{font-size:18px}
  .navlinks{gap:12px}
  .navlinks a.muted{display:none}
  /* Keep Log in reachable on phones -- hiding it with the other nav links left
     returning mobile visitors (most of the traffic) with no way back into the app. */
  .navlinks a.navlogin{display:inline-block;color:var(--head);font-weight:700}
  .btn{padding:11px 16px;font-size:14px}
  /* Adding the language toggle pushed the nav onto two lines on a phone. The hero has
     both CTAs immediately below, so the duplicate trial button in the nav is the one
     that goes -- the toggle and Log in are what a returning visitor needs up there. */
  .navlinks .btn-ghost{display:none}
  .lang-toggle{padding:5px 10px;font-size:12.5px}
  .hero{padding:36px 16px 34px}
  h1{font-size:29px;letter-spacing:-.5px;margin-bottom:16px}
  .sub{font-size:16.5px;margin-bottom:24px}
  .eyebrow{margin-bottom:18px;font-size:12.5px;padding:6px 13px}
  section{padding:54px 16px}
  /* One button, full width, allowed to wrap -- .btn sets white-space:nowrap globally,
     which would push a descriptive label off the side of a 375px screen. */
  .cta-row{flex-direction:column;gap:10px;margin-bottom:12px}
  .cta-row .btn{display:block;width:100%;white-space:normal;line-height:1.3;padding:15px 12px;font-size:16px}
  /* Pull the scanner mock up so its top edge shows above the fold: the product output is
     the thing that explains the product, and it used to start entirely below it. */
  .mock{margin-top:30px}
}
</style></head><body>
<header><div class="nav">
<a class="logo" href="/">QUANTIFY<span>.</span></a>
<div class="navlinks">
<a class="muted" href="#how">How it works</a>
<a class="muted" href="#proof">Results</a>
<a class="muted" href="#features">Features</a>
<a class="muted" href="/pricing">Pricing</a>
<a class="muted" href="/faq">FAQ</a>
<a class="muted navlogin" href="/login">Log in</a>
%%LANG_TOGGLE%%
<a class="btn btn-ghost" href="/signup">Start free trial</a>
</div>
</div></header>

<section class="hero" style="border-top:none">
<div class="eyebrow hero-in hero-in-1">%%EYEBROW%%</div>
<h1 class="hero-in hero-in-2">Is this dip worth buying?<br>Or is it a <span class="hl">falling knife</span>?</h1>
<p class="sub hero-in hero-in-3">A daily quant scan of the S&amp;P 500 and Nasdaq-100, with an AI second pass that checks every hit for blow-off-top and dead-cat-bounce risk.</p>
<div class="cta-row hero-in hero-in-4">
<a class="btn btn-hero" href="/demo">%%HERO_CTA%%</a>
<a class="btn btn-signup" href="/signup">Get Started Free</a>
</div>
<div class="cta-note hero-in hero-in-5">Demo needs no signup · trial is 7 days free, no credit card.</div>

<div class="mock" data-reveal>
<div class="mock-bar"><div class="mock-dot"></div><div class="mock-dot"></div><div class="mock-dot"></div></div>
<div class="mock-grid">
<div class="mock-col">
<div class="mock-h" id="mockH"><span class="live-dot"></span>MARKET SCANNER · 13 detected / 518</div>
<div id="mockList">
<div class="mock-row"><b>MRK</b><span>151.12 <span class="badge badge-ok">Favorable</span></span></div>
<div class="mock-row"><b>HOOD</b><span>110.71 <span class="badge badge-ok">Favorable</span></span></div>
<div class="mock-row"><b>DASH</b><span>232.00 <span class="badge badge-warn">Caution</span></span></div>
<div class="mock-row"><b>ADSK</b><span>270.54 <span class="badge badge-warn">Caution</span></span></div>
</div>
</div>
<div class="mock-col">
<div class="mock-h" id="mockChartTicker">MRK · TECHNICAL CHART</div>
<div class="mock-chartline"><svg viewBox="0 0 300 70" preserveAspectRatio="none" id="mockChartSvg"><polyline points="0,50 20,45 40,48 60,40 80,42 100,30 120,34 140,22 160,26 180,18 200,20 220,14 240,17 260,10 280,13 300,8" fill="none" stroke="#0e8a5f" stroke-width="2"/></svg></div>
<div class="mock-row"><span>RSI / MACD</span><b id="mockRsiMacd">64.58 / 1.4261</b></div>
<div class="mock-row"><span>52W HIGH / LOW</span><b id="mock52w">-4.42% / 97.2%</b></div>
</div>
<div class="mock-col">
<div class="mock-h">AI QUANT REPORT <span class="badge badge-ok" id="mockAiBadge" style="margin-left:4px">Favorable</span></div>
<div class="mock-ai">
<b>QUANT REVIEW</b>
<span id="mockQuantReview">RSI near 65, positive MACD, price 4.4% below its 52-week high.</span>
<b>RISK REVIEW</b>
<span id="mockRiskReview">Not extended near the high, well above its 52-week low — low blow-off-top and dead-cat-bounce risk.</span>
</div>
</div>
</div>
<div style="text-align:center;padding:8px 0 2px;font-size:10.5px;color:var(--dim)" id="mockFooterNote">Illustrative example</div>
</div>
<div style="text-align:center;margin-top:28px">
<p style="color:var(--dim2);font-size:13.5px;margin-bottom:14px">%%MOCK_NOTE%%</p>
<a class="btn" href="/demo">%%MOCK_CTA%%</a>
</div>
</section>

<section id="how">
<div class="section-head">
<div class="kicker">HOW IT WORKS</div>
<h2>Two filters. Not one.</h2>
<p>Most screeners stop at the math. We add a second pass that specifically hunts for the ways a pure quant signal can fool you.</p>
</div>
<div class="steps" data-reveal>
<div class="step"><div class="num">STEP 1</div><h3>Quant scan, every day</h3><p>Real price data across all 518 S&amp;P 500 + Nasdaq-100 tickers is pulled and scored on long-term trend (200-day moving average) and pullback depth from the recent high. Only the top-scoring names — usually a dozen or two — clear the bar.</p></div>
<div class="step"><div class="num">STEP 2</div><h3>AI risk cross-check</h3><p>Every ticker that clears the quant bar gets reviewed a second time by AI, specifically for two traps: chasing a stock already near a blow-off top, or mistaking a dead-cat bounce for a real recovery.</p></div>
<div class="step"><div class="num">STEP 3</div><h3>You decide</h3><p>You get the data, the reasoning, and a plain-language risk review — never a price target, never a "buy now." What you do with it is up to you.</p></div>
</div>
</section>

<section id="proof">
<div class="section-head">
<div class="kicker">PROVEN BY THE NUMBERS</div>
<h2>We tested it against 2 years of real data. Here's what happened.</h2>
<p>Not cherry-picked winners — replayed against 2 years of real price history using the exact formula running today. See exactly what's covered below.</p>
</div>
%%PROOF_CARDS%%
%%PROOF_NOTE%%
<div class="methodology">
<h4>How this was measured — read before you trust it</h4>
<ul>
<li><b>Universe:</b> %%UNIVERSE_NOTE%%</li>
<li><b>Signal counting:</b> only a fresh crossing above the score threshold counts as one signal — a stock staying "Favorable" for a week isn't counted 7 times.</li>
<li><b>Benchmark:</b> the S&amp;P figure is its return over <b>the same windows the strategy actually traded</b>, not its average over the whole period. This matters: a buy-the-dip rule fires after selloffs, and the market's own bounce after a selloff is better than its return on a random day. Measured the loose way, the edge looks about 40% larger than it is. The honest comparison is the one shown above.</li>
<li><b>Costs:</b> every signal is charged a 0.10% round trip for spread and slippage before anything is averaged, and a trade that made less than that is not counted as a win. Taxes are not modelled.</li>
<li><b>Survivorship bias:</b> this uses today's index membership applied to the past 2 years. Stocks removed from these indices during that window (delisted, acquired, or dropped for poor performance) aren't included, which flatters results by an amount that cannot be measured without historical index membership data we don't have. Assume the real numbers are lower than these.</li>
<li><b>Overlapping windows:</b> signals fire close together, so the 30/60/90-day windows overlap heavily. The observation counts are not independent samples, and the averages carry more uncertainty than a raw "n" suggests.</li>
<li><b>Out-of-sample check:</b> %%VALIDATION_NOTE%%</li>
<li><b>Recomputed periodically, not per-visit:</b> normally about once a week, though a server restart can also trigger a one-off recompute if there's no cached result yet. Either way the universe covered doesn't change; the numbers only move because new price data came in.</li>
</ul>
</div>
</section>

<section>
<div class="section-head">
<div class="kicker">WHY THIS IS DIFFERENT</div>
<h2>Most signal services show you a highlight reel.</h2>
<p>We show you the full validation — including the losses.</p>
</div>
<div class="diff-grid" data-reveal>
<div class="diff-col against">
<h4>The usual approach</h4>
<div class="diff-item"><span class="mark">&times;</span><span>A handful of cherry-picked win screenshots, no losing trades shown</span></div>
<div class="diff-item"><span class="mark">&times;</span><span>"This stock is about to explode" — hype with no disclosed methodology</span></div>
<div class="diff-item"><span class="mark">&times;</span><span>A black-box pick with no reasoning you can check</span></div>
<div class="diff-item"><span class="mark">&times;</span><span>Pay more for "VIP" or "premium" picks other subscribers don't see</span></div>
</div>
<div class="diff-col for">
<h4>QUANTIFY</h4>
<div class="diff-item"><span class="mark">&#10003;</span><span>Full backtest published on this page — wins and losses, in-sample and out-of-sample</span></div>
<div class="diff-item"><span class="mark">&#10003;</span><span>One entry rule, plainly disclosed: long-term uptrend, pulled back 10-25% from its recent high</span></div>
<div class="diff-item"><span class="mark">&#10003;</span><span>Every pick ships with the AI's actual reasoning and the two failure modes it checked</span></div>
<div class="diff-item"><span class="mark">&#10003;</span><span>One plan. Every subscriber sees the same data, the same day.</span></div>
</div>
</div>
</section>

<section id="features">
<div class="section-head">
<div class="kicker">FEATURES</div>
<h2>Built on real data, not vibes</h2>
</div>
<div class="features" data-reveal>
<div class="feature"><div class="icon"><svg width="22" height="22" viewBox="0 0 22 22" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M4 14v4M9 8v10M14 11v7M18 5v13"/></svg></div><h4>Live market data</h4><p>Real prices and volume from the actual market — no simulated or backfilled data.</p></div>
<div class="feature"><div class="icon"><svg width="22" height="22" viewBox="0 0 22 22" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M3 5h16v10H8l-4 4v-4H3z"/></svg></div><h4>Plain-language AI review</h4><p>Every detected ticker gets a written quant review and explicit risk check, in plain English.</p></div>
<div class="feature"><div class="icon"><svg width="22" height="22" viewBox="0 0 22 22" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M3 15l5-6 4 3 7-9M19 3h-4v4"/></svg></div><h4>52-week &amp; trend context</h4><p>Distance from the 52-week high/low and 200-day trend, so you see where a stock actually sits.</p></div>
<div class="feature"><div class="icon"><svg width="22" height="22" viewBox="0 0 22 22" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M11 3a5 5 0 00-5 5v3l-2 4h14l-2-4V8a5 5 0 00-5-5zM9 18a2 2 0 004 0"/></svg></div><h4>Price alerts &amp; news</h4><p>Set a target price and get emailed when it's hit, with live headlines next to the chart.</p></div>
</div>
</section>

<section>
<div class="final-wrap">
<h2>Know which one it is — before you buy, not after.</h2>
<p>See today's list, free for 7 days. Then $9.99/month. Takes under a minute to sign up.</p>
<a class="btn" href="/signup">Get Started Free</a>
</div>
<div class="disclaimer">
<div class="kicker">IMPORTANT</div>
<p>QUANTIFY is an informational and educational tool. Nothing on this site is investment advice, a recommendation, or a solicitation to buy or sell any security. Scores, badges, and AI commentary reflect a mathematical model's output on the data available at the time and can be wrong. Markets involve risk, including loss of principal. Do your own research and consult a licensed financial advisor before making investment decisions.</p>
</div>
</section>

<footer>
QUANTIFY. — informational and educational only, not investment advice.<br>
<a href="/login">Log in</a> · <a href="/signup">Sign up</a> · <a href="/pricing">Pricing</a> · <a href="/faq">FAQ</a> · <a href="/about">About</a> · <a href="/terms">Terms</a> · <a href="/privacy">Privacy</a>
</footer>
<div class="sticky-cta" id="stickyCta">
<span>%%STICKY_NOTE%%</span>
<a class="btn" href="/demo">%%STICKY_CTA%%</a>
</div>
<script>
if('IntersectionObserver' in window){
  const io=new IntersectionObserver((entries)=>{entries.forEach(e=>{if(e.isIntersecting){e.target.classList.add('is-visible');io.unobserve(e.target)}})},{threshold:0.05,rootMargin:'0px 0px 100px 0px'});
  document.querySelectorAll('[data-reveal]').forEach(el=>io.observe(el));
}else{
  document.querySelectorAll('[data-reveal]').forEach(el=>el.classList.add('is-visible'));
}
// Fast/flick scrolling (or any observer edge case) should never leave content stuck at
// opacity:0 -- this is a marketing page, a permanently blank section is worse than no
// animation at all. Belt-and-suspenders: force-reveal anything still hidden shortly after load.
setTimeout(()=>document.querySelectorAll('[data-reveal]:not(.is-visible)').forEach(el=>el.classList.add('is-visible')),2500);

function mockSparkPoints(arr){
  if(!arr||arr.length<2)return null;
  const w=300,h=70;
  const min=Math.min(...arr),max=Math.max(...arr),range=(max-min)||1;
  return arr.map((v,i)=>`${(i/(arr.length-1)*w).toFixed(1)},${(h-((v-min)/range*(h-10))-5).toFixed(1)}`).join(' ');
}
function mockBadgeClass(v){return v==='Favorable'?'badge-ok':v==='Risk'?'badge-danger':'badge-warn'}
async function loadPublicPreview(){
  try{
    const r=await fetch('/api/public-preview');
    const d=await r.json();
    if(!d.tickers||!d.tickers.length)return; // keep the illustrative fallback as-is
    document.getElementById('mockH').innerHTML=`<span class="live-dot"></span>MARKET SCANNER · ${d.detected_count} detected / ${d.universe_count}`;
    document.getElementById('mockList').innerHTML=d.tickers.map(t=>`<div class="mock-row"><b>${t.ticker}</b><span>${t.price} <span class="badge ${mockBadgeClass(t.timing_verdict)}">${t.timing_verdict}</span></span></div>`).join('');
    const top=d.tickers[0];
    document.getElementById('mockChartTicker').textContent=`${top.ticker} · TECHNICAL CHART`;
    const pts=mockSparkPoints(top.sparkline);
    if(pts){
      const color=top.change_pct>=0?'#0e8a5f':'#c8402c';
      document.getElementById('mockChartSvg').innerHTML=`<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1.5"/>`;
    }
    document.getElementById('mockRsiMacd').textContent=`${top.rsi} / ${top.macd}`;
    if(top.pct_from_52w_high!=null&&top.pct_from_52w_low!=null)document.getElementById('mock52w').textContent=`${top.pct_from_52w_high}% / ${top.pct_from_52w_low}%`;
    const badgeEl=document.getElementById('mockAiBadge');
    badgeEl.textContent=top.timing_verdict;badgeEl.className='badge '+mockBadgeClass(top.timing_verdict);
    if(top.quant_review)document.getElementById('mockQuantReview').textContent=top.quant_review;
    if(top.risk_review)document.getElementById('mockRiskReview').textContent=top.risk_review;
    const noteEl=document.getElementById('mockFooterNote');
    if(noteEl)noteEl.textContent=`Live — today's actual scan, as of ${d.scan_date}`;
  }catch(e){console.error('Public preview load failed',e)}
}
loadPublicPreview();

const stickyCta=document.getElementById('stickyCta');
const heroSection=document.querySelector('.hero');
if(stickyCta&&heroSection){
  // Belt-and-suspenders, deliberately: a scroll listener alone can miss the settled
  // state after a smooth/programmatic scroll if no further scroll event fires
  // afterward, and IntersectionObserver alone depends on the browser's own visibility
  // tracking actually running. Both are standard and either one is normally enough --
  // running both costs nothing (they converge on the same answer) and removes any
  // single point of failure for a purely cosmetic UI element.
  const updateSticky=()=>stickyCta.classList.toggle('show',heroSection.getBoundingClientRect().bottom<0);
  window.addEventListener('scroll',updateSticky,{passive:true});
  window.addEventListener('resize',updateSticky,{passive:true});
  if('IntersectionObserver' in window){
    new IntersectionObserver((entries)=>{entries.forEach(e=>stickyCta.classList.toggle('show',!e.isIntersecting))}).observe(heroSection);
  }
}
</script>
</body></html>"""


# -----------------------------------------------------------------------------
# Auth pages/endpoints
# -----------------------------------------------------------------------------
BASE_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{background:#ffffff;color:#3a4440;font:17px/1.65 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif}
.authwrap{display:flex;min-height:100vh}
.authbrand{flex:1;background:radial-gradient(ellipse 900px 600px at 30% 20%,rgba(14,138,95,.08),transparent 60%) #f5f7f6;padding:64px;display:flex;flex-direction:column;justify-content:center;border-right:1px solid #e2e6e3}
.brand{font-weight:700;font-size:21px;letter-spacing:.2px;color:#12201a;text-decoration:none;display:inline-block;margin-bottom:48px}
.brand span{color:#0e8a5f}
.authbrand h1{font-size:40px;font-weight:800;line-height:1.2;letter-spacing:-.5px;color:#12201a;max-width:460px;margin-bottom:18px}
.authbrand>p{color:#525f59;font-size:17px;max-width:420px;line-height:1.7}
.points{margin-top:40px;display:flex;flex-direction:column;gap:16px}
.point{color:#3a4440;font-size:15px}
.point b{color:#0e8a5f}
.authform{flex:1;display:flex;align-items:center;justify-content:center;padding:40px 24px}
.card{width:100%;max-width:420px;background:#ffffff;border:1px solid #e2e6e3;padding:40px 36px;border-radius:16px;box-shadow:0 20px 50px -25px rgba(18,32,26,.18)}
h2{color:#12201a;text-align:center;font-size:24px;font-weight:800;margin-bottom:8px}
.subtitle{text-align:center;color:#525f59;font-size:15px;margin-bottom:26px;line-height:1.6}
label{display:block;font-size:13.5px;font-weight:600;color:#525f59;margin:16px 0 7px}
input{width:100%;background:#f5f7f6;border:1.5px solid #dfe4e1;color:#12201a;padding:13px 15px;font-size:16px;border-radius:9px;transition:border-color .15s;font-family:inherit}
input:focus{outline:none;border-color:#0e8a5f}
button{width:100%;margin-top:22px;background:#0e8a5f;color:#ffffff;border:none;padding:14px;font-size:16px;font-weight:700;border-radius:9px;cursor:pointer;transition:opacity .15s,transform .1s;font-family:inherit}
button:hover{opacity:.9}
button:active{transform:scale(.98)}
.google-btn{display:flex;align-items:center;justify-content:center;gap:10px;width:100%;background:#fff;color:#1f1f1f;border:1px solid #dcdcdc;padding:13px;border-radius:9px;font-size:16px;font-weight:600;text-decoration:none;transition:background .15s}
.google-btn:hover{background:#f5f5f5}
.divider{display:flex;align-items:center;gap:12px;margin:24px 0;color:#8891a0;font-size:13.5px}
.divider::before,.divider::after{content:'';flex:1;height:1px;background:#e2e6e3}
.links{display:flex;justify-content:space-between;margin-top:24px;font-size:15px}
.links a{color:#1f5fbf;text-decoration:none}
.links a:hover{text-decoration:underline}
.error{color:#c8402c;font-size:14.5px;min-height:0;margin-bottom:8px}
.ok{color:#0e8a5f;font-size:14.5px;margin-bottom:8px}
.hint{font-size:13.5px;color:#77837e;margin-top:-8px;margin-bottom:18px}
details{margin-top:22px;font-size:15px;color:#525f59;text-align:center}
details summary{cursor:pointer;color:#1f5fbf}
details form{margin-top:12px;text-align:left}
details input{margin-bottom:10px}
details button{margin-top:6px;padding:12px}
@media(max-width:860px){
  .authwrap{flex-direction:column}
  .authbrand{border-right:none;border-bottom:1px solid #e2e6e3;padding:40px 24px;flex:none}
  .authbrand h1{font-size:28px}
  .points{display:none}
  .authform{flex:none;padding:40px 24px}
}
/* On phones the marketing panel used to eat the whole first screen, pushing the
   actual login fields below the fold -- trim it so the form is what you land on. */
@media(max-width:560px){
  .authbrand{padding:18px 20px 16px}
  .brand{margin-bottom:14px;font-size:19px}
  .authbrand h1{font-size:21px;margin-bottom:0}
  .authbrand>p{display:none}
  .authform{padding:20px 16px 40px}
  .card{padding:26px 20px}
  h2{font-size:22px}
  .subtitle{margin-bottom:20px;font-size:14.5px}
}
"""

AUTH_BRAND_HTML = """<div class="authbrand">
<a class="brand" href="/">QUANTIFY<span>.</span></a>
<h1>Quant-detected stocks,<br>AI risk-checked.</h1>
<p>A daily scan of the S&amp;P 500 and Nasdaq-100, cross-checked by AI for blow-off-top and dead-cat-bounce risk before it ever reaches your screen.</p>
<div class="points">
<div class="point"><b>&#9670;</b> Live market data, never simulated</div>
<div class="point"><b>&#9670;</b> Plain-language AI risk review on every pick</div>
<div class="point"><b>&#9670;</b> 7-day free trial, then $9.99/month</div>
</div>
</div>"""

GOOGLE_ICON_SVG = '<svg width="18" height="18" viewBox="0 0 18 18"><path fill="#4285F4" d="M17.64 9.2c0-.64-.06-1.25-.16-1.84H9v3.48h4.84a4.14 4.14 0 01-1.8 2.72v2.26h2.92c1.7-1.57 2.68-3.88 2.68-6.62z"/><path fill="#34A853" d="M9 18c2.43 0 4.47-.8 5.96-2.18l-2.92-2.26c-.81.54-1.84.86-3.04.86-2.34 0-4.32-1.58-5.03-3.7H.96v2.33A9 9 0 009 18z"/><path fill="#FBBC05" d="M3.97 10.72A5.4 5.4 0 013.68 9c0-.6.1-1.18.29-1.72V4.95H.96A9 9 0 000 9c0 1.45.35 2.83.96 4.05l3.01-2.33z"/><path fill="#EA4335" d="M9 3.58c1.32 0 2.5.45 3.44 1.35l2.58-2.58C13.46.89 11.43 0 9 0A9 9 0 00.96 4.95l3.01 2.33C4.68 5.16 6.66 3.58 9 3.58z"/></svg>'


def render_auth_page(title: str, form_html: str, path: str = "",
                     description: str = "Start your 7-day free trial of QUANTIFY — a daily quant scan of the S&P 500 and Nasdaq-100 with an AI risk review on every hit.",
                     lang: str = "en") -> HTMLResponse:
    og = og_head(title, description, path) + (hreflang_links(path) if path else "")
    toggle = lang_toggle_html(lang, path or "/login")
    return HTMLResponse(translate_public(f'''<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>{og}<style>{BASE_CSS}
.lang-toggle{{display:inline-block;padding:6px 12px;border:1px solid var(--border);border-radius:999px;color:var(--dim2,#525f59);font-size:13.5px;font-weight:700;text-decoration:none;line-height:1;white-space:nowrap}}
.lang-toggle:hover{{border-color:var(--green);color:var(--green)}}
.lang-toggle.app{{border-color:var(--sb-border,#232b2f);color:var(--sb-text,#9aa7ac);padding:5px 10px;font-size:12.5px}}
.lang-toggle.app:hover{{border-color:var(--green);color:var(--green)}}
</style></head><body><div class="authwrap">{AUTH_BRAND_HTML}<div class="authform"><div style="text-align:right;margin-bottom:10px">{toggle}</div>{form_html}</div></div></body></html>''', lang))


def _render_validation_note(results: dict) -> str:
    val = results.get("validation") or {}
    lines = []
    verdicts = []
    for h in (30, 60, 90):
        in_s = val.get(f"in_sample_{h}d")
        out_s = val.get(f"out_of_sample_{h}d")
        if not in_s or not out_s:
            continue
        in_avg, out_avg = in_s["avg_return_pct"], out_s["avg_return_pct"]
        in_wr, out_wr = in_s["win_rate_pct"], out_s["win_rate_pct"]
        avg_held = out_avg >= in_avg * 0.7
        wr_held = out_wr >= in_wr - 8  # allow some natural drift, but not a large drop
        if avg_held and wr_held:
            v = "held up"
        elif avg_held:
            v = "return held, win rate dropped"
        elif wr_held:
            v = "win rate held, return dropped"
        else:
            v = "weakened on both"
        verdicts.append(v)
        lines.append(f"{h}-day — in-sample: {'+' if in_avg>=0 else ''}{in_avg}% avg, {in_wr}% win rate (n={in_s['n']}); "
                      f"out-of-sample: {'+' if out_avg>=0 else ''}{out_avg}% avg, {out_wr}% win rate (n={out_s['n']}) — {v}.")
    if not lines:
        return "not enough data yet to report an in-sample vs out-of-sample split."
    if all(v == "held up" for v in verdicts):
        headline = "it held up across all three horizons"
    elif all(v == "weakened on both" for v in verdicts):
        headline = "it weakened across all three horizons — treat the full-period numbers above with real caution"
    else:
        headline = "it's mixed across horizons, not uniformly good or bad — see all three below rather than anchoring on one"
    detail = " ".join(lines)
    return (f"the scoring weights were tuned on the first 70% of this window and separately measured on the "
            f"untouched last 30%, at every horizon this page reports (not just the one that looks best): {headline}. {detail}")


def _render_universe_note(tickers_sampled) -> str:
    # Full-universe coverage only actually holds once the cache has been recomputed
    # since BACKTEST_SAMPLE_SIZE was raised -- until then this still honestly
    # describes whatever's in the currently-displayed numbers, not the code's intent.
    if isinstance(tickers_sampled, int) and tickers_sampled >= 500:
        return "every current S&amp;P 500 + Nasdaq-100 constituent (~518 tickers) — no sampling, nothing hand-picked or left out."
    return (f"a sample of {tickers_sampled} of the ~518 current S&amp;P 500 + Nasdaq-100 constituents. "
            f"(The site is moving to covering the full universe with no sampling — this note updates automatically once that recompute finishes.)")


def _render_proof_section() -> tuple[str, str, str, str]:
    results = BACKTEST_CACHE.get("results")
    if not results:
        placeholder = '<p class="proof-note">Backtest is computing on the server — check back shortly.</p>'
        return "", placeholder, "not available yet — check back after the first computation finishes.", "not available yet."
    horizons = results["horizons"]
    cards_html = ['<div class="proof-grid" data-reveal>']
    for h in ("30", "60", "90"):
        v = horizons.get(h, {})
        strat = v.get("strategy") or {}
        # Compare against the S&P over the SAME windows the strategy traded, not its
        # unconditional average. A pullback rule fires after dips, and the market's own
        # return following a dip is higher than its return on a random day -- measuring
        # against the random day overstated the edge by roughly 40%.
        matched = v.get("benchmark_matched") or {}
        bench = matched or (v.get("benchmark") or {})
        bench_label = ("the S&amp;P 500 over the same windows" if matched
                       else "the S&amp;P 500 over the whole period")
        avg = strat.get("avg_return_pct")
        bench_avg = bench.get("avg_return_pct")
        if avg is None:
            continue
        sign = "+" if avg >= 0 else ""
        bench_sign = "+" if (bench_avg or 0) >= 0 else ""
        avg_win = strat.get("avg_win_pct")
        avg_loss = strat.get("avg_loss_pct")
        worst = strat.get("worst_pct")
        risk_row = ""
        if avg_win is not None and avg_loss is not None:
            risk_row = (f'<div class="risk-row">When right: +{avg_win}% avg &middot; When wrong: {avg_loss}% avg</div>'
                        + (f'<div class="worst-case">Worst single case: {worst}%</div>' if worst is not None else ""))
        cards_html.append(
            f'<div class="proof-card"><div class="horizon">{h}-Day Forward Return</div>'
            f'<div class="num">{sign}{avg}%</div>'
            f'<div class="compare">vs {bench_sign}{bench_avg}% for {bench_label} &middot; {strat.get("win_rate_pct","-")}% win rate</div>'
            f'{risk_row}</div>'
        )
    cards_html.append("</div>")
    computed_at = BACKTEST_CACHE.get("computed_at")
    computed_str = datetime.fromtimestamp(computed_at).strftime("%B %d, %Y") if computed_at else "recently"
    tickers_sampled = results.get("tickers_sampled")
    note = (f'<p class="proof-note">Based on {results.get("signal_count","-")} historical signals across '
            f'{tickers_sampled if tickers_sampled is not None else "-"} tickers. Last computed: {computed_str}.<br>'
            f'Past performance does not guarantee future results. This is historical, informational analysis — '
            f'not a forecast, and not investment advice.</p>')
    return "".join(cards_html), note, _render_validation_note(results), _render_universe_note(tickers_sampled)



# Landing-page copy is filled server-side rather than by the /api/public-preview fetch so
# that a phone sees the actual numbers in the first paint -- no layout shift, and nothing
# above the fold depends on JS. Cached because / is the highest-traffic route and the
# underlying scan only changes four times a day.
_LANDING_COUNT_CACHE: dict = {"at": 0.0, "detected": None, "scan_date": None}
_LANDING_COUNT_TTL = 300.0


def _landing_scan_counts() -> tuple[int | None, str | None]:
    now = time.time()
    if now - _LANDING_COUNT_CACHE["at"] < _LANDING_COUNT_TTL:
        return _LANDING_COUNT_CACHE["detected"], _LANDING_COUNT_CACHE["scan_date"]
    detected = scan_date = None
    try:
        conn = db()
        # The landing headline counts the same list the demo shows, so it has to agree
        # with it -- quoting a partial day here would advertise 11 detected stocks and
        # then show 53 on the next page.
        scan_date = display_scan_date(conn)
        if scan_date:
            detected = conn.execute(
                "SELECT COUNT(*) FROM daily_scans WHERE scan_date=? AND quant_pass=1 AND timing_score IS NOT NULL",
                (scan_date,),
            ).fetchone()[0]
        conn.close()
    except Exception:
        # The landing page must render even if the DB is momentarily unavailable; the
        # copy below falls back to wording that needs no numbers.
        detected = scan_date = None
    _LANDING_COUNT_CACHE.update({"at": now, "detected": detected, "scan_date": scan_date})
    return detected, scan_date


def _landing_cta_copy(lang: str = "en") -> dict:
    detected, scan_date = _landing_scan_counts()
    # UNIVERSE is filled by a startup task and is empty until it finishes, so every line
    # that quotes it needs a version that reads correctly without it.
    universe = len(UNIVERSE)
    out_of = f" OUT OF {universe}" if universe else ""
    of_universe = f" of {universe}" if universe else ""
    if detected:
        # A quiet day can legitimately flag one or two names; the copy has to stay
        # grammatical and honest at those counts too.
        preview_n = min(3, detected)
        ko = lang == "ko"
        if detected > preview_n:
            mock_note = (f"오늘 스캔이 찾아낸 {detected}개 중 {preview_n}개입니다. 장 시작과 마감 전후로 하루 네 번 다시 돕니다."
                         if ko else
                         f"That's {preview_n} of the {detected} the scan flagged today. It reruns four times "
                         f"a day, around the open and the close.")
            mock_cta = (f"{detected}개 전부 보기 &mdash; 가입 불필요" if ko
                        else f"See all {detected} &mdash; no signup")
        else:
            mock_note = ("오늘 전체 목록입니다. 스캔은 장 시작과 마감 전후로 하루 네 번 돕니다." if ko else
                         "That's today's full list. The scan reruns four times a day, around the open "
                         "and the close.")
            mock_cta = ("실시간 스캔 열기 &mdash; 가입 불필요" if ko else "Open the live scan &mdash; no signup")
        if ko:
            noun_ko = "개 종목"
            return {
                "%%EYEBROW%%": f"오늘의 스캔 &middot; {universe}개 중 {detected}개 감지" if universe else f"오늘의 스캔 &middot; {detected}개 감지",
                "%%HERO_CTA%%": f"오늘 감지된 {detected}{noun_ko} 보기",
                "%%MOCK_NOTE%%": mock_note,
                "%%MOCK_CTA%%": mock_cta,
                "%%STICKY_NOTE%%": f"오늘의 스캔 &middot; <b>{detected}개 감지</b>",
                "%%STICKY_CTA%%": "무료로 보기",
            }
        noun = "stock" if detected == 1 else "stocks"
        return {
            "%%EYEBROW%%": f"TODAY'S SCAN &middot; {detected} DETECTED{out_of}",
            "%%HERO_CTA%%": f"See today's {detected} detected {noun}",
            "%%MOCK_NOTE%%": mock_note,
            "%%MOCK_CTA%%": mock_cta,
            "%%STICKY_NOTE%%": f"Today's scan &middot; <b>{detected}{of_universe} detected</b>",
            "%%STICKY_CTA%%": "See them free",
        }
    if lang == "ko":
        return {
            "%%EYEBROW%%": (f"매일 도는 퀀트 스캔 &middot; {universe}개 종목" if universe
                            else "매일 도는 퀀트 스캔 &middot; S&amp;P 500 + 나스닥100"),
            "%%HERO_CTA%%": "실시간 스캔 보기 &mdash; 가입 불필요",
            "%%MOCK_NOTE%%": "3개 미리보기입니다. 스캔은 장 시작과 마감 전후로 하루 네 번 돕니다.",
            "%%MOCK_CTA%%": "전체 목록 보기 &mdash; 가입 불필요",
            "%%STICKY_NOTE%%": "오늘의 스캔 &middot; <b>무료로 열람 가능</b>",
            "%%STICKY_CTA%%": "무료로 보기",
        }
    return {
        "%%EYEBROW%%": (f"DAILY QUANT SCAN &middot; {universe} TICKERS" if universe
                        else "DAILY QUANT SCAN &middot; S&amp;P 500 + NASDAQ-100"),
        "%%HERO_CTA%%": "See the live scan &mdash; no signup",
        "%%MOCK_NOTE%%": "That's a preview of 3. The scan reruns four times a day, around the open and the close.",
        "%%MOCK_CTA%%": "See the full list &mdash; no signup",
        "%%STICKY_NOTE%%": "Today's scan &middot; <b>free to look</b>",
        "%%STICKY_CTA%%": "See it free",
    }


def _safe_next(raw: Optional[str]) -> str:
    """Only ever redirect back to a path on this site. A bare "/" prefix is not enough:
    "//evil.com" and "/\\evil.com" are both read as protocol-relative URLs by browsers."""
    nxt = (raw or "/").strip()
    if not nxt.startswith("/") or nxt.startswith("//") or nxt.startswith("/\\"):
        return "/"
    return nxt


@app.get("/set-language")
async def set_language(request: Request, lang: str = "en", next: str = "/"):
    lang = (lang or "").strip().lower()
    if lang not in LANGUAGE_NAMES:
        lang = "en"
    response = RedirectResponse(_safe_next(next), status_code=303)
    response.set_cookie(LANG_COOKIE, lang, max_age=LANG_COOKIE_MAX_AGE, httponly=False, samesite="lax")
    # Keep the account in step so Settings never disagrees with what the reader is
    # looking at, and so the choice follows them to another browser.
    user = get_logged_in_user(request)
    if user:
        try:
            conn = db()
            conn.execute("UPDATE users SET pref_language=? WHERE email=?", (lang, user))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[Error: {type(e).__name__}] language preference save failed for {user}: {e}")
    return response


def lang_toggle_html(current: str, path: str = "/", style: str = "public") -> str:
    """The header switcher. Deliberately shows the language you would GET, not the one
    you are on -- a control labelled with the current state reads as a status display and
    people don't click it."""
    other = "ko" if current == "en" else "en"
    label = "한국어" if other == "ko" else "English"
    href = f"/set-language?lang={other}&next={urllib.parse.quote(path, safe='/?=&')}"
    cls = "lang-toggle" if style == "public" else "lang-toggle app"
    aria = "Switch to Korean" if other == "ko" else "Switch to English"
    return f'<a class="{cls}" href="{href}" hreflang="{other}" aria-label="{aria}" rel="nofollow">{label}</a>'


def hreflang_links(path: str) -> str:
    base = f"https://quantify.trading{path}"
    sep = "&" if "?" in path else "?"
    return (f'<link rel="alternate" hreflang="en" href="{base}">'
            f'<link rel="alternate" hreflang="ko" href="{base}{sep}lang=ko">'
            f'<link rel="alternate" hreflang="x-default" href="{base}">')


@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    if get_logged_in_user(request):
        return RedirectResponse("/terminal", status_code=303)
    lang = resolve_lang(request)
    cards, note, validation_note, universe_note = _render_proof_section()
    html = (LANDING_HTML.replace("%%PROOF_CARDS%%", cards).replace("%%PROOF_NOTE%%", note)
            .replace("%%VALIDATION_NOTE%%", validation_note).replace("%%UNIVERSE_NOTE%%", universe_note))
    for placeholder, value in _landing_cta_copy(lang).items():
        html = html.replace(placeholder, value)
    html = (html.replace("%%OG_IMAGE%%", _og_image_tags())
                .replace("%%LANG_TOGGLE%%", lang_toggle_html(lang, "/"))
                .replace("%%HREFLANG%%", hreflang_links("/"))
                .replace("%%LANG%%", lang))
    return HTMLResponse(translate_public(html, lang),
                        # Content differs by Accept-Language for a first-time visitor, so
                        # any shared cache must key on it or one reader's language gets
                        # served to another.
                        headers={"Vary": "Accept-Language, Cookie"})


_ROBOTS_GATED_PATHS = (
    "Allow: /\n"
    "Allow: /api/public-preview\n"
    "Disallow: /terminal\n"
    "Disallow: /market\n"
    "Disallow: /watchlist\n"
    "Disallow: /backtest\n"
    "Disallow: /portfolio\n"
    "Disallow: /settings\n"
    "Disallow: /accept-disclaimer\n"
    "Disallow: /api/\n\n"
)
# Same allow/disallow rules as the wildcard block below, spelled out per-bot on purpose:
# this file is sometimes read by a person (or an AI agent) checking specifically whether
# they're blocked, and an explicit "yes you're allowed" line answers that directly instead
# of relying on them to infer it from the wildcard.
_AI_BOT_USER_AGENTS = ("GPTBot", "ChatGPT-User", "ClaudeBot", "Claude-User", "PerplexityBot", "Google-Extended", "CCBot")


@app.get("/robots.txt", response_class=PlainTextResponse)
async def robots_txt(request: Request):
    base = str(request.base_url).rstrip("/")
    bot_blocks = "".join(f"User-agent: {ua}\n{_ROBOTS_GATED_PATHS}" for ua in _AI_BOT_USER_AGENTS)
    return (
        f"User-agent: *\n{_ROBOTS_GATED_PATHS}"
        f"{bot_blocks}"
        f"Sitemap: {base}/sitemap.xml\n"
    )


@app.get("/llms.txt", response_class=PlainTextResponse)
async def llms_txt():
    return (
        "# QUANTIFY\n\n"
        "> Daily quant scan of the S&P 500 and Nasdaq-100 that flags stocks pulling back "
        "inside a real uptrend, then AI-checked for blow-off-top and dead-cat-bounce risk. "
        "Informational and educational only -- never investment advice, never a buy or sell "
        "signal.\n\n"
        "QUANTIFY is a subscription web app: $9.99/month after a 7-day free trial. One "
        "validated strategy, published openly with in-sample and out-of-sample backtest "
        "results -- no cherry-picked wins, no personalized picks, no hidden paywalled tiers.\n\n"
        "## Key pages\n"
        "- [Home](https://quantify.trading/): overview, a live example of today's scan, and "
        "the published backtest results\n"
        "- [Pricing](https://quantify.trading/pricing): $9.99/month, 7-day free trial, one plan\n"
        "- [FAQ](https://quantify.trading/faq): strategy explanation, badge definitions, data "
        "update frequency\n"
        "- [About](https://quantify.trading/about): what QUANTIFY is and isn't\n"
        "- [Terms](https://quantify.trading/terms) / [Privacy](https://quantify.trading/privacy)\n\n"
        "## Data for agents\n"
        "- [Public preview API](https://quantify.trading/api/public-preview): unauthenticated "
        "JSON endpoint returning today's top detected tickers (price, RSI, MACD, 52-week "
        "range, AI-written quant/risk review, timing verdict). No API key required.\n\n"
        "## Important\n"
        "QUANTIFY is not a licensed investment adviser or broker-dealer. Scores and AI "
        "commentary are a model's output on available data and can be wrong. Nothing on this "
        "site is a recommendation to buy or sell any security.\n"
    )


@app.get("/sitemap.xml")
async def sitemap_xml(request: Request):
    base = str(request.base_url).rstrip("/")
    urls = ["/", "/login", "/signup", "/pricing", "/faq", "/about", "/demo", "/stocks", "/record", "/terms", "/privacy"]
    try:
        conn = db()
        latest = conn.execute("SELECT MAX(scan_date) FROM daily_scans").fetchone()[0]
        if latest:
            urls += [f"/stock/{r[0]}" for r in conn.execute(
                "SELECT DISTINCT ticker FROM daily_scans WHERE scan_date=? ORDER BY ticker", (latest,)
            ).fetchall()]
        conn.close()
    except Exception as exc:
        print(f"[Error: {type(exc).__name__}] sitemap ticker list failed: {exc}")
    items = "".join(f"<url><loc>{base}{u}</loc></url>" for u in urls)
    xml = f'<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{items}</urlset>'
    return Response(content=xml, media_type="application/xml")


MARKETING_CSS = """
:root{--bg:#ffffff;--bg-alt:#f6f8f7;--panel:#ffffff;--panel2:#f6f8f7;--border:#e1e6e3;--text:#2a3330;--head:#12201a;--dim:#6b7873;--dim2:#4d5854;--green:#0e8a5f;--green-bright:#17a374;--green-soft:#e6f5ee;--red:#c8402c;--orange:#a8660a}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font:17px/1.7 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}
a{color:var(--green);text-decoration:underline}
a:hover{color:var(--green-bright)}
.wrap{max-width:1080px;margin:0 auto;padding:0 28px}
header{border-bottom:1px solid var(--border);position:sticky;top:0;background:rgba(255,255,255,.96);backdrop-filter:blur(6px);z-index:10}
.nav{display:flex;align-items:center;justify-content:space-between;padding:18px 28px;max-width:1080px;margin:0 auto;flex-wrap:wrap;gap:14px}
.logo{color:var(--head);font-weight:700;font-size:20px;letter-spacing:-.2px;text-decoration:none}
.logo span{color:var(--green)}
.navlinks{display:flex;align-items:center;gap:26px;font-size:15px;flex-wrap:wrap}
.navlinks a.muted{color:var(--dim2);text-decoration:none;font-weight:500}
.navlinks a.muted:hover{color:var(--head)}
.btn{display:inline-block;padding:13px 24px;border:1px solid var(--green);color:#ffffff;background:var(--green);font:15px/1 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-weight:700;cursor:pointer;border-radius:8px;text-decoration:none}
.btn:hover{background:var(--green-bright);border-color:var(--green-bright);color:#ffffff}
main{padding:64px 28px 90px}
main h1{color:var(--head);font-size:46px;line-height:1.2;letter-spacing:-1px;margin-bottom:20px;font-weight:800;text-wrap:balance}
main h2{color:var(--head);font-size:26px;margin:44px 0 14px;font-weight:700;letter-spacing:-.3px}
main p,main li{font-size:17px;line-height:1.8;margin-bottom:16px;color:var(--text)}
main ul{padding-left:22px;margin-bottom:16px}
.eyebrow{display:inline-block;font-size:13px;font-weight:700;color:var(--orange);border:1px solid #ecdcb8;background:#fbf3e4;padding:6px 14px;border-radius:20px;letter-spacing:.4px;margin-bottom:26px}
.sublead{color:var(--dim2);font-size:19px;line-height:1.7;margin-bottom:34px;max-width:640px}
.price-card{max-width:440px;margin:14px 0 34px;border:1px solid var(--border);background:var(--panel2);padding:34px;border-radius:14px;box-shadow:0 1px 3px rgba(18,32,26,.05)}
.price-card .amount{color:var(--head);font-size:48px;font-weight:800}
.price-card .amount span{font-size:16px;color:var(--dim);font-weight:500}
.price-card ul{list-style:none;padding:0;text-align:left;margin:22px 0}
.price-card ul li{padding:8px 0;font-size:16px;margin-bottom:0}
.price-card ul li:before{content:"\\2713  ";color:var(--green);font-weight:700}
.faq-item{border-bottom:1px solid var(--border);padding:24px 0}
.faq-item h2{margin-top:0;font-size:19px}
.disclaimer{border:1px solid var(--border);background:var(--bg-alt);border-radius:10px;padding:18px 22px;margin-top:34px;font-size:14.5px;line-height:1.7;color:var(--dim2)}
footer{border-top:1px solid var(--border);padding:36px 28px;text-align:center;color:var(--dim);font-size:14px}
footer a{color:var(--dim2);text-decoration:underline}
@media(max-width:640px){main h1{font-size:32px}main h2{font-size:22px}.nav{justify-content:center}main{padding:44px 20px 70px}}
"""

_MARKETING_NAV = ('<a class="muted" href="/pricing">Pricing</a>'
                   '<a class="muted" href="/faq">FAQ</a>'
                   '<a class="muted" href="/about">About</a>'
                   '<a class="muted" href="/login">Log in</a>'
                   '<a class="btn" href="/signup">Get Started Free</a>')



# -----------------------------------------------------------------------------
# Public-page Korean copy
# -----------------------------------------------------------------------------
# Public pages had no language mechanism at all -- a Korean visitor arriving from
# Instagram met the whole funnel in English and could only switch after signing up and
# finding Settings. These are ordered longest-first at apply time so a long sentence is
# never broken by an earlier replacement of a phrase inside it.
PUBLIC_KO: dict[str, str] = {
    # Short keys are tag-bounded (">Word<") on purpose. A bare "or " or "/month" as a key
    # will happily rewrite the middle of an unrelated English sentence -- that produced
    # "acquired 또는 dropped" and "$9.99/월" on the first pass. Anything not tag-bounded
    # below is a full sentence, long enough to be unique on its own.

    # --- nav / chrome
    ">How it works<": ">작동 방식<",
    ">Results<": ">성과<",
    ">Features<": ">기능<",
    ">Pricing<": ">가격<",
    ">FAQ<": ">자주 묻는 질문<",
    ">Log in<": ">로그인<",
    ">Sign up<": ">회원가입<",
    ">About<": ">소개<",
    ">Terms<": ">이용약관<",
    ">Privacy<": ">개인정보처리방침<",
    ">Start free trial<": ">무료 체험 시작<",
    ">Get Started Free<": ">무료로 시작하기<",
    ">Email<": ">이메일<",
    ">Password<": ">비밀번호<",
    ">Create account<": ">계정 만들기<",
    ">Create an account<": ">계정 만들기<",
    ">Forgot password?<": ">비밀번호를 잊으셨나요?<",
    ">Already have an account? Log in<": ">이미 계정이 있으신가요? 로그인<",
    ">Log in<": ">로그인<",

    # --- landing hero
    "Is this dip worth buying?": "이 하락, 사도 되는 걸까?",
    "Or is it a <span class=\"hl\">falling knife</span>?": "아니면 <span class=\"hl\">떨어지는 칼날</span>일까?",
    "A daily quant scan of the S&amp;P 500 and Nasdaq-100, with an AI second pass that checks every hit for blow-off-top and dead-cat-bounce risk.":
        "S&amp;P 500과 나스닥100을 매일 퀀트로 스캔하고, 통과한 종목마다 AI가 급등 후 고점과 데드캣 바운스 위험을 한 번 더 점검합니다.",
    "Demo needs no signup &middot; trial is 7 days free, no credit card.":
        "데모는 가입 없이 볼 수 있습니다 &middot; 체험은 7일 무료, 카드 등록 없음.",
    ">Illustrative example<": ">예시 화면<",
    ">AI QUANT REPORT <": ">AI 퀀트 리포트 <",
    ">QUANT REVIEW<": ">퀀트 리뷰<",
    ">RISK REVIEW<": ">리스크 리뷰<",
    "Not extended near the high, well above its 52-week low — low blow-off-top and dead-cat-bounce risk.":
        "고점 부근까지 과열되지 않았고 52주 저점보다 충분히 높습니다 — 급등 후 고점 및 데드캣 바운스 위험이 낮습니다.",

    # --- how it works
    ">HOW IT WORKS<": ">작동 방식<",
    ">Two filters. Not one.<": ">필터는 하나가 아니라 둘입니다.<",
    "Most screeners stop at the math. We add a second pass that specifically hunts for the ways a pure quant signal can fool you.":
        "대부분의 스크리너는 계산에서 멈춥니다. QUANTIFY는 순수 퀀트 신호가 사람을 속이는 방식만 골라 찾아내는 2차 검증을 더합니다.",
    ">STEP 1<": ">1단계<", ">STEP 2<": ">2단계<", ">STEP 3<": ">3단계<",
    ">Quant scan, every day<": ">매일 도는 퀀트 스캔<",
    ">AI risk cross-check<": ">AI 리스크 교차 검증<",
    ">You decide<": ">판단은 당신이<",
    "Real price data across all 518 S&amp;P 500 + Nasdaq-100 tickers is pulled and scored on long-term trend (200-day moving average) and pullback depth from the recent high. Only the top-scoring names — usually a dozen or two — clear the bar.":
        "S&amp;P 500 + 나스닥100 518개 종목의 실제 가격 데이터를 받아 장기 추세(200일 이동평균)와 최근 고점 대비 하락 폭으로 점수를 매깁니다. 상위 점수 종목만 — 보통 10~20개 — 기준을 통과합니다.",
    "Every ticker that clears the quant bar gets reviewed a second time by AI, specifically for two traps: chasing a stock already near a blow-off top, or mistaking a dead-cat bounce for a real recovery.":
        "퀀트 기준을 통과한 모든 종목은 AI가 두 가지 함정을 집중적으로 다시 검토합니다. 이미 고점 부근까지 과열된 종목을 뒤늦게 쫓는 것, 그리고 데드캣 바운스를 진짜 회복으로 착각하는 것입니다.",
    "You get the data, the reasoning, and a plain-language risk review — never a price target, never a \"buy now.\" What you do with it is up to you.":
        "데이터와 근거, 그리고 쉬운 말로 된 리스크 리뷰를 제공합니다. 목표가도, \"지금 사세요\"도 없습니다. 그것으로 무엇을 할지는 당신이 정합니다.",

    # --- proof section
    ">PROVEN BY THE NUMBERS<": ">숫자로 검증<",
    ">We tested it against 2 years of real data. Here's what happened.<": ">2년치 실제 데이터로 검증했습니다. 결과는 이렇습니다.<",
    "Not cherry-picked winners — replayed against 2 years of real price history using the exact formula running today. See exactly what's covered below.":
        "골라낸 성공 사례가 아닙니다. 지금 돌아가는 것과 똑같은 공식으로 2년치 실제 가격 데이터를 다시 재생한 결과입니다. 무엇이 포함됐는지는 아래에 그대로 적어뒀습니다.",
    "Backtest is computing on the server — check back shortly.": "백테스트를 서버에서 계산하는 중입니다 — 잠시 후 다시 확인해주세요.",
    ">How this was measured — read before you trust it<": ">어떻게 측정했는지 — 믿기 전에 읽어보세요<",
    ">Universe:<": ">대상 종목:<",
    ">Signal counting:<": ">신호 집계 방식:<",
    "only a fresh crossing above the score threshold counts as one signal — a stock staying \"Favorable\" for a week isn't counted 7 times.":
        "점수 기준선을 새로 넘어선 순간만 신호 1건으로 셉니다. 한 종목이 일주일 내내 \"Favorable\"이어도 7번으로 세지 않습니다.",
    ">Benchmark:<": ">벤치마크:<",
    ", not its average over the whole period. This matters: a buy-the-dip rule fires after selloffs, and the market's own bounce after a selloff is better than its return on a random day. Measured the loose way, the edge looks about 40% larger than it is. The honest comparison is the one shown above.":
        " 기준으로 측정한 값이며, 전체 기간 평균이 아닙니다. 이게 중요한 이유는, 눌림목 매수 규칙은 하락 뒤에 신호가 뜨는데 하락 뒤 시장의 반등은 아무 날이나 잡았을 때보다 원래 좋기 때문입니다. 느슨하게 측정하면 우위가 실제보다 약 40% 커 보입니다. 위에 표시된 것이 정직한 비교입니다.",
    "the S&amp;P figure is its return over ": "S&amp;P 수치는 ",
    ">the same windows the strategy actually traded<": ">전략이 실제로 진입한 것과 같은 구간<",
    ">Costs:<": ">비용:<",
    "every signal is charged a 0.10% round trip for spread and slippage before anything is averaged, and a trade that made less than that is not counted as a win. Taxes are not modelled.":
        "평균을 내기 전에 모든 신호에 스프레드·슬리피지 명목으로 왕복 0.10% 비용을 부과하며, 그보다 적게 번 거래는 승리로 세지 않습니다. 세금은 반영하지 않았습니다.",
    ">Survivorship bias:<": ">생존 편향:<",
    "this uses today's index membership applied to the past 2 years. Stocks removed from these indices during that window (delisted, acquired, or dropped for poor performance) aren't included, which flatters results by an amount that cannot be measured without historical index membership data we don't have. Assume the real numbers are lower than these.":
        "오늘 기준 지수 구성 종목을 과거 2년에 적용한 것입니다. 그 기간에 지수에서 빠진 종목(상장폐지, 인수, 부진으로 제외)은 포함되지 않아 결과가 실제보다 좋게 나옵니다. 과거 지수 구성 데이터가 없으면 그 크기를 측정할 수 없으므로, 실제 수치는 이보다 낮다고 가정하세요.",
    ">Overlapping windows:<": ">구간 중첩:<",
    "signals fire close together, so the 30/60/90-day windows overlap heavily. The observation counts are not independent samples, and the averages carry more uncertainty than a raw \"n\" suggests.":
        "신호가 가까운 시점에 몰려서 발생하므로 30/60/90일 구간이 크게 겹칩니다. 관측 수는 독립 표본이 아니며, 평균값은 표시된 n이 시사하는 것보다 불확실성이 큽니다.",
    ">Out-of-sample check:<": ">Out-of-sample 검증:<",
    ">Recomputed periodically, not per-visit:<": ">방문할 때마다가 아니라 주기적으로 재계산:<",
    "normally about once a week, though a server restart can also trigger a one-off recompute if there's no cached result yet. Either way the universe covered doesn't change; the numbers only move because new price data came in.":
        "보통 주 1회 정도이며, 캐시된 결과가 없는 상태에서 서버가 재시작되면 한 번 더 계산될 수 있습니다. 어느 쪽이든 대상 종목은 바뀌지 않고, 숫자는 새 가격 데이터가 들어와서만 움직입니다.",
    "not available yet — check back after the first computation finishes.": "아직 준비되지 않았습니다 — 첫 계산이 끝난 뒤 다시 확인해주세요.",
    "not available yet.": "아직 준비되지 않았습니다.",

    # --- differentiation
    ">WHY THIS IS DIFFERENT<": ">무엇이 다른가<",
    "Most signal services show you a highlight reel.": "대부분의 시그널 서비스는 잘된 것만 모아 보여줍니다.",
    "We show you the full validation — including the losses.": "QUANTIFY는 검증 전체를 보여줍니다 — 손실까지 포함해서.",
    ">The usual approach<": ">흔한 방식<",
    "A handful of cherry-picked win screenshots, no losing trades shown": "골라낸 수익 인증 몇 장, 손실 거래는 보여주지 않음",
    "\"This stock is about to explode\" — hype with no disclosed methodology": "\"이 종목 곧 터진다\" — 방법론 공개 없는 과장",
    "A black-box pick with no reasoning you can check": "검증할 수 없는 블랙박스 추천",
    "Pay more for \"VIP\" or \"premium\" picks other subscribers don't see": "다른 구독자는 못 보는 \"VIP\"·\"프리미엄\" 추천에 추가 결제",
    "Full backtest published on this page — wins and losses, in-sample and out-of-sample":
        "이 페이지에 백테스트 전문 공개 — 수익과 손실, in-sample과 out-of-sample 모두",
    "One entry rule, plainly disclosed: long-term uptrend, pulled back 10-25% from its recent high":
        "진입 규칙 하나를 그대로 공개: 장기 상승 추세, 최근 고점 대비 10~25% 하락",
    "Every pick ships with the AI's actual reasoning and the two failure modes it checked":
        "모든 종목에 AI의 실제 판단 근거와 점검한 두 가지 실패 유형을 함께 제공",
    "One plan. Every subscriber sees the same data, the same day.": "요금제 하나. 모든 구독자가 같은 날 같은 데이터를 봅니다.",

    # --- features
    ">Built on real data, not vibes<": ">감이 아니라 실제 데이터 위에<",
    ">Live market data<": ">실시간 시장 데이터<",
    "Real prices and volume from the actual market — no simulated or backfilled data.":
        "실제 시장의 가격과 거래량 — 시뮬레이션이나 사후 보정 데이터 없음.",
    "Every detected ticker gets a written quant review and explicit risk check, in plain English.":
        "감지된 모든 종목에 서술형 퀀트 리뷰와 명시적인 리스크 점검을 쉬운 말로 제공합니다.",
    "Distance from the 52-week high/low and 200-day trend, so you see where a stock actually sits.":
        "52주 고점·저점 대비 거리와 200일 추세를 함께 보여줘서 종목의 현재 위치를 파악할 수 있습니다.",
    ">Price alerts &amp; news<": ">가격 알림 &amp; 뉴스<",
    "Set a target price and get emailed when it's hit, with live headlines next to the chart.":
        "목표가를 설정하면 도달했을 때 이메일로 알려주고, 차트 옆에 실시간 헤드라인을 함께 보여줍니다.",

    # --- final CTA + disclaimer
    ">Know which one it is — before you buy, not after.<": ">사고 나서가 아니라, 사기 전에 구분하세요.<",
    ">IMPORTANT<": ">중요<",
    "QUANTIFY is an informational and educational tool. Nothing on this site is investment advice, a recommendation, or a solicitation to buy or sell any security. Scores, badges, and AI commentary reflect a mathematical model's output on the data available at the time and can be wrong. Markets involve risk, including loss of principal. Do your own research and consult a licensed financial advisor before making investment decisions.":
        "QUANTIFY는 정보 제공 및 교육용 도구입니다. 이 사이트의 어떤 내용도 투자 조언이나 추천, 증권 매매 권유가 아닙니다. 점수, 배지, AI 코멘트는 그 시점에 이용 가능한 데이터에 대한 수학적 모델의 출력이며 틀릴 수 있습니다. 시장에는 원금 손실을 포함한 위험이 따릅니다. 투자 결정을 내리기 전에 직접 조사하고 허가받은 금융 전문가와 상담하세요.",
    "QUANTIFY. — informational and educational only, not investment advice.": "QUANTIFY. — 정보 제공 및 교육 목적일 뿐, 투자 조언이 아닙니다.",

    # --- pricing
    ">SIMPLE PRICING<": ">단순한 가격<",
    ">One plan. Everything included.<": ">요금제 하나. 전부 포함.<",
    "No tiers to compare and nothing paywalled inside the app — every subscriber gets the full quant scanner, AI risk review, market view, and portfolio tools.":
        "비교할 등급도 없고 앱 안에서 잠긴 기능도 없습니다. 모든 구독자가 퀀트 스캐너, AI 리스크 리뷰, 마켓 뷰, 포트폴리오 도구를 전부 사용합니다.",
    "7-day free trial, then billed monthly. Cancel anytime.": "7일 무료 체험 후 매월 청구. 언제든 해지할 수 있습니다.",
    "Full S&amp;P 500 + Nasdaq-100 quant scan, updated four times a day around the market open and close":
        "S&amp;P 500 + 나스닥100 전체 퀀트 스캔, 장 시작과 마감 전후로 하루 네 번 갱신",
    "AI risk review on every ticker that clears the quant bar": "퀀트 기준을 통과한 모든 종목에 대한 AI 리스크 리뷰",
    "Live charts, technicals, and a market-wide heatmap": "실시간 차트, 기술적 지표, 시장 전체 히트맵",
    "Watchlist, portfolio tracking, and price alerts": "관심종목, 포트폴리오 추적, 가격 알림",
    "Backtest methodology published openly, in-sample and out-of-sample": "백테스트 방법론 전면 공개 (in-sample 및 out-of-sample)",
    ">Start Free Trial<": ">무료 체험 시작<",
    ">What happens after the trial?<": ">체험이 끝나면 어떻게 되나요?<",
    "Your card is charged $9.99 when the 7-day trial ends, unless you cancel first from Settings. Cancel anytime — access continues through the end of whatever period you've already paid for.":
        "7일 체험이 끝나면 설정에서 미리 해지하지 않는 한 카드로 $9.99가 청구됩니다. 언제든 해지할 수 있고, 이미 결제한 기간이 끝날 때까지는 계속 이용할 수 있습니다.",
    ">Questions?<": ">궁금한 점이 있나요?<",
    ">email us directly<": ">직접 이메일로 문의<",
    "QUANTIFY is an informational and educational tool, not a licensed investment adviser or broker-dealer. Nothing on this page or in the app is investment advice.":
        "QUANTIFY는 정보 제공 및 교육용 도구이며, 허가받은 투자자문업자나 증권중개업자가 아닙니다. 이 페이지와 앱의 어떤 내용도 투자 조언이 아닙니다.",
    "QUANTIFY is an informational and educational tool, not a licensed investment adviser or broker-dealer.":
        "QUANTIFY는 정보 제공 및 교육용 도구이며, 허가받은 투자자문업자나 증권중개업자가 아닙니다.",

    # --- about
    ">ABOUT<": ">소개<",
    ">What QUANTIFY is — and isn't<": ">QUANTIFY가 무엇이고, 무엇이 아닌지<",
    ">What it is<": ">무엇인가<",
    ">What it isn't<": ">무엇이 아닌가<",
    ">Why one strategy<": ">왜 전략이 하나뿐인가<",
    "A quant scan of the S&amp;P 500 and Nasdaq-100, with an AI risk check layered on top. Built for people willing to do their own research, not people looking for someone to tell them what to buy.":
        "S&amp;P 500과 나스닥100을 퀀트로 스캔하고 그 위에 AI 리스크 점검을 얹습니다. 무엇을 사라고 말해줄 사람을 찾는 분이 아니라, 스스로 조사할 의향이 있는 분을 위해 만들었습니다.",
    "QUANTIFY is not a broker, not a licensed investment adviser, and not a signal service telling you when to buy or sell. It doesn't place trades, hold your money, or know your financial situation. It surfaces one specific pattern and flags risk around it — sizing, diversification, and the actual decision are entirely yours.":
        "QUANTIFY는 증권사도, 허가받은 투자자문업자도, 매수·매도 시점을 알려주는 시그널 서비스도 아닙니다. 주문을 넣지 않고, 당신의 돈을 보관하지 않으며, 당신의 재무 상황을 알지 못합니다. 하나의 특정 패턴을 찾아내고 그 주변의 위험을 표시할 뿐이며, 비중 조절과 분산 투자, 실제 결정은 전적으로 당신의 몫입니다.",
    "Most retail screening tools bury you in adjustable filters that are easy to overfit and hard to trust. QUANTIFY ships one entry rule at a time, validated against real out-of-sample data before it goes live, rather than a wall of knobs that look sophisticated but were never actually tested.":
        "대부분의 개인용 스크리닝 도구는 조절 가능한 필터를 잔뜩 제공하는데, 이런 필터는 과적합되기 쉽고 신뢰하기 어렵습니다. QUANTIFY는 정교해 보이지만 실제로 검증된 적 없는 수많은 옵션 대신, 실제 out-of-sample 데이터로 검증한 진입 규칙을 한 번에 하나씩만 제공합니다.",

    # --- auth
    ">Welcome back<": ">다시 오셨네요<",
    ">Log in to see today's detected tickers.<": ">오늘 감지된 종목을 보려면 로그인하세요.<",
    ">Create your account<": ">계정 만들기<",
    ">7-day free trial, then $9.99/month. Cancel anytime.<": ">7일 무료 체험 후 월 $9.99. 언제든 해지할 수 있습니다.<",
    "Continue with Google": "Google로 계속하기",
    ">10+ characters, with at least 1 letter and 1 number<": ">10자 이상, 영문 1자와 숫자 1자 이상 포함<",
    ">Quant-detected stocks,<": ">퀀트가 찾아낸 종목,<",
    ">AI risk-checked.<": ">AI가 리스크까지 점검.<",
    "Live market data, never simulated": "실제 시장 데이터, 시뮬레이션 아님",
    "Plain-language AI risk review on every pick": "모든 종목에 쉬운 말로 된 AI 리스크 리뷰",
    "7-day trial, then $9.99/month": "7일 무료 체험 후 월 $9.99",
    "A daily scan of the S&amp;P 500 and Nasdaq-100, cross-checked by AI for blow-off-top and dead-cat-bounce risk before it ever reaches your screen.":
        "S&amp;P 500과 나스닥100을 매일 스캔하고, 화면에 표시되기 전에 AI가 급등 후 고점과 데드캣 바운스 위험을 교차 검증합니다.",
    ">Didn't get a verification email?<": ">인증 메일을 받지 못하셨나요?<",
    ">Resend verification email<": ">인증 메일 다시 보내기<",
    "By creating an account you agree to our ": "계정을 만들면 다음에 동의하는 것으로 간주됩니다: ",

    # --- FAQ
    ">Frequently asked questions<": ">자주 묻는 질문<",
    ">Is this financial advice?<": ">이것은 투자 조언인가요?<",
    "No. QUANTIFY is an informational and educational tool. The quant score, badges, and AI commentary are a mathematical model's output on available data, and they can be wrong. Nothing here is a recommendation to buy or sell anything — consult a licensed financial advisor before making investment decisions.":
        "아닙니다. QUANTIFY는 정보 제공 및 교육용 도구입니다. 퀀트 점수, 배지, AI 코멘트는 주어진 데이터에 대한 수학적 모델의 출력이며 틀릴 수 있습니다. 어떤 것도 매수·매도 추천이 아니며, 투자 결정을 내리기 전에 허가받은 금융 전문가와 상담하세요.",
    ">What do the Favorable / Caution / Risk badges mean?<": ">Favorable / Caution / Risk 배지는 무슨 뜻인가요?<",
    "<b>Favorable</b> means the AI's entry-timing check found the setup clean. <b>Caution</b> means it found some risk worth knowing about before you look closer. <b>Risk</b> means it found something that argues for skipping or waiting. None of the three is ever a buy or sell order — they're a starting point for your own research.":
        "<b>Favorable</b>은 AI의 진입 타이밍 점검에서 문제가 발견되지 않았다는 뜻입니다. <b>Caution</b>은 더 자세히 보기 전에 알아둘 만한 위험이 있다는 뜻입니다. <b>Risk</b>는 건너뛰거나 기다릴 이유가 발견됐다는 뜻입니다. 셋 중 어느 것도 매수·매도 지시가 아니며, 본인의 조사를 시작하는 출발점일 뿐입니다.",
    ">What's the strategy behind the scan?<": ">스캔의 근거가 되는 전략은 무엇인가요?<",
    ">How often does the data update?<": ">데이터는 얼마나 자주 갱신되나요?<",
    "The full S&amp;P 500 + Nasdaq-100 scan recomputes four times a day on trading days — before the open, at the open, about an hour in, and after the close — using a licensed market data feed.":
        "S&amp;P 500 + 나스닥100 전체 스캔은 거래일마다 하루 네 번 다시 계산됩니다 — 장 시작 전, 장 시작 시점, 시작 후 약 1시간, 장 마감 후 — 정식 라이선스를 받은 시장 데이터 피드를 사용합니다.",
    ">Can I run my own custom screener?<": ">직접 스크리너를 설정할 수 있나요?<",
    "Not yet — today there's one validated strategy, and you can filter the results by badge and by index (S&amp;P 500 / Nasdaq-100). A configurable multi-strategy screener is on the roadmap.":
        "아직은 아닙니다. 현재는 검증된 전략 하나가 있고, 결과를 배지와 지수(S&amp;P 500 / 나스닥100)로 필터링할 수 있습니다. 설정 가능한 다중 전략 스크리너는 로드맵에 있습니다.",
    ">Is my payment information secure?<": ">결제 정보는 안전한가요?<",
    "Yes. Billing is handled by Gumroad — QUANTIFY never sees or stores your card details.":
        "네. 결제는 Gumroad가 처리하며, QUANTIFY는 카드 정보를 보거나 저장하지 않습니다.",
    ">How do I cancel?<": ">어떻게 해지하나요?<",
    "From Settings or the Subscription page once you're logged in. Cancelling stops future billing; you keep access through the end of the period you already paid for.":
        "로그인 후 설정 또는 구독 페이지에서 해지할 수 있습니다. 해지하면 다음 청구가 중단되며, 이미 결제한 기간이 끝날 때까지는 계속 이용할 수 있습니다.",
    # --- final six, several of which are split by inline links, so the anchor tags are
    # part of the key rather than translating the fragments around them separately.
    "Demo needs no signup \u00b7 trial is 7 days free, no credit card.":
        "데모는 가입 없이 볼 수 있습니다 \u00b7 체험은 7일 무료, 카드 등록 없음.",
    "See today's list, free for 7 days. Then $9.99/month. Takes under a minute to sign up.":
        "오늘의 목록을 7일 동안 무료로 보세요. 이후 월 $9.99. 가입은 1분이면 끝납니다.",
    "See the <a href=\"/faq\">FAQ</a>, or <a href=\"mailto:quantify.app.official@gmail.com\">email us directly</a>.":
        "<a href=\"/faq\">자주 묻는 질문</a>을 보시거나 <a href=\"mailto:quantify.app.official@gmail.com\">직접 이메일로 문의</a>해 주세요.",
    "One validated quant rule — a pullback in an established uptrend — run across 518 stocks, refreshed four times a day around the market open and close. Every ticker that clears the bar also gets an AI-written review checking for the specific ways that setup can fail (a blow-off top disguised as a pullback, a dead-cat bounce, a stock that's actually in a downtrend). The scoring and the backtest methodology behind it are published openly, in-sample and out-of-sample, not just cherry-picked results.":
        "검증된 퀀트 규칙 하나 — 확립된 상승 추세에서의 눌림목 — 를 518개 종목에 적용하고, 장 시작과 마감 전후로 하루 네 번 갱신합니다. 기준을 통과한 모든 종목은 그 설정이 실패할 수 있는 구체적인 방식(눌림목으로 위장한 급등 후 고점, 데드캣 바운스, 실제로는 하락 추세인 종목)을 AI가 서술형으로 점검합니다. 점수 산출과 그 뒤의 백테스트 방법론은 골라낸 결과가 아니라 in-sample과 out-of-sample 모두 공개되어 있습니다.",
    "<a href=\"/terms\">Terms</a> and <a href=\"/privacy\">Privacy Policy</a>":
        "<a href=\"/terms\">이용약관</a> 및 <a href=\"/privacy\">개인정보처리방침</a>",
    "home page</a>.": "홈페이지</a>에서 확인하세요.",
    "See the full numbers on the <a href=\"/#proof\">": "전체 수치는 <a href=\"/#proof\">",
    # --- Terms of Service. Headings are tag-bounded; clauses are whole sentences.
    ">Terms of Service<": ">이용약관<",
    "These Terms of Service (\"Terms\") govern your access to and use of QUANTIFY (the \"Service\"). By creating an account or using the Service, you agree to these Terms.":
        "본 이용약관(이하 \"약관\")은 QUANTIFY(이하 \"서비스\")에 대한 접근 및 이용에 적용됩니다. 계정을 만들거나 서비스를 이용하면 본 약관에 동의한 것으로 봅니다.",
    ">1. Description of the Service<": ">1. 서비스 설명<",
    "QUANTIFY is an informational and educational tool that runs a quantitative scan of the S&amp;P 500 and Nasdaq-100 and generates AI-written commentary about detected tickers. The Service is not a licensed investment adviser, broker-dealer, or financial planner.":
        "QUANTIFY는 S&amp;P 500과 나스닥100을 정량적으로 스캔하고 감지된 종목에 대해 AI가 작성한 코멘트를 생성하는 정보 제공 및 교육용 도구입니다. 본 서비스는 허가받은 투자자문업자, 증권중개업자, 재무설계사가 아닙니다.",
    ">2. Not Investment Advice<": ">2. 투자 조언이 아님<",
    "Nothing on the Service — including quant scores, badges, AI-generated commentary, or any other content — is investment advice, a recommendation, or a solicitation to buy or sell any security. All investment decisions, and all outcomes from those decisions, are solely your own responsibility. Markets involve risk, including the possible loss of your entire investment. Consult a licensed financial professional before making investment decisions.":
        "퀀트 점수, 배지, AI가 생성한 코멘트를 포함해 본 서비스의 어떤 내용도 투자 조언이나 추천, 증권 매매 권유가 아닙니다. 모든 투자 결정과 그 결과에 대한 책임은 전적으로 이용자 본인에게 있습니다. 시장에는 투자 원금 전액 손실을 포함한 위험이 따릅니다. 투자 결정을 내리기 전에 허가받은 금융 전문가와 상담하시기 바랍니다.",
    ">3. Eligibility and Your Account<": ">3. 이용 자격 및 계정<",
    "You must be at least 18 years old to use the Service. You are responsible for maintaining the confidentiality of your password and for all activity under your account. Notify us promptly of any unauthorized use.":
        "본 서비스는 만 18세 이상만 이용할 수 있습니다. 비밀번호의 기밀 유지와 계정에서 이루어지는 모든 활동에 대한 책임은 이용자에게 있습니다. 무단 사용을 발견하면 즉시 알려주시기 바랍니다.",
    ">4. Acceptable Use<": ">4. 허용되는 이용<",
    "You agree not to: scrape, reverse-engineer, or systematically extract data from the Service; interfere with or overload our infrastructure; resell or redistribute Service content as your own product; or use the Service for any unlawful purpose.":
        "이용자는 다음 행위를 하지 않는 데 동의합니다. 서비스의 데이터를 스크래핑·역설계하거나 조직적으로 추출하는 행위, 당사 인프라를 방해하거나 과부하시키는 행위, 서비스 콘텐츠를 자신의 제품인 것처럼 재판매·재배포하는 행위, 불법적인 목적으로 서비스를 이용하는 행위.",
    ">5. Third-Party Data<": ">5. 제3자 데이터<",
    "Market data displayed on the Service is sourced from third-party licensed data providers. News headlines and earnings dates are sourced from Yahoo Finance. This data may be delayed, incomplete, or inaccurate, and QUANTIFY is not affiliated with, endorsed by, or sponsored by any of these providers. We do not guarantee the accuracy, completeness, or timeliness of any data shown.":
        "본 서비스에 표시되는 시장 데이터는 제3자 라이선스 데이터 제공업체로부터 제공받습니다. 뉴스 헤드라인과 실적 발표일은 Yahoo Finance에서 가져옵니다. 이 데이터는 지연되거나 불완전하거나 부정확할 수 있으며, QUANTIFY는 해당 제공업체들과 제휴 관계에 있지 않고 이들로부터 보증이나 후원을 받지 않습니다. 당사는 표시되는 데이터의 정확성, 완전성, 적시성을 보장하지 않습니다.",
    ">6. Disclaimers and Limitation of Liability<": ">6. 면책 및 책임의 제한<",
    "The Service is provided \"as is\" and \"as available,\" without warranties of any kind, express or implied. To the fullest extent permitted by law, QUANTIFY and its operator will not be liable for any direct, indirect, incidental, or consequential damages arising from your use of, or inability to use, the Service, including any investment losses.":
        "본 서비스는 명시적이든 묵시적이든 어떠한 보증도 없이 \"있는 그대로\", \"이용 가능한 상태로\" 제공됩니다. 법이 허용하는 최대 범위에서, QUANTIFY와 그 운영자는 이용자의 서비스 이용 또는 이용 불가로 인해 발생하는 직접적·간접적·부수적·결과적 손해(투자 손실을 포함)에 대해 책임을 지지 않습니다.",
    ">7. Termination<": ">7. 이용 종료<",
    "We may suspend or terminate your account at any time, with or without notice, for conduct that violates these Terms or that we believe is harmful to the Service or other users.":
        "당사는 본 약관을 위반하거나 서비스 또는 다른 이용자에게 해가 된다고 판단되는 행위에 대해, 사전 통지 여부와 관계없이 언제든지 계정을 정지하거나 종료할 수 있습니다.",
    ">8. Changes to These Terms<": ">8. 약관의 변경<",
    "We may update these Terms from time to time. Continued use of the Service after a change constitutes acceptance of the revised Terms. We will update the \"Last updated\" date above when changes are made.":
        "당사는 본 약관을 수시로 변경할 수 있습니다. 변경 후에도 서비스를 계속 이용하면 변경된 약관에 동의한 것으로 봅니다. 변경이 있을 경우 위의 \"최종 수정일\"을 갱신합니다.",
    ">9. Contact<": ">9. 문의<",
    "Questions about these Terms can be sent to ": "본 약관에 관한 문의는 다음으로 보내주시기 바랍니다: ",

    # --- Privacy Policy
    ">Privacy Policy<": ">개인정보처리방침<",
    "This Privacy Policy explains what information QUANTIFY (\"we\") collects, how we use it, and the choices you have.":
        "본 개인정보처리방침은 QUANTIFY(이하 \"당사\")가 어떤 정보를 수집하고 어떻게 이용하는지, 그리고 이용자에게 어떤 선택권이 있는지를 설명합니다.",
    ">1. Information We Collect<": ">1. 수집하는 정보<",
    "your email address and a securely hashed password (or, if you sign in with Google, your verified Google email address).":
        "이메일 주소와 안전하게 해시 처리된 비밀번호(구글로 로그인하는 경우에는 인증된 구글 이메일 주소).",
    "saved portfolio tickers, price alerts, and display preferences (theme, language, default strategy).":
        "저장한 포트폴리오 종목, 가격 알림, 표시 설정(테마, 언어, 기본 전략).",
    "IP address and request timestamps, used only for security purposes such as rate-limiting login and signup attempts.":
        "IP 주소와 요청 시각. 로그인·가입 시도 횟수 제한 등 보안 목적으로만 사용됩니다.",
    ">2. How We Use Information<": ">2. 정보의 이용<",
    "We use this information to operate your account, send transactional emails (email verification, password reset codes, price alerts), generate the AI commentary you request, and keep the Service secure and reliable.":
        "당사는 이 정보를 계정 운영, 거래성 이메일 발송(이메일 인증, 비밀번호 재설정 코드, 가격 알림), 이용자가 요청한 AI 코멘트 생성, 그리고 서비스의 보안과 안정성 유지에 사용합니다.",
    ">3. Third-Party Services We Use<": ">3. 이용 중인 제3자 서비스<",
    "processes ticker/price/technical data to generate AI commentary. We do not send your name, email, or account details to Groq.":
        "AI 코멘트 생성을 위해 종목·가격·기술적 지표 데이터를 처리합니다. 당사는 이용자의 이름, 이메일, 계정 정보를 Groq에 전송하지 않습니다.",
    "sends transactional emails (verification, password reset, alerts) on our behalf.":
        "당사를 대신해 거래성 이메일(인증, 비밀번호 재설정, 알림)을 발송합니다.",
    ">4. Cookies<": ">4. 쿠키<",
    "We use a single functional session cookie to keep you logged in. We do not use advertising or cross-site tracking cookies.":
        "로그인 상태 유지를 위해 기능성 세션 쿠키 하나를 사용합니다. 광고나 교차 사이트 추적 쿠키는 사용하지 않습니다.",
    ">5. Data Retention<": ">5. 데이터 보관<",
    "We retain account and portfolio data for as long as your account is active. Daily market scan history is retained for a limited number of days for product features like score trends.":
        "계정과 포트폴리오 데이터는 계정이 활성 상태인 동안 보관합니다. 일별 시장 스캔 기록은 점수 추이 같은 기능을 위해 제한된 일수 동안만 보관합니다.",
    "When an account is deleted, we retain one piece of information indefinitely: a one-way cryptographic hash of the email address, recorded solely to enforce the one-free-trial-per-person limit. It cannot be reversed to recover the address, is never used to contact you, and is not shared with anyone. We keep it on the basis of our legitimate interest in preventing repeated free trials.":
        "계정이 삭제되면 단 하나의 정보만 무기한 보관합니다. 이메일 주소의 단방향 암호학적 해시값으로, 1인 1회 무료 체험 제한을 적용하기 위한 목적으로만 기록됩니다. 이 값으로는 원래 주소를 복원할 수 없고, 연락 목적으로 사용되지 않으며, 제3자에게 제공되지 않습니다. 무료 체험의 반복 이용을 방지하려는 당사의 정당한 이익에 근거해 보관합니다.",
    ">6. Data Security<": ">6. 데이터 보안<",
    "Passwords are hashed with PBKDF2-SHA256 and a unique salt per account — we never store your password in plain text. Traffic to the Service is encrypted in transit.":
        "비밀번호는 계정마다 고유한 솔트와 함께 PBKDF2-SHA256으로 해시 처리되며, 평문으로 저장하지 않습니다. 서비스로 오가는 통신은 전송 중 암호화됩니다.",
    ">7. Your Rights<": ">7. 이용자의 권리<",
    "You can update your preferences and password anytime from Settings. To request deletion of your account and associated data, email us and we will process the request.":
        "설정에서 언제든지 환경설정과 비밀번호를 변경할 수 있습니다. 계정 및 관련 데이터의 삭제를 요청하려면 이메일로 연락 주시면 처리해 드립니다.",
    ">8. Children's Privacy<": ">8. 아동의 개인정보<",
    "The Service is not directed at individuals under 18, and we do not knowingly collect information from children.":
        "본 서비스는 만 18세 미만을 대상으로 하지 않으며, 당사는 아동의 정보를 고의로 수집하지 않습니다.",
    ">9. Changes to This Policy<": ">9. 방침의 변경<",
    "We may update this Privacy Policy from time to time. We will update the \"Last updated\" date above when changes are made.":
        "당사는 본 개인정보처리방침을 수시로 변경할 수 있습니다. 변경이 있을 경우 위의 \"최종 수정일\"을 갱신합니다.",
    ">10. Contact<": ">10. 문의<",
    "Questions about this policy can be sent to ": "본 방침에 관한 문의는 다음으로 보내주시기 바랍니다: ",
    ">Account information:<": ">계정 정보:<",
    ">Product data you create:<": ">이용자가 생성하는 데이터:<",
    ">Basic usage data:<": ">기본 이용 데이터:<",
    " — if you choose to log in with Google, Google shares your verified email address with us to create your session.":
        " — 구글로 로그인을 선택하면, 구글이 세션 생성을 위해 인증된 이메일 주소를 당사에 제공합니다.",
    " — we query these for public market data and headlines; no personal data about you is sent to them.":
        " — 공개된 시장 데이터와 헤드라인을 조회하기 위해 이용하며, 이용자에 관한 개인정보는 전송하지 않습니다.",
    ">Market and news data providers (Polygon/Massive for market data, Yahoo Finance for news and earnings)<":
        ">시장·뉴스 데이터 제공업체 (시장 데이터는 Polygon/Massive, 뉴스와 실적은 Yahoo Finance)<",
    ">Google Sign-In<": ">구글 로그인<",
    # <title> and og:title read "QUANTIFY. <name>", so the ">Name<" keys above miss them.
    "QUANTIFY. Terms of Service": "QUANTIFY. 이용약관",
    "QUANTIFY. Privacy Policy": "QUANTIFY. 개인정보처리방침",
    # The "Last updated" date is passed in as literal text by both legal pages.
    "August 30, 2026": "2026년 8월 30일",
}




def translate_public(html: str, lang: str) -> str:
    """Replace the English source strings above with their Korean equivalents.

    Longest first, so a short key can never eat a fragment of a longer sentence that was
    still waiting to be matched. Same find-and-replace approach as translate_body(), for
    the same reason: these page bodies are full of JS template literals and turning them
    into f-strings would mean escaping every brace in them.
    """
    if lang == "en":
        return html
    table = PUBLIC_KO if lang == "ko" else {}
    if not table:
        return html
    ordered = sorted(table, key=len, reverse=True)

    def _swap(chunk: str) -> str:
        for src in ordered:
            chunk = chunk.replace(src, table[src])
        return chunk

    out, pos = [], 0
    for m in re.finditer(r"<(script|style)\b.*?</\1>", html, re.S | re.I):
        out.append(_swap(html[pos:m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(_swap(html[pos:]))
    return "".join(out)


def render_marketing_page(title: str, description: str, body_html: str, path: str = "",
                          extra_head: str = "", lang: str = "en") -> HTMLResponse:
    url = f"https://quantify.trading{path}"
    toggle = lang_toggle_html(lang, path or "/")
    return HTMLResponse(translate_public(f'''<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>QUANTIFY. — {title}</title><meta name="description" content="{description}">{og_head(f"QUANTIFY. — {title}", description, path)}{hreflang_links(path or "/")}<link rel="canonical" href="{url}">{extra_head}<style>{MARKETING_CSS}
.lang-toggle{{display:inline-block;padding:6px 12px;border:1px solid var(--border);border-radius:999px;color:var(--dim2,#525f59);font-size:13.5px;font-weight:700;text-decoration:none;line-height:1;white-space:nowrap}}
.lang-toggle:hover{{border-color:var(--green);color:var(--green)}}
.lang-toggle.app{{border-color:var(--sb-border,#232b2f);color:var(--sb-text,#9aa7ac);padding:5px 10px;font-size:12.5px}}
.lang-toggle.app:hover{{border-color:var(--green);color:var(--green)}}
</style></head><body>
<header><div class="nav"><a class="logo" href="/">QUANTIFY<span>.</span></a><div class="navlinks">{_MARKETING_NAV}{toggle}</div></div></header>
<main><div class="wrap">{body_html}</div></main>
<footer>QUANTIFY. — informational and educational only, not investment advice.<br>
<a href="/login">Log in</a> &middot; <a href="/signup">Sign up</a> &middot; <a href="/pricing">Pricing</a> &middot; <a href="/faq">FAQ</a> &middot; <a href="/about">About</a> &middot; <a href="/terms">Terms</a> &middot; <a href="/privacy">Privacy</a>
</footer>
</body></html>''', lang))


@app.get("/pricing", response_class=HTMLResponse)
async def pricing_page(request: Request):
    body = """
<div class="eyebrow">SIMPLE PRICING</div>
<h1>One plan. Everything included.</h1>
<p class="sublead">No tiers to compare and nothing paywalled inside the app — every subscriber gets the full quant scanner, AI risk review, market view, and portfolio tools.</p>
<div class="price-card">
<div class="amount">$9.99<span>/month</span></div>
<p style="color:var(--dim);font-size:13px;margin:6px 0 0">7-day free trial, then billed monthly. Cancel anytime.</p>
<ul>
<li>Full S&amp;P 500 + Nasdaq-100 quant scan, updated four times a day around the market open and close</li>
<li>AI risk review on every ticker that clears the quant bar</li>
<li>Live charts, technicals, and a market-wide heatmap</li>
<li>Watchlist, portfolio tracking, and price alerts</li>
<li>Backtest methodology published openly, in-sample and out-of-sample</li>
</ul>
<a class="btn" href="/signup" style="display:block;text-align:center">Start Free Trial</a>
</div>
<h2>What happens after the trial?</h2>
<p>Your card is charged $9.99 when the 7-day trial ends, unless you cancel first from Settings. Cancel anytime — access continues through the end of whatever period you've already paid for.</p>
<h2>Questions?</h2>
<p>See the <a href="/faq">FAQ</a>, or <a href="mailto:quantify.app.official@gmail.com">email us directly</a>.</p>
<div class="disclaimer">QUANTIFY is an informational and educational tool, not a licensed investment adviser or broker-dealer. Nothing on this page or in the app is investment advice.</div>
"""
    return render_marketing_page("Pricing", "QUANTIFY pricing: $9.99/month after a 7-day free trial, one plan, everything included.", body, path="/pricing", lang=resolve_lang(request, get_logged_in_user(request)))


@app.get("/faq", response_class=HTMLResponse)
async def faq_page(request: Request):
    faq_items = [
        ("Is this financial advice?", "No. QUANTIFY is an informational and educational tool. The quant score, badges, and AI commentary are a mathematical model's output on available data, and they can be wrong. Nothing here is a recommendation to buy or sell anything — consult a licensed financial advisor before making investment decisions."),
        ("What do the Favorable / Caution / Risk badges mean?", "<b>Favorable</b> means the AI's entry-timing check found the setup clean. <b>Caution</b> means it found some risk worth knowing about before you look closer. <b>Risk</b> means it found something that argues for skipping or waiting. None of the three is ever a buy or sell order — they're a starting point for your own research."),
        ("What's the strategy behind the scan?", 'QUANTIFY looks for stocks in a long-term uptrend (price above its 200-day moving average) that have pulled back 10-25% from their own recent 20-day high — a "buy the dip in an uptrend" pattern, not a breakout or momentum chase. This exact rule was validated by backtesting thousands of alternative entry rules against two years of real price history and comparing in-sample results against a held-out out-of-sample period never used for tuning. See the full numbers on the <a href="/#proof">home page</a>.'),
        ("How often does the data update?", "The full S&amp;P 500 + Nasdaq-100 scan recomputes four times a day on trading days — before the open, at the open, about an hour in, and after the close — using a licensed market data feed."),
        ("Can I run my own custom screener?", "Not yet — today there's one validated strategy, and you can filter the results by badge and by index (S&amp;P 500 / Nasdaq-100). A configurable multi-strategy screener is on the roadmap."),
        ("Is my payment information secure?", "Yes. Billing is handled by Gumroad — QUANTIFY never sees or stores your card details."),
        ("How do I cancel?", "From Settings or the Subscription page once you're logged in. Cancelling stops future billing; you keep access through the end of the period you already paid for."),
    ]
    body = ('<div class="eyebrow">FAQ</div>\n<h1>Frequently asked questions</h1>\n'
            + "".join(f'<div class="faq-item"><h2>{q}</h2><p>{a}</p></div>' for q, a in faq_items)
            + '<div class="disclaimer">QUANTIFY is an informational and educational tool, not a licensed investment adviser or broker-dealer.</div>')
    faq_jsonld = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {"@type": "Question", "name": q,
             "acceptedAnswer": {"@type": "Answer", "text": re.sub(r"<[^>]+>", "", a).replace("&amp;", "&")}}
            for q, a in faq_items
        ],
    }
    extra_head = f'<script type="application/ld+json">{json.dumps(faq_jsonld)}</script>'
    return render_marketing_page("FAQ", "Frequently asked questions about QUANTIFY: pricing, the strategy, data updates, and cancellation.", body, path="/faq", extra_head=extra_head, lang=resolve_lang(request, get_logged_in_user(request)))


@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    body = """
<div class="eyebrow">ABOUT</div>
<h1>What QUANTIFY is — and isn't</h1>
<p class="sublead">A quant scan of the S&amp;P 500 and Nasdaq-100, with an AI risk check layered on top. Built for people willing to do their own research, not people looking for someone to tell them what to buy.</p>
<h2>What it is</h2>
<p>One validated quant rule — a pullback in an established uptrend — run across 518 stocks, refreshed four times a day around the market open and close. Every ticker that clears the bar also gets an AI-written review checking for the specific ways that setup can fail (a blow-off top disguised as a pullback, a dead-cat bounce, a stock that's actually in a downtrend). The scoring and the backtest methodology behind it are published openly, in-sample and out-of-sample, not just cherry-picked results.</p>
<h2>What it isn't</h2>
<p>QUANTIFY is not a broker, not a licensed investment adviser, and not a signal service telling you when to buy or sell. It doesn't place trades, hold your money, or know your financial situation. It surfaces one specific pattern and flags risk around it — sizing, diversification, and the actual decision are entirely yours.</p>
<h2>Why one strategy</h2>
<p>Most retail screening tools bury you in adjustable filters that are easy to overfit and hard to trust. QUANTIFY ships one entry rule at a time, validated against real out-of-sample data before it goes live, rather than a wall of knobs that look sophisticated but were never actually tested.</p>
<div class="disclaimer">QUANTIFY is an informational and educational tool. Nothing on this page or in the app is investment advice. Questions: <a href="mailto:quantify.app.official@gmail.com">quantify.app.official@gmail.com</a>.</div>
"""
    return render_marketing_page("About", "What QUANTIFY is and isn't — one validated quant strategy with an AI risk check, not investment advice.", body, path="/about", lang=resolve_lang(request, get_logged_in_user(request)))


LEGAL_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{background:#ffffff;color:#2a3330;font:17px/1.75 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}
.wrap{max-width:760px;margin:0 auto;padding:56px 24px 110px}
header{display:flex;justify-content:space-between;align-items:center;margin-bottom:46px;flex-wrap:wrap;gap:10px}
.brand{font-weight:700;font-size:19px;color:#12201a;text-decoration:none}
.brand span{color:#0e8a5f}
a.back{color:#0e8a5f;text-decoration:none;font-size:15px;font-weight:700}
h1{color:#12201a;font-size:32px;font-weight:800;margin-bottom:10px}
.updated{color:#8891a0;font-size:14px;margin-bottom:40px}
h2{color:#12201a;font-size:20px;font-weight:700;margin:36px 0 14px}
p,li{color:#2a3330;font-size:16.5px;line-height:1.8;margin-bottom:14px}
ul{padding-left:22px;margin-bottom:14px}
a{color:#0e8a5f}
"""


LEGAL_GOVERNING_NOTICE_KO = (
    '<div class="governing">이 문서는 영문 원본을 번역한 것입니다. 번역본과 영문 원본의 내용이 다를 경우 '
    '<b>영문 원본이 우선</b>합니다. 원본은 <a href="?lang=en">여기에서</a> 확인할 수 있습니다.</div>'
)


def render_legal_page(title: str, updated: str, body_html: str, path: str = "",
                      lang: str = "en") -> HTMLResponse:
    og = og_head(f"QUANTIFY. {title}", f"QUANTIFY {title.lower()} — the terms that govern use of the service.", path)
    toggle = lang_toggle_html(lang, path or "/")
    # A translated contract is only safe if it says which version wins. Without this the
    # Korean and English texts are two competing agreements the moment a phrase drifts.
    notice = LEGAL_GOVERNING_NOTICE_KO if lang == "ko" else ""
    updated_label = "최종 수정일" if lang == "ko" else "Last updated"
    back_label = "&larr; 홈으로" if lang == "ko" else "&larr; Back to home"
    return HTMLResponse(translate_public(f'''<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>QUANTIFY. {title}</title>{og}{hreflang_links(path or "/")}<style>{LEGAL_CSS}
.governing{{background:#fbf9f2;border:1px solid #ecdcb8;border-radius:10px;padding:14px 18px;font-size:14px;line-height:1.7;color:#5a5344;margin:0 0 26px}}
.governing a{{color:#0e8a5f}}
.legalhead{{display:flex;align-items:center;gap:12px;flex-wrap:wrap}}
.lang-toggle{{display:inline-block;padding:5px 11px;border:1px solid #d9e0dc;border-radius:999px;color:#4d5a54;font-size:13px;font-weight:700;text-decoration:none;line-height:1;white-space:nowrap}}
.lang-toggle:hover{{border-color:#0e8a5f;color:#0e8a5f}}
</style></head><body><div class="wrap"><header><a class="brand" href="/">QUANTIFY<span>.</span></a><div class="legalhead">{toggle}<a class="back" href="/">{back_label}</a></div></header><h1>{title}</h1><div class="updated">{updated_label}: {updated}</div>{notice}{body_html}</div></body></html>''', lang))


# Reusable logged-in-app page shell (sidebar + base black-theme CSS lifted from the
# terminal page). The terminal/portfolio/subscription/contact/settings pages each still
# hand-roll their own complete document (pre-existing pattern, left as-is) -- this exists
# so any NEW logged-in page (starting with /market) doesn't paste an 8th copy of the same
# CSS. Not retrofitted onto the existing five pages.
APP_SHELL_CSS = """
:root{--bg:#ffffff;--panel:#ffffff;--panel2:#f5f7f6;--border:#e2e6e3;--text:#3a4440;--head:#12201a;--dim:#77837e;--green:#0e8a5f;--red:#c8402c;--orange:#a8660a;
--sb-bg:#12181b;--sb-border:#232b2f;--sb-text:#9aa7ac;--sb-text-active:#ffffff;--sb-hover:#1b2327;--sb-danger:#e57373}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--text);font:16px/1.65 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;margin:0;padding:14px 20px 20px 220px}
header,.panel{background:var(--panel);border:1px solid var(--border)}
.sidebar{position:fixed;left:0;top:0;bottom:0;width:196px;background:var(--sb-bg);border-right:1px solid var(--sb-border);display:flex;flex-direction:column;align-items:stretch;padding:18px 12px;gap:3px;z-index:40;overflow-y:auto}
.sidebar .side-brand{color:var(--sb-text-active);font-weight:800;font-size:17px;letter-spacing:-.2px;margin:2px 6px 18px;text-decoration:none}
.sidebar .side-brand span{color:var(--green)}
.side-link{display:flex;align-items:center;gap:12px;padding:10px 12px;border-radius:9px;color:var(--sb-text);text-decoration:none;font-size:14.5px;font-weight:600;cursor:pointer;background:transparent;border:1px solid transparent}
.side-link svg{flex-shrink:0}
.side-link:hover{background:var(--sb-hover);color:var(--sb-text-active)}
.side-link.active{background:var(--green);color:#ffffff;border-color:var(--green)}
.lang-toggle{display:inline-block;padding:5px 10px;border:1px solid var(--sb-border);border-radius:999px;color:var(--sb-text);font-size:12.5px;font-weight:700;text-decoration:none;line-height:1;white-space:nowrap}
.lang-toggle:hover{border-color:var(--green);color:var(--green)}

.side-spacer{flex:1;min-height:8px}
header{padding:14px 20px;display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;border-radius:10px}
.brand{font-weight:700;font-size:19px;color:var(--head);text-decoration:none;letter-spacing:.2px}
.brand span{color:var(--dim)}
.wrap{max-width:1180px;margin:0 auto}
h1.page-title{color:var(--head);font-size:25px;font-weight:800;margin:2px 0 18px}
.panel{border-radius:10px;padding:20px;margin-bottom:14px;box-shadow:0 1px 2px rgba(18,32,26,.04)}
.panel h3{margin:0 0 14px;font-size:14px;font-weight:700;text-transform:uppercase;letter-spacing:.4px;color:var(--dim);border-bottom:1px solid var(--border);padding-bottom:12px}
.notice{padding:14px 16px;background:var(--panel2);border:1px solid var(--border);margin-bottom:10px;line-height:1.6;border-radius:8px;font-size:15px}
.empty-hint{color:var(--dim);font-size:15px;padding:10px 0}
.summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}
.summary-tile{background:var(--panel2);border:1px solid var(--border);padding:14px;border-radius:8px;text-align:center}
.summary-tile .label{font-size:11.5px;color:var(--dim);text-transform:uppercase;letter-spacing:.3px;font-weight:600}
.summary-tile .value{color:var(--head);font-weight:800;font-size:20px;margin-top:6px}
.summary-tile .value.gain{color:var(--green)}
.summary-tile .value.loss{color:var(--red)}
.ai-market-summary{background:var(--panel2);border:1px solid var(--border);border-radius:10px;padding:18px 20px}
.ai-market-summary .headline{color:var(--head);font-weight:800;font-size:18px;margin-bottom:10px;line-height:1.4}
.ai-market-summary .body{font-size:15px;line-height:1.75;color:var(--text)}
.ai-market-summary .meta{margin-top:12px;padding-top:12px;border-top:1px solid var(--border);color:var(--dim);font-size:12.5px}
.heat-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(60px,1fr));gap:3px}
.heat-tile{padding:9px 3px;text-align:center;font-size:11px;font-weight:700;cursor:pointer;color:#0a1f14;border-radius:5px}
.heat-tile.badge-favorable{box-shadow:0 0 0 2px #0e8a5f inset}
.heat-tile.badge-caution{box-shadow:0 0 0 2px #a8660a inset}
.heat-tile.badge-risk{box-shadow:0 0 0 2px #c8402c inset}
.heat-tile.cap-1{grid-column:span 2;font-size:13px;padding-top:13px;padding-bottom:13px}
.heat-tile.cap-2{font-size:11.5px}
.heat-group-header{grid-column:1/-1;color:var(--dim);font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.3px;margin:14px 0 6px}
.groupby-row{display:flex;align-items:center;gap:10px;margin-bottom:12px}
.groupby-row select{background:var(--panel2);border:1px solid var(--border);color:var(--head);padding:7px 10px;border-radius:6px;font-size:14px}
.badge{padding:4px 11px;border-radius:12px;font-weight:700;font-size:13px;display:inline-block}
.badge-ok{background:#e6f5ee;color:var(--green)}
.badge-warn{background:#fbf1e0;color:var(--orange)}
.badge-danger{background:#fbe6e2;color:var(--red)}
.badge-pending{background:var(--panel2);color:var(--dim)}
button{background:var(--panel2);border:1px solid var(--border);color:var(--head);padding:9px 16px;font:14.5px -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;cursor:pointer;border-radius:7px;font-weight:600}
button:hover{background:var(--border)}
input[type=text],input[type=number]{background:var(--panel);border:1px solid var(--border);color:var(--head);padding:10px 12px;border-radius:7px;font-size:15px}
.add-row{display:flex;gap:10px;margin-bottom:14px}
.add-row input{flex:1}
.watch-row{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:14px 0;border-bottom:1px solid var(--border2,var(--border))}
.watch-row:last-child{border-bottom:none}
.watch-row .wt-name{display:flex;align-items:center;gap:12px;cursor:pointer}
.watch-row .wt-name b{color:var(--head);font-size:16px}
.watch-row .wt-price{text-align:right;font-size:14.5px}
.remove-btn{background:transparent;border:1px solid var(--red);color:var(--red);padding:6px 12px;font-size:13px;font-weight:600;border-radius:6px}
.gain{color:var(--green)}
.loss{color:var(--red)}
.backtest-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
.backtest-card{background:var(--panel2);border:1px solid var(--border);padding:16px;border-radius:8px}
.backtest-card h4{margin:0 0 12px;font-size:13px;color:var(--dim);font-weight:700;text-transform:uppercase;letter-spacing:.3px}
.backtest-row{display:flex;justify-content:space-between;padding:6px 0;font-size:14.5px}
.backtest-row b.gain{color:var(--green)}
.backtest-row b.loss{color:var(--red)}
.backtest-meta{font-size:13px;color:var(--dim);margin-top:12px;line-height:1.7}
@media(max-width:900px){
  body{padding-left:14px;padding-bottom:70px}
  .sidebar{left:0;right:0;top:auto;bottom:0;width:auto;height:60px;flex-direction:row;justify-content:space-around;align-items:center;border-right:none;border-top:1px solid var(--sb-border);padding:0;overflow-y:visible;overflow-x:auto}
  .sidebar .side-brand,.side-spacer{display:none}
  .side-link{flex-direction:column;gap:3px;padding:6px 8px;font-size:9.5px}
}
"""

_SIDEBAR_ICONS = {
    "scanner": '<svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.7"><circle cx="8.5" cy="8.5" r="5.5"/><path d="M16.5 16.5l-4-4"/></svg>',
    "market": '<svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><path d="M3 17V9M9.5 17V4M16 17v-6"/></svg>',
    "watchlist": '<svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"><path d="M10 2.5l2.4 4.9 5.4.8-3.9 3.8.9 5.4-4.8-2.5-4.8 2.5.9-5.4-3.9-3.8 5.4-.8z"/></svg>',
    "backtest": '<svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="10" cy="10" r="7"/><path d="M10 6.2v4l3 2"/></svg>',
    "portfolio": '<svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"><rect x="3" y="7" width="14" height="9" rx="1.5"/><path d="M7.2 7V5.5A1.5 1.5 0 018.7 4h2.6a1.5 1.5 0 011.5 1.5V7"/></svg>',
    "settings": '<svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"><circle cx="10" cy="10" r="2.6"/><path d="M10 3v2.2M10 14.8V17M17 10h-2.2M5.2 10H3M14.7 5.3l-1.5 1.5M6.8 13.2l-1.5 1.5M14.7 14.7l-1.5-1.5M6.8 6.8L5.3 5.3"/></svg>',
    "subscription": '<svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"><rect x="2.5" y="5" width="15" height="10" rx="1.5"/><path d="M2.5 8.3h15"/></svg>',
    "contact": '<svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"><rect x="2.5" y="4.5" width="15" height="11" rx="1.5"/><path d="M3 5.5l7 5.5 7-5.5"/></svg>',
    "logout": '<svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3.3H4.7A1.7 1.7 0 003 5v10a1.7 1.7 0 001.7 1.7H8M13.2 6.2l4 3.8-4 3.8M17 10H8"/></svg>',
}

_APP_SHELL_NAV_LINKS = [
    ("scanner", "/terminal", "nav_scanner"),
    ("market", "/market", "nav_market"),
    ("watchlist", "/watchlist", "nav_watchlist"),
    ("backtest", "/backtest", "nav_backtest"),
    ("portfolio", "/portfolio", "nav_portfolio"),
]
_SIDEBAR_BOTTOM_LINKS = [
    ("settings", "/settings", "nav_settings", ""),
    ("subscription", "/subscription", "nav_subscription", ""),
    ("contact", "/contact", "nav_contact", ""),
    ("logout", "/logout", "nav_logout", ' style="color:var(--sb-danger)"'),
]


def _render_sidebar(active_nav: str, lang: str = "en") -> str:
    def link(key, href, label_key, extra=""):
        cls = "side-link active" if key == active_nav else "side-link"
        return f'<a class="{cls}" href="{href}"{extra}>{_SIDEBAR_ICONS[key]}<span>{t(label_key, lang)}</span></a>'
    top = "".join(link(key, href, label_key) for key, href, label_key in _APP_SHELL_NAV_LINKS)
    bottom = "".join(link(key, href, label_key, extra) for key, href, label_key, extra in _SIDEBAR_BOTTOM_LINKS)
    switch = ('<div style="padding:8px 12px 4px">'
              + lang_toggle_html(lang, "/terminal", style="app") + '</div>')
    return (f'<nav class="sidebar"><a class="side-brand" href="/terminal">QUANTIFY<span>.</span></a>'
            f'{top}<div class="side-spacer"></div>{switch}{bottom}</nav>')


def render_app_shell(title: str, active_nav: str, body_html: str, extra_head: str = "",
                     lang: str = "en", theme: str = "light") -> HTMLResponse:
    return HTMLResponse(f'''<!doctype html><html lang="{lang}" data-theme="{theme}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>QUANTIFY. {title}</title>{extra_head}<style>{APP_SHELL_CSS}
{DARK_THEME_VARS}</style></head><body>
{_render_sidebar(active_nav, lang)}
<header><a class="brand" href="/terminal">QUANTIFY<span>.</span></a></header>
<div class="wrap"><h1 class="page-title">{title}</h1>{body_html}</div>
</body></html>''')


@app.get("/terms", response_class=HTMLResponse)
async def terms_page(request: Request):
    body = """
<p>These Terms of Service ("Terms") govern your access to and use of QUANTIFY (the "Service"). By creating an account or using the Service, you agree to these Terms.</p>
<h2>1. Description of the Service</h2>
<p>QUANTIFY is an informational and educational tool that runs a quantitative scan of the S&amp;P 500 and Nasdaq-100 and generates AI-written commentary about detected tickers. The Service is not a licensed investment adviser, broker-dealer, or financial planner.</p>
<h2>2. Not Investment Advice</h2>
<p>Nothing on the Service — including quant scores, badges, AI-generated commentary, or any other content — is investment advice, a recommendation, or a solicitation to buy or sell any security. All investment decisions, and all outcomes from those decisions, are solely your own responsibility. Markets involve risk, including the possible loss of your entire investment. Consult a licensed financial professional before making investment decisions.</p>
<h2>3. Eligibility and Your Account</h2>
<p>You must be at least 18 years old to use the Service. You are responsible for maintaining the confidentiality of your password and for all activity under your account. Notify us promptly of any unauthorized use.</p>
<h2>4. Acceptable Use</h2>
<p>You agree not to: scrape, reverse-engineer, or systematically extract data from the Service; interfere with or overload our infrastructure; resell or redistribute Service content as your own product; or use the Service for any unlawful purpose.</p>
<h2>5. Third-Party Data</h2>
<p>Market data displayed on the Service is sourced from third-party licensed data providers. News headlines and earnings dates are sourced from Yahoo Finance. This data may be delayed, incomplete, or inaccurate, and QUANTIFY is not affiliated with, endorsed by, or sponsored by any of these providers. We do not guarantee the accuracy, completeness, or timeliness of any data shown.</p>
<h2>6. Disclaimers and Limitation of Liability</h2>
<p>The Service is provided "as is" and "as available," without warranties of any kind, express or implied. To the fullest extent permitted by law, QUANTIFY and its operator will not be liable for any direct, indirect, incidental, or consequential damages arising from your use of, or inability to use, the Service, including any investment losses.</p>
<h2>7. Termination</h2>
<p>We may suspend or terminate your account at any time, with or without notice, for conduct that violates these Terms or that we believe is harmful to the Service or other users.</p>
<h2>8. Changes to These Terms</h2>
<p>We may update these Terms from time to time. Continued use of the Service after a change constitutes acceptance of the revised Terms. We will update the "Last updated" date above when changes are made.</p>
<h2>9. Contact</h2>
<p>Questions about these Terms can be sent to <a href="mailto:quantify.app.official@gmail.com">quantify.app.official@gmail.com</a>.</p>
"""
    return render_legal_page("Terms of Service", "August 30, 2026", body, path="/terms",
                             lang=resolve_lang(request, get_logged_in_user(request)))


@app.get("/privacy", response_class=HTMLResponse)
async def privacy_page(request: Request):
    body = """
<p>This Privacy Policy explains what information QUANTIFY ("we") collects, how we use it, and the choices you have.</p>
<h2>1. Information We Collect</h2>
<ul>
<li><b>Account information:</b> your email address and a securely hashed password (or, if you sign in with Google, your verified Google email address).</li>
<li><b>Product data you create:</b> saved portfolio tickers, price alerts, and display preferences (theme, language, default strategy).</li>
<li><b>Basic usage data:</b> IP address and request timestamps, used only for security purposes such as rate-limiting login and signup attempts.</li>
</ul>
<h2>2. How We Use Information</h2>
<p>We use this information to operate your account, send transactional emails (email verification, password reset codes, price alerts), generate the AI commentary you request, and keep the Service secure and reliable.</p>
<h2>3. Third-Party Services We Use</h2>
<ul>
<li><b>Groq</b> — processes ticker/price/technical data to generate AI commentary. We do not send your name, email, or account details to Groq.</li>
<li><b>Brevo</b> — sends transactional emails (verification, password reset, alerts) on our behalf.</li>
<li><b>Google Sign-In</b> — if you choose to log in with Google, Google shares your verified email address with us to create your session.</li>
<li><b>Market and news data providers (Polygon/Massive for market data, Yahoo Finance for news and earnings)</b> — we query these for public market data and headlines; no personal data about you is sent to them.</li>
</ul>
<h2>4. Cookies</h2>
<p>We use a single functional session cookie to keep you logged in. We do not use advertising or cross-site tracking cookies.</p>
<h2>5. Data Retention</h2>
<p>We retain account and portfolio data for as long as your account is active. Daily market scan history is retained for a limited number of days for product features like score trends.</p>
<p>When an account is deleted, we retain one piece of information indefinitely: a one-way cryptographic hash of the email address, recorded solely to enforce the one-free-trial-per-person limit. It cannot be reversed to recover the address, is never used to contact you, and is not shared with anyone. We keep it on the basis of our legitimate interest in preventing repeated free trials.</p>
<h2>6. Data Security</h2>
<p>Passwords are hashed with PBKDF2-SHA256 and a unique salt per account — we never store your password in plain text. Traffic to the Service is encrypted in transit.</p>
<h2>7. Your Rights</h2>
<p>You can update your preferences and password anytime from Settings. To request deletion of your account and associated data, email us and we will process the request.</p>
<h2>8. Children's Privacy</h2>
<p>The Service is not directed at individuals under 18, and we do not knowingly collect information from children.</p>
<h2>9. Changes to This Policy</h2>
<p>We may update this Privacy Policy from time to time. We will update the "Last updated" date above when changes are made.</p>
<h2>10. Contact</h2>
<p>Questions about this policy can be sent to <a href="mailto:quantify.app.official@gmail.com">quantify.app.official@gmail.com</a>.</p>
"""
    return render_legal_page("Privacy Policy", "August 30, 2026", body, path="/privacy",
                             lang=resolve_lang(request, get_logged_in_user(request)))


@app.get("/auth/google/login")
async def google_login(request: Request):
    if not GOOGLE_CLIENT_ID:
        return RedirectResponse("/login?error=Google+sign-in+is+not+configured+yet.", status_code=303)
    state = secrets.token_urlsafe(24)
    redirect_uri = f"{str(request.base_url).rstrip('/')}/auth/google/callback"
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email",
        "state": state,
        "prompt": "select_account",
    }
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)
    res = RedirectResponse(url, status_code=303)
    res.set_cookie("google_oauth_state", state, httponly=True, secure=bool(os.getenv("RENDER")), samesite="lax", max_age=600)
    return res


@app.get("/auth/google/callback")
async def google_callback(request: Request, code: Optional[str] = None, state: Optional[str] = None, error: Optional[str] = None):
    cookie_state = request.cookies.get("google_oauth_state")
    if error or not code or not state or not cookie_state or state != cookie_state:
        return RedirectResponse("/login?error=Google+sign-in+failed.+Please+try+again.", status_code=303)
    redirect_uri = f"{str(request.base_url).rstrip('/')}/auth/google/callback"
    try:
        token_resp = await asyncio.to_thread(requests.post, "https://oauth2.googleapis.com/token", data={
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }, timeout=10)
        token_resp.raise_for_status()
        access_token = token_resp.json()["access_token"]
        info_resp = await asyncio.to_thread(
            requests.get, "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {access_token}"}, timeout=10,
        )
        info_resp.raise_for_status()
        info = info_resp.json()
    except Exception as exc:
        print(f"[Error: {type(exc).__name__}] Google OAuth exchange failed: {exc}")
        return RedirectResponse("/login?error=Google+sign-in+failed.+Please+try+again.", status_code=303)

    email = (info.get("email") or "").strip().lower()
    if not email or not info.get("email_verified"):
        return RedirectResponse("/login?error=Could+not+verify+your+Google+email.", status_code=303)

    channel = getattr(request.state, "qtfy_channel", None)
    visitor_id = getattr(request.state, "qtfy_visitor_id", None)
    conn = db()
    row = conn.execute("SELECT email FROM users WHERE email=?", (email,)).fetchone()
    if not row:
        password_hash, salt = await asyncio.to_thread(make_password_hash, secrets.token_urlsafe(32))
        try:
            trial_ends_at = await asyncio.to_thread(claim_trial, email)
            conn.execute(
                "INSERT INTO users(email,password_hash,salt,is_active,created_at,trial_ends_at,pref_theme,signup_utm_source,auth_provider) VALUES(?,?,?,1,?,?,'light',?,'google')",
                (email, password_hash, salt, time.time(), trial_ends_at, channel),
            )
            conn.commit()
            # Only a genuinely new account is a signup -- a returning Google user hitting
            # this same callback is a login, and counting it would inflate the funnel.
            asyncio.create_task(asyncio.to_thread(_log_event, EVENT_SIGNUP, visitor_id, channel, email))
        except sqlite3.IntegrityError:
            # Two near-simultaneous callbacks for the same brand-new email (double-click,
            # redirected retry) can both pass the SELECT above before either INSERTs --
            # the account now exists (from the other request), so just proceed to log in.
            pass
    else:
        # Accounts created before auth_provider existed cannot be told apart in the
        # database, so they are labelled the first time they sign in with Google again.
        conn.execute("UPDATE users SET auth_provider='google' WHERE email=? AND auth_provider IS NULL", (email,))
        conn.commit()
    conn.close()

    await asyncio.to_thread(adopt_browser_language, request, email)
    token = create_session(email)
    res = RedirectResponse("/terminal", status_code=303)
    res.set_cookie("session_user", token, httponly=True, secure=bool(os.getenv("RENDER")), samesite="lax", max_age=SESSION_TTL)
    res.delete_cookie("google_oauth_state")
    return res


@app.get("/accept-disclaimer", response_class=HTMLResponse)
async def accept_disclaimer_page(request: Request, error: Optional[str] = None):
    user = get_logged_in_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if disclaimer_accepted(user):
        return RedirectResponse("/terminal", status_code=303)
    error = html_lib.escape(error) if error else ''
    form = f'''<div class="card">
<h2>Before you continue</h2>
<div class="subtitle">Please read and accept this to activate your account.</div>
<div class="error">{error}</div>
<p style="color:#9ab8af;font-size:13px;line-height:1.7;background:#0f1613;border:1px solid #223229;border-radius:9px;padding:16px;margin-bottom:18px">
QUANTIFY is an informational and educational tool. It is not a registered investment adviser, broker-dealer, or financial planner, and nothing on this site — including quant scores, badges, or AI-generated commentary — is investment advice, a recommendation, or a solicitation to buy or sell any security.
<br><br>
All investment decisions, and all outcomes from those decisions, are solely your own responsibility. Markets involve risk, including the possible loss of your entire investment. Consult a licensed financial professional before making investment decisions.
<br><br>
To the fullest extent permitted by law, QUANTIFY and its operator assume no legal liability for any losses, damages, or outcomes arising from your use of the Service or your investment decisions.
</p>
<form action="/api/accept-disclaimer" method="post">
<label style="display:flex;align-items:flex-start;gap:10px;font-weight:normal;cursor:pointer">
<input type="checkbox" name="agree" required style="width:auto;margin-top:3px">
<span>I have carefully read the <a href="/terms" target="_blank" rel="noopener">Terms of Service</a> in full, and I acknowledge that QUANTIFY assumes no legal liability and that all investment decisions are solely my own responsibility.</span>
</label>
<button>I Agree &amp; Continue</button>
</form>
</div>'''
    return render_auth_page("Before You Continue", form)


@app.post("/api/accept-disclaimer")
async def accept_disclaimer(request: Request, agree: Optional[str] = Form(None)):
    user = get_logged_in_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if agree != "on":
        return RedirectResponse("/accept-disclaimer?error=You+must+check+the+box+to+continue.", status_code=303)
    conn = db()
    conn.execute("UPDATE users SET disclaimer_accepted_at=? WHERE email=?", (time.time(), user))
    conn.commit(); conn.close()
    return RedirectResponse("/terminal?welcome=1", status_code=303)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: Optional[str] = None, msg: Optional[str] = None):
    error = html_lib.escape(error) if error else ''
    msg = html_lib.escape(msg) if msg else ''
    form = f'''<div class="card"><h2>Welcome back</h2><div class="subtitle">Log in to see today's detected tickers.</div><div class="error">{error}</div><div class="ok">{msg}</div><a class="google-btn" href="/auth/google/login">{GOOGLE_ICON_SVG}Continue with Google</a><div class="divider">or</div><form action="/api/auth/login" method="post"><label>Email</label><input type="email" name="email" required autocomplete="email" inputmode="email" autocapitalize="none" autocorrect="off"><label>Password</label><input type="password" name="password" required autocomplete="current-password"><button>Log in</button></form><div class="links"><a href="/signup">Create an account</a><a href="/forgot-password">Forgot password?</a></div><details><summary>Didn't get a verification email?</summary><form action="/api/auth/resend-verification" method="post"><label>Email</label><input type="email" name="email" required autocomplete="email" inputmode="email" autocapitalize="none" autocorrect="off"><button>Resend verification email</button></form></details></div>'''
    return render_auth_page("QUANTIFY. Log in", form, path="/login",
                            description="Log in to QUANTIFY — the daily quant scan of the S&P 500 and Nasdaq-100, AI risk-checked.",
                            lang=resolve_lang(request))


@app.post("/api/auth/login")
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    email = email.strip().lower()
    if _is_locked_out(LOGIN_ATTEMPTS, email, LOGIN_MAX_ATTEMPTS):
        return RedirectResponse("/login?error=Too+many+failed+attempts.+Try+again+later.", status_code=303)
    try:
        conn = db()
        row = conn.execute(
            """
            SELECT email,password_hash,salt,is_active
            FROM users
            WHERE email=?
            """,
            (email,),
        ).fetchone()
    except Exception as e:
        print(f"[Error: {type(e).__name__}] Login lookup error: {e}")
        return RedirectResponse("/login?error=Database+error", status_code=303)

    if row is None:
        conn.close()
        _register_failed_attempt(LOGIN_ATTEMPTS, email)
        return RedirectResponse("/login?error=Invalid+credentials", status_code=303)

    valid = False
    if row["password_hash"] and row["salt"]:
        try:
            valid = await asyncio.to_thread(
                verify_password,
                password,
                row["password_hash"],
                row["salt"],
            )
        except Exception:
            valid = False

    conn.close()

    if valid and not row["is_active"]:
        _register_failed_attempt(LOGIN_ATTEMPTS, email)
        return RedirectResponse("/login?error=Please+verify+your+email+first.+Check+your+inbox+or+resend+below.", status_code=303)

    if not valid:
        _register_failed_attempt(LOGIN_ATTEMPTS, email)
        return RedirectResponse("/login?error=Invalid+credentials", status_code=303)

    _clear_attempts(LOGIN_ATTEMPTS, email)
    await asyncio.to_thread(adopt_browser_language, request, email)
    token = create_session(email)
    res = RedirectResponse("/terminal", status_code=303)
    res.set_cookie(
        "session_user",
        token,
        httponly=True,
        secure=bool(os.getenv("RENDER")),
        samesite="lax",
        max_age=SESSION_TTL,
    )
    return res


@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request, error: Optional[str] = None):
    error = html_lib.escape(error) if error else ''
    form = f'''<div class="card"><h2>Create your account</h2><div class="subtitle">7-day free trial, then $9.99/month. Cancel anytime.</div><div class="error">{error}</div><a class="google-btn" href="/auth/google/login">{GOOGLE_ICON_SVG}Continue with Google</a><div class="divider">or</div><form action="/api/auth/signup" method="post"><label>Email</label><input type="email" name="email" required autocomplete="email" inputmode="email" autocapitalize="none" autocorrect="off"><label>Password</label><input type="password" name="password" required autocomplete="new-password"><p class="hint">10+ characters, with at least 1 letter and 1 number</p><button>Create account</button></form><p style="text-align:center;font-size:11.5px;color:#6b8a7e;margin-top:14px">By creating an account you agree to our <a href="/terms">Terms</a> and <a href="/privacy">Privacy Policy</a>.</p><div class="links"><a href="/login">Already have an account? Log in</a></div></div>'''
    return render_auth_page("QUANTIFY. Sign up", form, path="/signup", lang=resolve_lang(request))


VERIFY_TOKEN_TTL = 24 * 3600


def send_verification_email(request: Request, email: str, token: str) -> bool:
    link = f"{str(request.base_url).rstrip('/')}/verify-email?email={urllib.parse.quote(email)}&token={token}"
    body = (
        f"Welcome to QUANTIFY.\n\n"
        f"Click the link below to verify your email and activate your account:\n{link}\n\n"
        f"This link expires in 24 hours. If you didn't sign up, you can ignore this email."
    )
    return send_email_notification(email, "[QUANTIFY.] Verify your email", body)


@app.post("/api/auth/signup")
async def signup(request: Request, email: str = Form(...), password: str = Form(...)):
    client_ip = request.client.host if request.client else "unknown"
    if _is_locked_out(SIGNUP_ATTEMPTS, client_ip, SIGNUP_MAX_ATTEMPTS):
        return RedirectResponse("/signup?error=Too+many+signups+from+this+network.+Try+again+later.",status_code=303)
    _register_failed_attempt(SIGNUP_ATTEMPTS, client_ip)
    email=email.strip().lower()
    if not validate_email(email): return RedirectResponse("/signup?error=Invalid+email",status_code=303)
    ok,error=validate_password_policy(password)
    if not ok: return RedirectResponse("/signup?error="+urllib.parse.quote(error),status_code=303)
    password_hash,salt=await asyncio.to_thread(make_password_hash, password)
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    channel = getattr(request.state, "qtfy_channel", None)
    visitor_id = getattr(request.state, "qtfy_visitor_id", None)
    trial_ends_at = await asyncio.to_thread(claim_trial, email)
    try:
        conn=db()
        conn.execute(
            "INSERT INTO users(email,password_hash,salt,is_active,verify_token_hash,verify_expires,created_at,trial_ends_at,pref_theme,signup_utm_source) VALUES(?,?,?,0,?,?,?,?,'light',?)",
            (email,password_hash,salt,token_hash,time.time()+VERIFY_TOKEN_TTL,time.time(),trial_ends_at,channel),
        )
        conn.commit()
        conn.close()
    except sqlite3.IntegrityError:
        conn.close()
        return RedirectResponse("/signup?error=Email+already+registered",status_code=303)
    except Exception as e:
        print(f"[Error: {type(e).__name__}] Signup error: {e}")
        return RedirectResponse("/signup?error=Database+error",status_code=303)

    asyncio.create_task(asyncio.to_thread(_log_event, EVENT_SIGNUP, visitor_id, channel, email))
    await asyncio.to_thread(adopt_browser_language, request, email)
    if not await asyncio.to_thread(send_verification_email, request, email, token):
        return RedirectResponse("/login?msg=Account+created.+Verification+email+could+not+be+sent+-+contact+support.",status_code=303)
    return RedirectResponse("/check-email?email="+urllib.parse.quote(email),status_code=303)


@app.get("/check-email", response_class=HTMLResponse)
async def check_email_page(email: str = ""):
    email_esc = html_lib.escape(email)
    form = f'''<div class="card" style="text-align:center">
<div style="font-size:38px;margin-bottom:14px">&#9993;&#65039;</div>
<h2>Check your email</h2>
<div class="subtitle">We sent a verification link to<br><b style="color:#eef7f3">{email_esc}</b></div>
<p style="color:#7c9c90;font-size:13px;margin:18px 0 22px">Click the link in that email to activate your account, then come back and log in.</p>
<form action="/api/auth/resend-verification" method="post">
<input type="hidden" name="email" value="{email_esc}">
<button>Resend verification email</button>
</form>
<div class="links" style="justify-content:center;margin-top:18px"><a href="/login">Back to log in</a></div>
</div>'''
    return render_auth_page("Check your email", form)


@app.get("/verify-email", response_class=HTMLResponse)
async def verify_email(email: str, token: str):
    email = email.strip().lower()
    conn = db()
    row = conn.execute("SELECT verify_token_hash,verify_expires,is_active FROM users WHERE email=?", (email,)).fetchone()
    valid = (row and row["verify_token_hash"] and row["verify_expires"] and row["verify_expires"] > time.time()
             and hmac.compare_digest(row["verify_token_hash"], hashlib.sha256(token.encode()).hexdigest()))
    if not valid:
        conn.close()
        already = bool(row and row["is_active"])
        if already:
            return RedirectResponse("/login?msg=Email+already+verified.+You+can+log+in.", status_code=303)
        return RedirectResponse("/login?error=Invalid+or+expired+verification+link", status_code=303)
    conn.execute("UPDATE users SET is_active=1,verify_token_hash=NULL,verify_expires=NULL WHERE email=?", (email,))
    conn.commit(); conn.close()
    return RedirectResponse("/login?msg=Email+verified.+You+can+now+log+in.", status_code=303)


@app.post("/api/auth/resend-verification")
async def resend_verification(request: Request, email: str = Form(...)):
    email = email.strip().lower()
    if _is_locked_out(SEND_CODE_ATTEMPTS, "verify:"+email, SEND_CODE_MAX_ATTEMPTS):
        return RedirectResponse("/login?error=Too+many+requests.+Wait+15+minutes+and+try+again.", status_code=303)
    _register_failed_attempt(SEND_CODE_ATTEMPTS, "verify:"+email)
    conn = db()
    row = conn.execute("SELECT is_active FROM users WHERE email=?", (email,)).fetchone()
    if row and not row["is_active"]:
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        conn.execute("UPDATE users SET verify_token_hash=?,verify_expires=? WHERE email=?",
                     (token_hash, time.time()+VERIFY_TOKEN_TTL, email))
        conn.commit()
        conn.close()
        await asyncio.to_thread(send_verification_email, request, email, token)
    else:
        conn.close()
    return RedirectResponse("/login?msg=If+that+account+needs+verification,+a+new+email+was+sent.", status_code=303)


@app.get("/logout")
async def logout(request: Request):
    raw=request.cookies.get("session_user")
    if raw:
        try:
            conn=db()
            conn.execute("DELETE FROM sessions WHERE token_hash=?",(hashlib.sha256(raw.encode()).hexdigest(),))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[Error: {type(e).__name__}] Logout session deletion error: {e}")
    res=RedirectResponse("/login",status_code=303); res.delete_cookie("session_user"); return res


@app.get("/forgot-password", response_class=HTMLResponse)
async def forgot_page(error: Optional[str] = None):
    error = html_lib.escape(error) if error else ''
    form = f'''<div class="card"><h2>Reset your password</h2><div class="subtitle">Enter your email and we'll send you a 6-digit code.</div><div class="error">{error}</div><form action="/api/auth/send-code" method="post"><label>Email</label><input type="email" name="email" required autocomplete="email" inputmode="email" autocapitalize="none" autocorrect="off"><button>Send code</button></form><div class="links"><a href="/login">Back to log in</a></div></div>'''
    return render_auth_page("Password Recovery", form)


@app.post("/api/auth/send-code")
async def send_code(email: str = Form(...)):
    email=email.strip().lower()
    if _is_locked_out(SEND_CODE_ATTEMPTS, email, SEND_CODE_MAX_ATTEMPTS):
        return RedirectResponse("/forgot-password?error=Too+many+code+requests.+Wait+15+minutes+and+try+again.",status_code=303)
    _register_failed_attempt(SEND_CODE_ATTEMPTS, email)
    try:
        conn=db()
        row=conn.execute("SELECT email FROM users WHERE email=?",(email,)).fetchone()
        if row:
            code=f"{secrets.randbelow(1_000_000):06d}"; code_hash=hashlib.sha256(code.encode()).hexdigest()
            conn.execute("UPDATE users SET reset_code_hash=?,reset_expires=? WHERE email=?",(code_hash,time.time()+900,email))
            conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Error: {type(e).__name__}] Password reset code generation error: {e}")
        return RedirectResponse("/forgot-password?error=Database+error",status_code=303)

    if row:
        await asyncio.to_thread(send_email_notification, email,"[QUANTIFY.] Password Reset Code",f"Your code: {code}\nValid for: 15 minutes")
    # Respond identically regardless of whether the account exists, to avoid leaking registration status.
    return RedirectResponse("/reset-password?email="+urllib.parse.quote(email),status_code=303)


@app.get("/reset-password", response_class=HTMLResponse)
async def reset_page(email: str, error: Optional[str] = None):
    error = html_lib.escape(error) if error else ''
    email = html_lib.escape(email)
    form = f'''<div class="card"><h2>Choose a new password</h2><div class="subtitle">Enter the code we emailed you, plus a new password. Don't see it? Check spam, or wait a minute and try again.</div><div class="error">{error}</div><form action="/api/auth/verify-and-reset" method="post"><input type="hidden" name="email" value="{email}"><label>6-digit code</label><input name="code" required maxlength="6" inputmode="numeric" autocomplete="one-time-code" pattern="[0-9]*"><label>New password</label><input type="password" name="new_password" required autocomplete="new-password"><p class="hint">10+ characters, with at least 1 letter and 1 number</p><button>Reset password</button></form><div class="links"><a href="/forgot-password">Resend code</a></div></div>'''
    return render_auth_page("Reset Password", form)


@app.post("/api/auth/verify-and-reset")
async def verify_reset(email: str = Form(...), code: str = Form(...), new_password: str = Form(...)):
    email = email.strip().lower()
    if _is_locked_out(RESET_ATTEMPTS, email, RESET_MAX_ATTEMPTS):
        return RedirectResponse("/reset-password?email="+urllib.parse.quote(email)+"&error=Too+many+attempts.+Request+a+new+code.",status_code=303)
    ok,error=validate_password_policy(new_password)
    if not ok: return RedirectResponse("/reset-password?email="+urllib.parse.quote(email)+"&error="+urllib.parse.quote(error),status_code=303)
    try:
        conn=db()
        row=conn.execute("SELECT reset_code_hash,reset_expires FROM users WHERE email=?",(email,)).fetchone()
        valid=row and row["reset_code_hash"] and row["reset_expires"] and row["reset_expires"]>time.time() and hmac.compare_digest(row["reset_code_hash"],hashlib.sha256(code.encode()).hexdigest())
        if not valid:
            conn.close()
            _register_failed_attempt(RESET_ATTEMPTS, email)
            return RedirectResponse("/reset-password?email="+urllib.parse.quote(email)+"&error=Invalid+or+expired+code",status_code=303)
        _clear_attempts(RESET_ATTEMPTS, email)
        password_hash,salt=await asyncio.to_thread(make_password_hash, new_password)
        conn.execute("UPDATE users SET password_hash=?,salt=?,reset_code_hash=NULL,reset_expires=NULL WHERE email=?",(password_hash,salt,email))
        conn.execute("DELETE FROM sessions WHERE email=?",(email,))
        conn.commit(); conn.close()
    except Exception as e:
        print(f"[Error: {type(e).__name__}] Password reset processing error: {e}")
        return RedirectResponse("/reset-password?email="+urllib.parse.quote(email)+"&error=Database+error",status_code=303)
    return RedirectResponse("/login?msg=Password+reset+successfully",status_code=303)

# -----------------------------------------------------------------------------
# Public demo (no login) -- lets a skeptical visitor verify the product is real
# before being asked to create an account, instead of hitting the signup wall blind.
# -----------------------------------------------------------------------------
DEMO_AI_SECTION_LABELS = {
    "quant_review": "Quant Review",
    "supply_demand": "Supply/Demand",
    "risk_review": "Risk Review",
    "news_analysis": "News Analysis",
    "timing_reason": "Timing Rationale",
}

DEMO_CSS = """
:root{--bg:#000000;--panel:#000000;--panel2:#0a0a0a;--border:#222222;--border2:#181818;--text:#a8a8a8;--head:#ffffff;--dim:#787878;--green:#26a69a;--red:#ef5350;--orange:#ff9800}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--text);font:16px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;margin:0;padding:0 16px 40px;max-width:1280px;margin-left:auto;margin-right:auto}
a{color:var(--green)}
.topbar{display:flex;justify-content:space-between;align-items:center;padding:18px 4px;flex-wrap:wrap;gap:10px}
.brand{font-weight:700;font-size:19px;color:var(--head);text-decoration:none;letter-spacing:.2px}
.brand span{color:var(--dim)}
.signup-btn{background:var(--green);color:#fff;padding:9px 16px;border-radius:7px;font-weight:700;text-decoration:none;font-size:14px}
.panel{background:var(--panel);border:1px solid var(--border);padding:16px;border-radius:10px;overflow:hidden}
h3{font-size:14px;color:var(--dim);border-bottom:1px solid var(--border);padding-bottom:10px;margin:0 0 10px;text-transform:uppercase;letter-spacing:.4px;font-weight:700}
h3 small{color:var(--dim);font-weight:normal;text-transform:none;font-size:12px}
.intro-bar{margin-bottom:12px;line-height:1.6}
.intro-bar b{color:var(--head)}
.demo-grid{display:grid;grid-template-columns:330px 1fr;gap:12px;align-items:start}
.list{max-height:640px;overflow:auto}
.item{padding:12px;border-bottom:1px solid var(--border2);cursor:pointer;display:flex;justify-content:space-between;border-left:3px solid transparent}
.item:hover,.item.active-item{background:var(--panel2)}
.item.sig-favorable{border-left-color:var(--green)}
.item.sig-caution{border-left-color:var(--orange)}
.item.sig-risk{border-left-color:var(--red)}
.badge{padding:4px 11px;border-radius:12px;font-weight:700;display:inline-block;font-size:13px}
.badge-ok{background:rgba(38,166,154,.15);color:var(--green)}
.badge-warn{background:rgba(255,152,0,.15);color:var(--orange)}
.badge-danger{background:rgba(239,83,80,.15);color:var(--red)}
.badge-pending{background:var(--panel2);color:var(--dim)}
.metrics{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:10px}
.metric{background:var(--panel2);border:1px solid var(--border);padding:10px;text-align:center;border-radius:7px}
.metric>div:first-child{font-size:10.5px;color:var(--dim);text-transform:uppercase;letter-spacing:.2px}
.val{color:var(--head);font-weight:700;margin-top:5px;font-size:15px}
.ai-tldr{background:var(--panel2);border:1px solid var(--border);border-radius:10px;padding:14px 16px;margin:12px 0;font-size:15px;line-height:1.65}
.ai-tldr b{color:var(--head)}
.section{margin-bottom:14px}
.section b{color:var(--head);display:block;margin-bottom:5px;font-size:12.5px;text-transform:uppercase;letter-spacing:.3px}
.notice{padding:14px;background:var(--panel2);border:1px solid var(--border);margin-bottom:10px;line-height:1.6;border-radius:8px}
.action-bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:10px}
.action-btn{color:#fff;background:var(--green);border:1px solid var(--green);padding:9px 14px;border-radius:7px;font-weight:600;font-size:13.5px;text-decoration:none;display:inline-block}
.action-btn.locked-btn{background:var(--panel2);color:var(--dim);border-color:var(--border)}
.action-btn.locked-btn:hover{color:var(--head);border-color:var(--green)}
.demo-cta{background:var(--panel2);border:1px solid var(--green);border-radius:10px;padding:26px;text-align:center;margin-top:16px}
.demo-cta h2{color:var(--head);margin:0 0 8px;font-size:21px}
.demo-cta p{color:var(--dim);margin:0 0 16px;font-size:14px}
.demo-cta .action-btn{padding:12px 26px;font-size:15px}
.demo-cta .cta-note{color:var(--dim);font-size:12.5px;margin-top:10px}
@media(max-width:900px){.demo-grid{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}}
"""

DEMO_JS = """
function showDemoTicker(t){
  document.querySelectorAll('.demo-detail').forEach(function(el){el.style.display='none'});
  var d=document.getElementById('demo-detail-'+t);
  if(d)d.style.display='block';
  document.querySelectorAll('.item').forEach(function(el){el.classList.remove('active-item')});
  var it=document.getElementById('demo-item-'+t);
  if(it)it.classList.add('active-item');
}
"""


def _demo_badge_class(verdict):
    return {"Favorable": "badge-ok", "Caution": "badge-warn", "Risk": "badge-danger"}.get(verdict, "badge-pending")


def _demo_sig_class(verdict):
    return {"Favorable": "sig-favorable", "Caution": "sig-caution", "Risk": "sig-risk"}.get(verdict, "")


def _demo_sparkline_svg(arr):
    if not arr or len(arr) < 2:
        return ""
    w, h = 48, 18
    mn, mx = min(arr), max(arr)
    rng = (mx - mn) or 1
    pts = " ".join(f"{(i / (len(arr) - 1) * w):.1f},{(h - ((v - mn) / rng * h)):.1f}" for i, v in enumerate(arr))
    color = "#26a69a" if arr[-1] >= arr[0] else "#ef5350"
    return f'<svg width="{w}" height="{h}" style="vertical-align:middle;flex-shrink:0"><polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5"/></svg>'


def _demo_row_html(t):
    ticker = html_lib.escape(t["ticker"])
    change_pct = t["change_pct"]
    change_str = f"{'+' if change_pct is not None and change_pct >= 0 else ''}{change_pct}"
    return (
        f'<div class="item {_demo_sig_class(t["timing_verdict"])}" id="demo-item-{ticker}" onclick="showDemoTicker(\'{ticker}\')">'
        f'<b>{ticker}</b>'
        f'<span style="display:flex;align-items:center;gap:6px">{_demo_sparkline_svg(t["sparkline"])}'
        f'<span style="text-align:right">{t["price"]} · {change_str}%<br>'
        f'<small>Score {t["overall_score"]} · <span class="badge {_demo_badge_class(t["timing_verdict"])}">{html_lib.escape(t["timing_verdict"] or "Analyzing")}</span></small></span></span>'
        f'</div>'
    )


def _demo_detail_html(t, is_first):
    ticker = html_lib.escape(t["ticker"])
    high52 = f'{t["pct_from_52w_high"]}%' if t["pct_from_52w_high"] is not None else "N/A"
    low52 = f'{t["pct_from_52w_low"]}%' if t["pct_from_52w_low"] is not None else "N/A"
    trend = "N/A" if t["above_200d_sma"] is None else ("Uptrend" if t["above_200d_sma"] else "Downtrend")
    volume = f'{t["volume_ratio"]}x avg' if t["volume_ratio"] is not None else "N/A"
    sections = t["sections"] or {}
    sections_html = "".join(
        f'<div class="section"><b>{label}</b>{sections[key]}</div>'
        for key, label in DEMO_AI_SECTION_LABELS.items() if sections.get(key)
    ) or '<div class="notice">AI analysis unavailable for this ticker today.</div>'
    verdict = t["timing_verdict"] or "Analyzing"
    display = "block" if is_first else "none"
    return f'''<div class="demo-detail" id="demo-detail-{ticker}" style="display:{display}">
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px">
<h3 style="border:0;margin:0;padding:0">{ticker}</h3>
<span class="badge {_demo_badge_class(t["timing_verdict"])}">{html_lib.escape(verdict)}</span>
</div>
<div class="action-bar">
<a class="action-btn locked-btn" href="/signup" title="Sign up to use this">🔒 Set Alert</a>
<a class="action-btn locked-btn" href="/signup" title="Sign up to use this">🔒 Add to Watchlist</a>
<a class="action-btn locked-btn" href="/signup" title="Sign up to use this">🔒 Save to Portfolio</a>
</div>
<div class="metrics">
<div class="metric"><div>RSI / MACD</div><div class="val">{t["rsi"]} / {t["macd"]}</div></div>
<div class="metric"><div>52W High</div><div class="val">{high52}</div></div>
<div class="metric"><div>52W Low</div><div class="val">{low52}</div></div>
<div class="metric"><div>Trend</div><div class="val">{trend}</div></div>
<div class="metric"><div>Volume</div><div class="val">{volume}</div></div>
</div>
<h3 style="margin-top:16px">AI Quant Report <small>(informational only, not investment advice)</small></h3>
<div class="scroll">{sections_html}</div>
</div>'''


# -----------------------------------------------------------------------------
# Public per-ticker pages (/stock/<TICKER>, /stocks) -- search-visible surface built
# from the scan snapshot that's already stored. Read-only from daily_scans: no
# external API call ever runs on one of these requests.
# -----------------------------------------------------------------------------
STOCK_PAGE_CSS = """
.stock-head{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:6px}
.stock-badge{padding:5px 14px;border-radius:14px;font-weight:700;font-size:14px;display:inline-block}
.badge-ok{background:#e6f5ee;color:#0e8a5f}
.badge-warn{background:#fbf1e0;color:#a8660a}
.badge-danger{background:#fbe6e2;color:#c8402c}
.badge-pending{background:#f0f2f1;color:#6b7873}
.stock-score{font-size:40px;font-weight:800;color:#12201a;line-height:1}
.stock-score span{font-size:16px;color:#6b7873;font-weight:600}
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:26px 0 8px}
.metric-box{border:1px solid var(--border);background:var(--panel2);border-radius:10px;padding:14px 16px}
.metric-box .k{font-size:12px;color:var(--dim);text-transform:uppercase;letter-spacing:.3px;font-weight:700}
.metric-box .v{color:var(--head);font-weight:700;font-size:19px;margin-top:5px}
.locked-report{border:1px solid var(--border);background:var(--panel2);border-radius:12px;padding:22px;margin:10px 0 8px;text-align:center}
.locked-report p{color:var(--dim2);font-size:15px;margin-bottom:16px}
.updated{color:var(--dim);font-size:14px;margin-top:8px}
.stock-index{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px;margin:18px 0 34px}
.stock-index a{display:flex;justify-content:space-between;align-items:center;gap:8px;border:1px solid var(--border);border-radius:9px;padding:11px 13px;text-decoration:none;color:var(--head);font-weight:600}
.stock-index a:hover{border-color:var(--green);background:var(--panel2)}
.stock-index .sc{font-size:13px;color:var(--dim);font-weight:600}
"""


def _rsi_word(rsi):
    if rsi is None:
        return None
    return "oversold" if rsi < 30 else ("overbought" if rsi > 70 else "neutral")


def _stock_plain_english(d):
    """Deterministic, per-ticker English built from that row's own numbers.

    Every covered ticker gets a page, but only the handful that clear the quant filter
    ever get an AI write-up -- without this, the several hundred others would be
    identical number dumps (thin content, and useless to a reader).
    """
    t = d["ticker"]
    price, chg = d["price"], d["change_pct"]
    high, low = d["pct_from_52w_high"], d["pct_from_52w_low"]
    above = d["above_200d_sma"]
    rsi, vol = d["rsi"], d["volume_ratio"]
    alpha = d["alpha_score"]
    bits = []

    move = f"up {chg}%" if (chg or 0) > 0 else (f"down {abs(chg)}%" if (chg or 0) < 0 else "flat")
    bits.append(f"{t} last traded at ${price} ({move} on the day).")

    trend_txt = []
    if high is not None:
        trend_txt.append(f"{abs(high)}% below its 52-week high")
    if low is not None:
        trend_txt.append(f"{round(low)}% above its 52-week low")
    if trend_txt:
        sentence = f"It is {' and '.join(trend_txt)}"
        if above is not None:
            sentence += f", and trading {'above' if above else 'below'} its 200-day moving average"
        bits.append(sentence + ".")

    detail = []
    if rsi is not None:
        detail.append(f"RSI is {rsi} ({_rsi_word(rsi)})")
    if vol is not None:
        detail.append(f"volume is running at {vol}x its recent average")
    if detail:
        # No .capitalize() here -- it would lowercase the rest and turn "RSI" into "Rsi".
        sentence = " and ".join(detail)
        bits.append(sentence[0].upper() + sentence[1:] + ".")

    if d["quant_pass"]:
        verdict = d["timing_verdict"]
        score = d["overall_score"]
        if verdict and score is not None:
            bits.append(
                f"{t} cleared QUANTIFY's daily scan with a combined score of {score}/100 "
                f"and an AI timing read of {verdict}."
            )
        else:
            bits.append(
                f"{t} cleared QUANTIFY's quant filter in the latest scan; its AI timing "
                f"review has not finished running yet."
            )
    else:
        # Say *why* it didn't qualify -- that's the genuinely useful part for a reader
        # who searched this ticker, and it differs by ticker because it reads the data.
        # SQLite hands this back as 0/1, and `0 is False` is False in Python -- compare
        # by truthiness so a below-trend ticker actually gets the trend explanation.
        if above is not None and not above:
            why = (f"it is trading below its 200-day moving average, so it fails the "
                   f"long-term uptrend requirement outright")
        elif high is not None and abs(high) < 3:
            why = (f"it is only {abs(high)}% off its 52-week high, so there is no "
                   f"meaningful pullback to buy into")
        else:
            why = (f"its pullback setup scored {alpha}/100, short of the "
                   f"{round(QUANT_PASS_THRESHOLD)} the strategy requires")
        bits.append(
            f"{t} did not make QUANTIFY's list in the latest scan — {why}. The strategy "
            f"only flags stocks pulling back inside a confirmed long-term uptrend."
        )
    return " ".join(bits)


def _stock_metrics_html(d):
    def box(k, v):
        return f'<div class="metric-box"><div class="k">{k}</div><div class="v">{v}</div></div>'
    high = f'{d["pct_from_52w_high"]}%' if d["pct_from_52w_high"] is not None else "N/A"
    low = f'{d["pct_from_52w_low"]}%' if d["pct_from_52w_low"] is not None else "N/A"
    trend = "N/A" if d["above_200d_sma"] is None else ("Uptrend" if d["above_200d_sma"] else "Downtrend")
    vol = f'{d["volume_ratio"]}x avg' if d["volume_ratio"] is not None else "N/A"
    chg = d["change_pct"]
    chg_txt = f'{"+" if (chg or 0) >= 0 else ""}{chg}%' if chg is not None else "N/A"
    return (
        '<div class="metric-grid">'
        + box("Price", f'${d["price"]}')
        + box("Change", chg_txt)
        + box("RSI / MACD", f'{d["rsi"]} / {d["macd"]}')
        + box("52W High", high)
        + box("52W Low", low)
        + box("Trend", trend)
        + box("Volume", vol)
        + box("Quant score", f'{d["alpha_score"]}/100')
        + "</div>"
    )


def _latest_scan_row(ticker):
    conn = db()
    row = conn.execute("""
        SELECT ticker,universe,scan_date,price,change_pct,alpha_score,rsi,macd,
               pct_from_52w_high,pct_from_52w_low,above_200d_sma,volume_ratio,
               timing_score,timing_verdict,ai_report,ai_updated_at,quant_pass,created_at
        FROM daily_scans WHERE ticker=? ORDER BY scan_date DESC, id DESC LIMIT 1
    """, (ticker,)).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["overall_score"] = (
        round((d["alpha_score"] + d["timing_score"]) / 2, 1)
        if d["alpha_score"] is not None and d["timing_score"] is not None else None
    )
    return d


@app.get("/stock/{ticker}", response_class=HTMLResponse)
async def stock_page(ticker: str):
    ticker = normalize_ticker(ticker)
    if not re.fullmatch(r"[A-Z0-9.\-^=]{1,15}", ticker):
        return HTMLResponse("Not found", status_code=404)
    d = _latest_scan_row(ticker)
    if not d:
        return HTMLResponse("Not found", status_code=404)

    verdict = d["timing_verdict"]
    score = d["overall_score"] if d["overall_score"] is not None else d["alpha_score"]
    badge = (f'<span class="stock-badge {_demo_badge_class(verdict)}">{html_lib.escape(verdict)}</span>'
             if verdict else '<span class="stock-badge badge-pending">Not on today\'s list</span>')
    updated_ts = d["ai_updated_at"] or d["created_at"]
    updated = (datetime.fromtimestamp(updated_ts, ZoneInfo("America/New_York")).strftime("%b %d, %Y at %I:%M %p ET")
               if updated_ts else d["scan_date"])

    sections = {}
    if d["ai_report"]:
        try:
            sections = json.loads(d["ai_report"])
        except (json.JSONDecodeError, TypeError):
            sections = {}
    teaser = ""
    if sections.get("quant_review"):
        teaser = (f'<h2>AI quant review</h2><p>{sections["quant_review"]}</p>'
                  '<div class="locked-report"><p><b>Supply/Demand, Risk Review, News Analysis and Timing '
                  'Rationale</b> for this ticker are part of the full report.</p>'
                  '<a class="btn" href="/signup">Read the full report — 7 days free</a></div>')

    body = f'''<div class="eyebrow">{html_lib.escape(d["universe"] or "Scanned universe")} &middot; scan of {d["scan_date"]}</div>
<h1>Is {ticker} a buy right now?</h1>
<div class="stock-head"><div class="stock-score">{score if score is not None else "—"}<span>/100</span></div>{badge}</div>
<p class="sublead">QUANTIFY's quant scan and plain-English read on {ticker}, from the latest run.</p>
<p>{_stock_plain_english(d)}</p>
{_stock_metrics_html(d)}
<div class="updated">Last updated {updated}. The scanner recomputes four times each trading day.</div>
{teaser}
<h2>See the rest of today's scan</h2>
<p>QUANTIFY scans the S&amp;P 500 and Nasdaq-100 every trading day for stocks pulling back inside a
long-term uptrend, then has AI check each one for blow-off-top and dead-cat-bounce risk.
The full detected list, price alerts and portfolio tracking are available to members —
or <a href="/demo">try the live demo</a> with no signup.</p>
<p><a class="btn" href="/signup">Start your 7-day free trial</a></p>
<p style="margin-top:18px"><a href="/stocks">Browse every ticker QUANTIFY covers &rarr;</a></p>
<div class="disclaimer"><b>Not investment advice.</b> QUANTIFY is informational and educational only.
Scores and AI commentary describe historical and current data — they are never a recommendation to
buy or sell any security. Every investment decision, and its outcome, is your own.</div>'''

    return render_marketing_page(
        f"Is {ticker} a buy right now?",
        f"Is {ticker} a buy right now? QUANTIFY's quant score, {('AI verdict, ' if verdict else '')}"
        f"key indicators and a plain-English breakdown from the latest daily scan.",
        body,
        path=f"/stock/{ticker}",
        extra_head=f"<style>{STOCK_PAGE_CSS}</style>",
    )


RECORD_PAGE_CSS = """
.rec-summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:22px 0 10px}
.rec-box{border:1px solid var(--border);background:var(--panel2);border-radius:10px;padding:15px 16px}
.rec-box .k{font-size:12px;color:var(--dim);text-transform:uppercase;letter-spacing:.3px;font-weight:700}
.rec-box .v{color:var(--head);font-weight:800;font-size:22px;margin-top:5px}
.rec-box .v.up{color:#0e8a5f}.rec-box .v.down{color:#c8402c}
.rec-wrap{overflow-x:auto;margin:8px 0 6px}
table.rec{border-collapse:collapse;width:100%;font-size:15px;min-width:660px}
table.rec th{text-align:left;font-size:12px;text-transform:uppercase;letter-spacing:.3px;color:var(--dim);border-bottom:1px solid var(--border);padding:9px 10px;white-space:nowrap}
table.rec td{border-bottom:1px solid var(--border);padding:10px;white-space:nowrap}
table.rec td.up{color:#0e8a5f;font-weight:700}
table.rec td.down{color:#c8402c;font-weight:700}
.chain{font-size:13.5px;color:var(--dim2);background:var(--bg-alt);border:1px solid var(--border);border-radius:9px;padding:12px 15px;margin:14px 0}
.chain.bad{border-color:#c8402c;color:#c8402c}
.chain code{font-size:12px;color:var(--dim)}
.method{border:1px solid var(--border);border-radius:10px;padding:18px 22px;margin-top:26px;font-size:14.5px;line-height:1.75;color:var(--dim2)}
.method b{color:var(--head)}
.sep-note{border-left:3px solid var(--orange);background:#fbf3e4;padding:14px 18px;border-radius:0 8px 8px 0;margin:10px 0 18px;font-size:14.5px;color:#6b5320}
"""


def _record_rows(list_type):
    """Every row for one list, id-ascending (chain order), with live prices attached."""
    conn = db()
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM track_record WHERE list_type=? ORDER BY id ASC", (list_type,)
    ).fetchall()]
    latest_scan = conn.execute("SELECT MAX(scan_date) FROM daily_scans").fetchone()[0]
    prices = {}
    if latest_scan:
        prices = {r["ticker"]: r["price"] for r in conn.execute(
            "SELECT ticker,price FROM daily_scans WHERE scan_date=?", (latest_scan,)
        ).fetchall()}
    conn.close()
    bench_now = _benchmark_close()
    for d in rows:
        cur = prices.get(d["ticker"])
        d["current_price"] = cur
        d["change_pct"] = round((cur - d["entry_price"]) / d["entry_price"] * 100, 2) if cur and d["entry_price"] else None
        d["bench_change_pct"] = (
            round((bench_now - d["benchmark_price"]) / d["benchmark_price"] * 100, 2)
            if bench_now and d["benchmark_price"] else None
        )
    return rows


def _record_section(rows, label):
    """Summary + full table for one list. Renders every row -- no filtering, ever."""
    scored = [d for d in rows if d["change_pct"] is not None]
    winners = sum(1 for d in scored if d["change_pct"] > 0)
    losers = sum(1 for d in scored if d["change_pct"] < 0)
    avg = round(sum(d["change_pct"] for d in scored) / len(scored), 2) if scored else None
    bench = [d["bench_change_pct"] for d in scored if d["bench_change_pct"] is not None]
    avg_bench = round(sum(bench) / len(bench), 2) if bench else None

    def cls(v):
        return "up" if (v or 0) > 0 else ("down" if (v or 0) < 0 else "")

    def box(k, v, c=""):
        return f'<div class="rec-box"><div class="k">{k}</div><div class="v {c}">{v}</div></div>'

    summary = ('<div class="rec-summary">'
               + box("Records", len(rows))
               + box("Up", winners, "up") + box("Down", losers, "down")
               + box("Avg change", f"{avg:+}%" if avg is not None else "—", cls(avg))
               + box("S&P, same span", f"{avg_bench:+}%" if avg_bench is not None else "—", cls(avg_bench))
               + "</div>")

    if not rows:
        return summary + f"<p>No {label} recorded yet — the first entries appear after the next scan.</p>"

    body = "".join(
        f'<tr><td><a href="/stock/{d["ticker"]}">{d["ticker"]}</a></td>'
        f'<td>{d["record_date"]}</td>'
        f'<td>${d["entry_price"]}</td>'
        f'<td>{d["alpha_score"]}</td>'
        f'<td>{"$" + str(d["current_price"]) if d["current_price"] is not None else "—"}</td>'
        f'<td class="{cls(d["change_pct"])}">{f"{d['change_pct']:+}%" if d["change_pct"] is not None else "—"}</td>'
        f'<td>{f"{d['bench_change_pct']:+}%" if d["bench_change_pct"] is not None else "—"}</td></tr>'
        for d in sorted(rows, key=lambda x: (x["record_date"], x["ticker"]), reverse=True)
    )
    table = ('<div class="rec-wrap"><table class="rec"><thead><tr>'
             '<th>Ticker</th><th>Recorded</th><th>Price then</th><th>Score then</th>'
             '<th>Price now</th><th>Change</th><th>S&amp;P same span</th>'
             '</tr></thead><tbody>' + body + '</tbody></table></div>')
    return summary + table


@app.get("/unsubscribe", response_class=HTMLResponse)
async def unsubscribe_page(token: str = ""):
    return _do_unsubscribe(token)


@app.post("/unsubscribe", response_class=HTMLResponse)
async def unsubscribe_post(token: str = ""):
    # Gmail/Apple one-click unsubscribe POSTs to this same URL (RFC 8058).
    return _do_unsubscribe(token)


def _do_unsubscribe(token: str):
    ok = False
    if token:
        try:
            conn = db()
            row = conn.execute("SELECT email FROM users WHERE unsub_token=?", (token,)).fetchone()
            if row:
                conn.execute("UPDATE users SET pref_marketing_emails=0 WHERE unsub_token=?", (token,))
                conn.commit()
                ok = True
            conn.close()
        except Exception as exc:
            print(f"[Error: {type(exc).__name__}] Unsubscribe failed: {exc}")
    body = ('<h1>Unsubscribed</h1><p class="sublead">You won\'t get any more onboarding or '
            'update emails from QUANTIFY.</p><p>You will still get account emails — password '
            'resets, email verification, and price alerts you set up yourself.</p>'
            '<p>Changed your mind? Email support and we\'ll turn them back on.</p>'
            if ok else
            '<h1>Link not recognised</h1><p class="sublead">That unsubscribe link is not valid — '
            'it may have already been used or been cut short by your email client.</p>'
            '<p>Reply to any QUANTIFY email and we\'ll take you off the list manually.</p>')
    return render_marketing_page("Unsubscribe", "Manage QUANTIFY email preferences.", body, path="/unsubscribe")


@app.get("/record", response_class=HTMLResponse)
async def record_page():
    passed = _record_rows("passed")
    near = _record_rows("near_miss")
    ok, broken_at = verify_track_chain(passed)
    ok_near, _ = verify_track_chain(near)
    started = passed[0]["record_date"] if passed else None

    if ok and ok_near:
        chain = ('<div class="chain"><b>Chain verified.</b> Each row is hashed together with the '
                 f'hash of the row before it, so editing or removing any past entry breaks every '
                 f'hash after it. All {len(passed) + len(near)} rows currently verify.</div>')
    else:
        chain = ('<div class="chain bad"><b>Chain verification FAILED</b> — an entry does not match '
                 f'its hash{f" (row {broken_at + 1})" if broken_at is not None else ""}. '
                 'Treat these numbers as untrustworthy until this is explained.</div>')

    body = f'''<h1>Track record</h1>
<p class="sublead">Every stock QUANTIFY's scanner has flagged, recorded automatically on the day
it was flagged — winners and losers, nothing removed.</p>
{chain}
<h2>Flagged by the scanner</h2>
{_record_section(passed, "picks")}
<div class="method"><b>How this is recorded.</b> The scanner runs four times every trading day.
The first run that flags a ticker writes it here immediately, with the price and score it had at
that moment — later runs the same day cannot change that entry. There is no screen, button or API
anywhere in this product that lets anyone add, edit or delete a row: the scanner is the only writer.
{f"Recording began {started}." if started else "Recording begins with the next scan."}
<br><br>
<b>How the numbers are calculated.</b> Closing prices only. No commissions, no slippage, no
position sizing, no compounding, and no assumption about when you would have sold — "change" is
simply the move from the recorded price to the most recent scan price. The S&amp;P column is the
index's move over that same span, for context.</div>
<h2 style="margin-top:46px">Separate watch list — did not qualify</h2>
<div class="sep-note">These were the {NEAR_MISS_COUNT} highest-scoring tickers each day that
<b>did not pass</b> the filter, so they were never picks. They are tracked out of curiosity on a separate
chain, with separately calculated numbers, and are <b>not part of the record above</b> — they are
never counted, averaged or summed into it.</div>
{_record_section(near, "near misses")}
<h2 style="margin-top:46px">See the current scan</h2>
<p><a href="/demo">Try the live demo</a> with no signup, or <a href="/stocks">browse every ticker</a>
QUANTIFY covers.</p>
<p><a class="btn" href="/signup">Start your 7-day free trial</a></p>
<div class="disclaimer"><b>Not investment advice.</b> Past results — including everything on this
page — do not predict future returns. QUANTIFY is informational and educational only, and nothing
here is a recommendation to buy or sell any security. These figures ignore commissions, slippage,
taxes and position sizing, so they are not achievable real-world returns. Every investment
decision, and its outcome, is your own.</div>'''

    return render_marketing_page(
        "Track record",
        "Every stock QUANTIFY's scanner has flagged, recorded automatically on the day it was "
        "flagged — including the losers. No filtering, no hand-picked examples.",
        body,
        path="/record",
        extra_head=f"<style>{RECORD_PAGE_CSS}</style>",
    )


@app.get("/stocks", response_class=HTMLResponse)
async def stocks_index():
    conn = db()
    latest = conn.execute("SELECT MAX(scan_date) FROM daily_scans").fetchone()[0]
    rows = conn.execute("""
        SELECT ticker,universe,alpha_score,timing_verdict,
               ROUND((alpha_score+timing_score)/2.0,1) AS overall_score
        FROM daily_scans WHERE scan_date=? ORDER BY ticker
    """, (latest,)).fetchall() if latest else []
    conn.close()

    groups = {}
    for r in rows:
        groups.setdefault(r["universe"] or "Other", []).append(r)
    sections = []
    for uni, items in sorted(groups.items()):
        links = "".join(
            f'<a href="/stock/{r["ticker"]}"><span>{r["ticker"]}</span>'
            f'<span class="sc">{r["overall_score"] if r["overall_score"] is not None else r["alpha_score"]}'
            f'{" &middot; " + r["timing_verdict"] if r["timing_verdict"] else ""}</span></a>'
            for r in items
        )
        sections.append(f'<h2>{html_lib.escape(uni)} <span style="color:var(--dim);font-size:16px;font-weight:500">({len(items)})</span></h2>'
                        f'<div class="stock-index">{links}</div>')

    body = f'''<h1>Every stock QUANTIFY covers</h1>
<p class="sublead">Quant score, AI verdict and a plain-English breakdown for all
{len(rows)} tickers in the S&amp;P 500 and Nasdaq-100 — updated four times each trading day.</p>
{"".join(sections) if sections else "<p>The first scan has not completed yet — check back shortly.</p>"}
<div class="disclaimer"><b>Not investment advice.</b> QUANTIFY is informational and educational only.
Nothing here is a recommendation to buy or sell any security.</div>'''

    return render_marketing_page(
        "Every stock we cover",
        f"Quant scores and plain-English breakdowns for all {len(rows)} S&P 500 and Nasdaq-100 "
        "tickers QUANTIFY scans, updated four times every trading day.",
        body,
        path="/stocks",
        extra_head=f"<style>{STOCK_PAGE_CSS}</style>",
    )



# -----------------------------------------------------------------------------
# Open Graph preview image
# -----------------------------------------------------------------------------
# A real screenshot of the live product, not an illustration -- the whole point of the
# card is to show what someone actually gets when they click. Served from the repo so it
# needs no image library and no headless browser on the server; regenerate it by
# screenshotting /demo at 1200x630 whenever the UI changes.
OG_IMAGE_PATH = "/og/scanner.png"
OG_IMAGE_URL = f"https://quantify.trading{OG_IMAGE_PATH}"
OG_IMAGE_FILE = Path(__file__).resolve().parent / "static" / "og-scanner.png"
_OG_IMAGE_BYTES: Optional[bytes] = None
try:
    _OG_IMAGE_BYTES = OG_IMAGE_FILE.read_bytes()
except Exception as _og_err:
    print(f"[og] preview image not found at {OG_IMAGE_FILE} ({_og_err}) — og:image tags omitted", flush=True)


@app.get(OG_IMAGE_PATH)
async def og_scanner_image():
    if not _OG_IMAGE_BYTES:
        return Response(status_code=404)
    return Response(
        content=_OG_IMAGE_BYTES,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


def _og_image_tags() -> str:
    """Emitted only when the file is actually on disk -- an og:image pointing at a 404
    makes the preview card worse than having no image tag at all."""
    if not _OG_IMAGE_BYTES:
        return '<meta name="twitter:card" content="summary">'
    return ('<meta property="og:image" content="' + OG_IMAGE_URL + '">'
            '<meta property="og:image:width" content="1200">'
            '<meta property="og:image:height" content="630">'
            '<meta property="og:image:type" content="image/png">'
            '<meta property="og:image:alt" content="The QUANTIFY scanner showing today\'s detected stocks with scores and AI risk verdicts">'
            '<meta name="twitter:card" content="summary_large_image">'
            '<meta name="twitter:image" content="' + OG_IMAGE_URL + '">')


def og_head(title: str, description: str, path: str = "") -> str:
    """The full Open Graph / Twitter block for a public page. One helper so every public
    page gets the same tags and the image can never drift between them."""
    url = f"https://quantify.trading{path}"
    esc = html_lib.escape
    return (f'<meta property="og:title" content="{esc(title)}">'
            f'<meta property="og:description" content="{esc(description)}">'
            f'<meta property="og:type" content="website">'
            f'<meta property="og:url" content="{esc(url)}">'
            f'<meta property="og:site_name" content="QUANTIFY.">'
            + _og_image_tags())


@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    conn = db()
    # Same rule as the scanner: a day that is still being scored would otherwise show
    # here as a handful of tickers, and this is the page a first-time visitor lands on.
    latest_date = display_scan_date(conn)
    rows = []
    if latest_date:
        rows = conn.execute("""
            SELECT ticker,universe,price,change_pct,rsi,macd,pct_from_52w_high,pct_from_52w_low,
                   above_200d_sma,volume_ratio,timing_verdict,ai_report,
                   ROUND((alpha_score+timing_score)/2.0,1) AS overall_score
            FROM daily_scans WHERE scan_date=? AND quant_pass=1 AND timing_score IS NOT NULL
            ORDER BY overall_score DESC LIMIT 6
        """, (latest_date,)).fetchall()
    detected_total = conn.execute(
        "SELECT COUNT(*) FROM daily_scans WHERE scan_date=? AND quant_pass=1 AND timing_score IS NOT NULL",
        (latest_date,),
    ).fetchone()[0] if latest_date else 0
    conn.close()

    tickers = []
    for r in rows:
        d = dict(r)
        sections = {}
        if d.get("ai_report"):
            try:
                sections = json.loads(d["ai_report"])
            except (json.JSONDecodeError, TypeError):
                sections = {}
        cached = CACHE["historical"].get(f"single:{d['ticker']}:1d")
        sparkline = []
        if cached is not None:
            try:
                sparkline = [round(float(v), 2) for v in normalize_series(cached["data"], "Close").dropna().tail(20)]
            except Exception:
                sparkline = []
        tickers.append({**d, "sections": sections, "sparkline": sparkline})

    if detected_total and detected_total > len(tickers):
        scanner_caption = f"top {len(tickers)} of {detected_total} detected today"
    else:
        scanner_caption = f"{len(tickers)} real signals today"

    if not tickers:
        list_html = '<div class="notice">Demo is warming up — check back shortly.</div>'
        detail_html = ""
    else:
        rows_html = [_demo_row_html(t) for t in tickers]
        mid = len(tickers) // 2
        rows_html.insert(mid, '<div class="notice">Like what you see? Get the full daily scan — 7-day free trial. <a href="/signup">Sign up</a></div>')
        list_html = "".join(rows_html)
        detail_html = "".join(_demo_detail_html(t, i == 0) for i, t in enumerate(tickers))

    body = f'''<!doctype html><html lang="en" data-theme="dark"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>QUANTIFY. Live Demo</title><meta name="description" content="See QUANTIFY's real, current scan results — score, indicators, and AI risk review — no signup required.">{og_head("QUANTIFY. Live Demo", "See QUANTIFY's real, current scan results — score, indicators, and AI risk review. No signup required.", "/demo")}<link rel="canonical" href="https://quantify.trading/demo"><style>{DEMO_CSS}</style></head><body>
<div class="topbar"><a class="brand" href="/">QUANTIFY<span>.</span></a><a class="signup-btn" href="/signup">Sign up free</a></div>
<div class="panel intro-bar"><b>Live Demo</b> — real results from today's scan (informational only, not investment advice). <a href="/signup">Create a free account</a> to unlock alerts, watchlist, and portfolio tracking.</div>
<div class="demo-grid">
<section class="panel"><h3>Market Scanner <small>({scanner_caption})</small></h3><div class="list">{list_html}</div></section>
<section class="panel">{detail_html or '<div class="notice">No demo data yet — check back after the next scan.</div>'}</section>
</div>
<section class="panel demo-cta">
<h2>Like what you see?</h2>
<p>Get the full daily scan, real-time alerts, and portfolio tracking.</p>
<a class="action-btn" href="/signup">Start Free Trial — 7 Days Free</a>
<div class="cta-note">No credit card required.</div>
</section>
<script>{DEMO_JS}</script>
</body></html>'''
    return HTMLResponse(body)


# -----------------------------------------------------------------------------
# Dashboard
# -----------------------------------------------------------------------------
@app.get("/terminal", response_class=HTMLResponse)
async def dashboard(request: Request):
    user=get_logged_in_user(request)
    if not user: return RedirectResponse("/login",status_code=303)
    if not disclaimer_accepted(user): return RedirectResponse("/accept-disclaimer",status_code=303)
    if not has_active_access(user): return RedirectResponse("/subscription?reason=trial_ended",status_code=303)
    conn=db(); prefs=conn.execute("SELECT pref_theme,pref_language,pref_default_sort,pref_default_view,trial_ends_at FROM users WHERE email=?",(user,)).fetchone(); conn.close()
    theme = prefs["pref_theme"] if prefs and prefs["pref_theme"] in ("dark","light") else "light"
    pref_language = prefs["pref_language"] if prefs and prefs["pref_language"] in LANGUAGE_NAMES else "en"
    pref_default_sort = prefs["pref_default_sort"] if prefs and prefs["pref_default_sort"] in ("overall_score","change_pct","ticker") else "overall_score"
    pref_default_view = prefs["pref_default_view"] if prefs and prefs["pref_default_view"] in ("list","heatmap") else "list"
    trial_ends_at = prefs["trial_ends_at"] if prefs else None
    trial_ends_str = datetime.fromtimestamp(trial_ends_at).strftime("%B %d, %Y") if trial_ends_at else ""
    avatar_letter = html_lib.escape(user[0].upper()) if user else "?"
    user=html_lib.escape(user)
    lang = pref_language
    html = f'''<!doctype html><html lang="{lang}" data-theme="{theme}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>QUANTIFY.</title><script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script><style>
:root{{--bg:#ffffff;--panel:#ffffff;--panel2:#f5f7f6;--border:#e2e6e3;--border2:#ececec;--text:#3a4440;--head:#12201a;--dim:#77837e;--green:#0e8a5f;--red:#c8402c;--orange:#a8660a;--grid-line:#eef1ef;
--sb-bg:#12181b;--sb-border:#232b2f;--sb-text:#9aa7ac;--sb-text-active:#ffffff;--sb-hover:#1b2327;--sb-danger:#e57373}}
html[data-theme="dark"]{{--bg:#000000;--panel:#000000;--panel2:#0a0a0a;--border:#222222;--border2:#181818;--text:#a8a8a8;--head:#ffffff;--dim:#787878;--green:#26a69a;--red:#ef5350;--orange:#ff9800;--grid-line:#161616}}
*{{box-sizing:border-box}}body{{background:var(--bg);color:var(--text);font:16px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;margin:0;padding:12px 12px 12px 216px}}
header,.panel{{background:var(--panel);border:1px solid var(--border)}}
.sidebar{{position:fixed;left:0;top:0;bottom:0;width:196px;background:var(--sb-bg);border-right:1px solid var(--sb-border);display:flex;flex-direction:column;align-items:stretch;padding:18px 12px;gap:3px;z-index:40;overflow-y:auto}}
.sidebar .side-brand{{color:var(--sb-text-active);font-weight:800;font-size:17px;letter-spacing:-.2px;margin:2px 6px 18px;text-decoration:none}}
.sidebar .side-brand span{{color:var(--green)}}
.side-link{{display:flex;align-items:center;gap:12px;padding:10px 12px;border-radius:9px;color:var(--sb-text);text-decoration:none;font-size:14.5px;font-weight:600;cursor:pointer;background:transparent;border:1px solid transparent}}
.side-link svg{{flex-shrink:0}}
.side-link:hover{{background:var(--sb-hover);color:var(--sb-text-active)}}
.side-link.active{{background:var(--green);color:#ffffff;border-color:var(--green)}}
.lang-toggle{{display:inline-block;padding:5px 10px;border:1px solid var(--sb-border);border-radius:999px;color:var(--sb-text);font-size:12.5px;font-weight:700;text-decoration:none;line-height:1;white-space:nowrap}}
.lang-toggle:hover{{border-color:var(--green);color:var(--green)}}

.side-spacer{{flex:1;min-height:8px}}
@media(max-width:900px){{
  body{{padding-left:12px;padding-bottom:70px}}
  .sidebar{{left:0;right:0;top:auto;bottom:0;width:auto;height:60px;flex-direction:row;justify-content:space-around;align-items:center;border-right:none;border-top:1px solid var(--sb-border);padding:0;overflow-y:visible;overflow-x:auto}}
  .sidebar .side-brand,.side-spacer{{display:none}}
  .side-link{{flex-direction:column;gap:3px;padding:6px 8px;font-size:9.5px}}
}}

/* Score card -- the score used to live only as a line of text inside the AI panel,
   which meant the single number the whole product produces was the least visible thing
   on the page. */
.score-card{{background:var(--panel2);border:1px solid var(--border);border-radius:10px;padding:12px 14px;margin-bottom:10px}}
.score-head{{display:flex;gap:16px;align-items:center;flex-wrap:wrap}}
.snowflake-wrap{{flex-shrink:0;width:248px;height:248px}}
.sf-title{{font-size:11.5px;font-weight:800;letter-spacing:.6px;text-transform:uppercase;color:var(--dim);margin:14px 0 2px}}
.sf-sector{{font-size:12px;color:var(--dim);margin-bottom:8px}}
.sf-sector b{{color:var(--head)}}
.sf-legend{{display:grid;grid-template-columns:repeat(2,1fr);gap:4px 14px;margin-top:4px}}
.sf-item{{display:flex;justify-content:space-between;align-items:baseline;gap:8px;padding:4px 0;border-bottom:1px solid var(--border2);font-size:13.5px}}
.sf-item .nm{{font-weight:700;color:var(--head);display:flex;align-items:center;gap:4px;white-space:nowrap}}
.sf-item .sc{{font-weight:800;font-variant-numeric:tabular-nums}}
.sf-item .rw{{font-size:11.5px;color:var(--dim);text-align:right;flex:1}}
.sf-item.empty .nm,.sf-item.empty .sc{{color:var(--dim);font-weight:600}}
.sf-foot{{font-size:11.5px;color:var(--dim);line-height:1.6;margin-top:9px}}
@media(max-width:640px){{
  .snowflake-wrap{{width:min(300px,84vw);height:min(300px,84vw);margin:0 auto}}
  .sf-legend{{grid-template-columns:1fr}}
  .sf-item{{font-size:14px}}
}}
.snowflake-wrap svg{{width:100%;height:100%;display:block;overflow:visible}}
.score-main{{flex:1;min-width:0}}
.score-num{{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;margin-bottom:2px}}
.score-num b{{font-size:44px;line-height:1;color:var(--head);font-weight:800;letter-spacing:-1px;font-variant-numeric:tabular-nums}}
.score-num .outof{{font-size:15px;color:var(--dim);font-weight:600}}
.score-split{{font-size:12.5px;color:var(--dim);margin-bottom:10px}}
.score-split b{{color:var(--head);font-variant-numeric:tabular-nums}}
.score-axes{{display:grid;grid-template-columns:1fr;gap:1px;font-size:12px;margin-top:8px}}
.score-detail-btn{{margin-top:8px;padding:4px 10px;font-size:11.5px;font-weight:700;color:var(--dim);background:transparent;border:1px solid var(--border);border-radius:999px;cursor:pointer}}
.score-detail-btn:hover{{color:var(--head);border-color:var(--dim)}}
.score-axis{{display:flex;justify-content:space-between;gap:8px;align-items:center;padding:2px 0;border-bottom:1px solid var(--border2)}}
.score-axis .nm{{color:var(--dim);letter-spacing:.3px;display:flex;align-items:center;gap:3px}}
.score-axis.scored .nm{{color:var(--head);font-weight:700}}
.score-axis .rw{{color:var(--dim);font-variant-numeric:tabular-nums;text-align:right}}
.score-axis.scored .rw{{color:var(--head)}}
.score-note{{font-size:11.5px;color:var(--dim);line-height:1.6;margin-top:9px}}
.score-note b{{color:var(--head)}}
@media(max-width:640px){{
  /* This block sits after the snowflake rules, so a width here overrides them. Keep it
     to the score number only -- the radar's own mobile size is set with its other
     styles above, and shrinking it here is what made the axis labels unreadable. */
  .score-head{{gap:10px}}
  .score-num b{{font-size:34px}}
}}
header{{padding:12px 18px;display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;gap:12px;flex-wrap:wrap;border-radius:10px}}
.brand{{font-weight:700;font-size:19px;color:var(--head);text-decoration:none;letter-spacing:.2px}}
.brand span{{color:var(--dim)}}
.headerRight{{display:flex;align-items:center;gap:8px;flex-wrap:wrap}}
.userTag{{font-size:13px;color:var(--dim);margin-right:4px}}
button,input,select{{background:var(--panel2);border:1px solid var(--border);color:var(--text);padding:9px 12px;font:14px/1 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;border-radius:7px}}
button{{color:var(--head);cursor:pointer;font-weight:600;transition:background .15s,border-color .15s}}
button:hover{{background:var(--border)}}
button.danger{{color:var(--red);border-color:var(--red)}}
button.tf-btn.active{{color:#fff;background:var(--green);border-color:var(--green)}}
.grid{{display:grid;grid-template-columns:330px 1fr 360px;gap:12px;height:calc(100vh - 78px)}}
.panel{{padding:16px;overflow:hidden;display:flex;flex-direction:column;border-radius:10px}}
h3{{font-size:14px;color:var(--dim);border-bottom:1px solid var(--border);padding-bottom:10px;margin:0 0 10px;text-transform:uppercase;letter-spacing:.4px;font-weight:700}}
.list,.scroll{{overflow:auto;flex:1}}
.item{{padding:12px;border-bottom:1px solid var(--border2);cursor:pointer;display:flex;justify-content:space-between;border-left:3px solid transparent}}
.item:hover{{background:var(--panel2)}}
.item.sig-favorable{{border-left-color:var(--green)}}
.item.sig-caution{{border-left-color:var(--orange)}}
.item.sig-risk{{border-left-color:var(--red)}}
/* The middle panel's content (chart + legend + index minis + five metric tiles) has
   always been taller than the fixed grid height, and .panel clips rather than scrolls,
   so the bottom was being cut off -- the score card made an existing overflow obvious.
   Give the chart a real height and let the panel scroll instead of swallowing content. */
#detailPanel{{overflow-y:auto}}
.chart{{flex:none;height:min(42vh,400px);min-height:260px}}
.idx-row{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px}}
.idx-box{{background:var(--panel2);border:1px solid var(--border);padding:8px;border-radius:7px}}
.idx-label{{font-size:11px;color:var(--dim);margin-bottom:4px;display:flex;justify-content:space-between}}
.idx-chart{{height:60px}}
.metrics{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:10px}}
.metric{{background:var(--panel2);border:1px solid var(--border);padding:10px;text-align:center;border-radius:7px}}
.metric>div:first-child{{font-size:10.5px;color:var(--dim);text-transform:uppercase;letter-spacing:.2px}}
.val{{color:var(--head);font-weight:700;margin-top:5px;font-size:15px}}
.notice{{padding:14px;background:var(--panel2);border:1px solid var(--border);margin-bottom:10px;line-height:1.6;border-radius:8px}}
a{{color:var(--green);text-decoration:underline}}
.badge{{padding:4px 11px;border-radius:12px;font-weight:700;display:inline-block;font-size:13px}}
.badge-ok{{background:#e6f5ee;color:var(--green)}}
.badge-warn{{background:#fbf1e0;color:var(--orange)}}
.badge-danger{{background:#fbe6e2;color:var(--red)}}
.badge-pending{{background:var(--panel2);color:var(--dim)}}
html[data-theme="dark"] .badge-ok{{background:rgba(38,166,154,.15)}}
html[data-theme="dark"] .badge-warn{{background:rgba(255,152,0,.15)}}
html[data-theme="dark"] .badge-danger{{background:rgba(239,83,80,.15)}}
.section{{margin-bottom:14px}}
.section b{{color:var(--head);display:block;margin-bottom:5px;font-size:12.5px;text-transform:uppercase;letter-spacing:.3px}}
.tabs{{display:flex;gap:6px;margin-bottom:10px}}
.tab{{flex:1;padding:9px;font-size:13px;background:var(--panel2);color:var(--dim);border:1px solid var(--border);cursor:pointer;border-radius:7px;font-weight:600}}
.tab.active{{color:#fff;background:var(--green);border-color:var(--green)}}
.sortbar{{display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap}}.sortbar select{{flex:1;min-width:100px;font-size:13px}}
.heatmap{{overflow:auto;flex:1;display:flex;flex-direction:column;gap:4px}}
.heat-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(52px,1fr));gap:3px}}
.heat-tile{{padding:8px 3px;text-align:center;font-size:11px;font-weight:700;cursor:pointer;color:#0a1f14;border-radius:5px}}
.group-header,.heat-group-header{{font-size:12px;color:var(--dim);background:var(--panel2);padding:6px 9px;letter-spacing:.3px;position:sticky;top:0;z-index:1;text-transform:uppercase;font-weight:700}}
.legend{{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--dim);margin-top:8px}}
.legend span{{display:inline-flex;align-items:center;gap:5px}}
.legend i{{width:10px;height:2px;display:inline-block}}
.legend i.dash{{border-top:1px dashed currentColor;background:transparent;height:0;color:#9b6bff}}
.earnings-info{{font-size:13px;color:var(--dim);margin-top:10px;padding-top:10px;border-top:1px solid var(--border)}}
.earnings-info b{{color:var(--head)}}
.earnings-info .beat{{color:var(--green)}}
.earnings-info .miss{{color:var(--red)}}
.toast{{position:fixed;bottom:20px;right:20px;background:var(--panel);border:1px solid var(--border);color:var(--head);padding:14px 20px;border-radius:8px;box-shadow:0 10px 30px rgba(18,32,26,.18);font-size:14px;z-index:100;opacity:0;transform:translateY(10px);transition:opacity .2s,transform .2s;pointer-events:none;max-width:300px}}
.toast.show{{opacity:1;transform:translateY(0)}}
.toast.err{{border-color:var(--red);color:var(--red)}}
.help-icon{{display:inline-flex;align-items:center;justify-content:center;width:16px;height:16px;border-radius:50%;background:var(--panel2);border:1px solid var(--border);color:var(--dim);font-size:11px;font-weight:700;cursor:help;margin-left:5px;position:relative;vertical-align:middle}}
.help-icon .tip-bubble{{display:none;position:absolute;bottom:calc(100% + 6px);left:50%;transform:translateX(-50%);background:var(--panel);border:1px solid var(--border);color:var(--head);padding:10px 12px;border-radius:8px;font-size:13px;font-weight:400;line-height:1.55;width:220px;white-space:normal;z-index:60;box-shadow:0 10px 24px rgba(18,32,26,.18);text-align:left}}
.help-icon:hover .tip-bubble,.help-icon.open .tip-bubble{{display:block}}
.ai-tldr{{background:var(--panel2);border:1px solid var(--border);border-radius:10px;padding:14px 16px;margin-bottom:12px;font-size:15px;line-height:1.65}}
.ai-tldr b{{color:var(--head)}}
.tldr-next{{margin-top:10px;padding-top:10px;border-top:1px dashed var(--border);color:var(--dim);font-size:13.5px;line-height:1.55}}
.action-bar{{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start;padding:8px 10px;background:var(--green-soft,rgba(14,138,95,.08));border:1px solid var(--green);border-radius:8px}}
/* The shares and entry-price inputs already existed, but sat unlabelled among five
   other controls in one undifferentiated row -- on a phone, behind a collapsed bar
   called "Track this ticker". Two named groups make what the bar can do legible. */
.action-group{{display:flex;flex-direction:column;gap:5px}}
.action-group-label{{font-size:10.5px;font-weight:800;letter-spacing:.6px;text-transform:uppercase;color:var(--green)}}
.action-row{{display:flex;gap:7px;flex-wrap:wrap;align-items:center}}
.action-hint{{font-size:11.5px;color:var(--dim);line-height:1.45}}
.action-hint a{{color:var(--dim);text-decoration:underline}}
.action-hint a:hover{{color:var(--head)}}
.action-btn{{color:#fff;background:var(--green);border-color:var(--green)}}
.action-btn:hover{{opacity:.88;background:var(--green)}}
.manage-alerts-link{{font-size:12.5px;color:var(--dim);text-decoration:underline;padding:0 2px;white-space:nowrap}}
.manage-alerts-link:hover{{color:var(--text)}}
.tf-group{{display:flex;gap:5px}}
.usage-tip{{margin-top:12px;padding-top:12px;border-top:1px solid var(--border);color:var(--dim);font-size:13.5px;line-height:1.65}}
.onboard-overlay{{display:none;position:fixed;inset:0;background:rgba(18,32,26,.55);z-index:200;align-items:center;justify-content:center;padding:16px}}
.onboard-overlay.show{{display:flex}}
.onboard-card{{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:30px;max-width:440px;width:100%;max-height:85vh;overflow-y:auto;box-shadow:0 24px 60px rgba(18,32,26,.25)}}
.onboard-card h3{{color:var(--head);font-size:19px;margin-bottom:16px}}
.onboard-item{{display:flex;gap:12px;margin-bottom:16px;align-items:flex-start}}
.onboard-item .badge-demo{{flex-shrink:0;margin-top:1px}}
.onboard-item p{{color:var(--text);font-size:14.5px;line-height:1.65;margin:0}}
.onboard-item b{{color:var(--head)}}
.onboard-card button{{width:100%;margin-top:8px;background:var(--green);color:#fff;border:none;padding:13px;border-radius:9px;font-weight:700;font-size:15px;cursor:pointer}}
.help-fab{{position:fixed;bottom:20px;left:96px;width:36px;height:36px;border-radius:50%;background:var(--panel);border:1px solid var(--border);color:var(--head);font-weight:700;font-size:15px;cursor:pointer;z-index:50;display:flex;align-items:center;justify-content:center;box-shadow:0 4px 14px rgba(18,32,26,.12)}}
@media(max-width:900px){{.help-fab{{left:16px;bottom:76px}}}}
.avatar-wrap{{position:relative}}
.avatar{{width:36px;height:36px;border-radius:50%;background:var(--panel2);border:1px solid var(--border);color:var(--head);font-weight:700;cursor:pointer;display:flex;align-items:center;justify-content:center;padding:0;font-size:15px}}
.avatar-menu{{position:absolute;top:calc(100% + 8px);right:0;background:var(--panel);border:1px solid var(--border);border-radius:10px;min-width:190px;overflow:hidden;z-index:50;box-shadow:0 16px 36px rgba(18,32,26,.18)}}
.avatar-menu a{{display:block;padding:13px 16px;color:var(--head);text-decoration:none;font-size:14px;font-weight:600}}
.avatar-menu a:hover{{background:var(--panel2)}}
.avatar-menu a.danger-text{{color:var(--red)}}
.avatar-menu .email-row{{padding:13px 16px;color:var(--dim);font-size:12.5px;border-bottom:1px solid var(--border);word-break:break-all}}
.below-grid{{display:flex;flex-direction:column;gap:12px;margin-top:12px;max-width:1400px}}
.panel.wide{{overflow:visible}}
.empty-hint{{color:var(--dim);font-size:15px;padding:10px 0}}
.summary-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}}
.summary-tile{{background:var(--panel2);border:1px solid var(--border);padding:14px;border-radius:8px;text-align:center}}
.summary-tile .label{{font-size:11.5px;color:var(--dim);text-transform:uppercase;letter-spacing:.3px;font-weight:600}}
.summary-tile .value{{color:var(--head);font-weight:800;font-size:20px;margin-top:6px}}
.summary-tile .value.gain{{color:var(--green)}}
.summary-tile .value.loss{{color:var(--red)}}
.watch-add-row{{display:flex;gap:8px;margin-bottom:12px}}
.watch-add-row input{{flex:1}}
.watch-row{{display:flex;justify-content:space-between;align-items:center;padding:12px 0;border-bottom:1px solid var(--border2);font-size:14.5px}}
.watch-row:last-child{{border-bottom:none}}
.watch-row b{{cursor:pointer}}
.watch-row b:hover{{text-decoration:underline}}
.remove-btn{{background:transparent;border:1px solid var(--red);color:var(--red);padding:6px 12px;font-size:13px;font-weight:600;border-radius:6px}}
.backtest-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}}
.backtest-card{{background:var(--panel2);border:1px solid var(--border);padding:16px;border-radius:8px}}
.backtest-card h4{{margin:0 0 12px;font-size:13px;color:var(--dim);text-transform:uppercase;letter-spacing:.3px;font-weight:700}}
.backtest-row{{display:flex;justify-content:space-between;padding:6px 0;font-size:14.5px}}
.backtest-row b.gain{{color:var(--green)}}
.backtest-row b.loss{{color:var(--red)}}
.backtest-meta{{font-size:13px;color:var(--dim);margin-top:12px;line-height:1.6}}
@media(max-width:900px){{
  body{{padding:10px;height:auto;overflow-y:auto}}
  .grid{{grid-template-columns:1fr;height:auto}}
  .panel{{max-height:none}}
  #list,#heatmap{{max-height:360px}}
  .chart{{height:auto;min-height:300px}}
  .metrics{{grid-template-columns:repeat(2,1fr)}}
  .idx-row{{grid-template-columns:1fr}}
  header{{padding:12px}}
  .headerRight select{{flex:1;min-width:150px}}
}}
.mobile-actions-toggle{{display:none}}
@media(max-width:560px){{
  /* Seven alert/portfolio controls in one bar eat a third of a phone screen before
     the chart and AI report -- collapse them behind one tap instead. */
  .mobile-actions-toggle{{display:block;width:100%;background:var(--panel2);border:1px solid var(--green);color:var(--green);font-weight:700;padding:11px}}
  .action-bar{{display:none;width:100%;flex-direction:column;gap:12px}}
  .action-bar.open{{display:flex}}
  .action-group{{width:100%}}
  .action-bar input,.action-bar select{{flex:1;min-width:0;width:auto!important;padding:10px}}
  .action-bar .action-btn{{flex:1 1 100%;padding:11px}}
  .action-bar .manage-alerts-link{{flex:1 1 100%;text-align:center;padding-top:2px}}
  .tf-group{{width:100%}}
  .tf-group .tf-btn{{flex:1}}
  .metrics{{grid-template-columns:repeat(2,1fr);gap:6px}}
  .metric{{padding:8px}}
  .sortbar select{{min-width:calc(50% - 3px)}}
}}
</style></head><body>
{_render_sidebar("scanner", lang)}
<header><a class="brand" href="/terminal">QUANTIFY<span>.</span></a><div class="headerRight"><div class="avatar-wrap"><button class="avatar" onclick="event.stopPropagation();toggleAvatarMenu()" title="{user}">{avatar_letter}</button><div class="avatar-menu" id="avatarMenu" style="display:none"><div class="email-row">{user}</div><a href="/subscription">My Subscription</a><a href="/contact">Contact Us</a><a href="/logout" class="danger-text">Log out</a></div></div></div></header><div class="onboard-overlay" id="onboardOverlay"><div class="onboard-card"><h3>Quick guide to QUANTIFY</h3><div class="onboard-item"><span class="badge-demo"><span class="badge badge-ok">Favorable</span></span><p><b>Badges</b> are the AI's read on entry timing: <b>Favorable</b> (setup looks clean), <b>Caution</b> (some risk worth knowing about), or <b>Risk</b> (skip or wait). Never a buy/sell order.</p></div><div class="onboard-item"><span class="badge-demo">📊</span><p><b>Score (0-100)</b> combines the quant scan (is this a long-term uptrend that's pulled back to a good entry zone?) with the AI's risk check. Only names that clear the bar show up at all.</p></div><div class="onboard-item"><span class="badge-demo">🔍</span><p><b>The scanner list</b> updates a few times a day — tap any ticker to load its chart, technicals, and full AI report.</p></div><div class="onboard-item"><span class="badge-demo">❔</span><p>Little <b>?</b> icons next to unfamiliar terms (RSI, MACD, Trend...) explain what they mean — tap or hover any of them anytime.</p></div><button onclick="closeOnboarding()">Got it</button></div></div><button class="help-fab" onclick="openOnboarding()" title="Quick guide">?</button><div class="grid"><section class="panel"><h3>Market Scanner <span id="ucount"></span></h3><div class="tabs"><button class="tab active" id="tabList" onclick="showView('list')">List</button><button class="tab" id="tabHeatmap" onclick="showView('heatmap')">Heatmap</button></div><input id="tickerInput" placeholder="Jump to ticker (e.g. TSLA)" onkeydown="if(event.key==='Enter')loadTicker(this.value)"><div class="sortbar" id="sortbar"><select id="sortKey" onchange="renderList()"><option value="overall_score">Sort: Score</option><option value="change_pct">Sort: Change %</option><option value="ticker">Sort: Ticker A-Z</option></select><select id="filterBadge" onchange="renderList()"><option value="">All Badges</option><option value="Favorable">Favorable</option><option value="Caution">Caution</option><option value="Risk">Risk</option></select><select id="filterUniverse" onchange="renderList()"><option value="">All Markets</option><option value="S&amp;P 500">S&amp;P 500</option><option value="Nasdaq-100">Nasdaq-100</option></select><select id="filterSector" onchange="renderList()"><option value="">All Sectors</option></select></div><div id="scanAsOf" style="display:none;background:rgba(255,152,0,.1);border:1px solid rgba(255,152,0,.35);color:var(--orange);padding:6px 10px;border-radius:6px;font-size:11.5px;line-height:1.5;margin-bottom:8px"></div>
<div class="list" id="list">Preparing constituent list...</div><div class="heatmap" id="heatmap" style="display:none"></div></section><section class="panel" id="detailPanel"><div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px"><h3 id="title" style="border:0;margin:0;padding:0">AAPL</h3><div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center"><button class="mobile-actions-toggle" onclick="toggleActionBar()">&#9733; Set an alert or add to portfolio</button><div class="action-bar" id="actionBar" title="Track this ticker without deciding anything right now">
<div class="action-group">
<span class="action-group-label">Price alert</span>
<div class="action-row"><select id="targetDir" title="Alert when price rises to/above, or falls to/below, the target" style="padding:0 4px"><option value="above">&#8593; at/above</option><option value="below">&#8595; at/below</option></select><input id="target" type="number" placeholder="Target price $" style="width:100px" title="Get an email when the price reaches this value"><button class="action-btn" onclick="setAlert()" title="Email me when the price hits my target">Set Alert</button></div>
<span class="action-hint"><a href="/settings#alerts" class="manage-alerts-link" title="View or cancel your existing price alerts">Manage alerts</a></span>
</div>
<div class="action-group">
<span class="action-group-label">Add to portfolio</span>
<div class="action-row"><input id="portfolioShares" type="number" placeholder="Shares" style="width:76px" title="How many shares you own or are tracking"><input id="portfolioPrice" type="number" placeholder="Entry price $" style="width:104px" title="What you paid — defaults to today's scan price if left blank"><button class="action-btn" onclick="savePortfolio()" title="Add this ticker to My Portfolio">Save to Portfolio</button></div>
<span class="action-hint">Leave entry price blank to use today's scan price. Not sure how many shares? <a href="/portfolio#sizing">Position sizing calculator</a></span>
</div>
</div><div class="tf-group"><button class="tf-btn" data-tf="1h" onclick="changeTF('1h')">1H</button><button class="tf-btn active" data-tf="1d" onclick="changeTF('1d')">1D</button><button class="tf-btn" data-tf="1wk" onclick="changeTF('1wk')">1W</button><button class="tf-btn" data-tf="1mo" onclick="changeTF('1mo')">1M</button></div></div></div><div class="score-card" id="scoreCard" style="display:none">
<div class="score-head">
<div class="snowflake-wrap"><svg id="snowflake" viewBox="-30 -14 260 232" role="img" aria-label="Score breakdown radar"></svg></div>
<div class="score-main">
<div class="score-num"><b id="scoreBig">-</b><span class="outof" id="scoreOutOf">/ 100</span></div>
<div id="scoreBadge" style="margin:4px 0 6px"></div>
<div class="score-split" id="scoreSplit"></div>
</div>
</div>
<div class="score-axes" id="scoreAxes"></div>
<div class="score-note" id="scoreNote"></div>
</div>
<div id="staleWarning" style="display:none;background:rgba(255,152,0,.12);border:1px solid rgba(255,152,0,.4);color:var(--orange);padding:6px 10px;border-radius:6px;font-size:11.5px;font-weight:600;margin-bottom:6px"></div><div id="chart" class="chart"></div><div class="legend"><span><i style="background:var(--head)"></i>SMA 20</span><span><i style="background:var(--orange)"></i>SMA 50</span><span><i style="background:var(--red)"></i>SMA 200</span><span><i class="dash"></i>Bollinger Bands</span><span><i style="background:var(--green)"></i>Volume</span></div><div class="earnings-info" id="earningsInfo">Earnings: -</div><div class="idx-row"><div class="idx-box"><div class="idx-label"><span>S&amp;P 500 · 60D</span><span id="idx-sp500-val"></span></div><div id="idx-sp500" class="idx-chart"></div></div><div class="idx-box"><div class="idx-label"><span>NASDAQ-100 · 60D</span><span id="idx-ndx-val"></span></div><div id="idx-ndx" class="idx-chart"></div></div></div><div class="metrics"><div class="metric"><div>RSI / MACD<span class="help-icon" onclick="event.stopPropagation();this.classList.toggle('open')">?<span class="tip-bubble">RSI: below 30 usually means oversold, above 70 usually means overbought. MACD: positive means upward momentum, negative means downward.</span></span></div><div id="rsi" class="val">-</div></div><div class="metric"><div>52W High<span class="help-icon" onclick="event.stopPropagation();this.classList.toggle('open')">?<span class="tip-bubble">How far the price is below its highest point in the last 52 weeks. Closer to 0% means near the high.</span></span></div><div id="high52" class="val">-</div></div><div class="metric"><div>52W Low<span class="help-icon" onclick="event.stopPropagation();this.classList.toggle('open')">?<span class="tip-bubble">How far the price is above its lowest point in the last 52 weeks.</span></span></div><div id="low52" class="val">-</div></div><div class="metric"><div>Trend<span class="help-icon" onclick="event.stopPropagation();this.classList.toggle('open')">?<span class="tip-bubble">Whether the price is above (Uptrend) or below (Downtrend) its 200-day moving average — a common gauge of the long-term direction.</span></span></div><div id="trend" class="val">-</div></div><div class="metric"><div>Score Trend (Today)<span class="help-icon" onclick="event.stopPropagation();this.classList.toggle('open')">?<span class="tip-bubble">How this ticker's quant score has moved since today's first scan — rising or falling.</span></span></div><div id="scoretrend" class="val">-</div></div></div></section><section class="panel"><h3>AI Quant Report <small style="color:var(--dim);font-weight:normal;text-transform:none">(informational only, not investment advice)</small></h3><div id="aiTldr" class="ai-tldr" style="display:none"></div><div id="verdict" style="display:none;margin-bottom:10px"></div><div id="scoreDrift" style="display:none;margin-bottom:10px;padding:8px 10px;border-radius:6px;font-size:12.5px;line-height:1.5;background:rgba(255,152,0,.1);border:1px solid rgba(255,152,0,.35);color:var(--orange)"></div><div id="ai" class="scroll">Loading AI analysis based on real data...</div><div class="usage-tip">This flags entry timing on a single ticker, not a full plan. Many investors cap any one pick at a small slice of their total portfolio and spread bets across several signals rather than one — sizing and diversification are on you, not this tool.</div><h3 style="margin-top:12px">News</h3><div id="news" class="scroll">Waiting for news...</div></section></div>
<div class="toast" id="toast"></div><script>
const USER_LANGUAGE='{pref_language}';
const SCORE_DETAIL_SHOW={json.dumps(t("score_detail_show", lang))};
const SCORE_DETAIL_HIDE={json.dumps(t("score_detail_hide", lang))};
const SF_TEXT={json.dumps({k: t(k, lang) for k in (
    "sf_value","sf_growth","sf_profit","sf_health","sf_dividend","sf_ai",
    "sf_value_help","sf_growth_help","sf_profit_help","sf_health_help","sf_dividend_help","sf_ai_help",
    "sf_title","sf_vs_sector","sf_no_data","sf_footnote")})};
const SCORE_SPLIT_NOTE={json.dumps(t("score_split", lang))};
const SCORE_QUANT_ONLY={json.dumps(t("score_quant_only", lang))};
const SCAN_PREVIOUS_DAY={json.dumps(t("scan_previous_day", lang))};
const DEFAULT_SORT='{pref_default_sort}';
const DEFAULT_VIEW='{pref_default_view}';
const TRIAL_ENDS_STR='{trial_ends_str}';
const STRATEGY_MODE='Long-Term Momentum Pullback';
let ticker='AAPL',tf='1d',chart,candle,volume,smaLines={{}},idxCharts={{}},bbLines={{}},currentView='list',lastSignals=[],lastUpdated=null;
function showToast(msg,isErr,duration){{const t=document.getElementById('toast');t.textContent=msg;t.className='toast show'+(isErr?' err':'');clearTimeout(window._toastTimer);window._toastTimer=setTimeout(()=>t.classList.remove('show'),duration||3500)}}
function toggleActionBar(){{document.getElementById('actionBar').classList.toggle('open')}}
function openOnboarding(){{document.getElementById('onboardOverlay').classList.add('show')}}
function closeOnboarding(){{document.getElementById('onboardOverlay').classList.remove('show');try{{localStorage.setItem('onboarded','1')}}catch(e){{}}}}
document.addEventListener('click',(e)=>{{document.querySelectorAll('.help-icon.open').forEach(el=>{{if(!el.contains(e.target))el.classList.remove('open')}})}});
function toggleAvatarMenu(){{const m=document.getElementById('avatarMenu');m.style.display=m.style.display==='none'?'block':'none'}}
document.addEventListener('click',()=>{{const m=document.getElementById('avatarMenu');if(m)m.style.display='none'}});
function showView(v){{currentView=v;document.getElementById('tabList').classList.toggle('active',v==='list');document.getElementById('tabHeatmap').classList.toggle('active',v==='heatmap');document.getElementById('sortbar').style.display=v==='list'?'flex':'none';document.getElementById('list').style.display=v==='list'?'block':'none';document.getElementById('heatmap').style.display=v==='heatmap'?'flex':'none';if(v==='heatmap')loadHeatmap()}}
function heatColor(chg){{if(chg==null)return '#333';const c=Math.max(-5,Math.min(5,chg));const t=(c+5)/10;const r=Math.round(239+(38-239)*t),g=Math.round(83+(166-83)*t),b=Math.round(80+(154-80)*t);return `rgb(${{r}},${{g}},${{b}})`}}
function groupByUniverse(items){{const groups={{}};items.forEach(t=>{{const g=t.universe||'Other';(groups[g]=groups[g]||[]).push(t)}});return groups}}
async function loadHeatmap(){{const el=document.getElementById('heatmap');const r=await fetch('/api/heatmap');const d=await r.json();if(!d.tiles?.length){{el.innerHTML='<div class="notice">No scan data yet.</div>';return}}const groups=groupByUniverse(d.tiles);el.innerHTML=Object.entries(groups).map(([g,items])=>`<div class="heat-group-header">${{g}} (${{items.length}})</div><div class="heat-grid">`+items.map(t=>`<div class="heat-tile" style="background:${{heatColor(t.change_pct)}}" title="${{t.ticker}} · ${{t.change_pct??'-'}}% · Alpha ${{t.alpha_score??'-'}}" onclick="showView('list');loadTicker('${{t.ticker}}')">${{t.ticker}}</div>`).join('')+'</div>').join('')}}
function sparklineSVG(arr){{if(!arr||arr.length<2)return '';const w=48,h=18;const min=Math.min(...arr),max=Math.max(...arr),range=(max-min)||1;const pts=arr.map((v,i)=>`${{(i/(arr.length-1)*w).toFixed(1)}},${{(h-((v-min)/range*h)).toFixed(1)}}`).join(' ');const color=arr[arr.length-1]>=arr[0]?'#26a69a':'#ef5350';return `<svg width="${{w}}" height="${{h}}" style="vertical-align:middle;flex-shrink:0"><polyline points="${{pts}}" fill="none" stroke="${{color}}" stroke-width="1.5"/></svg>`}}
function verdictClass(v){{return v==='Favorable'?'badge-ok':v==='Caution'?'badge-warn':v==='Risk'?'badge-danger':'badge-pending'}}
function renderEarnings(e){{const el=document.getElementById('earningsInfo');if(!e||(!e.last&&!e.next)){{el.innerText='Earnings: no data available';return}}const parts=[];if(e.last){{const beat=e.last.beat;const cls=beat===true?'beat':beat===false?'miss':'';const label=beat===true?'Beat':beat===false?'Miss':'Met';const surprise=e.last.surprise_pct!=null?` (${{label}} ${{e.last.surprise_pct>0?'+':''}}${{e.last.surprise_pct}}%)`:'';parts.push(`Last earnings <b>${{e.last.date}}</b>: EPS $${{e.last.eps_actual}} vs $${{e.last.eps_estimate??'-'}} est.<span class="${{cls}}">${{surprise}}</span>`)}}if(e.next){{parts.push(`Next earnings: <b>${{e.next.date}}</b>`)}}el.innerHTML=parts.join(' &middot; ')}}
function itemSigClass(v){{return v==='Favorable'?'sig-favorable':v==='Caution'?'sig-caution':v==='Risk'?'sig-risk':''}}
function populateSectorFilter(){{const sel=document.getElementById('filterSector');const current=sel.value;const sectors=[...new Set(lastSignals.map(s=>s.sector).filter(Boolean))].sort();sel.innerHTML='<option value="">All Sectors</option>'+sectors.map(s=>`<option value="${{s}}">${{s}}</option>`).join('');if(sectors.includes(current))sel.value=current}}
function renderList(){{if(!lastSignals.length)return;const badge=document.getElementById('filterBadge').value;const uni=document.getElementById('filterUniverse').value;const sector=document.getElementById('filterSector').value;const key=document.getElementById('sortKey').value;const filtered=lastSignals.filter(s=>(!badge||s.timing_verdict===badge)&&(!uni||s.universe===uni)&&(!sector||s.sector===sector));const sorted=[...filtered].sort((a,b)=>key==='ticker'?a.ticker.localeCompare(b.ticker):(b[key]??-Infinity)-(a[key]??-Infinity));document.getElementById('list').innerHTML=sorted.length?sorted.map(s=>`<div class="item ${{itemSigClass(s.timing_verdict)}}" onclick="loadTicker('${{s.ticker}}')"><b>${{s.ticker}}</b><span style="display:flex;align-items:center;gap:6px">${{sparklineSVG(s.sparkline)}}<span style="text-align:right">${{s.price}} · ${{s.change_pct}}%<br><small>Score ${{s.overall_score}} · <span class="badge ${{verdictClass(s.timing_verdict)}}">${{s.timing_verdict||'Analyzing'}}</span></small></span></span></div>`).join(''):'<div class="notice">No tickers match this filter.</div>'}}
const CHART_THEME=document.documentElement.getAttribute('data-theme')==='dark'?{{bg:'#000000',text:'#a8a8a8',grid:'#161616',up:'#26a69a',down:'#ef5350',sma20:'#e8e8e8',sma50:'#ff9800',sma200:'#ef5350',bb:'#9b6bff',vol:'rgba(38,166,154,.5)'}}:{{bg:'#ffffff',text:'#77837e',grid:'#eef1ef',up:'#0e8a5f',down:'#c8402c',sma20:'#4a5750',sma50:'#a8660a',sma200:'#c8402c',bb:'#7c5cd4',vol:'rgba(14,138,95,.4)'}};
function watchChartSize(el){{
  // Nothing ever resized the main chart after init(), so it kept whatever height the
  // container had on first paint. The score card arrives with the async AI response and
  // shrinks that container, which is what was clipping the chart.
  if(!('ResizeObserver' in window))return;
  let last=0;
  new ResizeObserver(()=>{{
    const w=el.clientWidth,h=el.clientHeight;
    if(!w||!h)return;
    const key=w*10000+h;
    if(key===last)return;
    last=key;
    try{{chart.resize(w,h)}}catch(e){{}}
  }}).observe(el);
}}
function init(){{const c=document.getElementById('chart');chart=LightweightCharts.createChart(c,{{width:c.clientWidth,height:c.clientHeight,layout:{{background:{{type:'solid',color:CHART_THEME.bg}},textColor:CHART_THEME.text}},grid:{{vertLines:{{color:CHART_THEME.grid}},horzLines:{{color:CHART_THEME.grid}}}},timeScale:{{timeVisible:false}}}});candle=chart.addCandlestickSeries({{upColor:CHART_THEME.up,downColor:CHART_THEME.down,borderUpColor:CHART_THEME.up,borderDownColor:CHART_THEME.down,wickUpColor:CHART_THEME.up,wickDownColor:CHART_THEME.down}});volume=chart.addHistogramSeries({{color:CHART_THEME.vol,priceFormat:{{type:'volume'}},priceScaleId:''}});volume.priceScale().applyOptions({{scaleMargins:{{top:.8,bottom:0}}}});smaLines.sma20=chart.addLineSeries({{color:CHART_THEME.sma20,lineWidth:1,priceLineVisible:false,lastValueVisible:false}});smaLines.sma50=chart.addLineSeries({{color:CHART_THEME.sma50,lineWidth:1,priceLineVisible:false,lastValueVisible:false}});smaLines.sma200=chart.addLineSeries({{color:CHART_THEME.sma200,lineWidth:1,priceLineVisible:false,lastValueVisible:false}});bbLines.upper=chart.addLineSeries({{color:CHART_THEME.bb,lineWidth:1,lineStyle:2,priceLineVisible:false,lastValueVisible:false}});bbLines.lower=chart.addLineSeries({{color:CHART_THEME.bb,lineWidth:1,lineStyle:2,priceLineVisible:false,lastValueVisible:false}});watchChartSize(c);window.onresize=()=>{{chart.resize(c.clientWidth,c.clientHeight);Object.entries(idxCharts).forEach(([k,ic])=>{{const el=document.getElementById('idx-'+k);if(el)ic.resize(el.clientWidth,el.clientHeight)}})}};['sp500','ndx'].forEach(k=>{{const el=document.getElementById('idx-'+k);const ic=LightweightCharts.createChart(el,{{width:el.clientWidth,height:el.clientHeight,layout:{{background:{{type:'solid',color:CHART_THEME.bg}},textColor:CHART_THEME.text,fontSize:9}},grid:{{vertLines:{{visible:false}},horzLines:{{visible:false}}}},rightPriceScale:{{visible:false}},timeScale:{{visible:false}},handleScroll:false,handleScale:false}});idxCharts[k]=ic;idxCharts[k+'_line']=ic.addLineSeries({{color:CHART_THEME.sma20,lineWidth:1.5,priceLineVisible:false,lastValueVisible:false}})}})}}
async function loadIndices(){{try{{const r=await fetch('/api/market-indices');const d=await r.json();const map={{sp500:d.sp500,ndx:d.nasdaq100}};Object.entries(map).forEach(([k,series])=>{{if(!series?.length)return;const boxEl=document.getElementById('idx-'+k);if(boxEl&&boxEl.clientWidth&&boxEl.clientHeight)idxCharts[k].resize(boxEl.clientWidth,boxEl.clientHeight);idxCharts[k+'_line'].setData(series.map(p=>({{time:p.time,value:p.close}})));idxCharts[k].timeScale().fitContent();const first=series[0].close,last=series[series.length-1].close;const chg=((last/first-1)*100).toFixed(2);idxCharts[k+'_line'].applyOptions({{color:chg>=0?'#26a69a':'#ef5350'}});const valEl=document.getElementById('idx-'+k+'-val');if(valEl)valEl.innerHTML=`${{last}} <span style="color:${{chg>=0?'#26a69a':'#ef5350'}}">${{chg>=0?'+':''}}${{chg}}%</span>`}})}}catch(e){{console.warn('index load failed',e)}}}}
async function loadScoreHistory(t){{const el=document.getElementById('scoretrend');try{{const r=await fetch(`/api/score-history?ticker=${{encodeURIComponent(t)}}`);const d=await r.json();const scores=(d.points||[]).map(p=>p.alpha_score).filter(v=>v!=null);if(scores.length<2){{el.innerHTML=scores.length?scores[scores.length-1].toFixed(1):'-';return}}el.innerHTML=sparklineSVG(scores)+' '+scores[scores.length-1].toFixed(1)}}catch(e){{el.innerText='-'}}}}
async function autoScanOnOpen(){{try{{await fetch('/api/auto-scan',{{method:'POST'}});}}catch(e){{console.warn('auto-scan trigger failed',e)}};scan()}}
function updateUcount(d){{document.getElementById('ucount').innerText=d.universe_count?` · ${{d.quant_pass_count??0}} detected / ${{d.universe_count}} symbols`:'';
  const asOf=document.getElementById('scanAsOf');
  if(asOf){{
    if(d.showing_previous_day&&d.scan_date){{
      asOf.textContent=SCAN_PREVIOUS_DAY.replace('{{date}}',d.scan_date);
      asOf.style.display='block';
    }}else{{asOf.style.display='none';}}
  }}}}
async function scan(){{const r=await fetch('/api/scan');if(r.status===402){{location.href='/subscription';return}}const d=await r.json();updateUcount(d);lastUpdated=d.last_updated;lastSignals=d.signals||[];populateSectorFilter();if(!lastSignals.length){{const ready=d.universe_status?.ready;const err=d.universe_status?.error;const scanned=d.scanned_count>0;document.getElementById('list').innerHTML='<div class="notice">'+(scanned?'Scan complete — no tickers cleared the quant threshold today. You can still look up any ticker above.':(ready?'The server is preparing the next scan — check back shortly.':(err?'Could not prepare constituent data. The server will retry automatically.':'Preparing S&P 500 / Nasdaq-100 constituents...')))+'</div>';loadTicker(ticker);return}}renderList();loadTicker(lastSignals[0].ticker)}}
async function pollForUpdates(){{try{{const r=await fetch('/api/scan');if(r.status===402){{location.href='/subscription';return}}const d=await r.json();updateUcount(d);if(d.last_updated&&d.last_updated!==lastUpdated){{lastUpdated=d.last_updated;lastSignals=d.signals||[];populateSectorFilter();renderList();if(currentView==='heatmap')loadHeatmap();loadTicker(ticker);showToast('Updated with the latest scan.')}}}}catch(e){{}}}}

// ---- Company snowflake ------------------------------------------------------------
// Six fundamental axes, each a percentile against the ticker's own sector, so a wider
// shape means a cheaper / faster-growing / more profitable / sounder / better-paying
// company. An axis with no figure is drawn as a gap at the centre rather than a zero --
// a company with no dividend has no dividend rank, and denting the shape would be
// inventing bad news about it.
const SNOW_N=6;
const SNOW_KEYS=['sf_value','sf_growth','sf_profit','sf_health','sf_dividend','sf_ai'];
function snowPoint(cx,cy,r,i){{const a=(-90+i*360/SNOW_N)*Math.PI/180;return [cx+r*Math.cos(a),cy+r*Math.sin(a)];}}
function snowPoly(cx,cy,r){{let p=[];for(let i=0;i<SNOW_N;i++){{const q=snowPoint(cx,cy,r,i);p.push(q[0].toFixed(1)+','+q[1].toFixed(1));}}return p.join(' ');}}
function sfColor(v){{if(v===null||v===undefined)return 'var(--dim)';if(v>=66)return 'var(--green)';if(v>=33)return 'var(--orange)';return 'var(--red)';}}

function renderSnowflake(sf){{
  const card=document.getElementById('scoreCard');
  if(!sf){{card.style.display='none';return;}}
  card.style.display='block';
  const axes=sf.axes||[];
  const cx=100,cy=100,R=64;
  let g='';
  [0.25,0.5,0.75,1].forEach(f=>{{g+=`<polygon points="${{snowPoly(cx,cy,R*f)}}" fill="none" stroke="var(--border)" stroke-width="1"/>`;}});
  for(let i=0;i<SNOW_N;i++){{const q=snowPoint(cx,cy,R,i);
    const missing=(axes[i]&&(axes[i].value===null||axes[i].value===undefined));
    g+=`<line x1="${{cx}}" y1="${{cy}}" x2="${{q[0].toFixed(1)}}" y2="${{q[1].toFixed(1)}}" stroke="var(--border)" stroke-width="1"${{missing?' stroke-dasharray="3 3"':''}}/>`;}}

  const known=axes.filter(a=>a.value!==null&&a.value!==undefined).map(a=>a.value);
  const avg=known.length?known.reduce((s,v)=>s+v,0)/known.length:null;
  const col=sfColor(avg);
  // Draw across only the axes that have a figure. Collapsing a missing one to the
  // centre spikes the shape and reads as a catastrophic score -- and AI check is absent
  // for every ticker that did not clear the quant scan, which is most of them. The
  // spoke and its label stay visible, dimmed, so the gap is still legible.
  const pts=axes.map((a,i)=>({{a,i}}))
    .filter(o=>o.a.value!==null&&o.a.value!==undefined)
    .map(o=>{{const q=snowPoint(cx,cy,R*Math.max(0.02,o.a.value/100),o.i);
      return q[0].toFixed(1)+','+q[1].toFixed(1);}}).join(' ');
  g+=`<polygon points="${{pts}}" fill="${{col}}" fill-opacity="0.3" stroke="${{col}}" stroke-width="2.5" stroke-linejoin="round"/>`;
  axes.forEach((a,i)=>{{if(a.value===null||a.value===undefined)return;
    const q=snowPoint(cx,cy,R*Math.max(0.02,a.value/100),i);
    g+=`<circle cx="${{q[0].toFixed(1)}}" cy="${{q[1].toFixed(1)}}" r="3.2" fill="${{col}}"/>`;}});
  axes.forEach((a,i)=>{{const q=snowPoint(cx,cy,R+20,i);
    const anchor=Math.abs(q[0]-cx)<8?'middle':(q[0]>cx?'start':'end');
    const dim=(a.value===null||a.value===undefined);
    g+=`<text x="${{q[0].toFixed(1)}}" y="${{(q[1]+4).toFixed(1)}}" text-anchor="${{anchor}}" font-size="14" font-weight="800" letter-spacing="0.2" fill="${{dim?'var(--dim)':'var(--head)'}}">${{SF_TEXT[SNOW_KEYS[i]]||a.key}}</text>`;}});
  document.getElementById('snowflake').innerHTML=g;

  const q=sf.quant_score,ai=sf.ai_score;
  const hasOverall=sf.overall_score!==null&&sf.overall_score!==undefined;
  const shown=hasOverall?sf.overall_score:(q??null);
  const big=document.getElementById('scoreBig');
  big.innerText=shown!==null&&shown!==undefined?shown:'-';
  big.style.color=(shown===null||shown===undefined)?'var(--dim)':(shown>=83?'var(--green)':shown>=60?'var(--orange)':'var(--red)');
  document.getElementById('scoreBadge').innerHTML=sf.verdict?`<span class="badge ${{verdictClass(sf.verdict)}}">${{sf.verdict}}</span>`:'';
  document.getElementById('scoreSplit').innerHTML=hasOverall
    ? `Quant <b>${{q}}</b> &middot; AI <b>${{ai}}</b> &middot; ${{SCORE_SPLIT_NOTE}}`
    : SCORE_QUANT_ONLY;

  const legend=axes.map((a,i)=>{{
    const empty=(a.value===null||a.value===undefined);
    const label=SF_TEXT[SNOW_KEYS[i]]||a.key;
    const help=SF_TEXT[SNOW_KEYS[i]+'_help']||'';
    return `<div class="sf-item ${{empty?'empty':''}}"><span class="nm">${{label}}`+
      `<span class="help-icon" onclick="event.stopPropagation();this.classList.toggle('open')">?<span class="tip-bubble">${{help}}</span></span>`+
      `<span class="sc" style="color:${{empty?'var(--dim)':sfColor(a.value)}}">${{empty?'—':Math.round(a.value)}}</span></span>`+
      `<span class="rw">${{a.raw||''}}</span></div>`;}}).join('');
  document.getElementById('scoreAxes').innerHTML=
    `<div class="sf-title">${{SF_TEXT.sf_title}}</div>`+
    (sf.sector&&sf.sector!=='-'?`<div class="sf-sector">${{SF_TEXT.sf_vs_sector}} <b>${{sf.sector}}</b></div>`:'')+
    `<div class="sf-legend">${{legend}}</div>`;
  document.getElementById('scoreNote').innerHTML=
    sf.have_fundamentals?`<div class="sf-foot">${{SF_TEXT.sf_footnote}}</div>`
                        :`<div class="sf-foot">${{SF_TEXT.sf_no_data}}</div>`;
}}

async function loadTicker(t){{ticker=t.toUpperCase().trim();document.getElementById('title').innerText=ticker;document.getElementById('ai').innerText='Loading AI analysis based on real data...';document.getElementById('news').innerText='Waiting for news...';document.getElementById('verdict').style.display='none';document.getElementById('aiTldr').style.display='none';document.getElementById('scoreDrift').style.display='none';document.getElementById('scoreCard').style.display='none';const fastPromise=fetch(`/api/terminal-data-fast?ticker=${{encodeURIComponent(ticker)}}&timeframe=${{tf}}`);const aiPromise=fetch(`/api/terminal-data-ai?ticker=${{encodeURIComponent(ticker)}}&mode=${{encodeURIComponent(STRATEGY_MODE)}}&language=${{USER_LANGUAGE}}`);let d;try{{const fastRes=await fastPromise;if(fastRes.status===402){{location.href='/subscription';return}}d=await fastRes.json()}}catch(e){{document.getElementById('rsi').innerText='Could not load chart data.';console.error('Chart data load failed',e);return}}if(!d.fast?.data_ok){{document.getElementById('rsi').innerText=d.fast?.error||'No data';return}}const sw=document.getElementById('staleWarning');if(d.fast.stale_as_of){{const asOfDate=new Date(d.fast.stale_as_of*1000);sw.style.display='block';sw.innerText=`⚠ Live data temporarily unavailable — showing last known data from ${{asOfDate.toLocaleString()}}.`}}else if(d.fast.stale_db_date){{sw.style.display='block';sw.innerText=`⚠ Live data temporarily unavailable — showing indicators from the last scan on ${{d.fast.stale_db_date}}. No chart available for this snapshot.`}}else{{sw.style.display='none'}}const cd=d.fast.chart.map(x=>({{time:x.time,open:x.open,high:x.high,low:x.low,close:x.close}}));const vd=d.fast.chart.map(x=>({{time:x.time,value:x.volume}}));candle.setData(cd);volume.setData(vd);['sma20','sma50','sma200'].forEach(k=>{{const pts=d.fast.chart.filter(x=>x[k]!=null).map(x=>({{time:x.time,value:x[k]}}));smaLines[k].setData(pts)}});bbLines.upper.setData(d.fast.chart.filter(x=>x.bb_upper!=null).map(x=>({{time:x.time,value:x.bb_upper}})));bbLines.lower.setData(d.fast.chart.filter(x=>x.bb_lower!=null).map(x=>({{time:x.time,value:x.bb_lower}})));const cEl=document.getElementById('chart');if(cEl.clientWidth&&cEl.clientHeight)chart.resize(cEl.clientWidth,cEl.clientHeight);chart.timeScale().fitContent();document.getElementById('rsi').innerText=`RSI ${{d.fast.rsi}} / MACD ${{d.fast.macd}}`;document.getElementById('high52').innerText=d.fast.pct_from_52w_high==null?'N/A':d.fast.pct_from_52w_high+'%';document.getElementById('low52').innerText=d.fast.pct_from_52w_low==null?'N/A':d.fast.pct_from_52w_low+'%';document.getElementById('trend').innerText=d.fast.above_200d_sma==null?'N/A':(d.fast.above_200d_sma?'Uptrend':'Downtrend');document.getElementById('portfolioPrice').value=d.fast.price??'';document.getElementById('portfolioShares').value='';renderEarnings(d.fast.earnings);loadScoreHistory(ticker);try{{const aiRes=await aiPromise;if(aiRes.status===402){{location.href='/subscription';return}}const x=await aiRes.json();renderSnowflake(x.ai?.snowflake);const vEl=document.getElementById('verdict');if(x.ai?.timing_verdict){{vEl.style.display='block';const reviewedNote=x.ai.updated_at?` <span style="color:var(--dim);font-size:11px" title="Price/RSI/trend above refresh at each scan; this AI risk review only re-runs when the quant score has moved enough to matter">· AI reviewed ${{new Date(x.ai.updated_at*1000).toLocaleTimeString([],{{hour:'2-digit',minute:'2-digit'}})}}</span>`:'';vEl.innerHTML=`<span class="badge ${{verdictClass(x.ai.timing_verdict)}}">${{x.ai.timing_verdict}}</span> Score ${{x.ai.overall_score??'-'}} / 100${{reviewedNote}}`;const tldrEl=document.getElementById('aiTldr');const verdictPhrase={{Favorable:'looks like a reasonable entry point',Caution:'has some risk worth reading below',Risk:'looks risky right now'}}[x.ai.timing_verdict]||'has been reviewed';const trendPhrase=d.fast.above_200d_sma?'still in a long-term uptrend':'below its long-term trend';const pullbackPhrase=d.fast.pct_from_52w_high!=null?`, ${{Math.abs(d.fast.pct_from_52w_high)}}% off its 52-week high`:'';tldrEl.innerHTML=`<b>Bottom line:</b> ${{ticker}} ${{verdictPhrase}} — ${{trendPhrase}}${{pullbackPhrase}}. Score ${{x.ai.overall_score??'-'}}/100.<div class="tldr-next">Not a decision you need to make now — <b style="color:var(--head)">Set Alert</b> above to get emailed if it hits your price, or <b style="color:var(--head)">Save to Portfolio</b> to track it alongside your other picks.</div>`;tldrEl.style.display='block'}}else{{vEl.style.display='none';document.getElementById('aiTldr').style.display='none'}}const driftEl=document.getElementById('scoreDrift');if(x.ai?.scan_price&&d.fast?.price){{const drift=(d.fast.price-x.ai.scan_price)/x.ai.scan_price*100;if(Math.abs(drift)>=2){{const scanTimeStr=x.ai.updated_at?new Date(x.ai.updated_at*1000).toLocaleTimeString([],{{hour:'2-digit',minute:'2-digit'}}):'earlier today';driftEl.innerHTML=`⚠ This score was computed at $${{x.ai.scan_price}} (${{scanTimeStr}}) — price has moved ${{drift>=0?'+':''}}${{drift.toFixed(1)}}% since then, now $${{d.fast.price}}. The setup may no longer look the same.`;driftEl.style.display='block'}}else{{driftEl.style.display='none'}}}}else{{driftEl.style.display='none'}}const sec=x.ai?.report_sections;const aiEl=document.getElementById('ai');const langMismatch=x.ai?.language&&x.ai.language!==x.ai.language_requested;const langNote=langMismatch?`<div class="notice" style="margin-bottom:8px;font-size:12px">Showing in ${{x.ai.language==='ko'?'Korean':'English'}} — today's AI usage limit was reached before this could be regenerated in your preferred language. It switches automatically once quota resets.</div>`:'';if(sec){{const labels={{quant_review:'Quant Review',supply_demand:'Supply/Demand',risk_review:'Risk Review',news_analysis:'News Analysis',timing_reason:'Timing Rationale'}};aiEl.innerHTML=langNote+Object.keys(labels).filter(k=>sec[k]).map(k=>`<div class="section"><b>${{labels[k]}}</b>${{sec[k]}}</div>`).join('')}}else{{aiEl.innerText=!x.ai?.quant_pass?'AI analysis only runs for tickers that clear the daily quant scan — this one did not make the list today.':(x.ai?.status==='PENDING'||x.ai?.status==='RUNNING'?'Preparing AI analysis cache on the server...':(x.ai?.quota_exhausted?"Today's AI usage limit has been reached, so this review couldn't be generated right now — a shared daily limit, unrelated to your language setting. It resumes automatically tomorrow.":'AI analysis is unavailable.'))}}const news=x.ai?.news;if(!news)document.getElementById('news').innerText='Could not fetch a live news feed.';else document.getElementById('news').innerHTML=news.map(n=>`<div style="margin-bottom:8px"><a href="${{n.url}}" target="_blank" rel="noopener">${{n.title}}</a><br><small>${{n.published||''}}</small></div>`).join('')}}catch(e){{document.getElementById('ai').innerText='Could not load AI analysis. Please try again in a moment.';document.getElementById('news').innerText='Could not fetch a live news feed.';console.error('AI data load failed',e)}}}}
async function setAlert(){{const p=Number(document.getElementById('target').value);if(!(p>0))return showToast('Enter a target price first.',true);const dir=document.getElementById('targetDir').value;const f=new FormData();f.append('ticker',ticker);f.append('target_price',p);f.append('direction',dir);const r=await fetch('/api/alerts/set',{{method:'POST',body:f}});const d=await r.json();showToast(d.message||d.error,!r.ok)}}
async function savePortfolio(){{const sharesInput=document.getElementById('portfolioShares').value.trim();const priceInput=document.getElementById('portfolioPrice').value.trim();let shares='';if(sharesInput!==''){{const n=parseFloat(sharesInput);if(!isFinite(n)||n<=0){{showToast('Enter a positive number of shares, or leave it blank.',true);return}}shares=n}}let price='';if(priceInput!==''){{const p=parseFloat(priceInput);if(!isFinite(p)||p<=0){{showToast("Enter a positive entry price, or leave it blank to use today's scan price.",true);return}}price=p}}const f=new FormData();f.append('ticker',ticker);if(shares!=='')f.append('shares',shares);if(price!=='')f.append('price',price);const r=await fetch('/api/portfolio/save',{{method:'POST',body:f}});const d=await r.json();showToast(d.message||d.error,!r.ok)}}
function changeTF(x){{tf=x;document.querySelectorAll('.tf-btn').forEach(b=>b.classList.toggle('active',b.dataset.tf===x));chart.timeScale().applyOptions({{timeVisible:x==='1h'}});loadTicker(ticker)}}
window.onload=()=>{{const qp=new URLSearchParams(location.search);if(qp.get('welcome')==='1'){{showToast(`Welcome! Your 7-day free trial has started${{TRIAL_ENDS_STR?' — ends '+TRIAL_ENDS_STR:''}}.`,false,8000);history.replaceState(null,'','/terminal')}}const qTicker=qp.get('ticker');if(qTicker){{history.replaceState(null,'','/terminal')}}let seenOnboarding=false;try{{seenOnboarding=localStorage.getItem('onboarded')==='1'}}catch(e){{}}if(!seenOnboarding)openOnboarding();document.getElementById('sortKey').value=DEFAULT_SORT;if(DEFAULT_VIEW==='heatmap')showView('heatmap');init();autoScanOnOpen();loadIndices();if(qTicker)setTimeout(()=>loadTicker(qTicker),300);setInterval(pollForUpdates,20000)}};
</script></body></html>'''
    html = translate_body(html, lang, [
        (">My Subscription<", f">{t('my_subscription', lang)}<"),
        (">Contact Us<", f">{t('contact_us', lang)}<"),
        (">Log out<", f">{t('nav_logout', lang)}<"),
        (">Quick guide to QUANTIFY<", f">{t('onboarding_title', lang)}<"),
        ("<b>Badges</b> are the AI's read on entry timing: <b>Favorable</b> (setup looks clean), <b>Caution</b> (some risk worth knowing about), or <b>Risk</b> (skip or wait). Never a buy/sell order.", t("onboarding_badges_p", lang)),
        ("<b>Score (0-100)</b> combines the quant scan (is this a long-term uptrend that's pulled back to a good entry zone?) with the AI's risk check. Only names that clear the bar show up at all.", t("onboarding_score_p", lang)),
        ("<b>The scanner list</b> updates a few times a day — tap any ticker to load its chart, technicals, and full AI report.", t("onboarding_scanner_p", lang)),
        ("Little <b>?</b> icons next to unfamiliar terms (RSI, MACD, Trend...) explain what they mean — tap or hover any of them anytime.", t("onboarding_help_p", lang)),
        (">Got it<", f">{t('got_it', lang)}<"),
        (">Market Scanner <", f">{t('market_scanner', lang)} <"),
        (">List</button>", f">{t('opt_list', lang)}</button>"),
        (">Heatmap</button>", f">{t('heatmap', lang)}</button>"),
        ('placeholder="Jump to ticker (e.g. TSLA)"', f'placeholder="{t("jump_to_ticker", lang)}"'),
        (">Sort: Score<", f">{t('sort_score', lang)}<"),
        (">Sort: Change %<", f">{t('sort_change', lang)}<"),
        (">Sort: Ticker A-Z<", f">{t('sort_ticker', lang)}<"),
        (">All Badges<", f">{t('all_badges', lang)}<"),
        (">All Markets<", f">{t('all_markets', lang)}<"),
        (">All Sectors<", f">{t('all_sectors', lang)}<"),
        (">Preparing constituent list...<", f">{t('preparing_list', lang)}<"),
        ('placeholder="Target price $"', f'placeholder="{t("target_price_placeholder", lang)}"'),
        (">Set Alert</button>", f">{t('set_alert_btn', lang)}</button>"),
        ('placeholder="Shares"', f'placeholder="{t("shares_placeholder", lang)}"'),
        ('placeholder="Entry price $"', f'placeholder="{t("entry_price_placeholder", lang)}"'),
        (">Save to Portfolio</button>", f">{t('save_to_portfolio_btn', lang)}</button>"),
        (">Manage alerts<", f">{t('manage_alerts_link', lang)}<"),
        (">Price alert</span>", f">{t('act_price_alert', lang)}</span>"),
        (">Add to portfolio</span>", f">{t('act_add_portfolio', lang)}</span>"),
        ("Leave entry price blank to use today's scan price. Not sure how many shares?", t("act_portfolio_hint", lang)),
        (">Position sizing calculator</a>", f">{t('act_sizing_link', lang)}</a>"),
        ("&#9733; Set an alert or add to portfolio", f"&#9733; {t('act_toggle', lang)}"),
        ('aria-label="Score breakdown radar"', f'aria-label="{t("score_radar_alt", lang)}"'),
        ('title="Track this ticker without deciding anything right now"', f'title="{t("tip_track_ticker", lang)}"'),
        ('title="Alert when price rises to/above, or falls to/below, the target"', f'title="{t("tip_alert_dir", lang)}"'),
        ('title="Get an email when the price reaches this value"', f'title="{t("tip_target", lang)}"'),
        ('title="Email me when the price hits my target"', f'title="{t("tip_set_alert", lang)}"'),
        ('title="View or cancel your existing price alerts"', f'title="{t("tip_manage_alerts", lang)}"'),
        ('title="How many shares you own or are tracking"', f'title="{t("tip_shares", lang)}"'),
        ('title="What you paid \u2014 defaults to today\'s scan price if left blank"', f'title="{t("tip_entry", lang)}"'),
        ('title="Add this ticker to My Portfolio"', f'title="{t("tip_save_portfolio", lang)}"'),
        ('title="Price/RSI/trend above refresh at each scan; this AI risk review only re-runs when the quant score has moved enough to matter"',
         f'title="{t("tip_ai_refresh", lang)}"'),
        ("<div class=\"notice\">No scan data yet.</div>", f'<div class="notice">{t("no_scan_data", lang)}</div>'),
        (">SMA 20<", f">{t('sma20', lang)}<"),
        (">SMA 50<", f">{t('sma50', lang)}<"),
        (">SMA 200<", f">{t('sma200', lang)}<"),
        (">Bollinger Bands<", f">{t('bollinger', lang)}<"),
        (">Volume<", f">{t('volume_label', lang)}<"),
        (">Earnings: -<", f">{t('earnings_label', lang)} -<"),
        (">52W High<", f">{t('high52_label', lang)}<"),
        ("How far the price is below its highest point in the last 52 weeks. Closer to 0% means near the high.", t("high52_tip", lang)),
        (">52W Low<", f">{t('low52_label', lang)}<"),
        ("How far the price is above its lowest point in the last 52 weeks.", t("low52_tip", lang)),
        (">Trend<", f">{t('trend_label', lang)}<"),
        ("Whether the price is above (Uptrend) or below (Downtrend) its 200-day moving average — a common gauge of the long-term direction.", t("trend_tip", lang)),
        (">Score Trend (Today)<", f">{t('score_trend_label', lang)}<"),
        ("How this ticker's quant score has moved since today's first scan — rising or falling.", t("score_trend_tip", lang)),
        ("RSI: below 30 usually means oversold, above 70 usually means overbought. MACD: positive means upward momentum, negative means downward.", t("rsi_macd_tip", lang)),
        (">AI Quant Report <", f">{t('ai_quant_report', lang)} <"),
        ("(informational only, not investment advice)", t("informational_only", lang)),
        (">Loading AI analysis based on real data...<", f">{t('loading_ai_analysis', lang)}<"),
        ("This flags entry timing on a single ticker, not a full plan. Many investors cap any one pick at a small slice of their total portfolio and spread bets across several signals rather than one — sizing and diversification are on you, not this tool.", t("usage_tip", lang)),
        ('<h3 style="margin-top:12px">News</h3>', f'<h3 style="margin-top:12px">{t("news_label", lang)}</h3>'),
        (">Waiting for news...<", f">{t('waiting_for_news', lang)}<"),
    ])
    return HTMLResponse(html)


@app.get("/market", response_class=HTMLResponse)
async def market_page(request: Request):
    user = get_logged_in_user(request)
    if not user: return RedirectResponse("/login", status_code=303)
    if not disclaimer_accepted(user): return RedirectResponse("/accept-disclaimer", status_code=303)
    if not has_active_access(user): return RedirectResponse("/subscription?reason=trial_ended", status_code=303)
    lang = get_user_lang(user)
    body = """
<section class="panel"><h3>AI Market Summary <small style="color:var(--dim);font-weight:normal;text-transform:none">(informational only, not investment advice)</small></h3><div id="aiMarketSummaryBody"><div class="empty-hint">Loading...</div></div></section>
<section class="panel"><h3>Market Summary</h3><div id="marketSummaryBody" class="summary-grid"><div class="empty-hint">Loading...</div></div></section>
<section class="panel"><h3>By Universe</h3><div id="byUniverseBody" class="summary-grid"><div class="empty-hint">Loading...</div></div></section>
<section class="panel"><h3>Heatmap <small style="color:var(--dim);font-weight:normal;text-transform:none">click any tile to open its chart — bigger tiles are larger-cap</small></h3><div class="groupby-row"><span style="font-size:11.5px;color:var(--dim)">Group by</span><select id="heatGroupKey" onchange="renderHeatmap()"><option value="universe">Index</option><option value="sector">Sector</option></select></div><div id="heatmapBody"><div class="empty-hint">Loading...</div></div></section>
<script>
let lastHeatTiles=[];
function heatColor(chg){if(chg==null)return '#333';const c=Math.max(-5,Math.min(5,chg));const t=(c+5)/10;const r=Math.round(239+(38-239)*t),g=Math.round(83+(166-83)*t),b=Math.round(80+(154-80)*t);return `rgb(${r},${g},${b})`}
function badgeClass(v){return v==='Favorable'?'badge-favorable':v==='Caution'?'badge-caution':v==='Risk'?'badge-risk':''}
function groupTiles(items,key){const groups={};items.forEach(t=>{const g=t[key]||'Other';(groups[g]=groups[g]||[]).push(t)});return groups}
function renderHeatmap(){const el=document.getElementById('heatmapBody');if(!lastHeatTiles.length){el.innerHTML='<div class="notice">No scan data yet — check back after the next scan.</div>';return}const key=document.getElementById('heatGroupKey').value;const groups=groupTiles(lastHeatTiles,key);const sortedGroups=Object.entries(groups).sort((a,b)=>b[1].length-a[1].length);el.innerHTML=sortedGroups.map(([g,items])=>`<div class="heat-group-header">${g} (${items.length})</div><div class="heat-grid">`+[...items].sort((a,b)=>(a.cap_tier??3)-(b.cap_tier??3)).map(t=>`<div class="heat-tile cap-${t.cap_tier??3} ${badgeClass(t.timing_verdict)}" style="background:${heatColor(t.change_pct)}" title="${t.ticker} · ${t.change_pct??'-'}% · Score ${t.alpha_score??'-'} · ${t.timing_verdict||'Not yet reviewed'}${t.sector?' · '+t.sector:''}" onclick="location.href='/terminal?ticker=${t.ticker}'">${t.ticker}</div>`).join('')+'</div>').join('')}
async function loadHeatmap(){try{const r=await fetch('/api/heatmap');if(r.status===402){location.href='/subscription';return}const d=await r.json();lastHeatTiles=d.tiles||[];renderHeatmap()}catch(e){document.getElementById('heatmapBody').innerHTML='<div class="notice">Could not load the heatmap.</div>';console.error('Heatmap load failed',e)}}
async function loadMarketSummary(){try{const r=await fetch('/api/market-summary');if(r.status===402){location.href='/subscription';return}const d=await r.json();const chg=(v)=>v==null?'-':(v>=0?'+':'')+v+'%';const cls=(v)=>v==null?'':(v>=0?'gain':'loss');
const aiEl=document.getElementById('aiMarketSummaryBody');
if(d.ai_headline&&d.ai_summary){const genTime=d.ai_generated_at?new Date(d.ai_generated_at*1000).toLocaleString([],{hour:'2-digit',minute:'2-digit',month:'short',day:'numeric'}):null;aiEl.innerHTML=`<div class="ai-market-summary"><div class="headline">${d.ai_headline}</div><div class="body">${d.ai_summary}</div>${genTime?`<div class="meta">Generated from this scan cycle's data · ${genTime}</div>`:''}</div>`}
else{aiEl.innerHTML='<div class="empty-hint">AI summary is being prepared for today&#39;s scan — check back after the next cycle.</div>'}
document.getElementById('marketSummaryBody').innerHTML=`
<div class="summary-tile"><div class="label">S&amp;P 500</div><div class="value ${cls(d.sp500_change_pct)}">${chg(d.sp500_change_pct)}</div></div>
<div class="summary-tile"><div class="label">Nasdaq-100</div><div class="value ${cls(d.nasdaq_change_pct)}">${chg(d.nasdaq_change_pct)}</div></div>
<div class="summary-tile"><div class="label">Advancers</div><div class="value gain">${d.advancers??'-'}</div></div>
<div class="summary-tile"><div class="label">Decliners</div><div class="value loss">${d.decliners??'-'}</div></div>
<div class="summary-tile"><div class="label">Avg Change</div><div class="value ${cls(d.avg_change_pct)}">${chg(d.avg_change_pct)}</div></div>
<div class="summary-tile"><div class="label">Cleared Quant Bar</div><div class="value">${d.detected_count??'-'} / ${d.universe_count??'-'}</div></div>
<div class="summary-tile"><div class="label">Favorable</div><div class="value gain">${d.verdict_breakdown?.Favorable??0}</div></div>
<div class="summary-tile"><div class="label">Caution / Risk</div><div class="value loss">${(d.verdict_breakdown?.Caution??0)+(d.verdict_breakdown?.Risk??0)}</div></div>`;
const bu=d.by_universe||{};const names=Object.keys(bu);document.getElementById('byUniverseBody').innerHTML=names.length?names.map(name=>`<div class="summary-tile"><div class="label">${name}</div><div class="value">${bu[name].count} scanned</div><div style="margin-top:6px;font-size:11px;color:var(--dim)">${bu[name].advancers} up &middot; ${bu[name].decliners} down &middot; avg <span class="${cls(bu[name].avg_change_pct)}">${chg(bu[name].avg_change_pct)}</span></div></div>`).join(''):'<div class="empty-hint">No data yet.</div>'}catch(e){console.error('Market summary load failed',e)}}
loadMarketSummary();loadHeatmap();setInterval(()=>{loadMarketSummary();loadHeatmap()},60000);
</script>
"""
    body = translate_body(body, lang, [
        (">AI Market Summary <", f">{t('ai_market_summary', lang)} <"),
        ("(informational only, not investment advice)", t("informational_only", lang)),
        (">Market Summary<", f">{t('market_summary', lang)}<"),
        (">By Universe<", f">{t('by_universe', lang)}<"),
        (">Heatmap <", f">{t('heatmap', lang)} <"),
        ("click any tile to open its chart — bigger tiles are larger-cap", t("heatmap_hint", lang)),
        (">Group by<", f">{t('group_by', lang)}<"),
        ('value="universe">Index<', f'value="universe">{t("opt_index", lang)}<'),
        ('value="sector">Sector<', f'value="sector">{t("opt_sector", lang)}<'),
        (">Loading...<", f">{t('loading', lang)}<"),
            ("<div class=\"notice\">No scan data yet \u2014 check back after the next scan.</div>", f'<div class="notice">{t("no_scan_data_check", lang)}</div>'),
        ("<div class=\"notice\">Could not load the heatmap.</div>", f'<div class="notice">{t("heatmap_failed", lang)}</div>'),
        ("AI summary is being prepared for today&#39;s scan \u2014 check back after the next cycle.", t("ai_summary_pending", lang)),
        ("'Not yet reviewed'", f"'{t('not_yet_reviewed', lang)}'"),
        (">Cleared Quant Bar<", f">{t('cleared_quant_bar', lang)}<"),
])
    return render_app_shell(t("nav_market", lang), "market", body, lang=lang, theme=get_user_theme(user))


@app.get("/watchlist", response_class=HTMLResponse)
async def watchlist_page(request: Request):
    user = get_logged_in_user(request)
    if not user: return RedirectResponse("/login", status_code=303)
    if not disclaimer_accepted(user): return RedirectResponse("/accept-disclaimer", status_code=303)
    if not has_active_access(user): return RedirectResponse("/subscription?reason=trial_ended", status_code=303)
    lang = get_user_lang(user)
    body = """
<section class="panel"><h3>Watchlist</h3><div class="add-row"><input id="watchInput" type="text" placeholder="Add ticker (e.g. NVDA)" onkeydown="if(event.key==='Enter')addWatch()"><button onclick="addWatch()">Add</button></div><div id="watchlistBody"><div class="empty-hint">Loading...</div></div></section>
<script>
function sparklineSVG(arr){if(!arr||arr.length<2)return '';const w=60,h=22;const min=Math.min(...arr),max=Math.max(...arr),range=(max-min)||1;const pts=arr.map((v,i)=>`${(i/(arr.length-1)*w).toFixed(1)},${(h-((v-min)/range*h)).toFixed(1)}`).join(' ');const color=arr[arr.length-1]>=arr[0]?'#26a69a':'#ef5350';return `<svg width="${w}" height="${h}" style="vertical-align:middle;flex-shrink:0"><polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1.5"/></svg>`}
function verdictClass(v){return v==='Favorable'?'badge-ok':v==='Caution'?'badge-warn':v==='Risk'?'badge-danger':'badge-pending'}
async function addWatch(){const t=document.getElementById('watchInput').value.trim();if(!t)return;const f=new FormData();f.append('ticker',t);const r=await fetch('/api/watchlist/add',{method:'POST',body:f});const d=await r.json();if(!r.ok)alert(d.error||'Could not add ticker');document.getElementById('watchInput').value='';load()}
async function removeWatch(id){const f=new FormData();f.append('id',id);await fetch('/api/watchlist/remove',{method:'POST',body:f});load()}
async function load(){try{const r=await fetch('/api/watchlist');if(r.status===402){location.href='/subscription';return}const d=await r.json();const el=document.getElementById('watchlistBody');if(!d.items?.length){el.innerHTML='<div class="empty-hint">Nothing watched yet — add any ticker above, regardless of whether it clears the quant bar.</div>';return}el.innerHTML=d.items.map(it=>`<div class="watch-row"><div class="wt-name" onclick="location.href='/terminal?ticker=${it.ticker}'">${sparklineSVG(it.sparkline)}<div><b>${it.ticker}</b>${it.sector?`<div style="font-size:10.5px;color:var(--dim)">${it.sector}</div>`:''}</div></div><div class="wt-price">${it.price!=null?'$'+it.price:'-'}${it.change_pct!=null?` <span class="${it.change_pct>=0?'gain':'loss'}">${it.change_pct>=0?'+':''}${it.change_pct}%</span>`:''}${it.timing_verdict?`<div style="margin-top:4px"><span class="badge ${verdictClass(it.timing_verdict)}">${it.timing_verdict}</span></div>`:''}</div><button class="remove-btn" onclick="removeWatch(${it.id})">Remove</button></div>`).join('')}catch(e){console.error('Watchlist load failed',e)}}
load();
</script>
"""
    body = translate_body(body, lang, [
        (">Watchlist<", f">{t('watchlist_title', lang)}<"),
        ('placeholder="Add ticker (e.g. NVDA)"', f'placeholder="{t("add_ticker_placeholder", lang)}"'),
        (">Add</button>", f">{t('add_btn', lang)}</button>"),
        (">Loading...<", f">{t('loading', lang)}<"),
        ("Nothing watched yet — add any ticker above, regardless of whether it clears the quant bar.", t("nothing_watched", lang)),
        (">Remove</button>", f">{t('remove_btn', lang)}</button>"),
    ])
    return render_app_shell(t("nav_watchlist", lang), "watchlist", body, lang=lang, theme=get_user_theme(user))


@app.get("/backtest", response_class=HTMLResponse)
async def backtest_page(request: Request):
    user = get_logged_in_user(request)
    if not user: return RedirectResponse("/login", status_code=303)
    if not disclaimer_accepted(user): return RedirectResponse("/accept-disclaimer", status_code=303)
    if not has_active_access(user): return RedirectResponse("/subscription?reason=trial_ended", status_code=303)
    lang = get_user_lang(user)
    body = """
<section class="panel"><h3>Strategy Performance <small style="color:var(--dim);font-weight:normal;text-transform:none">(real historical replay, not a guarantee of future results)</small></h3><div id="backtestBody"><div class="empty-hint">Loading...</div></div></section>
<section class="panel"><h3>Methodology</h3><p style="font-size:12.5px;line-height:1.7;color:var(--text)">%%BT_METHOD%% %%BT_FAQ%%</p></section>
<script>
async function load(){try{const r=await fetch('/api/backtest-summary');if(r.status===402){location.href='/subscription';return}const d=await r.json();const el=document.getElementById('backtestBody');if(!d.results){el.innerHTML='<div class="empty-hint">Backtest is still computing on the server — check back soon.</div>';return}const res=d.results;const fmtPct=(v)=>v==null?'-':(v>=0?'+':'')+v+'%';const cls=(v)=>v==null?'':(v>=0?'gain':'loss');const cards=Object.entries(res.horizons).map(([h,v])=>`<div class="backtest-card"><h4>${h}-Day Forward Return</h4>
<div class="backtest-row"><span>Strategy avg</span><b class="${cls(v.strategy?.avg_return_pct)}">${fmtPct(v.strategy?.avg_return_pct)}</b></div>
<div class="backtest-row"><span>Strategy win rate</span><b>${v.strategy?.win_rate_pct??'-'}%</b></div>
<div class="backtest-row"><span>When right / wrong</span><b>${fmtPct(v.strategy?.avg_win_pct)} / ${fmtPct(v.strategy?.avg_loss_pct)}</b></div>
<div class="backtest-row"><span>Worst case</span><b class="loss">${fmtPct(v.strategy?.worst_pct)}</b></div>
<div class="backtest-row"><span>S&amp;P 500, same windows</span><b class="${cls((v.benchmark_matched||v.benchmark)?.avg_return_pct)}">${fmtPct((v.benchmark_matched||v.benchmark)?.avg_return_pct)}</b></div>
</div>`).join('');const val=res.validation;const valParts=[30,60,90].filter(h=>val?.[`in_sample_${h}d`]&&val?.[`out_of_sample_${h}d`]).map(h=>{const i=val[`in_sample_${h}d`],o=val[`out_of_sample_${h}d`];return `${h}d: in-sample ${fmtPct(i.avg_return_pct)} / ${i.win_rate_pct}% win (n=${i.n}) vs out-of-sample ${fmtPct(o.avg_return_pct)} / ${o.win_rate_pct}% win (n=${o.n})`});const valLine=valParts.length?`Out-of-sample check at all three horizons (not just the best-looking one) — tuned on the first 70% of the window, measured on the untouched last 30%: ${valParts.join(' &middot; ')}.`:'';const universeText=res.tickers_sampled>=500?`All ${res.tickers_sampled} tickers in the current S&amp;P 500 + Nasdaq-100 universe (no sampling)`:`${res.tickers_sampled} of the ~518 current S&amp;P 500 + Nasdaq-100 tickers`;el.innerHTML=`<div class="backtest-grid">${cards}</div><div class="backtest-meta">${universeText}, ${res.signal_count} historical signals (fresh threshold crossings, not repeat days) over the trailing 2 years. Uses today's index membership — stocks removed from these indices during that window aren't included, which flatters results. Returns are net of a 0.10% round-trip cost for spread and slippage; taxes are not modelled. The S&amp;P figure is its return over the same windows the strategy traded, not its average over the whole period. ${valLine} Last computed: ${d.computed_at?new Date(d.computed_at*1000).toLocaleDateString():'-'}. Past performance does not guarantee future results.</div>`}catch(e){console.error('Backtest load failed',e)}}
load();
</script>
"""
    # translate_body() no-ops for English, so a placeholder that has no English source
    # text in the template has to be filled here rather than through the list below --
    # otherwise an English visitor sees the raw %%BT_METHOD%% marker.
    body = (body.replace("%%BT_METHOD%%", t("backtest_methodology_body", lang))
                .replace("%%BT_FAQ%%", t("faq_more", lang)))
    body = translate_body(body, lang, [
        (">Strategy Performance <", f">{t('strategy_performance', lang)} <"),
        ("(real historical replay, not a guarantee of future results)", t("strategy_performance_hint", lang)),
        (">Loading...<", f">{t('loading', lang)}<"),
        (">Methodology<", f">{t('methodology', lang)}<"),
        ("Backtest is still computing on the server — check back soon.", t("backtest_computing", lang)),
        ("-Day Forward Return", t("day_forward_return", lang)),
        (">Strategy avg<", f">{t('strategy_avg', lang)}<"),
        (">Strategy win rate<", f">{t('strategy_win_rate', lang)}<"),
        (">When right / wrong<", f">{t('when_right_wrong', lang)}<"),
        (">Worst case<", f">{t('worst_case', lang)}<"),
        (">S&amp;P 500, same windows<", f">{t('sp500_avg_matched', lang)}<"),
    ])
    return render_app_shell(t("nav_backtest", lang), "backtest", body, lang=lang, theme=get_user_theme(user))


@app.get("/portfolio", response_class=HTMLResponse)
async def portfolio_page(request: Request):
    user = get_logged_in_user(request)
    if not user: return RedirectResponse("/login", status_code=303)
    if not disclaimer_accepted(user): return RedirectResponse("/accept-disclaimer", status_code=303)
    if not has_active_access(user): return RedirectResponse("/subscription?reason=trial_ended", status_code=303)
    lang = get_user_lang(user)
    theme = get_user_theme(user)
    user = html_lib.escape(user)
    html = f'''<!doctype html><html lang="{lang}" data-theme="{theme}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>QUANTIFY. Portfolio</title><style>{DARK_THEME_VARS}

:root{{--bg:#ffffff;--panel:#ffffff;--panel2:#f5f7f6;--border:#e2e6e3;--text:#3a4440;--head:#12201a;--dim:#77837e;--green:#0e8a5f;--red:#c8402c;--orange:#a8660a}}
*{{box-sizing:border-box}}body{{background:var(--bg);color:var(--text);font:16px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;margin:0;padding:14px}}
header{{background:var(--panel);border:1px solid var(--border);padding:14px 20px;display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;border-radius:10px;flex-wrap:wrap;gap:12px}}
.brand{{font-weight:700;font-size:19px;color:var(--head);text-decoration:none}}.brand span{{color:var(--dim)}}
.headerRight{{display:flex;align-items:center;gap:16px;flex-wrap:wrap}}
a.back{{color:var(--green);text-decoration:underline;font-size:15px;font-weight:600}}
button{{background:var(--panel2);border:1px solid var(--border);color:var(--head);padding:9px 15px;font:14.5px -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;cursor:pointer;border-radius:7px;font-weight:600}}
button:hover{{background:var(--border)}}
.wrap{{max-width:960px;margin:0 auto}}
.item{{background:var(--panel);border:1px solid var(--border);padding:20px;margin-bottom:12px;border-radius:10px;box-shadow:0 1px 2px rgba(18,32,26,.04)}}
.item-head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;gap:10px}}
.item-head b{{color:var(--head);font-size:18px}}
.meta{{color:var(--dim);font-size:13px;margin-bottom:12px}}
.badge{{padding:4px 11px;border-radius:12px;font-weight:700;font-size:13px}}
.badge-ok{{background:#e6f5ee;color:var(--green)}}
.badge-warn{{background:#fbf1e0;color:var(--orange)}}
.badge-danger{{background:#fbe6e2;color:var(--red)}}
.note{{color:var(--text);font-size:14.5px;line-height:1.7}}
.remove{{background:transparent;border:1px solid var(--red);color:var(--red);border-radius:6px}}
.empty{{color:var(--dim);padding:70px 20px;text-align:center;font-size:16px}}
.gain{{color:var(--green);font-weight:700}}
.loss{{color:var(--red);font-weight:700}}
.disclaimer-footer{{max-width:960px;margin:22px auto 0;color:var(--dim);font-size:13px;line-height:1.7;text-align:center;padding:14px;border-top:1px solid var(--border)}}
</style></head><body><header><a class="brand" href="/terminal">QUANTIFY<span>.</span></a><div class="headerRight"><span style="color:var(--dim);font-size:14px">Portfolio · {user}</span><a class="back" href="/settings">Settings</a><a class="back" href="/subscription">Subscription</a><a class="back" href="/contact">Contact</a><a class="back" href="/terminal">&larr; Back to Terminal</a></div></header>
<div class="wrap" id="list">Loading...</div>
<div class="wrap">
<div class="item"><b style="color:var(--head);font-size:17px">Sector Concentration</b><div id="concentrationBody" style="margin-top:12px"><div class="meta">Loading...</div></div></div>
<div class="item" id="sizing"><b style="color:var(--head);font-size:17px">Position Sizing Calculator</b>
<div style="margin-top:12px;display:flex;gap:12px;flex-wrap:wrap;align-items:flex-end">
<label style="font-size:13px;color:var(--dim)">Account size $<br><input id="szAccount" type="number" placeholder="10000" style="width:120px;background:var(--panel2);border:1px solid var(--border);color:var(--head);padding:8px;border-radius:6px;margin-top:5px;font-size:14.5px"></label>
<label style="font-size:13px;color:var(--dim)">Risk per trade %<br><input id="szRisk" type="number" placeholder="1" style="width:90px;background:var(--panel2);border:1px solid var(--border);color:var(--head);padding:8px;border-radius:6px;margin-top:5px;font-size:14.5px"></label>
<label style="font-size:13px;color:var(--dim)">Entry price $<br><input id="szEntry" type="number" placeholder="100" style="width:100px;background:var(--panel2);border:1px solid var(--border);color:var(--head);padding:8px;border-radius:6px;margin-top:5px;font-size:14.5px"></label>
<label style="font-size:13px;color:var(--dim)">Stop distance %<br><input id="szStop" type="number" placeholder="8" style="width:90px;background:var(--panel2);border:1px solid var(--border);color:var(--head);padding:8px;border-radius:6px;margin-top:5px;font-size:14.5px"></label>
<button onclick="calcSize()">Calculate</button>
</div>
<div id="szResult" style="margin-top:12px;font-size:15px;color:var(--head)"></div>
<div class="meta" style="margin-top:10px">Sizing arithmetic only — not a recommendation, and it doesn't account for correlation between your positions.</div>
</div>
</div>
<div class="disclaimer-footer">QUANTIFY is informational and educational only, not investment advice. Nothing here is a recommendation to buy or sell any security. All investment decisions are solely your own responsibility.</div>
<script>
function badgeClass(v){{return v==='Favorable'?'badge-ok':v==='Caution'?'badge-warn':v==='Risk'?'badge-danger':''}}
function renderConcentration(items){{const el=document.getElementById('concentrationBody');const withDollars=items.filter(it=>it.shares&&it.current_price!=null);const totalValue=withDollars.reduce((s,it)=>s+it.shares*it.current_price,0);if(!totalValue){{el.innerHTML='<div class="meta">Add a share count to your holdings (edit when saving from the terminal) to see sector concentration here.</div>';return}}const bySector={{}};withDollars.forEach(it=>{{const sec=it.sector||'Unclassified';bySector[sec]=(bySector[sec]||0)+it.shares*it.current_price}});const noDollar=items.length-withDollars.length;const rows=Object.entries(bySector).sort((a,b)=>b[1]-a[1]).map(([sec,val])=>{{const pct=val/totalValue*100;const warn=pct>30;return `<div style="margin-bottom:8px"><div style="display:flex;justify-content:space-between;font-size:12px"><span>${{sec}}${{warn?' ⚠':''}}</span><span>${{pct.toFixed(1)}}%</span></div><div style="background:var(--panel2);border-radius:3px;height:6px;margin-top:3px"><div style="background:${{warn?'var(--orange)':'var(--green)'}};height:6px;border-radius:3px;width:${{Math.min(100,pct)}}%"></div></div></div>`}}).join('');el.innerHTML=rows+(noDollar?`<div class="meta" style="margin-top:6px">${{noDollar}} holding(s) saved without a share count aren't included above.</div>`:'')+'<div class="meta" style="margin-top:6px">Rule of thumb only, not a risk model — flags over 30% in one sector.</div>'}}
function calcSize(){{const acct=parseFloat(document.getElementById('szAccount').value);const riskPct=parseFloat(document.getElementById('szRisk').value);const entry=parseFloat(document.getElementById('szEntry').value);const stopPct=parseFloat(document.getElementById('szStop').value);const el=document.getElementById('szResult');if(!(acct>0)||!(riskPct>0)||!(entry>0)||!(stopPct>0)){{el.innerHTML='<span style="color:var(--red)">Fill in all four fields with positive numbers.</span>';return}}const riskDollar=acct*(riskPct/100);const stopDollar=entry*(stopPct/100);const shares=Math.floor(riskDollar/stopDollar);const positionValue=shares*entry;el.innerHTML=`Suggested size: <b>${{shares}} shares</b> (~$${{positionValue.toLocaleString()}}) — risking ~$${{riskDollar.toFixed(2)}} if the stop is hit.`}}
async function load(){{const r=await fetch('/api/portfolio');const d=await r.json();const el=document.getElementById('list');if(!d.items?.length){{el.innerHTML='<div class="empty">Nothing saved yet. Open a ticker in the terminal and click ☆ Save to Portfolio.</div>';document.getElementById('concentrationBody').innerHTML='<div class="meta">Nothing saved yet.</div>';return}}el.innerHTML=d.items.map(it=>{{const sec=it.ai_report&&typeof it.ai_report==='object'?it.ai_report:null;const summary=sec?.quant_review||(typeof it.ai_report==='string'?it.ai_report:'');const date=new Date(it.saved_at*1000).toLocaleString();const plClass=it.return_pct==null?'':(it.return_pct>=0?'gain':'loss');const posLine=it.shares?`<div class="meta">${{it.shares}} sh @ $${{it.price}} → $${{it.current_price??'-'}}${{it.return_pct!=null?` · <span class="${{plClass}}">${{it.return_pct>=0?'+':''}}${{it.return_pct}}%${{it.pl_dollar!=null?` (${{it.pl_dollar>=0?'+':''}}$${{it.pl_dollar}})`:''}}</span>`:''}}</div>`:(it.return_pct!=null?`<div class="meta">$${{it.price}} → $${{it.current_price}} · <span class="${{plClass}}">${{it.return_pct>=0?'+':''}}${{it.return_pct}}%</span></div>`:'');return `<div class="item"><div class="item-head"><b>${{it.ticker}}</b><div>${{it.timing_verdict?`<span class="badge ${{badgeClass(it.timing_verdict)}}">${{it.timing_verdict}}</span> `:''}}<button class="remove" onclick="remove(${{it.id}})">Remove</button></div></div>${{posLine}}<div class="meta">Saved ${{date}} · Scan date ${{it.scan_date}} · RSI ${{it.rsi}}${{it.overall_score!=null?' · Score '+it.overall_score+'/100':''}}${{it.sector?' · '+it.sector:''}}</div>${{summary?`<div class="note">${{summary}}</div>`:''}}</div>`}}).join('');renderConcentration(d.items)}}
async function remove(id){{const f=new FormData();f.append('id',id);await fetch('/api/portfolio/remove',{{method:'POST',body:f}});load()}}
load();
</script></body></html>'''
    html = translate_body(html, lang, [
        (">Portfolio ·", f">{t('page_portfolio', lang)} ·"),
        (">Settings<", f">{t('nav_settings', lang)}<"),
        (">Subscription<", f">{t('nav_subscription', lang)}<"),
        (">Contact<", f">{t('nav_contact', lang)}<"),
        ("&larr; Back to Terminal", t("back_to_terminal", lang)),
        (">Loading...<", f">{t('loading', lang)}<"),
        (">Sector Concentration<", f">{t('sector_concentration', lang)}<"),
        (">Position Sizing Calculator<", f">{t('position_sizing', lang)}<"),
        ("Account size $<br>", f"{t('account_size', lang)}<br>"),
        ("Risk per trade %<br>", f"{t('risk_per_trade', lang)}<br>"),
        ("Entry price $<br>", f"{t('entry_price', lang)}<br>"),
        ("Stop distance %<br>", f"{t('stop_distance', lang)}<br>"),
        (">Calculate<", f">{t('calculate_btn', lang)}<"),
        ("Sizing arithmetic only — not a recommendation, and it doesn't account for correlation between your positions.", t("sizing_disclaimer", lang)),
        ("QUANTIFY is informational and educational only, not investment advice. Nothing here is a recommendation to buy or sell any security. All investment decisions are solely your own responsibility.", t("portfolio_disclaimer", lang)),
        ("Add a share count to your holdings (edit when saving from the terminal) to see sector concentration here.", t("add_share_count_hint", lang)),
        ("Rule of thumb only, not a risk model — flags over 30% in one sector.", t("sector_rule_of_thumb", lang)),
        ("Fill in all four fields with positive numbers.", t("fill_four_fields", lang)),
        ("Nothing saved yet. Open a ticker in the terminal and click ☆ Save to Portfolio.", t("nothing_saved_open_ticker", lang)),
        (">Nothing saved yet.<", f">{t('nothing_saved', lang)}<"),
        (">Remove</button>", f">{t('remove_btn', lang)}</button>"),
    ])
    return HTMLResponse(html)


@app.get("/subscription", response_class=HTMLResponse)
async def subscription_page(request: Request, reason: Optional[str] = None):
    user = get_logged_in_user(request)
    if not user: return RedirectResponse("/login", status_code=303)
    lang = get_user_lang(user)
    theme = get_user_theme(user)
    conn = db()
    row = conn.execute(
        "SELECT trial_ends_at,subscription_status,gumroad_subscription_id,ls_subscription_id FROM users WHERE email=?",
        (user,),
    ).fetchone()
    conn.close()
    trial_ends_at = row["trial_ends_at"] if row else None
    sub_status = (row["subscription_status"] if row else "trial") or "trial"
    days_left = max(0, int((trial_ends_at - time.time()) / 86400) + 1) if trial_ends_at else 0
    user_esc = html_lib.escape(user)
    ko = lang == "ko"

    trial_active = bool(trial_ends_at and time.time() < trial_ends_at)
    if sub_status == "active":
        plan_html = f'<span class="badge">{t("active_subscription", lang)}</span><p>{t("active_sub_thanks", lang)}</p>'
    elif trial_active:
        days_label = f'{days_left}일 남음' if ko else f'{days_left} day{"s" if days_left != 1 else ""} left'
        trial_desc = (f'모든 계정은 {TRIAL_DAYS}일 무료 체험 기간 동안 전체 기능을 이용할 수 있습니다. 체험 종료 후에도 계속 이용하려면 아래에서 구독하세요.' if ko
                      else f'Every account gets a {TRIAL_DAYS}-day free trial with full access. Subscribe below anytime to keep it going after your trial ends.')
        plan_html = (f'<span class="badge">{"무료 체험" if ko else "Free Trial"} &middot; {days_label}</span>'
                     f'<p>{trial_desc}</p>')
    else:
        trial_ended_desc = (f'{TRIAL_DAYS}일 무료 체험이 종료되었습니다. 스캐너와 AI 리뷰를 계속 이용하려면 아래에서 구독하세요.' if ko
                            else f'Your {TRIAL_DAYS}-day free trial has ended. Subscribe below to keep using the scanner and AI reports.')
        plan_html = f'<span class="badge warn">{t("trial_ended_badge", lang)}</span><p>{trial_ended_desc}</p>'

    if sub_status == "active":
        # Billing itself lives with the payment processor, not us -- we only ever
        # react to their webhooks -- so cancellation has to happen on their end too.
        # Point people at the right place instead of re-showing a "Subscribe" button,
        # which risked someone paying twice while already subscribed.
        if row and row["gumroad_subscription_id"]:
            checkout_html = (f'<p>{"결제 취소나 관리는 Gumroad 구매 영수증 이메일의 링크를 이용하거나, 결제할 때 사용한 계정으로 " if ko else "To cancel or manage billing, use the link in your Gumroad purchase receipt email, or go to "}'
                              '<a href="https://app.gumroad.com/library" target="_blank" rel="noopener">'
                              f'app.gumroad.com/library</a>{"에 로그인해서 확인하세요." if ko else " while logged into the account you paid with."}</p>')
        elif row and row["ls_subscription_id"]:
            checkout_html = (f'<p>{"결제 취소나 관리는 Lemon Squeezy 구매 영수증 이메일의 \'Manage Subscription\' 링크를 이용하세요." if ko else "To cancel or manage billing, use the \"Manage Subscription\" link in your Lemon Squeezy purchase receipt email."}</p>')
        else:
            checkout_html = f'<p>{"결제 취소나 관리는" if ko else "To cancel or manage billing,"} <a href="/contact">{"문의하기" if ko else "contact us"}</a>.</p>'
    else:
        checkout_url = GUMROAD_CHECKOUT_URL or LEMONSQUEEZY_CHECKOUT_URL
        if checkout_url:
            checkout_html = f'<a href="{checkout_url}" target="_blank" rel="noopener" class="subscribe-btn">{t("subscribe_btn", lang)}</a>'
        else:
            checkout_html = f'<div class="subscribe-btn disabled">{t("paid_plans_soon", lang)}</div>'

    reason_banner = ""
    if reason == "trial_ended" and sub_status != "active":
        banner_text = ('무료 체험이 종료되어 이 페이지로 이동되었습니다. 스캐너와 AI 리뷰를 다시 이용하려면 아래에서 구독하세요.' if ko
                       else 'Your free trial has ended — that\'s why you were sent here. Subscribe below to get back into the scanner and AI reports.')
        reason_banner = f'<div class="reason-banner">{banner_text}</div>'

    return HTMLResponse(f'''<!doctype html><html lang="{lang}" data-theme="{theme}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>QUANTIFY. Subscription</title><style>{DARK_THEME_VARS}

:root{{--bg:#ffffff;--panel:#ffffff;--panel2:#f5f7f6;--border:#e2e6e3;--text:#3a4440;--head:#12201a;--dim:#77837e;--green:#0e8a5f;--orange:#a8660a}}
*{{box-sizing:border-box}}body{{background:var(--bg);color:var(--text);font:16px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;margin:0;padding:14px}}
header{{background:var(--panel);border:1px solid var(--border);padding:14px 20px;display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;border-radius:10px;flex-wrap:wrap;gap:12px}}
.brand{{font-weight:700;font-size:19px;color:var(--head);text-decoration:none}}.brand span{{color:var(--dim)}}
.headerRight{{display:flex;align-items:center;gap:16px;flex-wrap:wrap}}
a.back{{color:var(--green);text-decoration:underline;font-size:15px;font-weight:600}}
.wrap{{max-width:560px;margin:0 auto}}
.card{{background:var(--panel);border:1px solid var(--border);padding:28px;margin-bottom:18px;border-radius:12px;box-shadow:0 1px 2px rgba(18,32,26,.04)}}
.card h2{{color:var(--head);font-size:15px;margin:0 0 18px;border-bottom:1px solid var(--border);padding-bottom:12px;text-transform:uppercase;letter-spacing:.4px;font-weight:700}}
.badge{{display:inline-block;padding:5px 12px;border-radius:12px;background:#e6f5ee;color:var(--green);font-weight:700;font-size:14px}}
.badge.warn{{background:#fbf1e0;color:var(--orange)}}
p{{color:var(--text);font-size:15.5px;line-height:1.75;margin-top:16px}}
.subscribe-btn{{display:block;text-align:center;margin-top:20px;background:var(--green);color:#ffffff;padding:14px;border-radius:8px;text-decoration:none;font-weight:700;font-size:15px}}
.subscribe-btn.disabled{{background:var(--panel2);color:var(--dim);border:1px solid var(--border);cursor:default}}
.reason-banner{{max-width:560px;margin:0 auto 14px;background:#fbf1e0;border:1px solid #ecdcb8;color:var(--orange);padding:14px 18px;border-radius:10px;font-size:14.5px;font-weight:600;line-height:1.6}}
</style></head><body><header><a class="brand" href="/terminal">QUANTIFY<span>.</span></a><div class="headerRight"><span style="color:var(--dim);font-size:14px">{t("page_subscription", lang)} · {user_esc}</span><a class="back" href="/portfolio">{t("nav_portfolio", lang)}</a><a class="back" href="/settings">{t("nav_settings", lang)}</a><a class="back" href="/contact">{t("nav_contact", lang)}</a><a class="back" href="/terminal">{t("back_to_terminal", lang)}</a></div></header>
<div class="wrap">{reason_banner}<div class="card">
<h2>{t("current_plan", lang)}</h2>
{plan_html}
{checkout_html}
</div></div>
</body></html>''')


@app.get("/contact", response_class=HTMLResponse)
async def contact_page(request: Request, msg: Optional[str] = None, error: Optional[str] = None):
    user = get_logged_in_user(request)
    if not user: return RedirectResponse("/login", status_code=303)
    lang = get_user_lang(user)
    theme = get_user_theme(user)
    user = html_lib.escape(user)
    msg = html_lib.escape(msg) if msg else ''
    error = html_lib.escape(error) if error else ''
    return HTMLResponse(f'''<!doctype html><html lang="{lang}" data-theme="{theme}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>QUANTIFY. Contact</title><style>{DARK_THEME_VARS}

:root{{--bg:#ffffff;--panel:#ffffff;--panel2:#f5f7f6;--border:#e2e6e3;--text:#3a4440;--head:#12201a;--dim:#77837e;--green:#0e8a5f;--red:#c8402c}}
*{{box-sizing:border-box}}body{{background:var(--bg);color:var(--text);font:16px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;margin:0;padding:14px}}
header{{background:var(--panel);border:1px solid var(--border);padding:14px 20px;display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;border-radius:10px;flex-wrap:wrap;gap:12px}}
.brand{{font-weight:700;font-size:19px;color:var(--head);text-decoration:none}}.brand span{{color:var(--dim)}}
.headerRight{{display:flex;align-items:center;gap:16px;flex-wrap:wrap}}
a.back{{color:var(--green);text-decoration:underline;font-size:15px;font-weight:600}}
.wrap{{max-width:560px;margin:0 auto}}
.card{{background:var(--panel);border:1px solid var(--border);padding:28px;margin-bottom:18px;border-radius:12px;box-shadow:0 1px 2px rgba(18,32,26,.04)}}
.card h2{{color:var(--head);font-size:15px;margin:0 0 18px;border-bottom:1px solid var(--border);padding-bottom:12px;text-transform:uppercase;letter-spacing:.4px;font-weight:700}}
label{{display:block;font-size:13.5px;font-weight:600;color:var(--dim);margin:16px 0 7px}}
textarea{{width:100%;background:var(--panel2);border:1px solid var(--border);color:var(--head);padding:12px;font:15px -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;border-radius:7px;resize:vertical;min-height:150px}}
button{{margin-top:18px;background:var(--green);border:1px solid var(--green);color:#ffffff;padding:12px 16px;font:14.5px -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-weight:700;cursor:pointer;width:100%;border-radius:7px}}
.msg{{font-size:14px;min-height:18px;margin-top:10px}}.ok{{color:var(--green)}}.err{{color:var(--red)}}
</style></head><body><header><a class="brand" href="/terminal">QUANTIFY<span>.</span></a><div class="headerRight"><span style="color:var(--dim);font-size:14px">{t("page_contact", lang)} · {user}</span><a class="back" href="/portfolio">{t("nav_portfolio", lang)}</a><a class="back" href="/settings">{t("nav_settings", lang)}</a><a class="back" href="/subscription">{t("nav_subscription", lang)}</a><a class="back" href="/terminal">{t("back_to_terminal", lang)}</a></div></header>
<div class="wrap"><div class="card">
<h2>{t("contact_us", lang)}</h2>
<div class="msg {'ok' if msg else 'err' if error else ''}">{msg or error}</div>
<label>{t("your_message", lang)}</label>
<textarea id="message" placeholder="{t("message_placeholder", lang)}"></textarea>
<button onclick="sendContact()">{t("send_message_btn", lang)}</button>
</div></div>
<script>
async function sendContact(){{const message=document.getElementById('message').value.trim();if(!message)return alert({json.dumps(t("write_message_first", lang))});const f=new FormData();f.append('message',message);const r=await fetch('/api/contact',{{method:'POST',body:f}});const d=await r.json();if(r.ok){{document.getElementById('message').value='';}}location.href='/contact?'+(r.ok?'msg=':'error=')+encodeURIComponent(d.message||d.error)}}
</script>
</body></html>''')


@app.post("/api/contact")
async def submit_contact(request: Request, message: str = Form(...)):
    user = get_logged_in_user(request)
    if not user: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if _is_locked_out(CONTACT_ATTEMPTS, user, CONTACT_MAX_ATTEMPTS):
        return JSONResponse({"error": "Too many messages sent recently. Try again later."}, status_code=429)
    message = message.strip()
    if not message:
        return JSONResponse({"error": "Message cannot be empty."}, status_code=400)
    if len(message) > 4000:
        return JSONResponse({"error": "Message is too long (4000 characters max)."}, status_code=400)
    _register_failed_attempt(CONTACT_ATTEMPTS, user)
    if not SENDER_EMAIL:
        return JSONResponse({"error": "Contact form is not configured yet. Please try again later."}, status_code=503)
    ok = await asyncio.to_thread(send_email_notification, CONTACT_NOTIFY_EMAIL, f"[QUANTIFY Contact] Message from {user}",
                                  f"From: {user}\n\n{message}", 3, user)
    if not ok:
        return JSONResponse({"error": "Could not send your message. Please try again later."}, status_code=500)
    return {"message": "Your message has been sent. We'll get back to you soon."}


@app.post("/api/lemonsqueezy/webhook")
async def lemonsqueezy_webhook(request: Request):
    body = await request.body()
    if not LEMONSQUEEZY_WEBHOOK_SECRET:
        print("[LemonSqueezy] Webhook received but LEMONSQUEEZY_WEBHOOK_SECRET is not set — ignoring.", flush=True)
        return JSONResponse({"error": "Webhook not configured"}, status_code=503)
    signature = request.headers.get("X-Signature", "")
    expected = hmac.new(LEMONSQUEEZY_WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        print("[Error: LemonSqueezySignature] Webhook signature mismatch — rejected.", flush=True)
        return JSONResponse({"error": "Invalid signature"}, status_code=401)
    try:
        payload = json.loads(body)
    except Exception as exc:
        print(f"[Error: {type(exc).__name__}] LemonSqueezy webhook payload parse error: {exc}", flush=True)
        return JSONResponse({"error": "Invalid payload"}, status_code=400)

    event_name = payload.get("meta", {}).get("event_name", "")
    data = payload.get("data", {})
    attrs = data.get("attributes", {})
    email = (attrs.get("user_email") or "").strip().lower()
    if not email:
        return {"ok": True}

    conn = db()
    try:
        if event_name in ("subscription_created", "subscription_updated", "subscription_resumed"):
            ls_status = attrs.get("status")
            sub_status = "active" if ls_status in ("active", "on_trial") else (ls_status or "active")
            was_active = (conn.execute("SELECT subscription_status FROM users WHERE email=?",
                                       (email,)).fetchone() or {"subscription_status": None})["subscription_status"] == "active"
            cur = conn.execute(
                "UPDATE users SET subscription_status=?,ls_customer_id=?,ls_subscription_id=? WHERE email=?",
                (sub_status, str(attrs.get("customer_id", "")), str(data.get("id", "")), email),
            )
            # subscription_updated fires on every renewal too; only the transition into
            # active is a conversion, or the count would climb every billing cycle.
            if sub_status == "active" and not was_active and cur.rowcount:
                asyncio.create_task(asyncio.to_thread(_log_payment_event, email))
        elif event_name in ("subscription_cancelled", "subscription_expired"):
            conn.execute("UPDATE users SET subscription_status='expired' WHERE email=?", (email,))
        conn.commit()
    except Exception as exc:
        print(f"[Error: {type(exc).__name__}] LemonSqueezy webhook DB update failed: {exc}", flush=True)
    finally:
        conn.close()
    return {"ok": True}


@app.post("/api/gumroad/webhook")
async def gumroad_webhook(request: Request):
    # Gumroad Ping payloads are unsigned form data (no secret to verify against —
    # see https://gumroad.com/ping) and Gumroad's own docs warn that a ping can be
    # forged or dropped, so it must be treated as a trigger, not as data: on a
    # "sale" ping we look the sale back up via the API (GUMROAD_ACCESS_TOKEN) and
    # only grant access if that lookup confirms it's real and matches the email
    # claimed. Without a configured token we can't verify, so we don't grant
    # access off an unverified ping — worst case a legitimate buyer waits for the
    # nightly reconciliation pass instead of getting access instantly.
    form = await request.form()
    email = (form.get("email") or "").strip().lower()
    resource_name = form.get("resource_name", "sale")
    sale_id = str(form.get("sale_id") or "")
    refunded = (form.get("refunded") or "").lower() == "true"
    print(f"[gumroad] Ping received: resource_name={resource_name} email={email or '(none)'} "
          f"sale_id={sale_id or '(none)'} test={form.get('test')}", flush=True)
    if not email:
        return {"ok": True}

    conn = db()
    try:
        if resource_name == "refund" or refunded:
            # Downgrading access on an unverified ping is the safe direction to
            # err in (worst case a legitimate subscriber has to re-subscribe),
            # unlike granting it, so this doesn't need API verification.
            cur = conn.execute("UPDATE users SET subscription_status='expired' WHERE email=?", (email,))
            conn.commit()
            if cur.rowcount == 0:
                print(f"[gumroad] Refund ping for {email} matched no QUANTIFY account.", flush=True)
        elif resource_name == "sale" and sale_id:
            data = await asyncio.to_thread(_gumroad_api_get, f"/sales/{sale_id}", {})
            sale = (data or {}).get("sale") or {}
            verified_email = (sale.get("email") or "").strip().lower()
            if verified_email and verified_email == email and not sale.get("ended") and not sale.get("cancelled"):
                cur = conn.execute(
                    "UPDATE users SET subscription_status='active',gumroad_subscription_id=? WHERE email=?",
                    (str(sale.get("subscription_id") or ""), email),
                )
                conn.commit()
                if cur.rowcount == 0:
                    print(f"[gumroad] Verified sale for {email} (sale_id={sale_id}) matched no QUANTIFY account "
                          f"— the buyer needs to sign up on QUANTIFY with this exact email.", flush=True)
                else:
                    print(f"[gumroad] Granted active access to {email} (sale_id={sale_id})", flush=True)
                    asyncio.create_task(asyncio.to_thread(_log_payment_event, email))
            else:
                print(f"[gumroad] Sale ping for {email} (sale_id={sale_id}) could not be verified "
                      f"(verified_email={verified_email or '(lookup failed)'}) — ignored.", flush=True)
    except Exception as exc:
        print(f"[Error: {type(exc).__name__}] Gumroad webhook DB update failed: {exc}", flush=True)
    finally:
        conn.close()
    return {"ok": True}


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    user = get_logged_in_user(request)
    if not user: return RedirectResponse("/login", status_code=303)
    if not disclaimer_accepted(user): return RedirectResponse("/accept-disclaimer", status_code=303)
    lang = get_user_lang(user)
    theme = get_user_theme(user)
    conn = db()
    prov_row = conn.execute("SELECT auth_provider FROM users WHERE email=?", (user,)).fetchone()
    conn.close()
    is_google = bool(prov_row and prov_row["auth_provider"] == "google")
    delete_field = (
        f'<label>{t("confirm_email_to_delete", lang)}</label>'
        '<input type="email" id="delete_confirm_email" autocomplete="off" autocapitalize="none">'
        if is_google else
        f'<label>{t("confirm_password", lang)}</label><input type="password" id="delete_password">'
    )
    user = html_lib.escape(user)
    return HTMLResponse(f'''<!doctype html><html lang="{lang}" data-theme="{theme}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>QUANTIFY. Settings</title><style>{DARK_THEME_VARS}

:root{{--bg:#ffffff;--panel:#ffffff;--panel2:#f5f7f6;--border:#e2e6e3;--text:#3a4440;--head:#12201a;--dim:#77837e;--green:#0e8a5f;--red:#c8402c}}
*{{box-sizing:border-box}}body{{background:var(--bg);color:var(--text);font:16px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;margin:0;padding:14px}}
header{{background:var(--panel);border:1px solid var(--border);padding:14px 20px;display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;border-radius:10px;flex-wrap:wrap;gap:12px}}
.brand{{font-weight:700;font-size:19px;color:var(--head);text-decoration:none}}.brand span{{color:var(--dim)}}
.headerRight{{display:flex;align-items:center;gap:16px;flex-wrap:wrap}}
a.back{{color:var(--green);text-decoration:underline;font-size:15px;font-weight:600}}
.wrap{{max-width:560px;margin:0 auto}}
.card{{background:var(--panel);border:1px solid var(--border);padding:28px;margin-bottom:18px;border-radius:12px;box-shadow:0 1px 2px rgba(18,32,26,.04)}}
.card h2{{color:var(--head);font-size:15px;margin:0 0 18px;border-bottom:1px solid var(--border);padding-bottom:12px;text-transform:uppercase;letter-spacing:.4px;font-weight:700}}
label{{display:block;font-size:13.5px;font-weight:600;color:var(--dim);margin:16px 0 7px}}
select,input{{width:100%;background:var(--panel2);border:1px solid var(--border);color:var(--head);padding:11px 12px;font:15px -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;border-radius:7px}}
button{{margin-top:18px;background:var(--green);border:1px solid var(--green);color:#ffffff;padding:12px 16px;font:14.5px -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-weight:700;cursor:pointer;width:100%;border-radius:7px}}
button:hover{{opacity:.9}}
.msg{{font-size:14px;min-height:18px;margin-top:10px}}.ok{{color:var(--green)}}.err{{color:var(--red)}}
.info-row{{display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid var(--border);font-size:14.5px}}
.info-row:last-child{{border-bottom:none}}
.info-row b{{color:var(--head)}}
.alert-row{{display:flex;justify-content:space-between;align-items:center;padding:11px 0;border-bottom:1px solid var(--border);font-size:14.5px}}
.alert-row:last-child{{border-bottom:none}}
.remove-btn{{background:transparent;border:1px solid var(--red);color:var(--red);width:auto;padding:6px 12px;margin:0;font-size:13px;font-weight:600;border-radius:6px}}
.empty-hint{{color:var(--dim);font-size:14.5px;padding:10px 0}}
.card.danger{{border-color:var(--red)}}
.card.danger h2{{color:var(--red)}}
button.danger-btn{{background:transparent;border:1px solid var(--red);color:var(--red)}}
</style></head><body><header><a class="brand" href="/terminal">QUANTIFY<span>.</span></a><div class="headerRight"><span style="color:var(--dim);font-size:14px">{t("page_settings", lang)} · {user}</span><a class="back" href="/portfolio">{t("nav_portfolio", lang)}</a><a class="back" href="/subscription">{t("nav_subscription", lang)}</a><a class="back" href="/contact">{t("nav_contact", lang)}</a><a class="back" href="/terminal">{t("back_to_terminal", lang)}</a></div></header>
<div class="wrap">
<div class="card"><h2>{t("account", lang)}</h2>
<div id="account-info"><div class="empty-hint">{t("loading", lang)}</div></div>
<a href="/subscription" style="display:block;text-align:center;margin-top:16px;color:var(--green);text-decoration:underline;font-size:14.5px;font-weight:600">{t("manage_subscription", lang)}</a>
</div>
<div class="card"><h2>{t("display", lang)}</h2>
<label>{t("theme_label", lang)}</label><select id="theme"><option value="light">{t("opt_light", lang)}</option><option value="dark">{t("opt_dark", lang)}</option></select>
<label>{t("ai_report_language", lang)}</label><select id="language"><option value="en">English</option><option value="ko">Korean (한국어)</option></select><p style="font-size:13px;color:var(--dim);margin-top:6px">{t("language_toggle_hint", lang)}</p>
<label>{t("default_scanner_sort", lang)}</label><select id="default_sort"><option value="overall_score">{t("opt_score", lang)}</option><option value="change_pct">{t("opt_change_pct", lang)}</option><option value="ticker">{t("opt_ticker_az", lang)}</option></select>
<label>{t("default_scanner_view", lang)}</label><select id="default_view"><option value="list">{t("opt_list", lang)}</option><option value="heatmap">{t("opt_heatmap", lang)}</option></select>
<label>{t("email_high_score", lang)}</label><select id="high_score_alerts"><option value="0">{t("opt_off", lang)}</option><option value="1">{t("opt_on", lang)}</option></select><p style="font-size:13px;color:var(--dim);margin-top:6px">{t("high_score_alert_hint", lang)}</p>
<button onclick="saveSettings()">{t("save_settings_btn", lang)}</button><div class="msg" id="settings-msg"></div>
</div>
<div class="card" id="alerts"><h2>{t("price_alerts", lang)}</h2>
<div id="alerts-list"><div class="empty-hint">{t("loading", lang)}</div></div>
</div>
<div class="card"><h2>{t("change_password", lang)}</h2>
<label>{t("current_password", lang)}</label><input type="password" id="current_password">
<label>{t("new_password", lang)}</label><input type="password" id="new_password">
<p style="font-size:13px;color:var(--dim);margin-top:8px">{t("password_char_hint", lang)}</p>
<button onclick="changePassword()">{t("change_password_btn", lang)}</button><div class="msg" id="password-msg"></div>
</div>
<div class="card danger"><h2>{t("danger_zone", lang)}</h2>
<p style="font-size:14.5px;color:var(--text);margin:0 0 12px">{t("delete_account_warning", lang)}</p>
{delete_field}
<button class="danger-btn" onclick="deleteAccount()">{t("delete_account_btn", lang)}</button><div class="msg" id="delete-msg"></div>
</div>
</div>
<script>
function fmtDate(ts){{return ts?new Date(ts*1000).toLocaleDateString():'-'}}
async function loadSettings(){{const r=await fetch('/api/settings');const d=await r.json();if(d.pref_theme)document.getElementById('theme').value=d.pref_theme;if(d.pref_language)document.getElementById('language').value=d.pref_language;if(d.pref_default_sort)document.getElementById('default_sort').value=d.pref_default_sort;if(d.pref_default_view)document.getElementById('default_view').value=d.pref_default_view;document.getElementById('high_score_alerts').value=d.pref_high_score_alerts?'1':'0';
const statusLabel={{active:{json.dumps(t("status_active", lang))},trial:{json.dumps(t("status_trial", lang))},expired:{json.dumps(t("status_expired", lang))},cancelled:{json.dumps(t("status_cancelled", lang))},paused:{json.dumps(t("status_paused", lang))}}}[d.subscription_status]||d.subscription_status;
document.getElementById('account-info').innerHTML=`<div class="info-row"><span>{t("email_label", lang)}</span><b>${{d.email||'-'}}</b></div><div class="info-row"><span>{t("member_since", lang)}</span><b>${{fmtDate(d.created_at)}}</b></div><div class="info-row"><span>{t("plan_status", lang)}</span><b>${{statusLabel||'-'}}</b></div>`}}
async function saveSettings(){{const f=new FormData();f.append('theme',document.getElementById('theme').value);f.append('language',document.getElementById('language').value);f.append('default_sort',document.getElementById('default_sort').value);f.append('default_view',document.getElementById('default_view').value);f.append('high_score_alerts',document.getElementById('high_score_alerts').value);const r=await fetch('/api/settings',{{method:'POST',body:f}});const d=await r.json();const el=document.getElementById('settings-msg');el.className='msg '+(r.ok?'ok':'err');el.innerText=d.message||d.error}}
async function changePassword(){{const f=new FormData();f.append('current_password',document.getElementById('current_password').value);f.append('new_password',document.getElementById('new_password').value);const r=await fetch('/api/settings/password',{{method:'POST',body:f}});const d=await r.json();const el=document.getElementById('password-msg');el.className='msg '+(r.ok?'ok':'err');el.innerText=d.message||d.error;if(r.ok){{document.getElementById('current_password').value='';document.getElementById('new_password').value=''}}}}
async function loadAlerts(){{const r=await fetch('/api/alerts/list');const d=await r.json();const el=document.getElementById('alerts-list');if(!d.alerts?.length){{el.innerHTML='<div class="empty-hint">'+{json.dumps(t("loading_empty_hint", lang))}+'</div>';return}}el.innerHTML=d.alerts.map(a=>`<div class="alert-row"><span>${{a.ticker}} ${{a.direction==='below'?{json.dumps(t("dir_below", lang))}:{json.dumps(t("dir_above", lang))}}} $${{a.target_price}}${{a.is_sent?' <span style="color:var(--dim)">'+{json.dumps(t("sent_suffix", lang))}+'</span>':''}}</span><button class="remove-btn" onclick="removeAlert(${{a.id}})">{t("remove_btn", lang)}</button></div>`).join('')}}
async function removeAlert(id){{const f=new FormData();f.append('id',id);await fetch('/api/alerts/remove',{{method:'POST',body:f}});loadAlerts()}}
async function deleteAccount(){{const pwEl=document.getElementById('delete_password');const emEl=document.getElementById('delete_confirm_email');const val=(pwEl||emEl).value;if(!val)return;if(!confirm({json.dumps(t("confirm_delete_account", lang))}))return;const f=new FormData();f.append(pwEl?'password':'confirm_email',val);const r=await fetch('/api/account/delete',{{method:'POST',body:f}});const d=await r.json();if(r.ok){{location.href='/'}}else{{const el=document.getElementById('delete-msg');el.className='msg err';el.innerText=d.error}}}}
loadSettings();loadAlerts();
</script></body></html>''')


if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8888")), reload=False)
