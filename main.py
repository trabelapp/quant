import asyncio
import gc
import hashlib
import html as html_lib
import hmac
import json
import os
import re
import secrets
import smtplib
import sqlite3
import time
import urllib.parse
from datetime import datetime, timedelta
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
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
BREVO_API_KEY = os.getenv("BREVO_API_KEY", "")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
LEMONSQUEEZY_CHECKOUT_URL = os.getenv("LEMONSQUEEZY_CHECKOUT_URL", "")
LEMONSQUEEZY_WEBHOOK_SECRET = os.getenv("LEMONSQUEEZY_WEBHOOK_SECRET", "")
GUMROAD_CHECKOUT_URL = os.getenv("GUMROAD_CHECKOUT_URL", "")
TRIAL_DAYS = 7

# Constituent refresh is deliberately infrequent. Market data is fetched in
# small batches with pauses, not as hundreds of simultaneous requests.
UNIVERSE_TTL = 24 * 3600
# Batch size and pause tuned to avoid the 3-minute delay issue.
MARKET_BATCH_SIZE = int(os.getenv("MARKET_BATCH_SIZE", "50"))
MARKET_BATCH_PAUSE = float(os.getenv("MARKET_BATCH_PAUSE", "0.5"))
HISTORICAL_TTL = 600
NEWS_TTL = 900
SHORT_INTEREST_TTL = 24 * 3600
CONSTITUENT_HTTP_TIMEOUT = 12

DATA_DIR = Path(os.getenv("DATA_DIR", "."))
UNIVERSE_FILE = DATA_DIR / "universe_cache.json"
BACKTEST_FILE = DATA_DIR / "backtest_cache.json"
BACKTEST_SAMPLE_SIZE = 200
BACKTEST_REFRESH_SECONDS = 7 * 24 * 3600
BACKTEST_CACHE = {"computed_at": None, "results": None, "error": None}

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
AI_CONCURRENCY = max(1, int(os.getenv("AI_CONCURRENCY", "4")))
QUANT_PASS_THRESHOLD = float(os.getenv("QUANT_PASS_THRESHOLD", "83"))
OVERALL_SCORE_THRESHOLD = float(os.getenv("OVERALL_SCORE_THRESHOLD", "50"))
AI_PROMPT_VERSION = 4
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
MARKET_SCAN_INTERVAL_SECONDS = 30 * 60

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
        """)
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
        if "pref_language" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN pref_language TEXT NOT NULL DEFAULT 'en'")
        if "pref_default_mode" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN pref_default_mode TEXT NOT NULL DEFAULT 'Long-Term Momentum Pullback'")
        if "pref_default_sort" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN pref_default_sort TEXT NOT NULL DEFAULT 'overall_score'")
        if "pref_default_view" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN pref_default_view TEXT NOT NULL DEFAULT 'list'")
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
        now_ts = time.time()
        conn.execute("UPDATE users SET created_at=? WHERE created_at IS NULL", (now_ts,))
        conn.execute("UPDATE users SET trial_ends_at=? WHERE trial_ends_at IS NULL", (now_ts + 7 * 86400,))

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


def calculate_alpha_score(close, rsi, macd_hist):
    if len(close) < 70:
        return None
    try:
        momentum_20 = float(close.iloc[-1] / close.iloc[-21] - 1)
        momentum_60 = float(close.iloc[-1] / close.iloc[-61] - 1)
        volatility = float(close.pct_change().rolling(20).std().iloc[-1])
        macd_scale = max(float(close.iloc[-1]) * 0.01, 1e-9)
        macd_component = max(-1.0, min(1.0, float(macd_hist.iloc[-1]) / macd_scale))
        momentum_component = max(-1.0, min(1.0, momentum_20 / 0.15))
        long_component = max(-1.0, min(1.0, momentum_60 / 0.30))
        rsi_component = 1.0 - min(abs(float(rsi.iloc[-1]) - 55.0) / 45.0, 1.0)
        risk_component = 1.0 - min(volatility / 0.06, 1.0)
        raw = (0.30 * momentum_component + 0.165 * long_component +
               0.17 * macd_component + 0.30 * (2 * rsi_component - 1) +
               0.065 * (2 * risk_component - 1))
        return round(max(0.0, min(100.0, 50.0 + raw * 50.0)), 1)
    except Exception as e:
        print(f"[Error: {type(e).__name__}] Alpha score calculation error: {e}")
        return None


def calculate_alpha_score_series(close, rsi, macd_hist):
    """Vectorized replay of calculate_alpha_score() across an entire price history,
    used only for the backtest — must stay numerically identical to the per-row version."""
    momentum_20 = close / close.shift(20) - 1
    momentum_60 = close / close.shift(60) - 1
    volatility = close.pct_change().rolling(20).std()
    macd_scale = (close * 0.01).clip(lower=1e-9)
    macd_component = (macd_hist / macd_scale).clip(-1, 1)
    momentum_component = (momentum_20 / 0.15).clip(-1, 1)
    long_component = (momentum_60 / 0.30).clip(-1, 1)
    rsi_component = 1.0 - (rsi - 55.0).abs().clip(upper=45.0) / 45.0
    risk_component = 1.0 - (volatility / 0.06).clip(upper=1.0)
    raw = (0.30 * momentum_component + 0.165 * long_component + 0.17 * macd_component +
           0.30 * (2 * rsi_component - 1) + 0.065 * (2 * risk_component - 1))
    return (50.0 + raw * 50.0).clip(0.0, 100.0).round(1)


def _is_rate_limit_error(exc) -> bool:
    msg = str(exc).lower()
    return "rate limit" in msg or "too many requests" in msg


def _download_group(tickers, period="2y", interval="1d", raise_on_rate_limit=False):
    try:
        return yf.download(
            tickers=tickers, period=period, interval=interval,
            auto_adjust=True, progress=False, threads=True, group_by="column"
        )
    except Exception as exc:
        if raise_on_rate_limit and _is_rate_limit_error(exc):
            raise
        print(f"[Error: {type(exc).__name__}] yfinance batch download error: {exc}")
        return None


RATE_LIMIT_BACKOFF_SECONDS = 10.0


async def _download_single_with_backoff(ticker, period, interval):
    for attempt in range(2):
        try:
            return await asyncio.to_thread(_download_group, [ticker], period, interval, True)
        except Exception as exc:
            if attempt == 0 and _is_rate_limit_error(exc):
                print(f"[RateLimit] yfinance rate limit hit on {ticker} — backing off {RATE_LIMIT_BACKOFF_SECONDS:.0f}s")
                await asyncio.sleep(RATE_LIMIT_BACKOFF_SECONDS)
                continue
            print(f"[Error: {type(exc).__name__}] Single download failed ({ticker}): {exc}")
            return None
    return None


INTERVAL_PERIODS = {"1h": "730d", "1d": "2y", "1wk": "5y", "1mo": "max"}


async def download_stock(ticker: str, interval="1d"):
    key = f"single:{ticker}:{interval}"
    cached = CACHE["historical"].get(key)
    if cached and time.time() - cached["ts"] < HISTORICAL_TTL:
        return cached["data"]
    period = INTERVAL_PERIODS.get(interval, "2y")
    data = await _download_single_with_backoff(ticker, period, interval)
    if data is None or data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data.xs(ticker, axis=1, level=1)
        except Exception:
            data = data.xs(ticker, axis=1, level=-1)
    data = data.dropna(how="all")
    if len(data) >= 30:
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

    for start in range(0, len(to_fetch), MARKET_BATCH_SIZE):
        chunk = to_fetch[start:start + MARKET_BATCH_SIZE]
        data = await asyncio.to_thread(_download_group, chunk, "2y", "1d")
        if data is not None and not data.empty:
            for ticker in chunk:
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        if ticker in data.columns.get_level_values(-1):
                            df = data.xs(ticker, axis=1, level=-1).dropna(how="all")
                        elif ticker in data.columns.get_level_values(1):
                            df = data.xs(ticker, axis=1, level=1).dropna(how="all")
                        else:
                            continue
                    else:
                        df = data.dropna(how="all") if len(chunk) == 1 else None
                    if df is not None and len(df) >= 30:
                        CACHE["historical"][f"single:{ticker}:1d"] = {"data": df, "ts": time.time()}
                        results[ticker] = df
                except Exception as exc:
                    print(f"[Error: {type(exc).__name__}] Batch parse error ({ticker}): {exc}")
        BATCH_STATUS["processed"] = (len(unique) - len(to_fetch)) + min(start + len(chunk), len(to_fetch))
        if start + MARKET_BATCH_SIZE < len(to_fetch):
            await asyncio.sleep(MARKET_BATCH_PAUSE)

    missing = [t for t in unique if t not in results]
    if missing:
        print(f"[Retry] Retrying {len(missing)} tickers missing from batch download individually: {missing}")
        for ticker in missing:
            try:
                data = await _download_single_with_backoff(ticker, "2y", "1d")
                if data is None or data.empty:
                    continue
                if isinstance(data.columns, pd.MultiIndex):
                    try:
                        df = data.xs(ticker, axis=1, level=1).dropna(how="all")
                    except Exception:
                        df = data.xs(ticker, axis=1, level=-1).dropna(how="all")
                else:
                    df = data.dropna(how="all")
                if df is not None and len(df) >= 30:
                    CACHE["historical"][f"single:{ticker}:1d"] = {"data": df, "ts": time.time()}
                    results[ticker] = df
            except Exception as exc:
                print(f"[Error: {type(exc).__name__}] Individual retry failed ({ticker}): {exc}")
            await asyncio.sleep(0.2)
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


def analyze_dataframe(ticker, df):
    if df is None or len(df) < 70:
        return None
    try:
        close = normalize_series(df, "Close").dropna()
        if len(close) < 70:
            return None
        rsi_series = calculate_rsi(close)
        _, _, macd_hist = calculate_macd(close)
        price = float(close.iloc[-1]); prev = float(close.iloc[-2])
        change = (price / prev - 1) * 100 if prev else 0.0
        score = calculate_alpha_score(close, rsi_series, macd_hist)
        if score is None:
            return None

        window = close.tail(252)
        high_52w = float(window.max()); low_52w = float(window.min())
        pct_from_high = round((price / high_52w - 1) * 100, 2) if high_52w else None
        pct_from_low = round((price / low_52w - 1) * 100, 2) if low_52w else None
        above_trend = None
        if len(close) >= 200:
            sma200 = close.rolling(200).mean().iloc[-1]
            if pd.notna(sma200):
                above_trend = bool(price > float(sma200))

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
                "above_200d_sma": above_trend, "volume_ratio": volume_ratio}
    except Exception as e:
        print(f"[Error: {type(e).__name__}] Dataframe analysis error ({ticker}): {e}")
        return None


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
                         ai_report,short_percent,ai_status,ai_mode,ai_updated_at,ai_error,quant_pass,created_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(scan_date,ticker) DO UPDATE SET
                            universe=excluded.universe, price=excluded.price, change_pct=excluded.change_pct,
                            alpha_score=excluded.alpha_score, rsi=excluded.rsi, macd=excluded.macd,
                            pct_from_52w_high=excluded.pct_from_52w_high, pct_from_52w_low=excluded.pct_from_52w_low,
                            above_200d_sma=excluded.above_200d_sma, volume_ratio=excluded.volume_ratio,
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
            cutoff_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
            conn.execute("DELETE FROM scan_history WHERE scan_date < ?", (cutoff_date,))
            conn.commit()
            alerts = conn.execute("SELECT id,email,ticker,target_price FROM user_alerts WHERE is_sent=0").fetchall()
            prices = {r["ticker"]: r["price"] for r in results}
            for alert in alerts:
                price = prices.get(alert["ticker"])
                if price is not None and price >= alert["target_price"]:
                    if send_email_notification(alert["email"],
                        f"[QUANTIFY Alert] {alert['ticker']} hit your target price",
                        f"{alert['ticker']} latest observed price: ${price:.2f}\nYour target price: ${alert['target_price']:.2f}"):
                        conn.execute("UPDATE user_alerts SET is_sent=1 WHERE id=?", (alert["id"],))
            conn.commit(); conn.close()
            start_ai_prefetch(mode)
            return results
        except Exception as exc:
            print(f"[Error: {type(exc).__name__}] run_eod_batch_process fatal error: {exc}")
            BATCH_STATUS.update({"error": str(exc)})
            raise
        finally:
            BATCH_STATUS.update({"running": False, "finished_at": time.time()})

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


def _summarize_returns(returns):
    if not returns:
        return None
    win_rate = sum(1 for x in returns if x > 0) / len(returns) * 100
    return {"avg_return_pct": round(sum(returns) / len(returns), 2), "win_rate_pct": round(win_rate, 1), "n": len(returns)}


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
    sample = tickers if len(tickers) <= BACKTEST_SAMPLE_SIZE else random.sample(tickers, BACKTEST_SAMPLE_SIZE)
    horizons = [30, 60, 90]
    forward_returns = {h: [] for h in horizons}
    bench_returns = {h: [] for h in horizons}
    signal_count = 0
    for ticker in sample:
        try:
            df = await download_stock(ticker, "1d")
            if df is None:
                continue
            close = normalize_series(df, "Close").dropna()
            if len(close) < 300:
                continue
            rsi = calculate_rsi(close)
            _, _, macd_hist = calculate_macd(close)
            scores = calculate_alpha_score_series(close, rsi, macd_hist)
            passed = (scores >= QUANT_PASS_THRESHOLD).to_numpy()
            n = len(close)
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
                for h in horizons:
                    if i + h < n and entry:
                        forward_returns[h].append(float(close.iloc[i + h] / entry - 1) * 100)
        except Exception as exc:
            print(f"[Error: {type(exc).__name__}] Backtest ticker error ({ticker}): {exc}")
            continue
    try:
        spx = await download_stock("^GSPC", "1d")
        spx_close = normalize_series(spx, "Close").dropna()
        n = len(spx_close)
        for i in range(0, n):
            entry = float(spx_close.iloc[i])
            for h in horizons:
                if i + h < n and entry:
                    bench_returns[h].append(float(spx_close.iloc[i + h] / entry - 1) * 100)
    except Exception as exc:
        print(f"[Error: {type(exc).__name__}] Backtest benchmark error: {exc}")

    results = {
        "signal_count": signal_count,
        "tickers_sampled": len(sample),
        "horizons": {
            str(h): {"strategy": _summarize_returns(forward_returns[h]), "benchmark": _summarize_returns(bench_returns[h])}
            for h in horizons
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
    load_backtest_cache()
    while True:
        needs_refresh = (not BACKTEST_CACHE.get("computed_at")
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


async def market_scan_scheduler():
    while True:
        await asyncio.sleep(MARKET_SCAN_INTERVAL_SECONDS)
        if BATCH_LOCK.locked():
            print("[scheduler] Skipping scheduled market scan — a scan is already running.", flush=True)
            continue
        try:
            print("[scheduler] Starting scheduled 30-minute market scan.", flush=True)
            await run_eod_batch_process()
        except Exception as exc:
            print(f"[Error: {type(exc).__name__}] Scheduled market scan failed: {exc}", flush=True)


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
            removed_hist = _prune_timed_cache(CACHE["historical"], HISTORICAL_TTL * 1.5)
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


@app.on_event("startup")
async def startup():
    init_db()
    load_universe_cache()
    asyncio.create_task(refresh_universe())
    asyncio.create_task(scheduler())
    asyncio.create_task(market_scan_scheduler())
    asyncio.create_task(index_warm_scheduler())
    asyncio.create_task(cache_prune_scheduler())
    # run_backtest() now shares BATCH_LOCK with the regular scan (see run_backtest),
    # so this can no longer run concurrently with it and double up on memory — that
    # was the actual cause of tonight's OOM restarts, not the scheduler itself.
    asyncio.create_task(backtest_scheduler())
    asyncio.create_task(asyncio.to_thread(check_email_config))
    asyncio.get_running_loop().call_later(3, start_server_warmup)


def _send_via_brevo(to_email, subject, body, max_retries=3):
    payload = {
        "sender": {"email": SENDER_EMAIL},
        "to": [{"email": to_email}],
        "subject": subject,
        "textContent": body,
    }
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


def _send_via_sendgrid(to_email, subject, body, max_retries=3):
    payload = {
        "personalizations": [{"to": [{"email": to_email}]}],
        "from": {"email": SENDER_EMAIL},
        "subject": subject,
        "content": [{"type": "text/plain", "value": body}],
    }
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


def _send_via_smtp(to_email, subject, body, max_retries=3):
    msg = MIMEMultipart(); msg["From"] = SENDER_EMAIL; msg["To"] = to_email; msg["Subject"] = subject
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


def send_email_notification(to_email, subject, body, max_retries=3):
    if BREVO_API_KEY and SENDER_EMAIL:
        return _send_via_brevo(to_email, subject, body, max_retries)
    if SENDGRID_API_KEY and SENDER_EMAIL:
        return _send_via_sendgrid(to_email, subject, body, max_retries)
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print(f"[Error: EmailNotConfigured] No email backend configured — could not send '{subject}' to {to_email}")
        return False
    return _send_via_smtp(to_email, subject, body, max_retries)


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
        SELECT ticker,price,change_pct,rsi,macd,pct_from_52w_high,pct_from_52w_low,above_200d_sma,volume_ratio
        FROM daily_scans
        WHERE scan_date=? AND quant_pass=1
          AND (ai_report IS NULL OR ai_mode<>? OR ai_prompt_version IS NULL OR ai_prompt_version<>?)
        ORDER BY alpha_score DESC
    """,(today_str(),mode,AI_PROMPT_VERSION)).fetchall()
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
                                        timing_score=?,timing_verdict=?,ai_prompt_version=?
                                    WHERE scan_date=? AND ticker=?""",
                                 (ai_result["report_json"],mode,time.time(),
                                  ai_result["timing_score"],ai_result["timing_verdict"],AI_PROMPT_VERSION,
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

# -----------------------------------------------------------------------------
# API endpoints
# -----------------------------------------------------------------------------
@app.get("/api/scan")
async def api_scan(request: Request):
    user = get_logged_in_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not has_active_access(user):
        return JSONResponse({"error": "Your free trial has ended. Subscribe to keep scanning."}, status_code=402)
    conn = db()
    scanned_count = conn.execute(
        "SELECT COUNT(*) FROM daily_scans WHERE scan_date=?", (today_str(),)
    ).fetchone()[0]
    last_updated = conn.execute(
        "SELECT MAX(created_at) FROM daily_scans WHERE scan_date=?", (today_str(),)
    ).fetchone()[0]
    rows = conn.execute("""
        SELECT ticker,universe,price,change_pct,alpha_score,quant_pass,timing_score,timing_verdict,
               ROUND((alpha_score+timing_score)/2.0,1) AS overall_score
        FROM daily_scans
        WHERE scan_date=? AND quant_pass=1 AND timing_score IS NOT NULL
          AND (alpha_score+timing_score)/2.0 >= ?
        ORDER BY overall_score DESC
    """, (today_str(), OVERALL_SCORE_THRESHOLD)).fetchall()
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
        signals.append(d)
    return {"scanned_count": scanned_count, "universe_count": len(UNIVERSE),
            "quant_pass_count": len(rows), "last_updated": last_updated,
            "universe_status": UNIVERSE_STATUS, "cache": dict(CACHE_STATUS),
            "signals": signals}


@app.get("/api/heatmap")
async def api_heatmap(request: Request):
    if not get_logged_in_user(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    conn = db()
    rows = conn.execute("""
        SELECT ticker,universe,change_pct,alpha_score FROM daily_scans
        WHERE scan_date=? ORDER BY ticker
    """, (today_str(),)).fetchall()
    conn.close()
    return {"tiles": [dict(r) for r in rows]}


@app.get("/api/score-history")
async def api_score_history(request: Request, ticker: str = "AAPL"):
    if not get_logged_in_user(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    ticker = normalize_ticker(ticker)
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
    if not has_active_access(user):
        return JSONResponse({"error": "Your free trial has ended. Subscribe to keep using QUANTIFY."}, status_code=402)
    conn = db()
    rows = conn.execute("SELECT change_pct,timing_verdict,quant_pass FROM daily_scans WHERE scan_date=?", (today_str(),)).fetchall()
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

    def day_change(df):
        try:
            close = normalize_series(df, "Close").dropna()
            return round((float(close.iloc[-1]) / float(close.iloc[-2]) - 1) * 100, 2)
        except Exception:
            return None

    sp500_df, ndx_df = await asyncio.gather(download_stock("^GSPC", "1d"), download_stock("^NDX", "1d"))
    return {"universe_count": total, "advancers": advancers, "decliners": decliners,
            "avg_change_pct": avg_change, "detected_count": detected, "verdict_breakdown": verdict_counts,
            "sp500_change_pct": day_change(sp500_df) if sp500_df is not None else None,
            "nasdaq_change_pct": day_change(ndx_df) if ndx_df is not None else None}


@app.get("/api/backtest-summary")
async def backtest_summary(request: Request):
    user = get_logged_in_user(request)
    if not user: return JSONResponse({"error": "Unauthorized"}, status_code=401)
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
    if not has_active_access(user):
        return JSONResponse({"error": "Your free trial has ended. Subscribe to keep using QUANTIFY."}, status_code=402)
    ticker = normalize_ticker(ticker)
    if not re.fullmatch(r"[A-Z0-9.\-^=]{1,15}", ticker):
        return JSONResponse({"error": "Invalid ticker"}, status_code=400)
    if timeframe not in ("1h", "1d", "1wk", "1mo"):
        return JSONResponse({"error": "Invalid timeframe"}, status_code=400)
    df = await download_stock(ticker, timeframe)
    if df is None or df.empty:
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
        earnings = await get_earnings(ticker)
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
        return {"fast": {"ticker": ticker, "data_ok": True, "price": round(price, 2), "change": round(change, 2),
            "pct_from_52w_high": pct_from_high, "pct_from_52w_low": pct_from_low, "above_200d_sma": above_trend,
            "rsi": round(float(rsi.iloc[-1]), 2), "macd": round(float(macd_hist.iloc[-1]), 4),
            "earnings": earnings, "chart": chart}}
    except Exception as e:
        print(f"[Error: {type(e).__name__}] terminal_data_fast processing error ({ticker}): {e}")
        return {"fast": {"ticker": ticker, "data_ok": False, "error": f"Data processing error: {type(e).__name__}"}}


@app.get("/api/terminal-data-ai")
async def terminal_data_ai(request: Request, ticker: str = "AAPL", mode: str = "Long-Term Momentum Pullback", language: str = "en"):
    user = get_logged_in_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not has_active_access(user):
        return JSONResponse({"error": "Your free trial has ended. Subscribe to keep using QUANTIFY."}, status_code=402)
    ticker=normalize_ticker(ticker)
    if not re.fullmatch(r"[A-Z0-9.\-^=]{1,15}",ticker):
        return JSONResponse({"error":"Invalid ticker"},status_code=400)
    if language not in LANGUAGE_NAMES:
        language = "en"

    conn=db()
    row=conn.execute("""SELECT ai_report,short_percent,ai_status,ai_mode,ai_language,ai_updated_at,ai_error,timing_score,timing_verdict,
                                price,change_pct,rsi,macd,pct_from_52w_high,pct_from_52w_low,above_200d_sma,volume_ratio,ai_prompt_version,alpha_score,quant_pass
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
                                          price,change_pct,rsi,macd,pct_from_52w_high,pct_from_52w_low,above_200d_sma,volume_ratio,ai_prompt_version,alpha_score,quant_pass
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
    }}


@app.post("/api/alerts/set")
async def set_alert(request: Request, ticker: str = Form(...), target_price: float = Form(...)):
    user = get_logged_in_user(request)
    if not user: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    ticker = normalize_ticker(ticker)
    if not re.fullmatch(r"[A-Z0-9.\-^=]{1,15}", ticker) or target_price <= 0:
        return JSONResponse({"error": "Invalid alert"}, status_code=400)
    try:
        conn = db(); conn.execute("INSERT INTO user_alerts(email,ticker,target_price,created_at) VALUES(?,?,?,?)", (user,ticker,target_price,time.time())); conn.commit(); conn.close()
    except Exception as e:
        print(f"[Error: {type(e).__name__}] Alert setup error: {e}")
        return JSONResponse({"error": f"Database error: {type(e).__name__}"}, status_code=500)
    return {"message": f"Alert set: {ticker} at ${target_price:.2f}."}


@app.get("/api/alerts/list")
async def alerts_list(request: Request):
    user = get_logged_in_user(request)
    if not user: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    conn = db()
    rows = conn.execute("SELECT id,ticker,target_price,is_sent,created_at FROM user_alerts WHERE email=? ORDER BY created_at DESC", (user,)).fetchall()
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
async def portfolio_save(request: Request, ticker: str = Form(...), note: str = Form(""), shares: str = Form("")):
    user = get_logged_in_user(request)
    if not user: return JSONResponse({"error": "Unauthorized"}, status_code=401)
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
    try:
        conn = db()
        row = conn.execute("""
            SELECT scan_date,price,change_pct,alpha_score,rsi,macd,timing_score,timing_verdict,ai_report
            FROM daily_scans WHERE ticker=? ORDER BY id DESC LIMIT 1
        """, (ticker,)).fetchone()
        if not row:
            conn.close()
            return JSONResponse({"error": "No scan data for this ticker yet. Run a scan first."}, status_code=400)
        conn.execute("""
            INSERT INTO portfolio_items
            (email,ticker,scan_date,price,change_pct,alpha_score,rsi,macd,timing_score,timing_verdict,ai_report,note,saved_at,shares)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (user, ticker, row["scan_date"], row["price"], row["change_pct"], row["alpha_score"],
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
    conn = db()
    rows = conn.execute("SELECT id,ticker,added_at FROM watchlist_items WHERE email=? ORDER BY added_at DESC", (user,)).fetchall()
    conn.close()
    tickers = [r["ticker"] for r in rows]
    prices = await asyncio.gather(*(get_current_price(t) for t in tickers))
    price_by_ticker = dict(zip(tickers, prices))
    items = []
    for r in rows:
        d = dict(r)
        d["price"] = price_by_ticker.get(d["ticker"])
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
        "SELECT email,pref_theme,pref_language,pref_default_sort,pref_default_view,created_at,trial_ends_at,subscription_status FROM users WHERE email=?",
        (user,)
    ).fetchone()
    conn.close()
    if not row:
        return JSONResponse({"error": "User not found"}, status_code=404)
    return dict(row)


@app.post("/api/account/delete")
async def delete_account(request: Request, password: str = Form(...)):
    user = get_logged_in_user(request)
    if not user: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    conn = db()
    row = conn.execute("SELECT password_hash,salt FROM users WHERE email=?", (user,)).fetchone()
    if not row or not row["password_hash"] or not verify_password(password, row["password_hash"], row["salt"]):
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
                           default_sort: str = Form("overall_score"), default_view: str = Form("list")):
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
    conn = db()
    conn.execute("UPDATE users SET pref_theme=?,pref_language=?,pref_default_sort=?,pref_default_view=? WHERE email=?",
                 (theme, language, default_sort, default_view, user))
    conn.commit(); conn.close()
    return {"message": "Settings saved."}


@app.post("/api/settings/password")
async def change_password(request: Request, current_password: str = Form(...), new_password: str = Form(...)):
    user = get_logged_in_user(request)
    if not user: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    conn = db()
    row = conn.execute("SELECT password_hash,salt FROM users WHERE email=?", (user,)).fetchone()
    if not row or not verify_password(current_password, row["password_hash"], row["salt"]):
        conn.close()
        return JSONResponse({"error": "Current password is incorrect."}, status_code=400)
    ok, error = validate_password_policy(new_password)
    if not ok:
        conn.close()
        return JSONResponse({"error": error}, status_code=400)
    password_hash, salt = make_password_hash(new_password)
    conn.execute("UPDATE users SET password_hash=?,salt=? WHERE email=?", (password_hash, salt, user))
    conn.commit(); conn.close()
    return {"message": "Password changed."}

# -----------------------------------------------------------------------------
# Landing page
# -----------------------------------------------------------------------------
LANDING_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QUANTIFY. — Quant-Detected Stocks, AI Risk-Checked</title>
<meta name="description" content="A daily quant scan of the S&P 500 and Nasdaq-100, cross-checked by AI for blow-off-top and dead-cat-bounce risk. Informational only — never a buy or sell signal.">
<meta property="og:title" content="QUANTIFY. — Quant-Detected Stocks, AI Risk-Checked">
<meta property="og:description" content="A daily quant scan of the S&P 500 and Nasdaq-100, cross-checked by AI for blow-off-top and dead-cat-bounce risk. Informational only — never a buy or sell signal.">
<meta property="og:type" content="website">
<meta name="twitter:card" content="summary">
<style>
:root{--bg:#050807;--panel:#030504;--panel2:#081310;--border:#14221b;--text:#9ab8af;--head:#dff5ed;--dim:#436659;--dim2:#567d6e;--green:#2ecc71;--red:#e74c3c;--orange:#f39c12;--blue:#3498db;--teal:#22d3c4}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font:15px/1.6 'Courier New',monospace;-webkit-font-smoothing:antialiased}
a{color:var(--blue);text-decoration:none}
.wrap{max-width:1080px;margin:0 auto;padding:0 24px}
header{border-bottom:1px solid var(--border);position:sticky;top:0;background:#050807;z-index:10}
.nav{display:flex;align-items:center;justify-content:space-between;padding:18px 24px;max-width:1080px;margin:0 auto}
.logo{color:var(--head);font-weight:bold;font-size:16px;letter-spacing:1px}
.logo span{color:var(--green)}
.navlinks{display:flex;align-items:center;gap:28px;font-size:13px}
.navlinks a.muted{color:var(--text)}
.btn{display:inline-block;padding:9px 18px;border:1px solid var(--green);color:var(--green);background:#0e241b;font:13px 'Courier New',monospace;font-weight:bold;cursor:pointer;letter-spacing:.3px}
.btn:hover{background:#123326}
.btn-ghost{border-color:var(--border);color:var(--text);background:transparent}
.btn-ghost:hover{border-color:var(--dim2)}
.hero{padding:90px 24px 70px;text-align:center;background:radial-gradient(ellipse 900px 500px at 50% -10%,rgba(46,204,113,.10),transparent 65%)}
section:nth-of-type(even){background:var(--panel2)}
.eyebrow{display:inline-block;font-size:12px;color:var(--orange);border:1px solid #3a2a08;background:#14100a;padding:5px 12px;letter-spacing:.5px;margin-bottom:22px}
h1{color:var(--head);font-size:42px;line-height:1.25;letter-spacing:-.5px;max-width:820px;margin:0 auto 20px}
h1 .hl{color:var(--green)}
.sub{color:var(--dim2);font-size:16px;max-width:600px;margin:0 auto 34px}
.cta-row{display:flex;gap:14px;justify-content:center;flex-wrap:wrap;margin-bottom:14px}
.cta-note{font-size:12px;color:var(--dim)}
.mock{max-width:900px;margin:64px auto 0;border:1px solid var(--border);background:var(--panel);text-align:left;box-shadow:0 40px 80px -30px rgba(0,0,0,.6)}
.mock-bar{display:flex;gap:6px;padding:10px 14px;border-bottom:1px solid var(--border)}
.mock-dot{width:9px;height:9px;border-radius:50%;background:#1a2e25}
.mock-grid{display:grid;grid-template-columns:1.1fr 1.6fr 1.3fr;gap:1px;background:var(--border)}
.mock-col{background:var(--panel);padding:14px}
.mock-h{font-size:10px;color:var(--dim);letter-spacing:.5px;margin-bottom:10px;border-bottom:1px solid var(--border);padding-bottom:6px}
.mock-row{display:flex;justify-content:space-between;padding:6px 0;font-size:12px;border-bottom:1px solid #0c1712}
.mock-row b{color:var(--head)}
.badge{padding:2px 7px;border-radius:2px;font-weight:bold;font-size:10px;display:inline-block}
.badge-ok{background:#0e241b;color:var(--green);border:1px solid var(--green)}
.badge-warn{background:#2a2008;color:var(--orange);border:1px solid var(--orange)}
.mock-chartline{height:70px;border:1px solid var(--border);background:linear-gradient(180deg,transparent,rgba(46,204,113,.06));position:relative;margin:10px 0}
.mock-chartline svg{width:100%;height:100%;display:block}
.mock-ai{font-size:11px;color:var(--text);line-height:1.7}
.mock-ai b{color:var(--head);display:block;font-size:10px;letter-spacing:.5px;margin:10px 0 3px}
section{padding:70px 24px;border-top:1px solid var(--border)}
.section-head{text-align:center;max-width:640px;margin:0 auto 46px}
.section-head .kicker{font-size:12px;color:var(--green);letter-spacing:1px;margin-bottom:10px}
.section-head h2{color:var(--head);font-size:28px;margin-bottom:12px}
.section-head p{color:var(--dim2);font-size:15px}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:24px;max-width:1000px;margin:0 auto}
.step{background:var(--panel);border:1px solid var(--border);border-left:2px solid var(--green);padding:26px}
.step:nth-child(2){border-left-color:var(--blue)}
.step:nth-child(3){border-left-color:var(--teal)}
.step .num{color:var(--green);font-size:12px;font-weight:bold;letter-spacing:1px;margin-bottom:14px}
.step:nth-child(2) .num{color:var(--blue)}
.step:nth-child(3) .num{color:var(--teal)}
.step h3{color:var(--head);font-size:16px;margin-bottom:10px}
.step p{color:var(--dim2);font-size:13.5px;line-height:1.7}
.features{display:grid;grid-template-columns:repeat(2,1fr);gap:1px;background:var(--border);max-width:760px;margin:0 auto;border:1px solid var(--border)}
.feature{background:var(--panel);padding:26px}
.feature .icon{color:var(--green);font-size:20px;margin-bottom:12px}
.feature:nth-child(4n+2) .icon{color:var(--blue)}
.feature:nth-child(4n+3) .icon{color:var(--teal)}
.proof-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:24px;max-width:900px;margin:0 auto 20px}
.proof-card{background:var(--panel);border:1px solid var(--border);padding:26px;text-align:center}
.proof-card .horizon{color:var(--dim);font-size:11px;letter-spacing:1px;text-transform:uppercase;margin-bottom:10px}
.proof-card .num{color:var(--green);font-size:34px;font-weight:800;margin-bottom:6px}
.proof-card .compare{color:var(--dim2);font-size:13px}
.proof-note{max-width:640px;margin:0 auto;text-align:center;color:var(--dim);font-size:12.5px;line-height:1.7}
.feature:nth-child(4n+4) .icon{color:var(--orange)}
.feature h4{color:var(--head);font-size:14.5px;margin-bottom:8px}
.feature p{color:var(--dim2);font-size:13px;line-height:1.65}
.final-wrap{text-align:center;max-width:820px;margin:0 auto 40px}
.final-wrap h2{color:var(--head);font-size:30px;margin-bottom:14px}
.final-wrap p{color:var(--dim2);margin-bottom:30px}
.disclaimer{max-width:820px;margin:0 auto;background:#0a0705;border:1px solid #2a2008;padding:22px 26px}
.disclaimer .kicker{color:var(--orange);font-size:11px;letter-spacing:1px;margin-bottom:10px;font-weight:bold}
.disclaimer p{color:var(--dim2);font-size:12.5px;line-height:1.7}
footer{border-top:1px solid var(--border);padding:34px 24px;text-align:center;color:var(--dim);font-size:12px}
footer a{color:var(--dim2)}
.btn{white-space:nowrap}
@media(max-width:820px){
  h1{font-size:30px}
  .steps,.features,.proof-grid{grid-template-columns:1fr}
  .mock-grid{grid-template-columns:1fr}
  .navlinks{gap:14px;font-size:12px}
}
@media(max-width:560px){
  .nav{padding:14px 16px}
  .logo{font-size:14px}
  .navlinks{gap:10px}
  .navlinks a.muted{display:none}
  .btn{padding:8px 12px;font-size:12px}
  .hero{padding:60px 16px 50px}
  h1{font-size:24px}
  .sub{font-size:14px}
  section{padding:50px 16px}
}
</style></head><body>
<header><div class="nav">
<a class="logo" href="/">QUANTIFY<span>.</span></a>
<div class="navlinks">
<a class="muted" href="#how">How it works</a>
<a class="muted" href="#proof">Results</a>
<a class="muted" href="#features">Features</a>
<a class="muted" href="/login">Log in</a>
<a class="btn" href="/signup">Get Started Free</a>
</div>
</div></header>

<section class="hero" style="border-top:none">
<div class="eyebrow">7-DAY FREE TRIAL · THEN $9.99/MONTH</div>
<h1>Stop scanning <span class="hl">518</span> stocks by hand.<br>See the ones that actually <span class="hl">cleared the bar</span>.</h1>
<p class="sub">A daily quant scan of the S&amp;P 500 and Nasdaq-100, double-checked by AI for blow-off-top and dead-cat-bounce risk before it reaches your screen.</p>
<div class="cta-row">
<a class="btn" href="/signup">Get Started Free</a>
<a class="btn btn-ghost" href="#how">See how it works</a>
</div>
<div class="cta-note">Same data for every subscriber — never personalized picks. Cancel anytime during the trial.</div>

<div class="mock">
<div class="mock-bar"><div class="mock-dot"></div><div class="mock-dot"></div><div class="mock-dot"></div></div>
<div class="mock-grid">
<div class="mock-col">
<div class="mock-h">MARKET SCANNER · 13 detected / 518</div>
<div class="mock-row"><b>MRK</b><span>151.12 <span class="badge badge-ok">Favorable</span></span></div>
<div class="mock-row"><b>HOOD</b><span>110.71 <span class="badge badge-ok">Favorable</span></span></div>
<div class="mock-row"><b>DASH</b><span>232.00 <span class="badge badge-warn">Caution</span></span></div>
<div class="mock-row"><b>ADSK</b><span>270.54 <span class="badge badge-warn">Caution</span></span></div>
</div>
<div class="mock-col">
<div class="mock-h">MRK · TECHNICAL CHART</div>
<div class="mock-chartline"><svg viewBox="0 0 300 70" preserveAspectRatio="none"><polyline points="0,50 20,45 40,48 60,40 80,42 100,30 120,34 140,22 160,26 180,18 200,20 220,14 240,17 260,10 280,13 300,8" fill="none" stroke="#2ecc71" stroke-width="1.5"/></svg></div>
<div class="mock-row"><span>RSI / MACD</span><b>64.58 / 1.4261</b></div>
<div class="mock-row"><span>52W HIGH / LOW</span><b>-4.42% / 97.2%</b></div>
</div>
<div class="mock-col">
<div class="mock-h">AI QUANT REPORT <span class="badge badge-ok" style="margin-left:4px">Favorable</span></div>
<div class="mock-ai">
<b>QUANT REVIEW</b>
RSI near 65, positive MACD, price 4.4% below its 52-week high.
<b>RISK REVIEW</b>
Not extended near the high, well above its 52-week low — low blow-off-top and dead-cat-bounce risk.
</div>
</div>
</div>
</div>
</section>

<section id="how">
<div class="section-head">
<div class="kicker">HOW IT WORKS</div>
<h2>Two filters. Not one.</h2>
<p>Most screeners stop at the math. We add a second pass that specifically hunts for the ways a pure quant signal can fool you.</p>
</div>
<div class="steps">
<div class="step"><div class="num">STEP 1</div><h3>Quant scan, every day</h3><p>Real price data across all 518 S&amp;P 500 + Nasdaq-100 tickers is pulled and scored on momentum, RSI, and MACD. Only the top-scoring names — usually a dozen or two — clear the bar.</p></div>
<div class="step"><div class="num">STEP 2</div><h3>AI risk cross-check</h3><p>Every ticker that clears the quant bar gets reviewed a second time by AI, specifically for two traps: chasing a stock already near a blow-off top, or mistaking a dead-cat bounce for a real recovery.</p></div>
<div class="step"><div class="num">STEP 3</div><h3>You decide</h3><p>You get the data, the reasoning, and a plain-language risk review — never a price target, never a "buy now." What you do with it is up to you.</p></div>
</div>
</section>

<section id="proof">
<div class="section-head">
<div class="kicker">PROVEN BY THE NUMBERS</div>
<h2>We tested it against 2 years of real data. Here's what happened.</h2>
<p>Not backtested on cherry-picked winners — every ticker that historically cleared today's quant bar, across the full S&amp;P 500 + Nasdaq-100, over the trailing 2 years.</p>
</div>
%%PROOF_CARDS%%
%%PROOF_NOTE%%
</section>

<section id="features">
<div class="section-head">
<div class="kicker">FEATURES</div>
<h2>Built on real data, not vibes</h2>
</div>
<div class="features">
<div class="feature"><div class="icon">◆</div><h4>Live market data</h4><p>Real prices and volume from the actual market — no simulated or backfilled data.</p></div>
<div class="feature"><div class="icon">◆</div><h4>Plain-language AI review</h4><p>Every detected ticker gets a written quant review and explicit risk check, in plain English.</p></div>
<div class="feature"><div class="icon">◆</div><h4>52-week &amp; trend context</h4><p>Distance from the 52-week high/low and 200-day trend, so you see where a stock actually sits.</p></div>
<div class="feature"><div class="icon">◆</div><h4>Price alerts &amp; news</h4><p>Set a target price and get emailed when it's hit, with live headlines next to the chart.</p></div>
</div>
</section>

<section>
<div class="final-wrap">
<h2>See today's detected tickers.</h2>
<p>7-day free trial, then $9.99/month. Takes under a minute to sign up.</p>
<a class="btn" href="/signup">Get Started Free</a>
</div>
<div class="disclaimer">
<div class="kicker">IMPORTANT</div>
<p>QUANTIFY is an informational and educational tool. Nothing on this site is investment advice, a recommendation, or a solicitation to buy or sell any security. Scores, badges, and AI commentary reflect a mathematical model's output on the data available at the time and can be wrong. Markets involve risk, including loss of principal. Do your own research and consult a licensed financial advisor before making investment decisions.</p>
</div>
</section>

<footer>
QUANTIFY. — informational and educational only, not investment advice.<br>
<a href="/login">Log in</a> · <a href="/signup">Sign up</a> · <a href="/terms">Terms</a> · <a href="/privacy">Privacy</a>
</footer>
</body></html>"""


# -----------------------------------------------------------------------------
# Auth pages/endpoints
# -----------------------------------------------------------------------------
BASE_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{background:#060a08;color:#c3d6cf;font:15px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}
.authwrap{display:flex;min-height:100vh}
.authbrand{flex:1;background:radial-gradient(ellipse 900px 600px at 30% 20%,rgba(46,204,113,.14),transparent 60%) #050807;padding:60px;display:flex;flex-direction:column;justify-content:center;border-right:1px solid #14221b}
.brand{font-weight:700;font-size:19px;letter-spacing:.2px;color:#eef7f3;text-decoration:none;display:inline-block;margin-bottom:44px}
.brand span{color:#2ecc71}
.authbrand h1{font-size:36px;line-height:1.25;color:#eef7f3;max-width:440px;margin-bottom:16px}
.authbrand>p{color:#7c9c90;font-size:15px;max-width:400px;line-height:1.65}
.points{margin-top:38px;display:flex;flex-direction:column;gap:14px}
.point{color:#9ab8af;font-size:13.5px}
.point b{color:#2ecc71}
.authform{flex:1;display:flex;align-items:center;justify-content:center;padding:40px 24px}
.card{width:100%;max-width:400px;background:#0b100e;border:1px solid #1c2b24;padding:36px 32px;border-radius:16px;box-shadow:0 30px 60px -30px rgba(0,0,0,.6)}
h2{color:#eef7f3;text-align:center;font-size:21px;font-weight:700;margin-bottom:6px}
.subtitle{text-align:center;color:#7c9c90;font-size:13.5px;margin-bottom:24px;line-height:1.5}
label{display:block;font-size:12.5px;font-weight:600;color:#9ab8af;margin:14px 0 6px}
input{width:100%;background:#0f1613;border:1.5px solid #223229;color:#eef7f3;padding:12px 14px;font-size:14.5px;border-radius:9px;transition:border-color .15s;font-family:inherit}
input:focus{outline:none;border-color:#2ecc71}
button{width:100%;margin-top:20px;background:linear-gradient(135deg,#2ecc71,#22c58f);color:#04150c;border:none;padding:13px;font-size:14.5px;font-weight:700;border-radius:9px;cursor:pointer;transition:opacity .15s,transform .1s;font-family:inherit}
button:hover{opacity:.9}
button:active{transform:scale(.98)}
.google-btn{display:flex;align-items:center;justify-content:center;gap:10px;width:100%;background:#fff;color:#1f1f1f;border:1px solid #dcdcdc;padding:12px;border-radius:9px;font-size:14.5px;font-weight:600;text-decoration:none;transition:background .15s}
.google-btn:hover{background:#f2f2f2}
.divider{display:flex;align-items:center;gap:12px;margin:22px 0;color:#5c7a6e;font-size:12.5px}
.divider::before,.divider::after{content:'';flex:1;height:1px;background:#1c2b24}
.links{display:flex;justify-content:space-between;margin-top:22px;font-size:13.5px}
.links a{color:#4db8ff;text-decoration:none}
.links a:hover{text-decoration:underline}
.error{color:#ff6b6b;font-size:13px;min-height:0;margin-bottom:6px}
.ok{color:#2ecc71;font-size:13px;margin-bottom:6px}
.hint{font-size:12.5px;color:#6b8a7e;margin-top:-8px;margin-bottom:16px}
details{margin-top:20px;font-size:13px;color:#7c9c90;text-align:center}
details summary{cursor:pointer;color:#4db8ff}
details form{margin-top:12px;text-align:left}
details input{margin-bottom:10px}
details button{margin-top:6px;padding:11px}
@media(max-width:860px){
  .authwrap{flex-direction:column}
  .authbrand{border-right:none;border-bottom:1px solid #14221b;padding:36px 24px;flex:none}
  .authbrand h1{font-size:26px}
  .points{display:none}
  .authform{flex:none;padding:36px 24px}
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


def render_auth_page(title: str, form_html: str) -> HTMLResponse:
    return HTMLResponse(f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title><style>{BASE_CSS}</style></head><body><div class="authwrap">{AUTH_BRAND_HTML}<div class="authform">{form_html}</div></div></body></html>''')


def _render_proof_section() -> tuple[str, str]:
    results = BACKTEST_CACHE.get("results")
    if not results:
        return "", '<p class="proof-note">Backtest is computing on the server — check back shortly.</p>'
    horizons = results["horizons"]
    cards_html = ['<div class="proof-grid">']
    for h in ("30", "60", "90"):
        v = horizons.get(h, {})
        strat = v.get("strategy") or {}
        bench = v.get("benchmark") or {}
        avg = strat.get("avg_return_pct")
        bench_avg = bench.get("avg_return_pct")
        if avg is None:
            continue
        sign = "+" if avg >= 0 else ""
        bench_sign = "+" if (bench_avg or 0) >= 0 else ""
        cards_html.append(
            f'<div class="proof-card"><div class="horizon">{h}-Day Forward Return</div>'
            f'<div class="num">{sign}{avg}%</div>'
            f'<div class="compare">vs {bench_sign}{bench_avg}% for the S&amp;P 500 &middot; {strat.get("win_rate_pct","-")}% win rate</div></div>'
        )
    cards_html.append("</div>")
    computed_at = BACKTEST_CACHE.get("computed_at")
    computed_str = datetime.fromtimestamp(computed_at).strftime("%B %d, %Y") if computed_at else "recently"
    note = (f'<p class="proof-note">Based on {results.get("signal_count","-")} historical signals across '
            f'{results.get("tickers_sampled","-")} sampled tickers. Last computed: {computed_str}.<br>'
            f'Past performance does not guarantee future results. This is historical, informational analysis — '
            f'not a forecast, and not investment advice.</p>')
    return "".join(cards_html), note


@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    if get_logged_in_user(request):
        return RedirectResponse("/terminal", status_code=303)
    cards, note = _render_proof_section()
    html = LANDING_HTML.replace("%%PROOF_CARDS%%", cards).replace("%%PROOF_NOTE%%", note)
    return HTMLResponse(html)


@app.get("/robots.txt", response_class=PlainTextResponse)
async def robots_txt(request: Request):
    base = str(request.base_url).rstrip("/")
    return (
        "User-agent: *\n"
        "Allow: /\n"
        "Disallow: /terminal\n"
        "Disallow: /portfolio\n"
        "Disallow: /settings\n"
        "Disallow: /accept-disclaimer\n"
        "Disallow: /api/\n\n"
        f"Sitemap: {base}/sitemap.xml\n"
    )


@app.get("/sitemap.xml")
async def sitemap_xml(request: Request):
    base = str(request.base_url).rstrip("/")
    urls = ["/", "/login", "/signup", "/terms", "/privacy"]
    items = "".join(f"<url><loc>{base}{u}</loc></url>" for u in urls)
    xml = f'<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{items}</urlset>'
    return Response(content=xml, media_type="application/xml")


LEGAL_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#b2b5be;font:15px/1.7 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}
.wrap{max-width:760px;margin:0 auto;padding:50px 24px 100px}
header{display:flex;justify-content:space-between;align-items:center;margin-bottom:40px;flex-wrap:wrap;gap:10px}
.brand{font-weight:700;font-size:17px;color:#d1d4dc;text-decoration:none}
.brand span{color:#2962ff}
a.back{color:#2962ff;text-decoration:none;font-size:13px;font-weight:600}
h1{color:#d1d4dc;font-size:28px;margin-bottom:8px}
.updated{color:#787b86;font-size:13px;margin-bottom:36px}
h2{color:#d1d4dc;font-size:17px;margin:32px 0 12px}
p,li{color:#b2b5be;font-size:14.5px;line-height:1.75;margin-bottom:12px}
ul{padding-left:20px;margin-bottom:12px}
a{color:#2962ff}
"""


def render_legal_page(title: str, updated: str, body_html: str) -> HTMLResponse:
    return HTMLResponse(f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>QUANTIFY. {title}</title><style>{LEGAL_CSS}</style></head><body><div class="wrap"><header><a class="brand" href="/">QUANTIFY<span>.</span></a><a class="back" href="/">&larr; Back to home</a></header><h1>{title}</h1><div class="updated">Last updated: {updated}</div>{body_html}</div></body></html>''')


@app.get("/terms", response_class=HTMLResponse)
async def terms_page():
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
<p>Market data, news, and other information displayed on the Service are sourced from third parties, including Yahoo Finance, and may be delayed, incomplete, or inaccurate. QUANTIFY is not affiliated with, endorsed by, or sponsored by Yahoo, Inc. We do not guarantee the accuracy, completeness, or timeliness of any data shown.</p>
<h2>6. Disclaimers and Limitation of Liability</h2>
<p>The Service is provided "as is" and "as available," without warranties of any kind, express or implied. To the fullest extent permitted by law, QUANTIFY and its operator will not be liable for any direct, indirect, incidental, or consequential damages arising from your use of, or inability to use, the Service, including any investment losses.</p>
<h2>7. Termination</h2>
<p>We may suspend or terminate your account at any time, with or without notice, for conduct that violates these Terms or that we believe is harmful to the Service or other users.</p>
<h2>8. Changes to These Terms</h2>
<p>We may update these Terms from time to time. Continued use of the Service after a change constitutes acceptance of the revised Terms. We will update the "Last updated" date above when changes are made.</p>
<h2>9. Contact</h2>
<p>Questions about these Terms can be sent to <a href="mailto:quantify.app.official@gmail.com">quantify.app.official@gmail.com</a>.</p>
"""
    return render_legal_page("Terms of Service", "August 30, 2026", body)


@app.get("/privacy", response_class=HTMLResponse)
async def privacy_page():
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
<li><b>Market and news data providers (e.g. Yahoo Finance, Google News)</b> — we query these for public market data and headlines; no personal data about you is sent to them.</li>
</ul>
<h2>4. Cookies</h2>
<p>We use a single functional session cookie to keep you logged in. We do not use advertising or cross-site tracking cookies.</p>
<h2>5. Data Retention</h2>
<p>We retain account and portfolio data for as long as your account is active. Daily market scan history is retained for a limited number of days for product features like score trends.</p>
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
    return render_legal_page("Privacy Policy", "August 30, 2026", body)


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
        token_resp = requests.post("https://oauth2.googleapis.com/token", data={
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }, timeout=10)
        token_resp.raise_for_status()
        access_token = token_resp.json()["access_token"]
        info_resp = requests.get("https://www.googleapis.com/oauth2/v3/userinfo",
                                  headers={"Authorization": f"Bearer {access_token}"}, timeout=10)
        info_resp.raise_for_status()
        info = info_resp.json()
    except Exception as exc:
        print(f"[Error: {type(exc).__name__}] Google OAuth exchange failed: {exc}")
        return RedirectResponse("/login?error=Google+sign-in+failed.+Please+try+again.", status_code=303)

    email = (info.get("email") or "").strip().lower()
    if not email or not info.get("email_verified"):
        return RedirectResponse("/login?error=Could+not+verify+your+Google+email.", status_code=303)

    conn = db()
    row = conn.execute("SELECT email FROM users WHERE email=?", (email,)).fetchone()
    if not row:
        password_hash, salt = make_password_hash(secrets.token_urlsafe(32))
        conn.execute(
            "INSERT INTO users(email,password_hash,salt,is_active,created_at,trial_ends_at) VALUES(?,?,?,1,?,?)",
            (email, password_hash, salt, time.time(), time.time() + 7 * 86400),
        )
        conn.commit()
    conn.close()

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
    return RedirectResponse("/terminal", status_code=303)


@app.get("/login", response_class=HTMLResponse)
async def login_page(error: Optional[str] = None, msg: Optional[str] = None):
    error = html_lib.escape(error) if error else ''
    msg = html_lib.escape(msg) if msg else ''
    form = f'''<div class="card"><h2>Welcome back</h2><div class="subtitle">Log in to see today's detected tickers.</div><div class="error">{error}</div><div class="ok">{msg}</div><a class="google-btn" href="/auth/google/login">{GOOGLE_ICON_SVG}Continue with Google</a><div class="divider">or</div><form action="/api/auth/login" method="post"><label>Email</label><input type="email" name="email" required><label>Password</label><input type="password" name="password" required><button>Log in</button></form><div class="links"><a href="/signup">Create an account</a><a href="/forgot-password">Forgot password?</a></div><details><summary>Didn't get a verification email?</summary><form action="/api/auth/resend-verification" method="post"><label>Email</label><input type="email" name="email" required><button>Resend verification email</button></form></details></div>'''
    return render_auth_page("QUANTIFY. Login", form)


@app.post("/api/auth/login")
async def login(email: str = Form(...), password: str = Form(...)):
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
            valid = verify_password(
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
async def signup_page(error: Optional[str] = None):
    error = html_lib.escape(error) if error else ''
    form = f'''<div class="card"><h2>Create your account</h2><div class="subtitle">7-day free trial, then $9.99/month. Cancel anytime.</div><div class="error">{error}</div><a class="google-btn" href="/auth/google/login">{GOOGLE_ICON_SVG}Continue with Google</a><div class="divider">or</div><form action="/api/auth/signup" method="post"><label>Email</label><input type="email" name="email" required><label>Password</label><input type="password" name="password" required><p class="hint">10+ characters, with at least 1 letter and 1 number</p><button>Create account</button></form><p style="text-align:center;font-size:11.5px;color:#6b8a7e;margin-top:14px">By creating an account you agree to our <a href="/terms">Terms</a> and <a href="/privacy">Privacy Policy</a>.</p><div class="links"><a href="/login">Already have an account? Log in</a></div></div>'''
    return render_auth_page("QUANTIFY. Sign Up", form)


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
    password_hash,salt=make_password_hash(password)
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    try:
        conn=db()
        conn.execute(
            "INSERT INTO users(email,password_hash,salt,is_active,verify_token_hash,verify_expires,created_at,trial_ends_at) VALUES(?,?,?,0,?,?,?,?)",
            (email,password_hash,salt,token_hash,time.time()+VERIFY_TOKEN_TTL,time.time(),time.time()+7*86400),
        )
        conn.commit()
        conn.close()
    except sqlite3.IntegrityError:
        conn.close()
        return RedirectResponse("/signup?error=Email+already+registered",status_code=303)
    except Exception as e:
        print(f"[Error: {type(e).__name__}] Signup error: {e}")
        return RedirectResponse("/signup?error=Database+error",status_code=303)

    if not send_verification_email(request, email, token):
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
        send_verification_email(request, email, token)
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
    form = f'''<div class="card"><h2>Reset your password</h2><div class="subtitle">Enter your email and we'll send you a 6-digit code.</div><div class="error">{error}</div><form action="/api/auth/send-code" method="post"><label>Email</label><input type="email" name="email" required><button>Send code</button></form><div class="links"><a href="/login">Back to log in</a></div></div>'''
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
        send_email_notification(email,"[QUANTIFY.] Password Reset Code",f"Your code: {code}\nValid for: 15 minutes")
    # Respond identically regardless of whether the account exists, to avoid leaking registration status.
    return RedirectResponse("/reset-password?email="+urllib.parse.quote(email),status_code=303)


@app.get("/reset-password", response_class=HTMLResponse)
async def reset_page(email: str, error: Optional[str] = None):
    error = html_lib.escape(error) if error else ''
    email = html_lib.escape(email)
    form = f'''<div class="card"><h2>Choose a new password</h2><div class="subtitle">Enter the code we emailed you, plus a new password. Don't see it? Check spam, or wait a minute and try again.</div><div class="error">{error}</div><form action="/api/auth/verify-and-reset" method="post"><input type="hidden" name="email" value="{email}"><label>6-digit code</label><input name="code" required maxlength="6"><label>New password</label><input type="password" name="new_password" required><p class="hint">10+ characters, with at least 1 letter and 1 number</p><button>Reset password</button></form><div class="links"><a href="/forgot-password">Resend code</a></div></div>'''
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
        password_hash,salt=make_password_hash(new_password)
        conn.execute("UPDATE users SET password_hash=?,salt=?,reset_code_hash=NULL,reset_expires=NULL WHERE email=?",(password_hash,salt,email))
        conn.execute("DELETE FROM sessions WHERE email=?",(email,))
        conn.commit(); conn.close()
    except Exception as e:
        print(f"[Error: {type(e).__name__}] Password reset processing error: {e}")
        return RedirectResponse("/reset-password?email="+urllib.parse.quote(email)+"&error=Database+error",status_code=303)
    return RedirectResponse("/login?msg=Password+reset+successfully",status_code=303)

# -----------------------------------------------------------------------------
# Dashboard
# -----------------------------------------------------------------------------
@app.get("/terminal", response_class=HTMLResponse)
async def dashboard(request: Request):
    user=get_logged_in_user(request)
    if not user: return RedirectResponse("/login",status_code=303)
    if not disclaimer_accepted(user): return RedirectResponse("/accept-disclaimer",status_code=303)
    if not has_active_access(user): return RedirectResponse("/subscription",status_code=303)
    conn=db(); prefs=conn.execute("SELECT pref_theme,pref_language,pref_default_sort,pref_default_view FROM users WHERE email=?",(user,)).fetchone(); conn.close()
    theme = prefs["pref_theme"] if prefs and prefs["pref_theme"] in ("dark","light") else "dark"
    pref_language = prefs["pref_language"] if prefs and prefs["pref_language"] in LANGUAGE_NAMES else "en"
    pref_default_sort = prefs["pref_default_sort"] if prefs and prefs["pref_default_sort"] in ("overall_score","change_pct","ticker") else "overall_score"
    pref_default_view = prefs["pref_default_view"] if prefs and prefs["pref_default_view"] in ("list","heatmap") else "list"
    avatar_letter = html_lib.escape(user[0].upper()) if user else "?"
    user=html_lib.escape(user)
    return HTMLResponse(f'''<!doctype html><html lang="en" data-theme="{theme}"><head><meta charset="utf-8"><title>QUANTIFY.</title><script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script><style>
:root{{--bg:#000000;--panel:#000000;--panel2:#0a0a0a;--border:#222222;--border2:#181818;--text:#a8a8a8;--head:#ffffff;--dim:#787878;--green:#26a69a;--red:#ef5350;--orange:#ff9800;--grid-line:#161616}}
html[data-theme="light"]{{--bg:#ffffff;--panel:#ffffff;--panel2:#f2f2f2;--border:#dedede;--border2:#e8e8e8;--text:#4a4a4a;--head:#000000;--dim:#8a8a8a;--green:#089981;--red:#e64545;--orange:#c17900;--grid-line:#e8e8e8}}
*{{box-sizing:border-box}}body{{background:var(--bg);color:var(--text);font:13px/1.4 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;margin:0;padding:8px 8px 8px 68px}}
header,.panel{{background:var(--panel);border:1px solid var(--border)}}
.sidebar{{position:fixed;left:0;top:0;bottom:0;width:56px;background:var(--panel);border-right:1px solid var(--border);display:flex;flex-direction:column;align-items:center;padding:12px 0;gap:4px;z-index:40}}
.sidebar .side-brand{{color:var(--head);font-weight:800;font-size:15px;margin-bottom:14px;text-decoration:none}}
.side-link{{width:40px;height:40px;display:flex;align-items:center;justify-content:center;border-radius:8px;color:var(--dim);text-decoration:none;font-size:9.5px;font-weight:700;letter-spacing:.3px;cursor:pointer;background:transparent;border:1px solid transparent;position:relative}}
.side-link:hover{{background:var(--panel2);color:var(--head)}}
.side-link.active{{background:var(--panel2);color:var(--head);border-color:var(--border)}}
.side-link .side-tip{{position:absolute;left:52px;top:50%;transform:translateY(-50%);background:var(--panel2);border:1px solid var(--border);color:var(--head);padding:4px 9px;border-radius:4px;font-size:11px;font-weight:600;white-space:nowrap;opacity:0;pointer-events:none;transition:opacity .1s}}
.side-link:hover .side-tip{{opacity:1}}
.side-spacer{{flex:1}}
@media(max-width:900px){{
  body{{padding-left:8px;padding-bottom:64px}}
  .sidebar{{left:0;right:0;top:auto;bottom:0;width:auto;height:56px;flex-direction:row;justify-content:space-around;border-right:none;border-top:1px solid var(--border);padding:0}}
  .sidebar .side-brand,.side-spacer{{display:none}}
  .side-link .side-tip{{display:none}}
}}
header{{padding:10px 16px;display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;gap:12px;flex-wrap:wrap;border-radius:6px}}
.brand{{font-weight:700;font-size:16px;color:var(--head);text-decoration:none;letter-spacing:.2px}}
.brand span{{color:var(--dim)}}
.headerRight{{display:flex;align-items:center;gap:8px;flex-wrap:wrap}}
.userTag{{font-size:11px;color:var(--dim);margin-right:4px}}
button,input,select{{background:var(--panel2);border:1px solid var(--border);color:var(--text);padding:7px 10px;font:12px/1 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;border-radius:4px}}
button{{color:var(--head);cursor:pointer;font-weight:600;transition:background .15s,border-color .15s}}
button:hover{{background:var(--border)}}
button.danger{{color:var(--red);border-color:var(--red)}}
button.tf-btn.active{{color:var(--bg);background:var(--head);border-color:var(--head)}}
.grid{{display:grid;grid-template-columns:310px 1fr 340px;gap:8px;height:calc(100vh - 66px)}}
.panel{{padding:12px;overflow:hidden;display:flex;flex-direction:column;border-radius:6px}}
h3{{font-size:11px;color:var(--dim);border-bottom:1px solid var(--border);padding-bottom:8px;margin:0 0 8px;text-transform:uppercase;letter-spacing:.5px;font-weight:600}}
.list,.scroll{{overflow:auto;flex:1}}
.item{{padding:9px 10px;border-bottom:1px solid var(--border2);cursor:pointer;display:flex;justify-content:space-between;border-left:3px solid transparent}}
.item:hover{{background:var(--panel2)}}
.item.sig-favorable{{border-left-color:var(--green)}}
.item.sig-caution{{border-left-color:var(--orange)}}
.item.sig-risk{{border-left-color:var(--red)}}
.chart{{flex:1;min-height:220px}}
.idx-row{{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px}}
.idx-box{{background:var(--panel2);border:1px solid var(--border);padding:6px;border-radius:4px}}
.idx-label{{font-size:9px;color:var(--dim);margin-bottom:3px;display:flex;justify-content:space-between}}
.idx-chart{{height:60px}}
.metrics{{display:grid;grid-template-columns:repeat(5,1fr);gap:6px;margin-top:8px}}
.metric{{background:var(--panel2);border:1px solid var(--border);padding:8px;text-align:center;border-radius:4px}}
.metric>div:first-child{{font-size:9px;color:var(--dim);text-transform:uppercase;letter-spacing:.3px}}
.val{{color:var(--head);font-weight:700;margin-top:4px;font-size:13px}}
.notice{{padding:10px;background:var(--panel2);border:1px solid var(--border);margin-bottom:8px;line-height:1.5;border-radius:4px}}
a{{color:var(--head);text-decoration:underline}}
.badge{{padding:3px 9px;border-radius:12px;font-weight:700;display:inline-block;font-size:11px}}
.badge-ok{{background:rgba(38,166,154,.15);color:var(--green)}}
.badge-warn{{background:rgba(255,152,0,.15);color:var(--orange)}}
.badge-danger{{background:rgba(239,83,80,.15);color:var(--red)}}
.badge-pending{{background:rgba(120,123,134,.15);color:var(--dim)}}
.section{{margin-bottom:12px}}
.section b{{color:var(--head);display:block;margin-bottom:4px;font-size:11px;text-transform:uppercase;letter-spacing:.3px}}
.tabs{{display:flex;gap:4px;margin-bottom:8px}}
.tab{{flex:1;padding:7px;font-size:11px;background:var(--panel2);color:var(--dim);border:1px solid var(--border);cursor:pointer;border-radius:4px;font-weight:600}}
.tab.active{{color:var(--bg);background:var(--head);border-color:var(--head)}}
.sortbar{{display:flex;gap:4px;margin-bottom:8px}}.sortbar select{{flex:1;font-size:11px}}
.heatmap{{overflow:auto;flex:1;display:flex;flex-direction:column;gap:4px}}
.heat-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(46px,1fr));gap:2px}}
.heat-tile{{padding:6px 2px;text-align:center;font-size:9px;font-weight:700;cursor:pointer;color:#04150c;border-radius:3px}}
.group-header,.heat-group-header{{font-size:10px;color:var(--dim);background:var(--panel2);padding:5px 8px;letter-spacing:.5px;position:sticky;top:0;z-index:1;text-transform:uppercase;font-weight:600}}
.legend{{display:flex;gap:12px;flex-wrap:wrap;font-size:10px;color:var(--dim);margin-top:6px}}
.legend span{{display:inline-flex;align-items:center;gap:4px}}
.legend i{{width:10px;height:2px;display:inline-block}}
.legend i.dash{{border-top:1px dashed currentColor;background:transparent;height:0;color:#9b6bff}}
.earnings-info{{font-size:11px;color:var(--dim);margin-top:8px;padding-top:8px;border-top:1px solid var(--border)}}
.earnings-info b{{color:var(--head)}}
.earnings-info .beat{{color:var(--green)}}
.earnings-info .miss{{color:var(--red)}}
.toast{{position:fixed;bottom:20px;right:20px;background:var(--panel);border:1px solid var(--border);color:var(--head);padding:12px 18px;border-radius:6px;box-shadow:0 10px 30px rgba(0,0,0,.5);font-size:12px;z-index:100;opacity:0;transform:translateY(10px);transition:opacity .2s,transform .2s;pointer-events:none;max-width:280px}}
.toast.show{{opacity:1;transform:translateY(0)}}
.toast.err{{border-color:var(--red);color:var(--red)}}
.avatar-wrap{{position:relative}}
.avatar{{width:32px;height:32px;border-radius:50%;background:var(--panel2);border:1px solid var(--border);color:var(--head);font-weight:700;cursor:pointer;display:flex;align-items:center;justify-content:center;padding:0;font-size:13px}}
.avatar-menu{{position:absolute;top:calc(100% + 8px);right:0;background:var(--panel2);border:1px solid var(--border);border-radius:8px;min-width:170px;overflow:hidden;z-index:50;box-shadow:0 12px 30px rgba(0,0,0,.7)}}
.avatar-menu a{{display:block;padding:11px 14px;color:var(--head);text-decoration:none;font-size:12.5px;font-weight:600}}
.avatar-menu a:hover{{background:var(--border)}}
.avatar-menu a.danger-text{{color:var(--red)}}
.avatar-menu .email-row{{padding:11px 14px;color:var(--dim);font-size:11px;border-bottom:1px solid var(--border);word-break:break-all}}
.below-grid{{display:flex;flex-direction:column;gap:8px;margin-top:8px;max-width:1400px}}
.panel.wide{{overflow:visible}}
.empty-hint{{color:var(--dim);font-size:12px;padding:8px 0}}
.summary-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}}
.summary-tile{{background:var(--panel2);border:1px solid var(--border);padding:10px;border-radius:4px;text-align:center}}
.summary-tile .label{{font-size:9px;color:var(--dim);text-transform:uppercase;letter-spacing:.3px}}
.summary-tile .value{{color:var(--head);font-weight:700;font-size:16px;margin-top:4px}}
.summary-tile .value.gain{{color:var(--green)}}
.summary-tile .value.loss{{color:var(--red)}}
.watch-add-row{{display:flex;gap:6px;margin-bottom:10px}}
.watch-add-row input{{flex:1}}
.watch-row{{display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:1px solid var(--border2);font-size:12.5px}}
.watch-row:last-child{{border-bottom:none}}
.watch-row b{{cursor:pointer}}
.watch-row b:hover{{text-decoration:underline}}
.remove-btn{{background:transparent;border:1px solid var(--red);color:var(--red);padding:4px 10px;font-size:11px;font-weight:600}}
.backtest-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}}
.backtest-card{{background:var(--panel2);border:1px solid var(--border);padding:12px;border-radius:4px}}
.backtest-card h4{{margin:0 0 10px;font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:.3px}}
.backtest-row{{display:flex;justify-content:space-between;padding:4px 0;font-size:12.5px}}
.backtest-row b.gain{{color:var(--green)}}
.backtest-row b.loss{{color:var(--red)}}
.backtest-meta{{font-size:11px;color:var(--dim);margin-top:10px}}
@media(max-width:900px){{
  body{{padding:6px;height:auto;overflow-y:auto}}
  .grid{{grid-template-columns:1fr;height:auto}}
  .panel{{max-height:none}}
  #list,#heatmap{{max-height:340px}}
  .chart{{min-height:280px}}
  .metrics{{grid-template-columns:repeat(2,1fr)}}
  .idx-row{{grid-template-columns:1fr}}
  header{{padding:10px}}
  .headerRight select{{flex:1;min-width:140px}}
}}
</style></head><body>
<nav class="sidebar">
<a class="side-brand" href="/terminal" title="QUANTIFY.">Q</a>
<a class="side-link active" href="/terminal">SCN<span class="side-tip">Scanner</span></a>
<a class="side-link" onclick="document.getElementById('marketSummarySection').scrollIntoView({{behavior:'smooth'}})">SUM<span class="side-tip">Market Summary</span></a>
<a class="side-link" onclick="document.getElementById('watchlistSection').scrollIntoView({{behavior:'smooth'}})">WL<span class="side-tip">Watchlist</span></a>
<a class="side-link" onclick="document.getElementById('backtestSection').scrollIntoView({{behavior:'smooth'}})">BT<span class="side-tip">Backtest</span></a>
<a class="side-link" href="/portfolio">PF<span class="side-tip">Portfolio</span></a>
<div class="side-spacer"></div>
<a class="side-link" href="/settings">SET<span class="side-tip">Settings</span></a>
<a class="side-link" href="/subscription">SUB<span class="side-tip">Subscription</span></a>
<a class="side-link" href="/contact">CT<span class="side-tip">Contact Us</span></a>
<a class="side-link" href="/logout" style="color:var(--red)">OUT<span class="side-tip">Log out</span></a>
</nav>
<header><a class="brand" href="/terminal">QUANTIFY<span>.</span></a><div class="headerRight"><div class="avatar-wrap"><button class="avatar" onclick="event.stopPropagation();toggleAvatarMenu()" title="{user}">{avatar_letter}</button><div class="avatar-menu" id="avatarMenu" style="display:none"><div class="email-row">{user}</div><a href="/subscription">My Subscription</a><a href="/contact">Contact Us</a><a href="/logout" class="danger-text">Log out</a></div></div></div></header><div class="grid"><section class="panel"><h3>Market Scanner <span id="ucount"></span></h3><div class="tabs"><button class="tab active" id="tabList" onclick="showView('list')">List</button><button class="tab" id="tabHeatmap" onclick="showView('heatmap')">Heatmap</button></div><input id="tickerInput" placeholder="Jump to ticker (e.g. TSLA)" onkeydown="if(event.key==='Enter')loadTicker(this.value)"><div class="sortbar" id="sortbar"><select id="sortKey" onchange="renderList()"><option value="overall_score">Sort: Score</option><option value="change_pct">Sort: Change %</option><option value="ticker">Sort: Ticker A-Z</option></select></div><div class="list" id="list">Preparing constituent list...</div><div class="heatmap" id="heatmap" style="display:none"></div></section><section class="panel"><div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px"><h3 id="title" style="border:0;margin:0;padding:0">AAPL</h3><div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center"><input id="target" type="number" placeholder="Target price $" style="width:110px" title="Get an email when the price reaches this value"><button onclick="setAlert()" title="Email me when the price hits my target">&#128276; Set Alert</button><button onclick="savePortfolio()" title="Add this ticker to My Portfolio">&#9734; Save to Portfolio</button><button class="tf-btn" data-tf="1h" onclick="changeTF('1h')">1H</button><button class="tf-btn active" data-tf="1d" onclick="changeTF('1d')">1D</button><button class="tf-btn" data-tf="1wk" onclick="changeTF('1wk')">1W</button><button class="tf-btn" data-tf="1mo" onclick="changeTF('1mo')">1M</button></div></div><div id="chart" class="chart"></div><div class="legend"><span><i style="background:#e8e8e8"></i>SMA 20</span><span><i style="background:#ff9800"></i>SMA 50</span><span><i style="background:#ef5350"></i>SMA 200</span><span><i class="dash"></i>Bollinger Bands</span><span><i style="background:#26a69a"></i>Volume</span></div><div class="earnings-info" id="earningsInfo">Earnings: -</div><div class="idx-row"><div class="idx-box"><div class="idx-label"><span>S&amp;P 500 · 60D</span><span id="idx-sp500-val"></span></div><div id="idx-sp500" class="idx-chart"></div></div><div class="idx-box"><div class="idx-label"><span>NASDAQ-100 · 60D</span><span id="idx-ndx-val"></span></div><div id="idx-ndx" class="idx-chart"></div></div></div><div class="metrics"><div class="metric"><div>RSI / MACD</div><div id="rsi" class="val">-</div></div><div class="metric"><div>52W High</div><div id="high52" class="val">-</div></div><div class="metric"><div>52W Low</div><div id="low52" class="val">-</div></div><div class="metric"><div>Trend</div><div id="trend" class="val">-</div></div><div class="metric"><div>Score Trend (Today)</div><div id="scoretrend" class="val">-</div></div></div></section><section class="panel"><h3>AI Quant Report <small style="color:var(--dim);font-weight:normal;text-transform:none">(informational only, not investment advice)</small></h3><div id="verdict" style="display:none;margin-bottom:10px"></div><div id="ai" class="scroll">Loading AI analysis based on real data...</div><h3 style="margin-top:12px">News</h3><div id="news" class="scroll">Waiting for news...</div></section></div>
<div class="below-grid">
<section class="panel wide" id="marketSummarySection"><h3>Market Summary</h3><div id="marketSummaryBody" class="summary-grid"><div class="empty-hint">Loading...</div></div></section>
<section class="panel wide" id="watchlistSection"><h3>Watchlist</h3><div class="watch-add-row"><input id="watchInput" placeholder="Add ticker (e.g. NVDA)" onkeydown="if(event.key==='Enter')addWatch()"><button onclick="addWatch()">Add</button></div><div id="watchlistBody"><div class="empty-hint">Loading...</div></div></section>
<section class="panel wide" id="backtestSection"><h3>Strategy Performance <small style="color:var(--dim);font-weight:normal;text-transform:none">(real historical replay, not a guarantee of future results)</small></h3><div id="backtestBody"><div class="empty-hint">Loading...</div></div></section>
</div>
<div class="toast" id="toast"></div><script>
const USER_LANGUAGE='{pref_language}';
const DEFAULT_SORT='{pref_default_sort}';
const DEFAULT_VIEW='{pref_default_view}';
const STRATEGY_MODE='Long-Term Momentum Pullback';
let ticker='AAPL',tf='1d',chart,candle,volume,smaLines={{}},idxCharts={{}},bbLines={{}},currentView='list',lastSignals=[],lastUpdated=null;
function showToast(msg,isErr){{const t=document.getElementById('toast');t.textContent=msg;t.className='toast show'+(isErr?' err':'');clearTimeout(window._toastTimer);window._toastTimer=setTimeout(()=>t.classList.remove('show'),3500)}}
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
function renderList(){{if(!lastSignals.length)return;const key=document.getElementById('sortKey').value;const sorted=[...lastSignals].sort((a,b)=>key==='ticker'?a.ticker.localeCompare(b.ticker):(b[key]??-Infinity)-(a[key]??-Infinity));document.getElementById('list').innerHTML=sorted.map(s=>`<div class="item ${{itemSigClass(s.timing_verdict)}}" onclick="loadTicker('${{s.ticker}}')"><b>${{s.ticker}}</b><span style="display:flex;align-items:center;gap:6px">${{sparklineSVG(s.sparkline)}}<span style="text-align:right">${{s.price}} · ${{s.change_pct}}%<br><small>Score ${{s.overall_score}} · <span class="badge ${{verdictClass(s.timing_verdict)}}">${{s.timing_verdict||'Analyzing'}}</span></small></span></span></div>`).join('')}}
function init(){{const c=document.getElementById('chart');chart=LightweightCharts.createChart(c,{{width:c.clientWidth,height:c.clientHeight,layout:{{background:{{type:'solid',color:'#000000'}},textColor:'#a8a8a8'}},grid:{{vertLines:{{color:'#161616'}},horzLines:{{color:'#161616'}}}},timeScale:{{timeVisible:true}}}});candle=chart.addCandlestickSeries({{upColor:'#26a69a',downColor:'#ef5350',borderUpColor:'#26a69a',borderDownColor:'#ef5350',wickUpColor:'#26a69a',wickDownColor:'#ef5350'}});volume=chart.addHistogramSeries({{color:'rgba(38,166,154,.5)',priceFormat:{{type:'volume'}},priceScaleId:''}});volume.priceScale().applyOptions({{scaleMargins:{{top:.8,bottom:0}}}});smaLines.sma20=chart.addLineSeries({{color:'#e8e8e8',lineWidth:1,priceLineVisible:false,lastValueVisible:false}});smaLines.sma50=chart.addLineSeries({{color:'#ff9800',lineWidth:1,priceLineVisible:false,lastValueVisible:false}});smaLines.sma200=chart.addLineSeries({{color:'#ef5350',lineWidth:1,priceLineVisible:false,lastValueVisible:false}});bbLines.upper=chart.addLineSeries({{color:'#9b6bff',lineWidth:1,lineStyle:2,priceLineVisible:false,lastValueVisible:false}});bbLines.lower=chart.addLineSeries({{color:'#9b6bff',lineWidth:1,lineStyle:2,priceLineVisible:false,lastValueVisible:false}});window.onresize=()=>{{chart.resize(c.clientWidth,c.clientHeight);Object.entries(idxCharts).forEach(([k,ic])=>{{const el=document.getElementById('idx-'+k);if(el)ic.resize(el.clientWidth,el.clientHeight)}})}};['sp500','ndx'].forEach(k=>{{const el=document.getElementById('idx-'+k);const ic=LightweightCharts.createChart(el,{{width:el.clientWidth,height:el.clientHeight,layout:{{background:{{type:'solid',color:'#000000'}},textColor:'#a8a8a8',fontSize:9}},grid:{{vertLines:{{visible:false}},horzLines:{{visible:false}}}},rightPriceScale:{{visible:false}},timeScale:{{visible:false}},handleScroll:false,handleScale:false}});idxCharts[k]=ic;idxCharts[k+'_line']=ic.addLineSeries({{color:'#e8e8e8',lineWidth:1.5,priceLineVisible:false,lastValueVisible:false}})}})}}
async function loadIndices(){{try{{const r=await fetch('/api/market-indices');const d=await r.json();const map={{sp500:d.sp500,ndx:d.nasdaq100}};Object.entries(map).forEach(([k,series])=>{{if(!series?.length)return;const boxEl=document.getElementById('idx-'+k);if(boxEl&&boxEl.clientWidth&&boxEl.clientHeight)idxCharts[k].resize(boxEl.clientWidth,boxEl.clientHeight);idxCharts[k+'_line'].setData(series.map(p=>({{time:p.time,value:p.close}})));idxCharts[k].timeScale().fitContent();const first=series[0].close,last=series[series.length-1].close;const chg=((last/first-1)*100).toFixed(2);idxCharts[k+'_line'].applyOptions({{color:chg>=0?'#26a69a':'#ef5350'}});const valEl=document.getElementById('idx-'+k+'-val');if(valEl)valEl.innerHTML=`${{last}} <span style="color:${{chg>=0?'#26a69a':'#ef5350'}}">${{chg>=0?'+':''}}${{chg}}%</span>`}})}}catch(e){{console.warn('index load failed',e)}}}}
async function loadScoreHistory(t){{const el=document.getElementById('scoretrend');try{{const r=await fetch(`/api/score-history?ticker=${{encodeURIComponent(t)}}`);const d=await r.json();const scores=(d.points||[]).map(p=>p.alpha_score).filter(v=>v!=null);if(scores.length<2){{el.innerHTML=scores.length?scores[scores.length-1].toFixed(1):'-';return}}el.innerHTML=sparklineSVG(scores)+' '+scores[scores.length-1].toFixed(1)}}catch(e){{el.innerText='-'}}}}
async function autoScanOnOpen(){{try{{await fetch('/api/auto-scan',{{method:'POST'}});}}catch(e){{console.warn('auto-scan trigger failed',e)}};scan()}}
function updateUcount(d){{document.getElementById('ucount').innerText=d.universe_count?` · ${{d.quant_pass_count??0}} detected / ${{d.universe_count}} symbols`:''}}
async function scan(){{const r=await fetch('/api/scan');if(r.status===402){{location.href='/subscription';return}}const d=await r.json();updateUcount(d);lastUpdated=d.last_updated;lastSignals=d.signals||[];if(!lastSignals.length){{const ready=d.universe_status?.ready;const err=d.universe_status?.error;const scanned=d.scanned_count>0;document.getElementById('list').innerHTML='<div class="notice">'+(scanned?'Scan complete — no tickers cleared the quant threshold today. You can still look up any ticker above.':(ready?'The server is preparing the next scan — check back shortly.':(err?'Could not prepare constituent data. The server will retry automatically.':'Preparing S&P 500 / Nasdaq-100 constituents...')))+'</div>';loadTicker(ticker);return}}renderList();loadTicker(lastSignals[0].ticker)}}
async function pollForUpdates(){{try{{const r=await fetch('/api/scan');if(r.status===402){{location.href='/subscription';return}}const d=await r.json();updateUcount(d);if(d.last_updated&&d.last_updated!==lastUpdated){{lastUpdated=d.last_updated;lastSignals=d.signals||[];renderList();if(currentView==='heatmap')loadHeatmap();loadTicker(ticker);showToast('Updated with the latest scan.')}}}}catch(e){{}}}}
async function loadTicker(t){{ticker=t.toUpperCase().trim();document.getElementById('title').innerText=ticker;document.getElementById('ai').innerText='Loading AI analysis based on real data...';document.getElementById('news').innerText='Waiting for news...';document.getElementById('verdict').style.display='none';const fastPromise=fetch(`/api/terminal-data-fast?ticker=${{encodeURIComponent(ticker)}}&timeframe=${{tf}}`);const aiPromise=fetch(`/api/terminal-data-ai?ticker=${{encodeURIComponent(ticker)}}&mode=${{encodeURIComponent(STRATEGY_MODE)}}&language=${{USER_LANGUAGE}}`);let d;try{{const fastRes=await fastPromise;if(fastRes.status===402){{location.href='/subscription';return}}d=await fastRes.json()}}catch(e){{document.getElementById('rsi').innerText='Could not load chart data.';console.error('Chart data load failed',e);return}}if(!d.fast?.data_ok){{document.getElementById('rsi').innerText=d.fast?.error||'No data';return}}const cd=d.fast.chart.map(x=>({{time:x.time,open:x.open,high:x.high,low:x.low,close:x.close}}));const vd=d.fast.chart.map(x=>({{time:x.time,value:x.volume}}));candle.setData(cd);volume.setData(vd);['sma20','sma50','sma200'].forEach(k=>{{const pts=d.fast.chart.filter(x=>x[k]!=null).map(x=>({{time:x.time,value:x[k]}}));smaLines[k].setData(pts)}});bbLines.upper.setData(d.fast.chart.filter(x=>x.bb_upper!=null).map(x=>({{time:x.time,value:x.bb_upper}})));bbLines.lower.setData(d.fast.chart.filter(x=>x.bb_lower!=null).map(x=>({{time:x.time,value:x.bb_lower}})));const cEl=document.getElementById('chart');if(cEl.clientWidth&&cEl.clientHeight)chart.resize(cEl.clientWidth,cEl.clientHeight);chart.timeScale().fitContent();document.getElementById('rsi').innerText=`RSI ${{d.fast.rsi}} / MACD ${{d.fast.macd}}`;document.getElementById('high52').innerText=d.fast.pct_from_52w_high==null?'N/A':d.fast.pct_from_52w_high+'%';document.getElementById('low52').innerText=d.fast.pct_from_52w_low==null?'N/A':d.fast.pct_from_52w_low+'%';document.getElementById('trend').innerText=d.fast.above_200d_sma==null?'N/A':(d.fast.above_200d_sma?'Uptrend':'Downtrend');renderEarnings(d.fast.earnings);loadScoreHistory(ticker);try{{const aiRes=await aiPromise;if(aiRes.status===402){{location.href='/subscription';return}}const x=await aiRes.json();const vEl=document.getElementById('verdict');if(x.ai?.timing_verdict){{vEl.style.display='block';vEl.innerHTML=`<span class="badge ${{verdictClass(x.ai.timing_verdict)}}">${{x.ai.timing_verdict}}</span> Score ${{x.ai.overall_score??'-'}} / 100`}}else{{vEl.style.display='none'}}const sec=x.ai?.report_sections;const aiEl=document.getElementById('ai');if(sec){{const labels={{quant_review:'Quant Review',supply_demand:'Supply/Demand',risk_review:'Risk Review',news_analysis:'News Analysis',timing_reason:'Timing Rationale'}};aiEl.innerHTML=Object.keys(labels).filter(k=>sec[k]).map(k=>`<div class="section"><b>${{labels[k]}}</b>${{sec[k]}}</div>`).join('')}}else{{aiEl.innerText=x.ai?.ai_report||(!x.ai?.quant_pass?'AI analysis only runs for tickers that clear the daily quant scan — this one did not make the list today.':(x.ai?.status==='PENDING'||x.ai?.status==='RUNNING'?'Preparing AI analysis cache on the server...':'AI analysis is unavailable.'))}}const news=x.ai?.news;if(!news)document.getElementById('news').innerText='Could not fetch a live news feed.';else document.getElementById('news').innerHTML=news.map(n=>`<div style="margin-bottom:8px"><a href="${{n.url}}" target="_blank" rel="noopener">${{n.title}}</a><br><small>${{n.published||''}}</small></div>`).join('')}}catch(e){{document.getElementById('ai').innerText='Could not load AI analysis. Please try again in a moment.';document.getElementById('news').innerText='Could not fetch a live news feed.';console.error('AI data load failed',e)}}}}
async function setAlert(){{const p=Number(document.getElementById('target').value);if(!(p>0))return showToast('Enter a target price first.',true);const f=new FormData();f.append('ticker',ticker);f.append('target_price',p);const r=await fetch('/api/alerts/set',{{method:'POST',body:f}});const d=await r.json();showToast(d.message||d.error,!r.ok)}}
async function savePortfolio(){{const input=prompt('How many shares? (optional — leave blank to just track the ticker)');if(input===null)return;let shares='';if(input.trim()!==''){{const n=parseFloat(input);if(!isFinite(n)||n<=0){{showToast('Enter a positive number of shares, or leave it blank.',true);return}}shares=n}}const f=new FormData();f.append('ticker',ticker);if(shares!=='')f.append('shares',shares);const r=await fetch('/api/portfolio/save',{{method:'POST',body:f}});const d=await r.json();showToast(d.message||d.error,!r.ok)}}
function changeTF(x){{tf=x;document.querySelectorAll('.tf-btn').forEach(b=>b.classList.toggle('active',b.dataset.tf===x));loadTicker(ticker)}}
async function loadMarketSummary(){{try{{const r=await fetch('/api/market-summary');if(r.status===402)return;const d=await r.json();const el=document.getElementById('marketSummaryBody');const chg=(v)=>v==null?'-':(v>=0?'+':'')+v+'%';const cls=(v)=>v==null?'':(v>=0?'gain':'loss');el.innerHTML=`
<div class="summary-tile"><div class="label">S&amp;P 500</div><div class="value ${{cls(d.sp500_change_pct)}}">${{chg(d.sp500_change_pct)}}</div></div>
<div class="summary-tile"><div class="label">Nasdaq-100</div><div class="value ${{cls(d.nasdaq_change_pct)}}">${{chg(d.nasdaq_change_pct)}}</div></div>
<div class="summary-tile"><div class="label">Advancers</div><div class="value gain">${{d.advancers??'-'}}</div></div>
<div class="summary-tile"><div class="label">Decliners</div><div class="value loss">${{d.decliners??'-'}}</div></div>
<div class="summary-tile"><div class="label">Avg Change</div><div class="value ${{cls(d.avg_change_pct)}}">${{chg(d.avg_change_pct)}}</div></div>
<div class="summary-tile"><div class="label">Cleared Quant Bar</div><div class="value">${{d.detected_count??'-'}} / ${{d.universe_count??'-'}}</div></div>
<div class="summary-tile"><div class="label">Favorable</div><div class="value gain">${{d.verdict_breakdown?.Favorable??0}}</div></div>
<div class="summary-tile"><div class="label">Caution / Risk</div><div class="value loss">${{(d.verdict_breakdown?.Caution??0)+(d.verdict_breakdown?.Risk??0)}}</div></div>`}}catch(e){{console.error('Market summary load failed',e)}}}}
async function addWatch(){{const t=document.getElementById('watchInput').value.trim();if(!t)return;const f=new FormData();f.append('ticker',t);const r=await fetch('/api/watchlist/add',{{method:'POST',body:f}});const d=await r.json();showToast(d.message||d.error,!r.ok);if(r.ok){{document.getElementById('watchInput').value='';loadWatchlist()}}}}
async function removeWatch(id){{const f=new FormData();f.append('id',id);await fetch('/api/watchlist/remove',{{method:'POST',body:f}});loadWatchlist()}}
async function loadWatchlist(){{try{{const r=await fetch('/api/watchlist');if(r.status===402)return;const d=await r.json();const el=document.getElementById('watchlistBody');if(!d.items?.length){{el.innerHTML='<div class="empty-hint">Nothing watched yet — add any ticker above, regardless of whether it clears the quant bar.</div>';return}}el.innerHTML=d.items.map(it=>`<div class="watch-row"><b onclick="loadTicker('${{it.ticker}}');document.querySelector('.grid').scrollIntoView({{behavior:'smooth'}})">${{it.ticker}}</b><span>${{it.price!=null?'$'+it.price:'-'}}</span><button class="remove-btn" onclick="removeWatch(${{it.id}})">Remove</button></div>`).join('')}}catch(e){{console.error('Watchlist load failed',e)}}}}
async function loadBacktest(){{try{{const r=await fetch('/api/backtest-summary');if(r.status===402)return;const d=await r.json();const el=document.getElementById('backtestBody');if(!d.results){{el.innerHTML='<div class="empty-hint">Backtest is still computing on the server — check back soon.</div>';return}}const res=d.results;const fmtPct=(v)=>v==null?'-':(v>=0?'+':'')+v+'%';const cls=(v)=>v==null?'':(v>=0?'gain':'loss');const cards=Object.entries(res.horizons).map(([h,v])=>`<div class="backtest-card"><h4>${{h}}-Day Forward Return</h4>
<div class="backtest-row"><span>Strategy avg</span><b class="${{cls(v.strategy?.avg_return_pct)}}">${{fmtPct(v.strategy?.avg_return_pct)}}</b></div>
<div class="backtest-row"><span>Strategy win rate</span><b>${{v.strategy?.win_rate_pct??'-'}}%</b></div>
<div class="backtest-row"><span>S&amp;P 500 avg (same period)</span><b class="${{cls(v.benchmark?.avg_return_pct)}}">${{fmtPct(v.benchmark?.avg_return_pct)}}</b></div>
</div>`).join('');el.innerHTML=`<div class="backtest-grid">${{cards}}</div><div class="backtest-meta">Based on ${{res.signal_count}} historical signals across ${{res.tickers_sampled}} sampled tickers over the trailing 2 years. Last computed: ${{d.computed_at?new Date(d.computed_at*1000).toLocaleDateString():'-'}}. Past performance does not guarantee future results.</div>`}}catch(e){{console.error('Backtest load failed',e)}}}}
window.onload=()=>{{document.getElementById('sortKey').value=DEFAULT_SORT;if(DEFAULT_VIEW==='heatmap')showView('heatmap');init();autoScanOnOpen();loadIndices();loadMarketSummary();loadWatchlist();loadBacktest();setInterval(pollForUpdates,20000)}};
</script></body></html>''')


@app.get("/portfolio", response_class=HTMLResponse)
async def portfolio_page(request: Request):
    user = get_logged_in_user(request)
    if not user: return RedirectResponse("/login", status_code=303)
    if not disclaimer_accepted(user): return RedirectResponse("/accept-disclaimer", status_code=303)
    user = html_lib.escape(user)
    return HTMLResponse(f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>QUANTIFY. Portfolio</title><style>
:root{{--bg:#000000;--panel:#000000;--panel2:#0a0a0a;--border:#222222;--text:#a8a8a8;--head:#ffffff;--dim:#787878;--green:#26a69a;--red:#ef5350;--orange:#ff9800}}
*{{box-sizing:border-box}}body{{background:var(--bg);color:var(--text);font:13px/1.4 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;margin:0;padding:8px}}
header{{background:var(--panel);border:1px solid var(--border);padding:10px 16px;display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;border-radius:6px;flex-wrap:wrap;gap:10px}}
.brand{{font-weight:700;font-size:16px;color:var(--head);text-decoration:none}}.brand span{{color:var(--dim)}}
.headerRight{{display:flex;align-items:center;gap:14px}}
a.back{{color:var(--head);text-decoration:underline;font-size:13px;font-weight:600}}
button{{background:var(--panel2);border:1px solid var(--border);color:var(--head);padding:7px 12px;font:12px -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;cursor:pointer;border-radius:4px;font-weight:600}}
button:hover{{background:var(--border)}}
.wrap{{max-width:900px;margin:0 auto}}
.item{{background:var(--panel);border:1px solid var(--border);padding:16px;margin-bottom:10px;border-radius:6px}}
.item-head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;gap:10px}}
.item-head b{{color:var(--head);font-size:16px}}
.meta{{color:var(--dim);font-size:11px;margin-bottom:10px}}
.badge{{padding:3px 9px;border-radius:12px;font-weight:700;font-size:11px}}
.badge-ok{{background:rgba(38,166,154,.15);color:var(--green)}}
.badge-warn{{background:rgba(255,152,0,.15);color:var(--orange)}}
.badge-danger{{background:rgba(239,83,80,.15);color:var(--red)}}
.note{{color:var(--text);font-size:12.5px;line-height:1.6}}
.remove{{background:transparent;border:1px solid var(--red);color:var(--red)}}
.empty{{color:var(--dim);padding:60px 20px;text-align:center}}
.gain{{color:var(--green);font-weight:700}}
.loss{{color:var(--red);font-weight:700}}
.disclaimer-footer{{max-width:900px;margin:20px auto 0;color:var(--dim);font-size:10.5px;line-height:1.6;text-align:center;padding:12px;border-top:1px solid var(--border)}}
</style></head><body><header><a class="brand" href="/terminal">QUANTIFY<span>.</span></a><div class="headerRight"><span style="color:var(--dim);font-size:12.5px">Portfolio · {user}</span><a class="back" href="/settings">Settings</a><a class="back" href="/subscription">Subscription</a><a class="back" href="/contact">Contact</a><a class="back" href="/terminal">&larr; Back to Terminal</a></div></header>
<div class="wrap" id="list">Loading...</div>
<div class="disclaimer-footer">QUANTIFY is informational and educational only, not investment advice. Nothing here is a recommendation to buy or sell any security. All investment decisions are solely your own responsibility.</div>
<script>
function badgeClass(v){{return v==='Favorable'?'badge-ok':v==='Caution'?'badge-warn':v==='Risk'?'badge-danger':''}}
async function load(){{const r=await fetch('/api/portfolio');const d=await r.json();const el=document.getElementById('list');if(!d.items?.length){{el.innerHTML='<div class="empty">Nothing saved yet. Open a ticker in the terminal and click ☆ Save to Portfolio.</div>';return}}el.innerHTML=d.items.map(it=>{{const sec=it.ai_report&&typeof it.ai_report==='object'?it.ai_report:null;const summary=sec?.quant_review||(typeof it.ai_report==='string'?it.ai_report:'');const date=new Date(it.saved_at*1000).toLocaleString();const plClass=it.return_pct==null?'':(it.return_pct>=0?'gain':'loss');const posLine=it.shares?`<div class="meta">${{it.shares}} sh @ $${{it.price}} → $${{it.current_price??'-'}}${{it.return_pct!=null?` · <span class="${{plClass}}">${{it.return_pct>=0?'+':''}}${{it.return_pct}}%${{it.pl_dollar!=null?` (${{it.pl_dollar>=0?'+':''}}$${{it.pl_dollar}})`:''}}</span>`:''}}</div>`:(it.return_pct!=null?`<div class="meta">$${{it.price}} → $${{it.current_price}} · <span class="${{plClass}}">${{it.return_pct>=0?'+':''}}${{it.return_pct}}%</span></div>`:'');return `<div class="item"><div class="item-head"><b>${{it.ticker}}</b><div>${{it.timing_verdict?`<span class="badge ${{badgeClass(it.timing_verdict)}}">${{it.timing_verdict}}</span> `:''}}<button class="remove" onclick="remove(${{it.id}})">Remove</button></div></div>${{posLine}}<div class="meta">Saved ${{date}} · Scan date ${{it.scan_date}} · RSI ${{it.rsi}}${{it.overall_score!=null?' · Score '+it.overall_score+'/100':''}}</div>${{summary?`<div class="note">${{summary}}</div>`:''}}</div>`}}).join('')}}
async function remove(id){{const f=new FormData();f.append('id',id);await fetch('/api/portfolio/remove',{{method:'POST',body:f}});load()}}
load();
</script></body></html>''')


@app.get("/subscription", response_class=HTMLResponse)
async def subscription_page(request: Request):
    user = get_logged_in_user(request)
    if not user: return RedirectResponse("/login", status_code=303)
    conn = db()
    row = conn.execute("SELECT trial_ends_at,subscription_status FROM users WHERE email=?", (user,)).fetchone()
    conn.close()
    trial_ends_at = row["trial_ends_at"] if row else None
    sub_status = (row["subscription_status"] if row else "trial") or "trial"
    days_left = max(0, int((trial_ends_at - time.time()) / 86400) + 1) if trial_ends_at else 0
    user_esc = html_lib.escape(user)

    trial_active = bool(trial_ends_at and time.time() < trial_ends_at)
    if sub_status == "active":
        plan_html = '<span class="badge">Active Subscription</span><p>Your subscription is active. Thanks for supporting QUANTIFY.</p>'
    elif trial_active:
        plan_html = (f'<span class="badge">Free Trial &middot; {days_left} day{"s" if days_left != 1 else ""} left</span>'
                     f'<p>Every account gets a {TRIAL_DAYS}-day free trial with full access. Subscribe below anytime to keep it going after your trial ends.</p>')
    else:
        plan_html = (f'<span class="badge warn">Trial Ended</span>'
                     f'<p>Your {TRIAL_DAYS}-day free trial has ended. Subscribe below to keep using the scanner and AI reports.</p>')

    checkout_url = GUMROAD_CHECKOUT_URL or LEMONSQUEEZY_CHECKOUT_URL
    if checkout_url:
        checkout_html = f'<a href="{checkout_url}" target="_blank" rel="noopener" class="subscribe-btn">Subscribe</a>'
    else:
        checkout_html = '<div class="subscribe-btn disabled">Paid plans coming soon</div>'

    return HTMLResponse(f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>QUANTIFY. Subscription</title><style>
:root{{--bg:#000000;--panel:#000000;--panel2:#0a0a0a;--border:#222222;--text:#a8a8a8;--head:#ffffff;--dim:#787878;--green:#26a69a;--orange:#ff9800}}
*{{box-sizing:border-box}}body{{background:var(--bg);color:var(--text);font:13px/1.4 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;margin:0;padding:8px}}
header{{background:var(--panel);border:1px solid var(--border);padding:10px 16px;display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;border-radius:6px;flex-wrap:wrap;gap:10px}}
.brand{{font-weight:700;font-size:16px;color:var(--head);text-decoration:none}}.brand span{{color:var(--dim)}}
.headerRight{{display:flex;align-items:center;gap:14px}}
a.back{{color:var(--head);text-decoration:underline;font-size:13px;font-weight:600}}
.wrap{{max-width:520px;margin:0 auto}}
.card{{background:var(--panel);border:1px solid var(--border);padding:22px;margin-bottom:16px;border-radius:8px}}
.card h2{{color:var(--head);font-size:13px;margin:0 0 16px;border-bottom:1px solid var(--border);padding-bottom:10px;text-transform:uppercase;letter-spacing:.5px;font-weight:600}}
.badge{{display:inline-block;padding:4px 10px;border-radius:12px;background:rgba(38,166,154,.15);color:var(--green);font-weight:700;font-size:12px}}
.badge.warn{{background:rgba(255,152,0,.15);color:var(--orange)}}
p{{color:var(--text);font-size:13px;line-height:1.7;margin-top:14px}}
.subscribe-btn{{display:block;text-align:center;margin-top:18px;background:var(--head);color:var(--bg);padding:12px;border-radius:6px;text-decoration:none;font-weight:700;font-size:13px}}
.subscribe-btn.disabled{{background:var(--panel2);color:var(--dim);border:1px solid var(--border);cursor:default}}
</style></head><body><header><a class="brand" href="/terminal">QUANTIFY<span>.</span></a><div class="headerRight"><span style="color:var(--dim);font-size:12.5px">Subscription · {user_esc}</span><a class="back" href="/portfolio">Portfolio</a><a class="back" href="/settings">Settings</a><a class="back" href="/contact">Contact</a><a class="back" href="/terminal">&larr; Back to Terminal</a></div></header>
<div class="wrap"><div class="card">
<h2>Current Plan</h2>
{plan_html}
{checkout_html}
</div></div>
</body></html>''')


@app.get("/contact", response_class=HTMLResponse)
async def contact_page(request: Request, msg: Optional[str] = None, error: Optional[str] = None):
    user = get_logged_in_user(request)
    if not user: return RedirectResponse("/login", status_code=303)
    user = html_lib.escape(user)
    msg = html_lib.escape(msg) if msg else ''
    error = html_lib.escape(error) if error else ''
    return HTMLResponse(f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>QUANTIFY. Contact</title><style>
:root{{--bg:#000000;--panel:#000000;--panel2:#0a0a0a;--border:#222222;--text:#a8a8a8;--head:#ffffff;--dim:#787878;--green:#26a69a;--red:#ef5350}}
*{{box-sizing:border-box}}body{{background:var(--bg);color:var(--text);font:13px/1.4 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;margin:0;padding:8px}}
header{{background:var(--panel);border:1px solid var(--border);padding:10px 16px;display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;border-radius:6px;flex-wrap:wrap;gap:10px}}
.brand{{font-weight:700;font-size:16px;color:var(--head);text-decoration:none}}.brand span{{color:var(--dim)}}
.headerRight{{display:flex;align-items:center;gap:14px}}
a.back{{color:var(--head);text-decoration:underline;font-size:13px;font-weight:600}}
.wrap{{max-width:520px;margin:0 auto}}
.card{{background:var(--panel);border:1px solid var(--border);padding:22px;margin-bottom:16px;border-radius:8px}}
.card h2{{color:var(--head);font-size:13px;margin:0 0 16px;border-bottom:1px solid var(--border);padding-bottom:10px;text-transform:uppercase;letter-spacing:.5px;font-weight:600}}
label{{display:block;font-size:11.5px;font-weight:600;color:var(--dim);margin:14px 0 6px}}
textarea{{width:100%;background:var(--panel2);border:1px solid var(--border);color:var(--head);padding:10px;font:12.5px -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;border-radius:4px;resize:vertical;min-height:140px}}
button{{margin-top:16px;background:var(--head);border:1px solid var(--head);color:var(--bg);padding:10px 14px;font:12.5px -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-weight:700;cursor:pointer;width:100%;border-radius:4px}}
.msg{{font-size:12px;min-height:16px;margin-top:8px}}.ok{{color:var(--green)}}.err{{color:var(--red)}}
</style></head><body><header><a class="brand" href="/terminal">QUANTIFY<span>.</span></a><div class="headerRight"><span style="color:var(--dim);font-size:12.5px">Contact · {user}</span><a class="back" href="/portfolio">Portfolio</a><a class="back" href="/settings">Settings</a><a class="back" href="/subscription">Subscription</a><a class="back" href="/terminal">&larr; Back to Terminal</a></div></header>
<div class="wrap"><div class="card">
<h2>Contact Us</h2>
<div class="msg {'ok' if msg else 'err' if error else ''}">{msg or error}</div>
<label>Your message</label>
<textarea id="message" placeholder="Bug report, feedback, question — anything."></textarea>
<button onclick="sendContact()">Send Message</button>
</div></div>
<script>
async function sendContact(){{const message=document.getElementById('message').value.trim();if(!message)return alert('Write a message first.');const f=new FormData();f.append('message',message);const r=await fetch('/api/contact',{{method:'POST',body:f}});const d=await r.json();if(r.ok){{document.getElementById('message').value='';}}location.href='/contact?'+(r.ok?'msg=':'error=')+encodeURIComponent(d.message||d.error)}}
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
    ok = send_email_notification(SENDER_EMAIL, f"[QUANTIFY Contact] Message from {user}",
                                  f"From: {user}\n\n{message}")
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
            conn.execute(
                "UPDATE users SET subscription_status=?,ls_customer_id=?,ls_subscription_id=? WHERE email=?",
                (sub_status, str(attrs.get("customer_id", "")), str(data.get("id", "")), email),
            )
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
    # see https://gumroad.com/ping). Acceptable for now since no paywall is enforced;
    # revisit with API-verified sale lookups before this ever gates access.
    form = await request.form()
    email = (form.get("email") or "").strip().lower()
    if not email:
        return {"ok": True}
    resource_name = form.get("resource_name", "sale")
    subscription_id = str(form.get("subscription_id") or "")
    refunded = (form.get("refunded") or "").lower() == "true"

    conn = db()
    try:
        if resource_name == "refund" or refunded:
            conn.execute("UPDATE users SET subscription_status='expired' WHERE email=?", (email,))
        elif resource_name == "sale":
            conn.execute(
                "UPDATE users SET subscription_status='active',gumroad_subscription_id=? WHERE email=?",
                (subscription_id, email),
            )
        conn.commit()
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
    user = html_lib.escape(user)
    return HTMLResponse(f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>QUANTIFY. Settings</title><style>
:root{{--bg:#000000;--panel:#000000;--panel2:#0a0a0a;--border:#222222;--text:#a8a8a8;--head:#ffffff;--dim:#787878;--green:#26a69a;--red:#ef5350}}
*{{box-sizing:border-box}}body{{background:var(--bg);color:var(--text);font:13px/1.4 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;margin:0;padding:8px}}
header{{background:var(--panel);border:1px solid var(--border);padding:10px 16px;display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;border-radius:6px;flex-wrap:wrap;gap:10px}}
.brand{{font-weight:700;font-size:16px;color:var(--head);text-decoration:none}}.brand span{{color:var(--dim)}}
.headerRight{{display:flex;align-items:center;gap:14px}}
a.back{{color:var(--head);text-decoration:underline;font-size:13px;font-weight:600}}
.wrap{{max-width:520px;margin:0 auto}}
.card{{background:var(--panel);border:1px solid var(--border);padding:22px;margin-bottom:16px;border-radius:8px}}
.card h2{{color:var(--head);font-size:13px;margin:0 0 16px;border-bottom:1px solid var(--border);padding-bottom:10px;text-transform:uppercase;letter-spacing:.5px;font-weight:600}}
label{{display:block;font-size:11.5px;font-weight:600;color:var(--dim);margin:14px 0 6px}}
select,input{{width:100%;background:var(--panel2);border:1px solid var(--border);color:var(--head);padding:9px 10px;font:12.5px -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;border-radius:4px}}
button{{margin-top:16px;background:var(--head);border:1px solid var(--head);color:var(--bg);padding:10px 14px;font:12.5px -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-weight:700;cursor:pointer;width:100%;border-radius:4px}}
button:hover{{opacity:.9}}
.msg{{font-size:12px;min-height:16px;margin-top:8px}}.ok{{color:var(--green)}}.err{{color:var(--red)}}
.info-row{{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--border);font-size:12.5px}}
.info-row:last-child{{border-bottom:none}}
.info-row b{{color:var(--head)}}
.alert-row{{display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:1px solid var(--border);font-size:12.5px}}
.alert-row:last-child{{border-bottom:none}}
.remove-btn{{background:transparent;border:1px solid var(--red);color:var(--red);width:auto;padding:4px 10px;margin:0;font-size:11px;font-weight:600}}
.empty-hint{{color:var(--dim);font-size:12px;padding:8px 0}}
.card.danger{{border-color:var(--red)}}
.card.danger h2{{color:var(--red)}}
button.danger-btn{{background:transparent;border:1px solid var(--red);color:var(--red)}}
</style></head><body><header><a class="brand" href="/terminal">QUANTIFY<span>.</span></a><div class="headerRight"><span style="color:var(--dim);font-size:12.5px">Settings · {user}</span><a class="back" href="/portfolio">Portfolio</a><a class="back" href="/subscription">Subscription</a><a class="back" href="/contact">Contact</a><a class="back" href="/terminal">&larr; Back to Terminal</a></div></header>
<div class="wrap">
<div class="card"><h2>Account</h2>
<div id="account-info"><div class="empty-hint">Loading...</div></div>
<a href="/subscription" style="display:block;text-align:center;margin-top:14px;color:var(--head);text-decoration:underline;font-size:12.5px;font-weight:600">Manage Subscription &rarr;</a>
</div>
<div class="card"><h2>Display</h2>
<label>Theme</label><select id="theme"><option value="dark">Dark</option><option value="light">Light</option></select>
<label>AI report language</label><select id="language"><option value="en">English</option><option value="ko">Korean (한국어)</option></select>
<label>Default scanner sort</label><select id="default_sort"><option value="overall_score">Score</option><option value="change_pct">Change %</option><option value="ticker">Ticker A-Z</option></select>
<label>Default scanner view</label><select id="default_view"><option value="list">List</option><option value="heatmap">Heatmap</option></select>
<button onclick="saveSettings()">Save Settings</button><div class="msg" id="settings-msg"></div>
</div>
<div class="card"><h2>Price Alerts</h2>
<div id="alerts-list"><div class="empty-hint">Loading...</div></div>
</div>
<div class="card"><h2>Change Password</h2>
<label>Current password</label><input type="password" id="current_password">
<label>New password</label><input type="password" id="new_password">
<p style="font-size:11px;color:var(--dim);margin-top:6px">10+ characters, with at least 1 letter and 1 number</p>
<button onclick="changePassword()">Change Password</button><div class="msg" id="password-msg"></div>
</div>
<div class="card danger"><h2>Danger Zone</h2>
<p style="font-size:12px;color:var(--text);margin:0 0 10px">Permanently delete your account and all associated data (portfolio, watchlist, alerts). This cannot be undone.</p>
<label>Confirm password</label><input type="password" id="delete_password">
<button class="danger-btn" onclick="deleteAccount()">Delete My Account</button><div class="msg" id="delete-msg"></div>
</div>
</div>
<script>
function fmtDate(ts){{return ts?new Date(ts*1000).toLocaleDateString():'-'}}
async function loadSettings(){{const r=await fetch('/api/settings');const d=await r.json();if(d.pref_theme)document.getElementById('theme').value=d.pref_theme;if(d.pref_language)document.getElementById('language').value=d.pref_language;if(d.pref_default_sort)document.getElementById('default_sort').value=d.pref_default_sort;if(d.pref_default_view)document.getElementById('default_view').value=d.pref_default_view;
const statusLabel={{active:'Active Subscription',trial:'Free Trial',expired:'Trial Ended',cancelled:'Cancelled',paused:'Paused'}}[d.subscription_status]||d.subscription_status;
document.getElementById('account-info').innerHTML=`<div class="info-row"><span>Email</span><b>${{d.email||'-'}}</b></div><div class="info-row"><span>Member since</span><b>${{fmtDate(d.created_at)}}</b></div><div class="info-row"><span>Plan status</span><b>${{statusLabel||'-'}}</b></div>`}}
async function saveSettings(){{const f=new FormData();f.append('theme',document.getElementById('theme').value);f.append('language',document.getElementById('language').value);f.append('default_sort',document.getElementById('default_sort').value);f.append('default_view',document.getElementById('default_view').value);const r=await fetch('/api/settings',{{method:'POST',body:f}});const d=await r.json();const el=document.getElementById('settings-msg');el.className='msg '+(r.ok?'ok':'err');el.innerText=d.message||d.error}}
async function changePassword(){{const f=new FormData();f.append('current_password',document.getElementById('current_password').value);f.append('new_password',document.getElementById('new_password').value);const r=await fetch('/api/settings/password',{{method:'POST',body:f}});const d=await r.json();const el=document.getElementById('password-msg');el.className='msg '+(r.ok?'ok':'err');el.innerText=d.message||d.error;if(r.ok){{document.getElementById('current_password').value='';document.getElementById('new_password').value=''}}}}
async function loadAlerts(){{const r=await fetch('/api/alerts/list');const d=await r.json();const el=document.getElementById('alerts-list');if(!d.alerts?.length){{el.innerHTML='<div class="empty-hint">No alerts set. Open a ticker in the terminal and click Set Alert.</div>';return}}el.innerHTML=d.alerts.map(a=>`<div class="alert-row"><span>${{a.ticker}} @ $${{a.target_price}}${{a.is_sent?' <span style="color:var(--dim)">(sent)</span>':''}}</span><button class="remove-btn" onclick="removeAlert(${{a.id}})">Remove</button></div>`).join('')}}
async function removeAlert(id){{const f=new FormData();f.append('id',id);await fetch('/api/alerts/remove',{{method:'POST',body:f}});loadAlerts()}}
async function deleteAccount(){{const pw=document.getElementById('delete_password').value;if(!pw)return;if(!confirm('Are you sure? This permanently deletes your account and cannot be undone.'))return;const f=new FormData();f.append('password',pw);const r=await fetch('/api/account/delete',{{method:'POST',body:f}});const d=await r.json();if(r.ok){{location.href='/'}}else{{const el=document.getElementById('delete-msg');el.className='msg err';el.innerText=d.error}}}}
loadSettings();loadAlerts();
</script></body></html>''')


if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8888")), reload=False)
