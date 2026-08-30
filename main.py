import asyncio
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
    "short_interest": {},
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
AI_PROMPT_VERSION = 2
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
MARKET_SCAN_INTERVAL_SECONDS = 30 * 60

LOGIN_ATTEMPTS = {}
LOGIN_MAX_ATTEMPTS = 5
LOGIN_LOCKOUT_SECONDS = 15 * 60

RESET_ATTEMPTS = {}
RESET_MAX_ATTEMPTS = 5

SEND_CODE_ATTEMPTS = {}
SEND_CODE_MAX_ATTEMPTS = 3

SIGNUP_ATTEMPTS = {}
SIGNUP_MAX_ATTEMPTS = 5


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
        if "verify_token_hash" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN verify_token_hash TEXT")
        if "verify_expires" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN verify_expires REAL")
        if "disclaimer_accepted_at" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN disclaimer_accepted_at REAL")

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
        raw = (0.30 * momentum_component + 0.25 * long_component +
               0.20 * macd_component + 0.15 * (2 * rsi_component - 1) +
               0.10 * (2 * risk_component - 1))
        return round(max(0.0, min(100.0, 50.0 + raw * 50.0)), 1)
    except Exception as e:
        print(f"[Error: {type(e).__name__}] Alpha score calculation error: {e}")
        return None


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
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode({
        "q": f"{ticker} stock", "hl": "en-US", "gl": "US", "ceid": "US:en"
    })
    try:
        response = requests.get(url, headers={"User-Agent": "QUANTIFY/2.0"}, timeout=6)
        response.raise_for_status()
        soup = bs4.BeautifulSoup(response.text, "xml")
        result = []
        for item in soup.find_all("item")[:5]:
            title, link, pub = item.find("title"), item.find("link"), item.find("pubDate")
            if title and link:
                result.append({"title": title.get_text(strip=True),
                               "url": link.get_text(strip=True),
                               "published": pub.get_text(strip=True) if pub else None})
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


def fetch_short_interest_sync(ticker, attempt: int = 0):
    try:
        pct = yf.Ticker(ticker).info.get("shortPercentOfFloat")
        if pct is None:
            return None
        pct = float(pct) * 100
        return round(pct, 2) if 0 <= pct <= 100 else None
    except Exception as exc:
        if attempt == 0:
            time.sleep(1.5)
            return fetch_short_interest_sync(ticker, attempt=1)
        print(f"[Error: {type(exc).__name__}] Short interest fetch error ({ticker}): {exc}")
        return None


async def get_short_interest(ticker):
    cached = CACHE["short_interest"].get(ticker)
    if cached and time.time() - cached["ts"] < SHORT_INTEREST_TTL:
        return cached["data"]
    data = await asyncio.to_thread(fetch_short_interest_sync, ticker)
    if data is not None:
        CACHE["short_interest"][ticker] = {"data": data, "ts": time.time()}
    return data


LANGUAGE_NAMES = {"en": "English", "ko": "Korean"}


MODE_CRITERIA = {
    "Long-Term Momentum Pullback": (
        "This mode looks for stocks in a confirmed long-term uptrend (price above its 200-day moving average) "
        "that have pulled back somewhat from recent highs — a potential entry within an established trend, not a "
        "reversal bet. Weight the 200-day trend and the pullback-from-high distance heavily; a stock far below its "
        "200-day MA does not fit this mode even if other numbers look attractive."
    ),
    "Short-Term Volatility Breakout": (
        "This mode looks for a sharp short-term price move accompanied by a genuine volume surge — momentum "
        "continuation over the next few days. Weight recent price change and volume-vs-average heavily; a big price "
        "move on below-average volume is a weak, less credible breakout for this mode."
    ),
    "Institutional Flow Leaders": (
        "This mode looks for signs of sustained, steady buying pressure (low short interest, MACD confirming an "
        "uptrend, volume near or above its average without an extreme single-day spike) consistent with gradual "
        "accumulation rather than a retail-driven pop. Weight short interest and MACD confirmation heavily; a huge "
        "one-day volume spike is more consistent with retail hype than institutional accumulation for this mode."
    ),
}


def ai_report_sync(ticker, price, change, mode, rsi, macd, short_pct,
                    pct_from_high=None, pct_from_low=None, above_trend=None, language="en",
                    volume_ratio=None, news=None):
    if not ai_client:
        return None
    language_name = LANGUAGE_NAMES.get(language, "English")
    short_text = (f"Short interest: {short_pct:.2f}%" if short_pct is not None
                  else "Short interest data unavailable")
    high_text = f"{pct_from_high:+.1f}% vs 52-week high" if pct_from_high is not None else "52-week high data unavailable"
    low_text = f"{pct_from_low:+.1f}% vs 52-week low" if pct_from_low is not None else "52-week low data unavailable"
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
        f"Change: {change:.2f}%\nRSI: {rsi:.2f}\nMACD histogram: {macd:.4f}\n"
        f"{high_text}\n{low_text}\n{trend_text}\n{volume_text}\n{short_text}\n"
        f"Strategy mode: {mode}\n{mode_criteria}\n\n{news_text}\n\n"
        "This ticker was flagged automatically by a quant algorithm based purely on numeric thresholds, not "
        "hand-picked by a person. You must check for these two failure modes: "
        "(1) Has it already spiked short-term, sitting near its 52-week high with RSI in overbought territory, "
        "risking a blow-off top if someone chases it now? "
        "(2) Is this a bounce inside a long-term downtrend (below the 200-day MA) or near its 52-week low, "
        "risking a dead-cat bounce in a financially weak stock?\n"
        "Base your analysis only on the numbers and headlines actually provided. Do not estimate or invent missing "
        "data, and do not assume a headline's content beyond its title. This is informational analysis, not "
        "investment advice. Do not state a specific buy price, price target, or stop-loss price in any form. Be "
        "direct and specific rather than hedging every sentence — state what the numbers show and commit to a view.\n\n"
        f"The values of quant_review, supply_demand, risk_review, news_analysis, and timing_reason must be written "
        f"in {language_name} only. The timing_verdict value itself must still be exactly one of the three English "
        "words below, never translated.\n"
        "Respond with ONLY the JSON object below, no other text:\n"
        '{"quant_review":"2-3 sentences on RSI/MACD/trend, and whether the setup actually fits the stated strategy mode\'s criteria",'
        '"supply_demand":"2-3 sentences on volume-vs-average and short interest together — do they agree or conflict?",'
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
        short_pct = await get_short_interest(ticker)
        row["short_percent"] = short_pct
        ai_result = await generate_ai_report(
            ticker, row["price"], row["change"], mode, row["rsi"], row["macd"], short_pct,
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


@app.on_event("startup")
async def startup():
    init_db()
    load_universe_cache()
    asyncio.create_task(refresh_universe())
    asyncio.create_task(scheduler())
    asyncio.create_task(market_scan_scheduler())
    asyncio.create_task(index_warm_scheduler())
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

                short_pct=await get_short_interest(ticker)
                above_trend = bool(row["above_200d_sma"]) if row["above_200d_sma"] is not None else None
                news = await fetch_stock_news(ticker)
                ai_result=await generate_ai_report(
                    ticker,row["price"],row["change_pct"],mode,row["rsi"],row["macd"],short_pct,
                    row["pct_from_52w_high"],row["pct_from_52w_low"],above_trend,"en",
                    row["volume_ratio"],news,
                )

                conn=db()
                if ai_result:
                    conn.execute("""UPDATE daily_scans
                                    SET short_percent=?,ai_report=?,ai_status='READY',
                                        ai_mode=?,ai_updated_at=?,ai_error=NULL,
                                        timing_score=?,timing_verdict=?,ai_prompt_version=?
                                    WHERE scan_date=? AND ticker=?""",
                                 (short_pct,ai_result["report_json"],mode,time.time(),
                                  ai_result["timing_score"],ai_result["timing_verdict"],AI_PROMPT_VERSION,
                                  today_str(),ticker))
                    AI_STATUS["ready"]+=1
                else:
                    conn.execute("""UPDATE daily_scans
                                    SET short_percent=?,ai_status='ERROR',ai_mode=?,
                                        ai_updated_at=?,ai_error=?
                                    WHERE scan_date=? AND ticker=?""",
                                 (short_pct,mode,time.time(),"AI returned no report",today_str(),ticker))
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
    if not get_logged_in_user(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    conn = db()
    scanned_count = conn.execute(
        "SELECT COUNT(*) FROM daily_scans WHERE scan_date=?", (today_str(),)
    ).fetchone()[0]
    last_updated = conn.execute(
        "SELECT MAX(created_at) FROM daily_scans WHERE scan_date=?", (today_str(),)
    ).fetchone()[0]
    rows = conn.execute("""
        SELECT ticker,universe,price,change_pct,alpha_score,quant_pass,timing_score,timing_verdict FROM daily_scans
        WHERE scan_date=? AND quant_pass=1 ORDER BY alpha_score DESC
    """, (today_str(),)).fetchall()
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
    if not get_logged_in_user(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
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
        short_pct = await get_short_interest(ticker)
        chart = []
        for idx, row in df.tail(150).iterrows():
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
            "long_ratio": round(100 - short_pct, 1) if short_pct is not None else None,
            "short_ratio": short_pct, "short_percent": short_pct, "chart": chart}}
    except Exception as e:
        print(f"[Error: {type(e).__name__}] terminal_data_fast processing error ({ticker}): {e}")
        return {"fast": {"ticker": ticker, "data_ok": False, "error": f"Data processing error: {type(e).__name__}"}}


@app.get("/api/terminal-data-ai")
async def terminal_data_ai(request: Request, ticker: str = "AAPL", mode: str = "Long-Term Momentum Pullback", language: str = "en"):
    user = get_logged_in_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    ticker=normalize_ticker(ticker)
    if not re.fullmatch(r"[A-Z0-9.\-^=]{1,15}",ticker):
        return JSONResponse({"error":"Invalid ticker"},status_code=400)
    if language not in LANGUAGE_NAMES:
        language = "en"

    conn=db()
    row=conn.execute("""SELECT ai_report,short_percent,ai_status,ai_mode,ai_language,ai_updated_at,ai_error,timing_score,timing_verdict,
                                price,change_pct,rsi,macd,pct_from_52w_high,pct_from_52w_low,above_200d_sma,volume_ratio,ai_prompt_version
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
                                          price,change_pct,rsi,macd,pct_from_52w_high,pct_from_52w_low,above_200d_sma,volume_ratio,ai_prompt_version
                                  FROM daily_scans WHERE scan_date=? AND ticker=?""", (today_str(), ticker)).fetchone()

    # A cached report only satisfies this request if it matches both the selected
    # strategy mode AND language, and was generated under the current prompt schema.
    # On a mismatch, generate a fresh one on demand (this is a shared cache slot, so
    # the next viewer with different mode/language will likewise regenerate it —
    # acceptable for this scale, not per-user cached).
    needs_regen = row is not None and row["price"] is not None and (
        row["ai_report"] is None or row["ai_mode"] != mode or (row["ai_language"] or "en") != language
        or row["ai_prompt_version"] != AI_PROMPT_VERSION
    )
    news=await fetch_stock_news(ticker)
    if needs_regen:
        short_pct = await get_short_interest(ticker)
        above_trend = bool(row["above_200d_sma"]) if row["above_200d_sma"] is not None else None
        ai_result = await generate_ai_report(
            ticker, row["price"], row["change_pct"], mode, row["rsi"], row["macd"], short_pct,
            row["pct_from_52w_high"], row["pct_from_52w_low"], above_trend, language,
            row["volume_ratio"], news,
        )
        if ai_result:
            conn.execute("""UPDATE daily_scans
                            SET short_percent=?,ai_report=?,ai_status='READY',ai_mode=?,ai_language=?,
                                ai_updated_at=?,ai_error=NULL,timing_score=?,timing_verdict=?,ai_prompt_version=?
                            WHERE scan_date=? AND ticker=?""",
                         (short_pct, ai_result["report_json"], mode, language, time.time(),
                          ai_result["timing_score"], ai_result["timing_verdict"], AI_PROMPT_VERSION, today_str(), ticker))
            conn.commit()
            row = conn.execute("""SELECT ai_report,short_percent,ai_status,ai_mode,ai_language,ai_updated_at,ai_error,timing_score,timing_verdict,
                                          price,change_pct,rsi,macd,pct_from_52w_high,pct_from_52w_low,above_200d_sma,volume_ratio
                                  FROM daily_scans WHERE scan_date=? AND ticker=?""", (today_str(), ticker)).fetchone()
    conn.close()

    report_sections = None
    if row and row["ai_report"]:
        try:
            report_sections = json.loads(row["ai_report"])
        except json.JSONDecodeError:
            report_sections = {"quant_review": row["ai_report"]}

    return {"ai":{
        "ai_report": row["ai_report"] if row else None,
        "report_sections": report_sections,
        "timing_score": row["timing_score"] if row else None,
        "timing_verdict": row["timing_verdict"] if row else None,
        "news": news,
        "market_summary": None,
        "status": row["ai_status"] if row else "PENDING",
        "mode": row["ai_mode"] if row else mode,
        "updated_at": row["ai_updated_at"] if row else None,
        "error": row["ai_error"] if row else None,
        "short_info":{
            "short_percent_of_float": row["short_percent"] if row else None,
            "long_ratio": round(100-row["short_percent"],1) if row and row["short_percent"] is not None else None,
            "short_ratio": row["short_percent"] if row else None
        }
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


@app.post("/api/portfolio/save")
async def portfolio_save(request: Request, ticker: str = Form(...), note: str = Form("")):
    user = get_logged_in_user(request)
    if not user: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    ticker = normalize_ticker(ticker)
    if not re.fullmatch(r"[A-Z0-9.\-^=]{1,15}", ticker):
        return JSONResponse({"error": "Invalid ticker"}, status_code=400)
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
            (email,ticker,scan_date,price,change_pct,alpha_score,rsi,macd,timing_score,timing_verdict,ai_report,note,saved_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (user, ticker, row["scan_date"], row["price"], row["change_pct"], row["alpha_score"],
              row["rsi"], row["macd"], row["timing_score"], row["timing_verdict"], row["ai_report"],
              note.strip()[:500], time.time()))
        conn.commit(); conn.close()
    except Exception as e:
        print(f"[Error: {type(e).__name__}] Portfolio save error: {e}")
        return JSONResponse({"error": f"Database error: {type(e).__name__}"}, status_code=500)
    return {"message": f"Saved {ticker} to your portfolio."}


@app.get("/api/portfolio")
async def portfolio_list(request: Request):
    user = get_logged_in_user(request)
    if not user: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    conn = db()
    rows = conn.execute("""
        SELECT id,ticker,scan_date,price,change_pct,alpha_score,rsi,macd,timing_score,timing_verdict,ai_report,note,saved_at
        FROM portfolio_items WHERE email=? ORDER BY saved_at DESC
    """, (user,)).fetchall()
    conn.close()
    items = []
    for r in rows:
        d = dict(r)
        if d.get("ai_report"):
            try:
                d["ai_report"] = json.loads(d["ai_report"])
            except json.JSONDecodeError:
                pass
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


@app.get("/api/settings")
async def get_settings(request: Request):
    user = get_logged_in_user(request)
    if not user: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    conn = db()
    row = conn.execute("SELECT email,pref_theme,pref_language,pref_default_mode FROM users WHERE email=?", (user,)).fetchone()
    conn.close()
    if not row:
        return JSONResponse({"error": "User not found"}, status_code=404)
    return dict(row)


@app.post("/api/settings")
async def update_settings(request: Request, theme: str = Form(...), language: str = Form(...), default_mode: str = Form(...)):
    user = get_logged_in_user(request)
    if not user: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if theme not in ("dark", "light"):
        return JSONResponse({"error": "Invalid theme"}, status_code=400)
    if language not in LANGUAGE_NAMES:
        return JSONResponse({"error": "Invalid language"}, status_code=400)
    if default_mode not in ("Long-Term Momentum Pullback", "Short-Term Volatility Breakout", "Institutional Flow Leaders"):
        return JSONResponse({"error": "Invalid mode"}, status_code=400)
    conn = db()
    conn.execute("UPDATE users SET pref_theme=?,pref_language=?,pref_default_mode=? WHERE email=?",
                 (theme, language, default_mode, user))
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
  .steps,.features{grid-template-columns:1fr}
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
<a class="muted" href="#features">Features</a>
<a class="muted" href="/login">Log in</a>
<a class="btn" href="/signup">Get Started Free</a>
</div>
</div></header>

<section class="hero" style="border-top:none">
<div class="eyebrow">EARLY ACCESS · FREE WHILE IN BETA</div>
<h1>Stop scanning <span class="hl">518</span> stocks by hand.<br>See the ones that actually <span class="hl">cleared the bar</span>.</h1>
<p class="sub">A daily quant scan of the S&amp;P 500 and Nasdaq-100, double-checked by AI for blow-off-top and dead-cat-bounce risk before it reaches your screen.</p>
<div class="cta-row">
<a class="btn" href="/signup">Get Started Free</a>
<a class="btn btn-ghost" href="#how">See how it works</a>
</div>
<div class="cta-note">No credit card. Same data for every subscriber — never personalized picks.</div>

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
<p>Free during early access. Takes under a minute to sign up.</p>
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
<div class="point"><b>&#9670;</b> Free during early access, no credit card</div>
</div>
</div>"""

GOOGLE_ICON_SVG = '<svg width="18" height="18" viewBox="0 0 18 18"><path fill="#4285F4" d="M17.64 9.2c0-.64-.06-1.25-.16-1.84H9v3.48h4.84a4.14 4.14 0 01-1.8 2.72v2.26h2.92c1.7-1.57 2.68-3.88 2.68-6.62z"/><path fill="#34A853" d="M9 18c2.43 0 4.47-.8 5.96-2.18l-2.92-2.26c-.81.54-1.84.86-3.04.86-2.34 0-4.32-1.58-5.03-3.7H.96v2.33A9 9 0 009 18z"/><path fill="#FBBC05" d="M3.97 10.72A5.4 5.4 0 013.68 9c0-.6.1-1.18.29-1.72V4.95H.96A9 9 0 000 9c0 1.45.35 2.83.96 4.05l3.01-2.33z"/><path fill="#EA4335" d="M9 3.58c1.32 0 2.5.45 3.44 1.35l2.58-2.58C13.46.89 11.43 0 9 0A9 9 0 00.96 4.95l3.01 2.33C4.68 5.16 6.66 3.58 9 3.58z"/></svg>'


def render_auth_page(title: str, form_html: str) -> HTMLResponse:
    return HTMLResponse(f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title><style>{BASE_CSS}</style></head><body><div class="authwrap">{AUTH_BRAND_HTML}<div class="authform">{form_html}</div></div></body></html>''')


@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    if get_logged_in_user(request):
        return RedirectResponse("/terminal", status_code=303)
    return HTMLResponse(LANDING_HTML)


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
<p>Market data, news, and other information displayed on the Service are sourced from third parties and may be delayed, incomplete, or inaccurate. We do not guarantee the accuracy, completeness, or timeliness of any data shown.</p>
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
            "INSERT INTO users(email,password_hash,salt,is_active) VALUES(?,?,?,1)",
            (email, password_hash, salt),
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
    form = f'''<div class="card"><h2>Create your account</h2><div class="subtitle">Free while in early access — no credit card required.</div><div class="error">{error}</div><a class="google-btn" href="/auth/google/login">{GOOGLE_ICON_SVG}Continue with Google</a><div class="divider">or</div><form action="/api/auth/signup" method="post"><label>Email</label><input type="email" name="email" required><label>Password</label><input type="password" name="password" required><p class="hint">10+ characters, with at least 1 letter and 1 number</p><button>Create account</button></form><p style="text-align:center;font-size:11.5px;color:#6b8a7e;margin-top:14px">By creating an account you agree to our <a href="/terms">Terms</a> and <a href="/privacy">Privacy Policy</a>.</p><div class="links"><a href="/login">Already have an account? Log in</a></div></div>'''
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
            "INSERT INTO users(email,password_hash,salt,is_active,verify_token_hash,verify_expires) VALUES(?,?,?,0,?,?)",
            (email,password_hash,salt,token_hash,time.time()+VERIFY_TOKEN_TTL),
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
        return RedirectResponse("/login?msg=If+that+account+needs+verification,+a+new+email+was+sent.", status_code=303)
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
        return RedirectResponse("/reset-password?email="+urllib.parse.quote(email),status_code=303)
    _register_failed_attempt(SEND_CODE_ATTEMPTS, email)
    try:
        conn=db()
        row=conn.execute("SELECT email FROM users WHERE email=?",(email,)).fetchone()
        if row:
            code=f"{secrets.randbelow(1_000_000):06d}"; code_hash=hashlib.sha256(code.encode()).hexdigest()
            conn.execute("UPDATE users SET reset_code_hash=?,reset_expires=? WHERE email=?",(code_hash,time.time()+300,email))
            conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Error: {type(e).__name__}] Password reset code generation error: {e}")
        return RedirectResponse("/forgot-password?error=Database+error",status_code=303)

    if row:
        send_email_notification(email,"[QUANTIFY.] Password Reset Code",f"Your code: {code}\nValid for: 5 minutes")
    # Respond identically regardless of whether the account exists, to avoid leaking registration status.
    return RedirectResponse("/reset-password?email="+urllib.parse.quote(email),status_code=303)


@app.get("/reset-password", response_class=HTMLResponse)
async def reset_page(email: str, error: Optional[str] = None):
    error = html_lib.escape(error) if error else ''
    email = html_lib.escape(email)
    form = f'''<div class="card"><h2>Choose a new password</h2><div class="subtitle">Enter the code we emailed you, plus a new password.</div><div class="error">{error}</div><form action="/api/auth/verify-and-reset" method="post"><input type="hidden" name="email" value="{email}"><label>6-digit code</label><input name="code" required maxlength="6"><label>New password</label><input type="password" name="new_password" required><p class="hint">10+ characters, with at least 1 letter and 1 number</p><button>Reset password</button></form></div>'''
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
    conn=db(); prefs=conn.execute("SELECT pref_theme,pref_language,pref_default_mode FROM users WHERE email=?",(user,)).fetchone(); conn.close()
    theme = prefs["pref_theme"] if prefs and prefs["pref_theme"] in ("dark","light") else "dark"
    pref_language = prefs["pref_language"] if prefs and prefs["pref_language"] in LANGUAGE_NAMES else "en"
    pref_default_mode = prefs["pref_default_mode"] if prefs and prefs["pref_default_mode"] else "Long-Term Momentum Pullback"
    user=html_lib.escape(user)
    return HTMLResponse(f'''<!doctype html><html lang="en" data-theme="{theme}"><head><meta charset="utf-8"><title>QUANTIFY.</title><script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script><style>
:root{{--bg:#0a0a0a;--panel:#141414;--panel2:#1e1e1e;--border:#2e2e2e;--border2:#252525;--text:#a8a8a8;--head:#ffffff;--dim:#787878;--green:#26a69a;--red:#ef5350;--orange:#ff9800;--grid-line:#1e1e1e}}
html[data-theme="light"]{{--bg:#ffffff;--panel:#ffffff;--panel2:#f2f2f2;--border:#dedede;--border2:#e8e8e8;--text:#4a4a4a;--head:#000000;--dim:#8a8a8a;--green:#089981;--red:#e64545;--orange:#c17900;--grid-line:#e8e8e8}}
*{{box-sizing:border-box}}body{{background:var(--bg);color:var(--text);font:13px/1.4 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;margin:0;padding:8px}}
header,.panel{{background:var(--panel);border:1px solid var(--border)}}
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
.ratio{{height:6px;background:var(--border);margin-top:6px;display:flex;border-radius:3px;overflow:hidden}}
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
.toast{{position:fixed;bottom:20px;right:20px;background:var(--panel);border:1px solid var(--border);color:var(--head);padding:12px 18px;border-radius:6px;box-shadow:0 10px 30px rgba(0,0,0,.5);font-size:12px;z-index:100;opacity:0;transform:translateY(10px);transition:opacity .2s,transform .2s;pointer-events:none;max-width:280px}}
.toast.show{{opacity:1;transform:translateY(0)}}
.toast.err{{border-color:var(--red);color:var(--red)}}
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
</style></head><body><header><a class="brand" href="/terminal">QUANTIFY<span>.</span></a><div class="headerRight"><span class="userTag">{user}</span><select id="mode" title="The AI review is measured against this strategy">
<option>Long-Term Momentum Pullback</option><option>Short-Term Volatility Breakout</option><option>Institutional Flow Leaders</option></select><button onclick="location='/portfolio'">Portfolio</button><button onclick="location='/settings'">Settings</button><button class="danger" onclick="location='/logout'">Log out</button></div></header><div class="grid"><section class="panel"><h3>Market Scanner <span id="ucount"></span></h3><div class="tabs"><button class="tab active" id="tabList" onclick="showView('list')">List</button><button class="tab" id="tabHeatmap" onclick="showView('heatmap')">Heatmap</button></div><input id="tickerInput" placeholder="Jump to ticker (e.g. TSLA)" onkeydown="if(event.key==='Enter')loadTicker(this.value)"><div class="sortbar" id="sortbar"><select id="sortKey" onchange="renderList()"><option value="alpha_score">Sort: Alpha Score</option><option value="change_pct">Sort: Change %</option><option value="timing_score">Sort: AI Timing Score</option><option value="ticker">Sort: Ticker A-Z</option></select></div><div class="list" id="list">Preparing constituent list...</div><div class="heatmap" id="heatmap" style="display:none"></div></section><section class="panel"><div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px"><h3 id="title" style="border:0;margin:0;padding:0">AAPL</h3><div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center"><input id="target" type="number" placeholder="Target price $" style="width:110px" title="Get an email when the price reaches this value"><button onclick="setAlert()" title="Email me when the price hits my target">&#128276; Set Alert</button><button onclick="savePortfolio()" title="Add this ticker to My Portfolio">&#9734; Save to Portfolio</button><button class="tf-btn" data-tf="1h" onclick="changeTF('1h')">1H</button><button class="tf-btn active" data-tf="1d" onclick="changeTF('1d')">1D</button><button class="tf-btn" data-tf="1wk" onclick="changeTF('1wk')">1W</button><button class="tf-btn" data-tf="1mo" onclick="changeTF('1mo')">1M</button></div></div><div id="chart" class="chart"></div><div class="legend"><span><i style="background:#e8e8e8"></i>SMA 20</span><span><i style="background:#ff9800"></i>SMA 50</span><span><i style="background:#ef5350"></i>SMA 200</span><span><i class="dash"></i>Bollinger Bands</span><span><i style="background:#26a69a"></i>Volume</span></div><div class="idx-row"><div class="idx-box"><div class="idx-label"><span>S&amp;P 500 · 60D</span><span id="idx-sp500-val"></span></div><div id="idx-sp500" class="idx-chart"></div></div><div class="idx-box"><div class="idx-label"><span>NASDAQ-100 · 60D</span><span id="idx-ndx-val"></span></div><div id="idx-ndx" class="idx-chart"></div></div></div><div class="metrics"><div class="metric"><div>RSI / MACD</div><div id="rsi" class="val">-</div></div><div class="metric"><div>52W High</div><div id="high52" class="val">-</div></div><div class="metric"><div>52W Low</div><div id="low52" class="val">-</div></div><div class="metric"><div>Trend</div><div id="trend" class="val">-</div></div><div class="metric"><div>Score Trend (Today)</div><div id="scoretrend" class="val">-</div></div></div><div class="notice">Short interest: <b id="short">No data</b><div class="ratio"><div id="longbar" style="background:#26a69a"></div><div id="shortbar" style="background:#ef5350"></div></div></div></section><section class="panel"><h3>AI Quant Report <small style="color:var(--dim);font-weight:normal;text-transform:none">(informational only, not investment advice)</small></h3><div id="verdict" style="display:none;margin-bottom:10px"></div><div id="ai" class="scroll">Loading AI analysis based on real data...</div><h3 style="margin-top:12px">News</h3><div id="news" class="scroll">Waiting for news...</div></section></div><div class="toast" id="toast"></div><script>
const USER_LANGUAGE='{pref_language}';
document.getElementById('mode').value='{pref_default_mode}';
let ticker='AAPL',tf='1d',chart,candle,volume,smaLines={{}},idxCharts={{}},bbLines={{}},currentView='list',lastSignals=[],lastUpdated=null;
function showToast(msg,isErr){{const t=document.getElementById('toast');t.textContent=msg;t.className='toast show'+(isErr?' err':'');clearTimeout(window._toastTimer);window._toastTimer=setTimeout(()=>t.classList.remove('show'),3500)}}
function showView(v){{currentView=v;document.getElementById('tabList').classList.toggle('active',v==='list');document.getElementById('tabHeatmap').classList.toggle('active',v==='heatmap');document.getElementById('sortbar').style.display=v==='list'?'flex':'none';document.getElementById('list').style.display=v==='list'?'block':'none';document.getElementById('heatmap').style.display=v==='heatmap'?'flex':'none';if(v==='heatmap')loadHeatmap()}}
function heatColor(chg){{if(chg==null)return '#333';const c=Math.max(-5,Math.min(5,chg));const t=(c+5)/10;const r=Math.round(239+(38-239)*t),g=Math.round(83+(166-83)*t),b=Math.round(80+(154-80)*t);return `rgb(${{r}},${{g}},${{b}})`}}
function groupByUniverse(items){{const groups={{}};items.forEach(t=>{{const g=t.universe||'Other';(groups[g]=groups[g]||[]).push(t)}});return groups}}
async function loadHeatmap(){{const el=document.getElementById('heatmap');const r=await fetch('/api/heatmap');const d=await r.json();if(!d.tiles?.length){{el.innerHTML='<div class="notice">No scan data yet.</div>';return}}const groups=groupByUniverse(d.tiles);el.innerHTML=Object.entries(groups).map(([g,items])=>`<div class="heat-group-header">${{g}} (${{items.length}})</div><div class="heat-grid">`+items.map(t=>`<div class="heat-tile" style="background:${{heatColor(t.change_pct)}}" title="${{t.ticker}} · ${{t.change_pct??'-'}}% · Alpha ${{t.alpha_score??'-'}}" onclick="showView('list');loadTicker('${{t.ticker}}')">${{t.ticker}}</div>`).join('')+'</div>').join('')}}
function sparklineSVG(arr){{if(!arr||arr.length<2)return '';const w=48,h=18;const min=Math.min(...arr),max=Math.max(...arr),range=(max-min)||1;const pts=arr.map((v,i)=>`${{(i/(arr.length-1)*w).toFixed(1)}},${{(h-((v-min)/range*h)).toFixed(1)}}`).join(' ');const color=arr[arr.length-1]>=arr[0]?'#26a69a':'#ef5350';return `<svg width="${{w}}" height="${{h}}" style="vertical-align:middle;flex-shrink:0"><polyline points="${{pts}}" fill="none" stroke="${{color}}" stroke-width="1.5"/></svg>`}}
function verdictClass(v){{return v==='Favorable'?'badge-ok':v==='Caution'?'badge-warn':v==='Risk'?'badge-danger':'badge-pending'}}
function itemSigClass(v){{return v==='Favorable'?'sig-favorable':v==='Caution'?'sig-caution':v==='Risk'?'sig-risk':''}}
function renderList(){{if(!lastSignals.length)return;const key=document.getElementById('sortKey').value;const sorted=[...lastSignals].sort((a,b)=>key==='ticker'?a.ticker.localeCompare(b.ticker):(b[key]??-Infinity)-(a[key]??-Infinity));const groups=groupByUniverse(sorted);document.getElementById('list').innerHTML=Object.entries(groups).map(([g,items])=>`<div class="group-header">${{g}} (${{items.length}})</div>`+items.map(s=>`<div class="item ${{itemSigClass(s.timing_verdict)}}" onclick="loadTicker('${{s.ticker}}')"><b>${{s.ticker}}</b><span style="display:flex;align-items:center;gap:6px">${{sparklineSVG(s.sparkline)}}<span style="text-align:right">${{s.price}} · ${{s.change_pct}}%<br><small>Alpha ${{s.alpha_score}} · <span class="badge ${{verdictClass(s.timing_verdict)}}">${{s.timing_verdict||'Analyzing'}}</span></small></span></span></div>`).join('')).join('')}}
function init(){{const c=document.getElementById('chart');chart=LightweightCharts.createChart(c,{{width:c.clientWidth,height:c.clientHeight,layout:{{background:{{type:'solid',color:'#141414'}},textColor:'#a8a8a8'}},grid:{{vertLines:{{color:'#1e1e1e'}},horzLines:{{color:'#1e1e1e'}}}},timeScale:{{timeVisible:true}}}});candle=chart.addCandlestickSeries({{upColor:'#26a69a',downColor:'#ef5350',borderUpColor:'#26a69a',borderDownColor:'#ef5350',wickUpColor:'#26a69a',wickDownColor:'#ef5350'}});volume=chart.addHistogramSeries({{color:'rgba(38,166,154,.5)',priceFormat:{{type:'volume'}},priceScaleId:''}});volume.priceScale().applyOptions({{scaleMargins:{{top:.8,bottom:0}}}});smaLines.sma20=chart.addLineSeries({{color:'#e8e8e8',lineWidth:1,priceLineVisible:false,lastValueVisible:false}});smaLines.sma50=chart.addLineSeries({{color:'#ff9800',lineWidth:1,priceLineVisible:false,lastValueVisible:false}});smaLines.sma200=chart.addLineSeries({{color:'#ef5350',lineWidth:1,priceLineVisible:false,lastValueVisible:false}});bbLines.upper=chart.addLineSeries({{color:'#9b6bff',lineWidth:1,lineStyle:2,priceLineVisible:false,lastValueVisible:false}});bbLines.lower=chart.addLineSeries({{color:'#9b6bff',lineWidth:1,lineStyle:2,priceLineVisible:false,lastValueVisible:false}});window.onresize=()=>{{chart.resize(c.clientWidth,c.clientHeight);Object.entries(idxCharts).forEach(([k,ic])=>{{const el=document.getElementById('idx-'+k);if(el)ic.resize(el.clientWidth,el.clientHeight)}})}};['sp500','ndx'].forEach(k=>{{const el=document.getElementById('idx-'+k);const ic=LightweightCharts.createChart(el,{{width:el.clientWidth,height:el.clientHeight,layout:{{background:{{type:'solid',color:'#141414'}},textColor:'#a8a8a8',fontSize:9}},grid:{{vertLines:{{visible:false}},horzLines:{{visible:false}}}},rightPriceScale:{{visible:false}},timeScale:{{visible:false}},handleScroll:false,handleScale:false}});idxCharts[k]=ic;idxCharts[k+'_line']=ic.addLineSeries({{color:'#e8e8e8',lineWidth:1.5,priceLineVisible:false,lastValueVisible:false}})}})}}
async function loadIndices(){{try{{const r=await fetch('/api/market-indices');const d=await r.json();const map={{sp500:d.sp500,ndx:d.nasdaq100}};Object.entries(map).forEach(([k,series])=>{{if(!series?.length)return;const boxEl=document.getElementById('idx-'+k);if(boxEl&&boxEl.clientWidth&&boxEl.clientHeight)idxCharts[k].resize(boxEl.clientWidth,boxEl.clientHeight);idxCharts[k+'_line'].setData(series.map(p=>({{time:p.time,value:p.close}})));idxCharts[k].timeScale().fitContent();const first=series[0].close,last=series[series.length-1].close;const chg=((last/first-1)*100).toFixed(2);idxCharts[k+'_line'].applyOptions({{color:chg>=0?'#26a69a':'#ef5350'}});const valEl=document.getElementById('idx-'+k+'-val');if(valEl)valEl.innerHTML=`${{last}} <span style="color:${{chg>=0?'#26a69a':'#ef5350'}}">${{chg>=0?'+':''}}${{chg}}%</span>`}})}}catch(e){{console.warn('index load failed',e)}}}}
async function loadScoreHistory(t){{const el=document.getElementById('scoretrend');try{{const r=await fetch(`/api/score-history?ticker=${{encodeURIComponent(t)}}`);const d=await r.json();const scores=(d.points||[]).map(p=>p.alpha_score).filter(v=>v!=null);if(scores.length<2){{el.innerHTML=scores.length?scores[scores.length-1].toFixed(1):'-';return}}el.innerHTML=sparklineSVG(scores)+' '+scores[scores.length-1].toFixed(1)}}catch(e){{el.innerText='-'}}}}
async function autoScanOnOpen(){{try{{await fetch('/api/auto-scan',{{method:'POST'}});}}catch(e){{console.warn('auto-scan trigger failed',e)}};scan()}}
function updateUcount(d){{document.getElementById('ucount').innerText=d.universe_count?` · ${{d.quant_pass_count??0}} detected / ${{d.universe_count}} symbols`:''}}
async function scan(){{const r=await fetch('/api/scan');const d=await r.json();updateUcount(d);lastUpdated=d.last_updated;lastSignals=d.signals||[];if(!lastSignals.length){{const ready=d.universe_status?.ready;const err=d.universe_status?.error;const scanned=d.scanned_count>0;document.getElementById('list').innerHTML='<div class="notice">'+(scanned?'Scan complete — no tickers cleared the quant threshold today. You can still look up any ticker above.':(ready?'The server is preparing the next scan — check back shortly.':(err?'Could not prepare constituent data. The server will retry automatically.':'Preparing S&P 500 / Nasdaq-100 constituents...')))+'</div>';loadTicker(ticker);return}}renderList();loadTicker(lastSignals[0].ticker)}}
async function pollForUpdates(){{try{{const r=await fetch('/api/scan');const d=await r.json();updateUcount(d);if(d.last_updated&&d.last_updated!==lastUpdated){{lastUpdated=d.last_updated;lastSignals=d.signals||[];renderList();if(currentView==='heatmap')loadHeatmap();loadTicker(ticker);showToast('Updated with the latest scan.')}}}}catch(e){{}}}}
async function loadTicker(t){{ticker=t.toUpperCase().trim();document.getElementById('title').innerText=ticker;document.getElementById('ai').innerText='Loading AI analysis based on real data...';document.getElementById('news').innerText='Waiting for news...';document.getElementById('verdict').style.display='none';const fastPromise=fetch(`/api/terminal-data-fast?ticker=${{encodeURIComponent(ticker)}}&timeframe=${{tf}}`).then(r=>r.json());const aiPromise=fetch(`/api/terminal-data-ai?ticker=${{encodeURIComponent(ticker)}}&mode=${{encodeURIComponent(document.getElementById('mode').value)}}&language=${{USER_LANGUAGE}}`).then(r=>r.json());const d=await fastPromise;if(!d.fast?.data_ok){{document.getElementById('rsi').innerText=d.fast?.error||'No data';return}}const cd=d.fast.chart.map(x=>({{time:x.time,open:x.open,high:x.high,low:x.low,close:x.close}}));const vd=d.fast.chart.map(x=>({{time:x.time,value:x.volume}}));candle.setData(cd);volume.setData(vd);['sma20','sma50','sma200'].forEach(k=>{{const pts=d.fast.chart.filter(x=>x[k]!=null).map(x=>({{time:x.time,value:x[k]}}));smaLines[k].setData(pts)}});bbLines.upper.setData(d.fast.chart.filter(x=>x.bb_upper!=null).map(x=>({{time:x.time,value:x.bb_upper}})));bbLines.lower.setData(d.fast.chart.filter(x=>x.bb_lower!=null).map(x=>({{time:x.time,value:x.bb_lower}})));const cEl=document.getElementById('chart');if(cEl.clientWidth&&cEl.clientHeight)chart.resize(cEl.clientWidth,cEl.clientHeight);chart.timeScale().fitContent();document.getElementById('rsi').innerText=`RSI ${{d.fast.rsi}} / MACD ${{d.fast.macd}}`;document.getElementById('high52').innerText=d.fast.pct_from_52w_high==null?'N/A':d.fast.pct_from_52w_high+'%';document.getElementById('low52').innerText=d.fast.pct_from_52w_low==null?'N/A':d.fast.pct_from_52w_low+'%';document.getElementById('trend').innerText=d.fast.above_200d_sma==null?'N/A':(d.fast.above_200d_sma?'Uptrend':'Downtrend');loadScoreHistory(ticker);document.getElementById('short').innerText=d.fast.short_percent==null?'No data':d.fast.short_percent+'%';document.getElementById('longbar').style.width=(d.fast.long_ratio??0)+'%';document.getElementById('shortbar').style.width=(d.fast.short_ratio??0)+'%';const x=await aiPromise;const vEl=document.getElementById('verdict');if(x.ai?.timing_verdict){{vEl.style.display='block';vEl.innerHTML=`<span class="badge ${{verdictClass(x.ai.timing_verdict)}}">${{x.ai.timing_verdict}}</span> Timing score ${{x.ai.timing_score??'-'}} / 100`}}else{{vEl.style.display='none'}}const sec=x.ai?.report_sections;const aiEl=document.getElementById('ai');if(sec){{const labels={{quant_review:'Quant Review',supply_demand:'Supply/Demand',risk_review:'Risk Review',news_analysis:'News Analysis',timing_reason:'Timing Rationale'}};aiEl.innerHTML=Object.keys(labels).filter(k=>sec[k]).map(k=>`<div class="section"><b>${{labels[k]}}</b>${{sec[k]}}</div>`).join('')}}else{{aiEl.innerText=x.ai?.ai_report||(x.ai?.status==='PENDING'||x.ai?.status==='RUNNING'?'Preparing AI analysis cache on the server...':'AI analysis is unavailable.')}}const news=x.ai?.news;if(!news)document.getElementById('news').innerText='Could not fetch a live news feed.';else document.getElementById('news').innerHTML=news.map(n=>`<div style="margin-bottom:8px"><a href="${{n.url}}" target="_blank" rel="noopener">${{n.title}}</a><br><small>${{n.published||''}}</small></div>`).join('')}}
async function setAlert(){{const p=Number(document.getElementById('target').value);if(!(p>0))return showToast('Enter a target price first.',true);const f=new FormData();f.append('ticker',ticker);f.append('target_price',p);const r=await fetch('/api/alerts/set',{{method:'POST',body:f}});const d=await r.json();showToast(d.message||d.error,!r.ok)}}
async function savePortfolio(){{const f=new FormData();f.append('ticker',ticker);const r=await fetch('/api/portfolio/save',{{method:'POST',body:f}});const d=await r.json();showToast(d.message||d.error,!r.ok)}}
function changeTF(x){{tf=x;document.querySelectorAll('.tf-btn').forEach(b=>b.classList.toggle('active',b.dataset.tf===x));loadTicker(ticker)}}
window.onload=()=>{{init();autoScanOnOpen();loadIndices();setInterval(pollForUpdates,20000)}};
</script></body></html>''')


@app.get("/portfolio", response_class=HTMLResponse)
async def portfolio_page(request: Request):
    user = get_logged_in_user(request)
    if not user: return RedirectResponse("/login", status_code=303)
    if not disclaimer_accepted(user): return RedirectResponse("/accept-disclaimer", status_code=303)
    user = html_lib.escape(user)
    return HTMLResponse(f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>QUANTIFY. Portfolio</title><style>
:root{{--bg:#0a0a0a;--panel:#141414;--panel2:#1e1e1e;--border:#2e2e2e;--text:#a8a8a8;--head:#ffffff;--dim:#787878;--green:#26a69a;--red:#ef5350;--orange:#ff9800}}
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
.disclaimer-footer{{max-width:900px;margin:20px auto 0;color:var(--dim);font-size:10.5px;line-height:1.6;text-align:center;padding:12px;border-top:1px solid var(--border)}}
</style></head><body><header><a class="brand" href="/terminal">QUANTIFY<span>.</span></a><div class="headerRight"><span style="color:var(--dim);font-size:12.5px">Portfolio · {user}</span><a class="back" href="/terminal">&larr; Back to Terminal</a></div></header>
<div class="wrap" id="list">Loading...</div>
<div class="disclaimer-footer">QUANTIFY is informational and educational only, not investment advice. Nothing here is a recommendation to buy or sell any security. All investment decisions are solely your own responsibility.</div>
<script>
function badgeClass(v){{return v==='Favorable'?'badge-ok':v==='Caution'?'badge-warn':v==='Risk'?'badge-danger':''}}
async function load(){{const r=await fetch('/api/portfolio');const d=await r.json();const el=document.getElementById('list');if(!d.items?.length){{el.innerHTML='<div class="empty">Nothing saved yet. Open a ticker in the terminal and click ☆ Save to Portfolio.</div>';return}}el.innerHTML=d.items.map(it=>{{const sec=it.ai_report&&typeof it.ai_report==='object'?it.ai_report:null;const summary=sec?.quant_review||(typeof it.ai_report==='string'?it.ai_report:'');const date=new Date(it.saved_at*1000).toLocaleString();return `<div class="item"><div class="item-head"><b>${{it.ticker}}</b><div>${{it.timing_verdict?`<span class="badge ${{badgeClass(it.timing_verdict)}}">${{it.timing_verdict}}</span> `:''}}<button class="remove" onclick="remove(${{it.id}})">Remove</button></div></div><div class="meta">Saved ${{date}} · Scan date ${{it.scan_date}} · Price $${{it.price}} (${{it.change_pct}}%) · Alpha ${{it.alpha_score}} · RSI ${{it.rsi}}${{it.timing_score!=null?' · Timing score '+it.timing_score+'/100':''}}</div>${{summary?`<div class="note">${{summary}}</div>`:''}}</div>`}}).join('')}}
async function remove(id){{const f=new FormData();f.append('id',id);await fetch('/api/portfolio/remove',{{method:'POST',body:f}});load()}}
load();
</script></body></html>''')


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    user = get_logged_in_user(request)
    if not user: return RedirectResponse("/login", status_code=303)
    if not disclaimer_accepted(user): return RedirectResponse("/accept-disclaimer", status_code=303)
    user = html_lib.escape(user)
    return HTMLResponse(f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>QUANTIFY. Settings</title><style>
:root{{--bg:#0a0a0a;--panel:#141414;--panel2:#1e1e1e;--border:#2e2e2e;--text:#a8a8a8;--head:#ffffff;--dim:#787878;--green:#26a69a;--red:#ef5350}}
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
</style></head><body><header><a class="brand" href="/terminal">QUANTIFY<span>.</span></a><div class="headerRight"><span style="color:var(--dim);font-size:12.5px">Settings · {user}</span><a class="back" href="/terminal">&larr; Back to Terminal</a></div></header>
<div class="wrap">
<div class="card"><h2>Display</h2>
<label>Theme</label><select id="theme"><option value="dark">Dark</option><option value="light">Light</option></select>
<label>AI report language</label><select id="language"><option value="en">English</option><option value="ko">Korean (한국어)</option></select>
<label>Default strategy mode</label><select id="default_mode"><option>Long-Term Momentum Pullback</option><option>Short-Term Volatility Breakout</option><option>Institutional Flow Leaders</option></select>
<button onclick="saveSettings()">Save Settings</button><div class="msg" id="settings-msg"></div>
</div>
<div class="card"><h2>Change Password</h2>
<label>Current password</label><input type="password" id="current_password">
<label>New password</label><input type="password" id="new_password">
<p style="font-size:11px;color:var(--dim);margin-top:6px">10+ characters, with at least 1 letter and 1 number</p>
<button onclick="changePassword()">Change Password</button><div class="msg" id="password-msg"></div>
</div>
</div>
<script>
async function loadSettings(){{const r=await fetch('/api/settings');const d=await r.json();if(d.pref_theme)document.getElementById('theme').value=d.pref_theme;if(d.pref_language)document.getElementById('language').value=d.pref_language;if(d.pref_default_mode)document.getElementById('default_mode').value=d.pref_default_mode}}
async function saveSettings(){{const f=new FormData();f.append('theme',document.getElementById('theme').value);f.append('language',document.getElementById('language').value);f.append('default_mode',document.getElementById('default_mode').value);const r=await fetch('/api/settings',{{method:'POST',body:f}});const d=await r.json();const el=document.getElementById('settings-msg');el.className='msg '+(r.ok?'ok':'err');el.innerText=d.message||d.error}}
async function changePassword(){{const f=new FormData();f.append('current_password',document.getElementById('current_password').value);f.append('new_password',document.getElementById('new_password').value);const r=await fetch('/api/settings/password',{{method:'POST',body:f}});const d=await r.json();const el=document.getElementById('password-msg');el.className='msg '+(r.ok?'ok':'err');el.innerText=d.message||d.error;if(r.ok){{document.getElementById('current_password').value='';document.getElementById('new_password').value=''}}}}
loadSettings();
</script></body></html>''')


if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8888")), reload=False)
