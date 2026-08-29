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
from datetime import datetime
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
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from openai import OpenAI, RateLimitError

load_dotenv()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DB_FILE = os.getenv("DB_FILE", "users.db")
AI_BASE_URL = os.getenv("AI_BASE_URL", "http://localhost:11434/v1")
AI_API_KEY = os.getenv("AI_API_KEY", "ollama")
AI_MODEL = os.getenv("AI_MODEL", "llama3")
SESSION_TTL = 24 * 3600

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "")

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

LOGIN_ATTEMPTS = {}
LOGIN_MAX_ATTEMPTS = 5
LOGIN_LOCKOUT_SECONDS = 15 * 60

RESET_ATTEMPTS = {}
RESET_MAX_ATTEMPTS = 5

SEND_CODE_ATTEMPTS = {}
SEND_CODE_MAX_ATTEMPTS = 3


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
    if len(password) < 16:
        return False, "Password must be at least 16 characters long."
    if not re.search(r"[A-Z]", password):
        return False, "Password must include at least 1 uppercase letter."
    if not re.search(r"[a-z]", password):
        return False, "Password must include at least 1 lowercase letter."
    if len(re.findall(r"[0-9]", password)) < 4:
        return False, "Password must include at least 4 digits."
    if not re.search(r"[^A-Za-z0-9]", password):
        return False, "Password must include at least 1 special character."
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


def _download_group(tickers, period="2y", interval="1d"):
    try:
        return yf.download(
            tickers=tickers, period=period, interval=interval,
            auto_adjust=True, progress=False, threads=True, group_by="column"
        )
    except Exception as exc:
        print(f"[Error: {type(exc).__name__}] yfinance batch download error: {exc}")
        return None


INTERVAL_PERIODS = {"1h": "730d", "1d": "2y", "1wk": "5y", "1mo": "max"}


async def download_stock(ticker: str, interval="1d"):
    key = f"single:{ticker}:{interval}"
    cached = CACHE["historical"].get(key)
    if cached and time.time() - cached["ts"] < HISTORICAL_TTL:
        return cached["data"]
    period = INTERVAL_PERIODS.get(interval, "2y")
    data = await asyncio.to_thread(_download_group, [ticker], period, interval)
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
                data = await asyncio.to_thread(_download_group, [ticker], "2y", "1d")
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
def fetch_news_sync(ticker: str):
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
        print(f"[Error: {type(exc).__name__}] News fetch error ({ticker}): {exc}")
        return None


async def fetch_stock_news(ticker):
    cached = CACHE["news"].get(ticker)
    if cached and time.time() - cached["ts"] < NEWS_TTL:
        return cached["data"]
    data = await asyncio.to_thread(fetch_news_sync, ticker)
    CACHE["news"][ticker] = {"data": data, "ts": time.time()}
    return data


def fetch_short_interest_sync(ticker):
    try:
        pct = yf.Ticker(ticker).info.get("shortPercentOfFloat")
        if pct is None:
            return None
        pct = float(pct) * 100
        return round(pct, 2) if 0 <= pct <= 100 else None
    except Exception as exc:
        print(f"[Error: {type(exc).__name__}] Short interest fetch error ({ticker}): {exc}")
        return None


async def get_short_interest(ticker):
    cached = CACHE["short_interest"].get(ticker)
    if cached and time.time() - cached["ts"] < SHORT_INTEREST_TTL:
        return cached["data"]
    data = await asyncio.to_thread(fetch_short_interest_sync, ticker)
    CACHE["short_interest"][ticker] = {"data": data, "ts": time.time()}
    return data


LANGUAGE_NAMES = {"en": "English", "ko": "Korean"}


def ai_report_sync(ticker, price, change, mode, rsi, macd, short_pct,
                    pct_from_high=None, pct_from_low=None, above_trend=None, language="en"):
    if not ai_client:
        return None
    language_name = LANGUAGE_NAMES.get(language, "English")
    short_text = (f"Short interest: {short_pct:.2f}%" if short_pct is not None
                  else "Short interest data unavailable")
    high_text = f"{pct_from_high:+.1f}% vs 52-week high" if pct_from_high is not None else "52-week high data unavailable"
    low_text = f"{pct_from_low:+.1f}% vs 52-week low" if pct_from_low is not None else "52-week low data unavailable"
    trend_text = ("Above the 200-day moving average" if above_trend is True else
                  "Below the 200-day moving average" if above_trend is False else "200-day moving average data unavailable")
    prompt = (
        f"Date: {display_date()}\nTicker: {ticker}\nPrice: ${price:.2f}\n"
        f"Change: {change:.2f}%\nRSI: {rsi:.2f}\nMACD histogram: {macd:.4f}\n"
        f"{high_text}\n{low_text}\n{trend_text}\n{short_text}\nStrategy mode: {mode}\n\n"
        "This ticker was flagged automatically by a quant algorithm based purely on numeric thresholds, not "
        "hand-picked by a person. You must check for these two failure modes: "
        "(1) Has it already spiked short-term, sitting near its 52-week high with RSI in overbought territory, "
        "risking a blow-off top if someone chases it now? "
        "(2) Is this a bounce inside a long-term downtrend (below the 200-day MA) or near its 52-week low, "
        "risking a dead-cat bounce in a financially weak stock?\n"
        "Base your analysis only on the numbers actually provided. Do not estimate or invent missing data. "
        "This is informational analysis, not investment advice. Do not state a specific buy price, price target, "
        "or stop-loss price in any form.\n\n"
        f"The values of quant_review, supply_demand, risk_review, and timing_reason must be written in {language_name} only. "
        "The timing_verdict value itself must still be exactly one of the three English words below, never translated.\n"
        "Respond with ONLY the JSON object below, no other text:\n"
        '{"quant_review":"2-3 sentence assessment of the quant indicators","supply_demand":"2-3 sentence assessment of supply/demand and short interest",'
        '"risk_review":"2-3 sentence assessment of the two failure modes above",'
        '"timing_verdict":"one of: Favorable, Caution, or Risk",'
        '"timing_score":an integer from 0 to 100,'
        '"timing_reason":"2-3 sentence explanation for the score"}'
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

        return {"ticker": ticker, "price": round(price, 2), "change": round(change, 2),
                "alpha_score": score, "rsi": round(float(rsi_series.iloc[-1]), 2),
                "macd": round(float(macd_hist.iloc[-1]), 4),
                "pct_from_52w_high": pct_from_high, "pct_from_52w_low": pct_from_low,
                "above_200d_sma": above_trend}
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
                         pct_from_52w_high,pct_from_52w_low,above_200d_sma,
                         ai_report,short_percent,ai_status,ai_mode,ai_updated_at,ai_error,quant_pass,created_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(scan_date,ticker) DO UPDATE SET
                            universe=excluded.universe, price=excluded.price, change_pct=excluded.change_pct,
                            alpha_score=excluded.alpha_score, rsi=excluded.rsi, macd=excluded.macd,
                            pct_from_52w_high=excluded.pct_from_52w_high, pct_from_52w_low=excluded.pct_from_52w_low,
                            above_200d_sma=excluded.above_200d_sma,
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
                        r.get("ai_report"),
                        r.get("short_percent"), "PENDING", mode, None, None, r.get("quant_pass", 0), now
                    ))
                    BATCH_STATUS["saved"] += 1
                except Exception as exc:
                    print(f"[Error: {type(exc).__name__}] Batch DB save error ({r.get('ticker', '?')}): {exc}", flush=True)

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


@app.on_event("startup")
async def startup():
    init_db()
    load_universe_cache()
    asyncio.create_task(refresh_universe())
    asyncio.create_task(scheduler())
    asyncio.create_task(asyncio.to_thread(check_email_config))
    asyncio.get_running_loop().call_later(3, start_server_warmup)


def send_email_notification(to_email, subject, body, max_retries=3):
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print(f"[Error: EmailNotConfigured] SENDER_EMAIL/SENDER_PASSWORD not set — could not send '{subject}' to {to_email}")
        return False
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


def check_email_config():
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("[email] SENDER_EMAIL/SENDER_PASSWORD not set — verification, password reset, and alert emails will not be sent.")
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
        SELECT ticker,price,change_pct,rsi,macd,pct_from_52w_high,pct_from_52w_low,above_200d_sma
        FROM daily_scans
        WHERE scan_date=? AND quant_pass=1
          AND (ai_report IS NULL OR ai_mode<>?)
        ORDER BY alpha_score DESC
    """,(today_str(),mode)).fetchall()
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
                ai_result=await generate_ai_report(
                    ticker,row["price"],row["change_pct"],mode,row["rsi"],row["macd"],short_pct,
                    row["pct_from_52w_high"],row["pct_from_52w_low"],above_trend,
                )

                conn=db()
                if ai_result:
                    conn.execute("""UPDATE daily_scans
                                    SET short_percent=?,ai_report=?,ai_status='READY',
                                        ai_mode=?,ai_updated_at=?,ai_error=NULL,
                                        timing_score=?,timing_verdict=?
                                    WHERE scan_date=? AND ticker=?""",
                                 (short_pct,ai_result["report_json"],mode,time.time(),
                                  ai_result["timing_score"],ai_result["timing_verdict"],
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
    rows = conn.execute("""
        SELECT ticker,universe,price,change_pct,alpha_score,quant_pass,timing_score,timing_verdict FROM daily_scans
        WHERE scan_date=? AND quant_pass=1 ORDER BY alpha_score DESC
    """, (today_str(),)).fetchall()
    conn.close()
    return {"scanned_count": scanned_count, "universe_count": len(UNIVERSE),
            "quant_pass_count": len(rows),
            "universe_status": UNIVERSE_STATUS, "cache": dict(CACHE_STATUS),
            "signals": [dict(r) | {"status": "ACTIVE DATA"} for r in rows]}


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


@app.get("/api/admin/run-batch")
async def api_run_batch(request: Request):
    if not get_logged_in_user(request):
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
async def api_ai_status(request: Request):
    if not get_logged_in_user(request):
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
async def api_batch_status(request: Request):
    if not get_logged_in_user(request):
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
                                price,change_pct,rsi,macd,pct_from_52w_high,pct_from_52w_low,above_200d_sma
                        FROM daily_scans WHERE scan_date=? AND ticker=?
                        ORDER BY id DESC LIMIT 1""",(today_str(),ticker)).fetchone()

    # A cached report only satisfies this request if it matches both the selected
    # strategy mode AND language. On a mismatch, generate a fresh one on demand
    # (this is a shared cache slot, so the next viewer with different mode/language
    # will likewise regenerate it — acceptable for this scale, not per-user cached).
    needs_regen = row is not None and row["price"] is not None and (
        row["ai_report"] is None or row["ai_mode"] != mode or (row["ai_language"] or "en") != language
    )
    if needs_regen:
        short_pct = await get_short_interest(ticker)
        above_trend = bool(row["above_200d_sma"]) if row["above_200d_sma"] is not None else None
        ai_result = await generate_ai_report(
            ticker, row["price"], row["change_pct"], mode, row["rsi"], row["macd"], short_pct,
            row["pct_from_52w_high"], row["pct_from_52w_low"], above_trend, language,
        )
        if ai_result:
            conn.execute("""UPDATE daily_scans
                            SET short_percent=?,ai_report=?,ai_status='READY',ai_mode=?,ai_language=?,
                                ai_updated_at=?,ai_error=NULL,timing_score=?,timing_verdict=?
                            WHERE scan_date=? AND ticker=?""",
                         (short_pct, ai_result["report_json"], mode, language, time.time(),
                          ai_result["timing_score"], ai_result["timing_verdict"], today_str(), ticker))
            conn.commit()
            row = conn.execute("""SELECT ai_report,short_percent,ai_status,ai_mode,ai_language,ai_updated_at,ai_error,timing_score,timing_verdict,
                                          price,change_pct,rsi,macd,pct_from_52w_high,pct_from_52w_low,above_200d_sma
                                  FROM daily_scans WHERE scan_date=? AND ticker=?""", (today_str(), ticker)).fetchone()
    conn.close()
    news=await fetch_stock_news(ticker)

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
<style>
:root{--bg:#050807;--panel:#030504;--panel2:#060908;--border:#14221b;--text:#9ab8af;--head:#dff5ed;--dim:#436659;--dim2:#567d6e;--green:#2ecc71;--red:#e74c3c;--orange:#f39c12;--blue:#3498db}
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
.hero{padding:90px 24px 70px;text-align:center}
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
.step{background:var(--panel);border:1px solid var(--border);padding:26px}
.step .num{color:var(--green);font-size:12px;font-weight:bold;letter-spacing:1px;margin-bottom:14px}
.step h3{color:var(--head);font-size:16px;margin-bottom:10px}
.step p{color:var(--dim2);font-size:13.5px;line-height:1.7}
.features{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--border);max-width:1000px;margin:0 auto;border:1px solid var(--border)}
.feature{background:var(--panel);padding:26px}
.feature .icon{color:var(--green);font-size:20px;margin-bottom:12px}
.feature h4{color:var(--head);font-size:14.5px;margin-bottom:8px}
.feature p{color:var(--dim2);font-size:13px;line-height:1.65}
.trust{max-width:820px;margin:0 auto;background:var(--panel);border:1px solid var(--border);padding:36px}
.trust h3{color:var(--head);font-size:16px;margin-bottom:14px}
.trust ul{list-style:none}
.trust li{color:var(--text);font-size:13.5px;padding:9px 0 9px 22px;position:relative;border-bottom:1px solid #0c1712}
.trust li:last-child{border-bottom:none}
.trust li:before{content:"—";position:absolute;left:0;color:var(--green)}
.disclaimer{max-width:820px;margin:0 auto;background:#0a0705;border:1px solid #2a2008;padding:22px 26px}
.disclaimer .kicker{color:var(--orange);font-size:11px;letter-spacing:1px;margin-bottom:10px;font-weight:bold}
.disclaimer p{color:var(--dim2);font-size:12.5px;line-height:1.7}
.final{text-align:center;padding:90px 24px}
.final h2{color:var(--head);font-size:30px;margin-bottom:14px}
.final p{color:var(--dim2);margin-bottom:30px}
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
<p class="sub">QUANTIFY runs a quantitative scan across the S&amp;P 500 and Nasdaq-100 every day, then makes an AI double-check every hit for blow-off-top and dead-cat-bounce risk — before it ever reaches your screen.</p>
<div class="cta-row">
<a class="btn" href="/signup">Get Started Free</a>
<a class="btn btn-ghost" href="#how">See how it works</a>
</div>
<div class="cta-note">No credit card. Real market data from day one.</div>

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
<div class="feature"><div class="icon">◆</div><h4>Live market data</h4><p>Real prices and volume from the actual market — no simulated or backfilled data, ever.</p></div>
<div class="feature"><div class="icon">◆</div><h4>Daily automated scan</h4><p>The full S&amp;P 500 + Nasdaq-100 universe is rescanned automatically, so you're not the one doing the legwork.</p></div>
<div class="feature"><div class="icon">◆</div><h4>Plain-language AI review</h4><p>Every detected ticker gets a written quant review, supply/demand read, and explicit risk check — in plain English.</p></div>
<div class="feature"><div class="icon">◆</div><h4>52-week &amp; trend context</h4><p>Distance from the 52-week high/low and 200-day trend, so you can see where a stock actually sits.</p></div>
<div class="feature"><div class="icon">◆</div><h4>Price alerts</h4><p>Set a target price on any ticker and get notified by email when it's reached.</p></div>
<div class="feature"><div class="icon">◆</div><h4>Live news feed</h4><p>Recent headlines pulled per ticker, right next to the chart — no extra tab-switching.</p></div>
</div>
</section>

<section>
<div class="trust">
<h3>Why this looks different from most stock-tip sites</h3>
<ul>
<li>The scan is math-first: a ticker is flagged by its score, not by a person's opinion.</li>
<li>Every flagged ticker is specifically screened for blow-off-top and dead-cat-bounce risk before you ever see it.</li>
<li>Every subscriber sees the same data for the same ticker on the same day — nothing here is personalized trading advice.</li>
<li>We show you the reasoning, not just a verdict — every score comes with a written explanation.</li>
</ul>
</div>
</section>

<section>
<div class="disclaimer">
<div class="kicker">IMPORTANT</div>
<p>QUANTIFY is an informational and educational tool. Nothing on this site is investment advice, a recommendation, or a solicitation to buy or sell any security. Scores, badges, and AI commentary reflect a mathematical model's output on the data available at the time and can be wrong. Markets involve risk, including loss of principal. Do your own research and consult a licensed financial advisor before making investment decisions.</p>
</div>
</section>

<div class="final">
<h2>See today's detected tickers.</h2>
<p>Free during early access. Takes under a minute to sign up.</p>
<a class="btn" href="/signup">Get Started Free</a>
</div>

<footer>
QUANTIFY. — informational and educational only, not investment advice.<br>
<a href="/login">Log in</a> · <a href="/signup">Sign up</a>
</footer>
</body></html>"""


# -----------------------------------------------------------------------------
# Auth pages/endpoints
# -----------------------------------------------------------------------------
BASE_CSS = """
*{box-sizing:border-box}body{margin:0;background:#050807;color:#9ab8af;font:11px 'Courier New',monospace;display:flex;align-items:center;justify-content:center;height:100vh}.card{width:380px;background:#030504;border:1px solid #14221b;padding:24px;border-radius:4px}h2{color:#dff5ed;text-align:center;font-size:16px;letter-spacing:1px}label{display:block;font-size:10px;color:#436659;margin:8px 0 4px}input,button{width:100%;background:#060908;border:1px solid #1a2e25;color:#9ab8af;padding:9px;font:11px 'Courier New',monospace}button{margin-top:12px;background:#0e241b;color:#2ecc71;border-color:#2ecc71;font-weight:bold;cursor:pointer}.links{display:flex;justify-content:space-between;margin-top:14px}.links a{color:#3498db}.error{color:#e74c3c;min-height:14px}.ok{color:#2ecc71}.hint{font-size:9px;color:#567d6e}
"""


@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    if get_logged_in_user(request):
        return RedirectResponse("/terminal", status_code=303)
    return HTMLResponse(LANDING_HTML)


@app.get("/login", response_class=HTMLResponse)
async def login_page(error: Optional[str] = None, msg: Optional[str] = None):
    error = html_lib.escape(error) if error else ''
    msg = html_lib.escape(msg) if msg else ''
    return HTMLResponse(f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>QUANTIFY. Login</title><style>{BASE_CSS}details{{margin-top:14px;font-size:10px;color:#567d6e}}details summary{{cursor:pointer;color:#3498db}}details form{{margin-top:8px}}details input{{margin-bottom:6px}}details button{{margin-top:4px;padding:6px}}</style></head><body><div class="card"><h2>QUANTIFY. ACCESS</h2><div class="error">{error}</div><div class="ok">{msg}</div><form action="/api/auth/login" method="post"><label>EMAIL</label><input type="email" name="email" required><label>PASSWORD</label><input type="password" name="password" required><button>LOGIN TO TERMINAL</button></form><div class="links"><a href="/signup">Sign up</a><a href="/forgot-password">Forgot password</a></div><details><summary>Didn't get a verification email?</summary><form action="/api/auth/resend-verification" method="post"><label>EMAIL</label><input type="email" name="email" required><button>RESEND VERIFICATION EMAIL</button></form></details></div></body></html>''')


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
        secure=False,
        samesite="lax",
        max_age=SESSION_TTL,
    )
    return res


@app.get("/signup", response_class=HTMLResponse)
async def signup_page(error: Optional[str] = None):
    error = html_lib.escape(error) if error else ''
    return HTMLResponse(f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>QUANTIFY. Sign Up</title><style>{BASE_CSS}</style></head><body><div class="card"><h2>SECURE REGISTER</h2><div class="error">{error}</div><p class="hint">16+ characters · upper/lowercase · special character · 4+ digits</p><form action="/api/auth/signup" method="post"><label>EMAIL</label><input type="email" name="email" required><label>PASSWORD</label><input type="password" name="password" required><button>CREATE ACCOUNT</button></form><div class="links"><a href="/login">Log in</a></div></div></body></html>''')


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
    return RedirectResponse("/login?msg=Account+created.+Check+your+email+to+verify+before+logging+in.",status_code=303)


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
    return HTMLResponse(f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Password Recovery</title><style>{BASE_CSS}</style></head><body><div class="card"><h2>PASSWORD RECOVERY</h2><div class="error">{error}</div><form action="/api/auth/send-code" method="post"><label>EMAIL</label><input type="email" name="email" required><button>SEND CODE</button></form><div class="links"><a href="/login">Log in</a></div></div></body></html>''')


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
    return HTMLResponse(f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Reset Password</title><style>{BASE_CSS}</style></head><body><div class="card"><h2>RESET CREDENTIALS</h2><div class="error">{error}</div><form action="/api/auth/verify-and-reset" method="post"><input type="hidden" name="email" value="{email}"><label>6-DIGIT CODE</label><input name="code" required maxlength="6"><label>NEW PASSWORD</label><input type="password" name="new_password" required><button>RESET PASSWORD</button></form></div></body></html>''')


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
    conn=db(); prefs=conn.execute("SELECT pref_theme,pref_language,pref_default_mode FROM users WHERE email=?",(user,)).fetchone(); conn.close()
    theme = prefs["pref_theme"] if prefs and prefs["pref_theme"] in ("dark","light") else "dark"
    pref_language = prefs["pref_language"] if prefs and prefs["pref_language"] in LANGUAGE_NAMES else "en"
    pref_default_mode = prefs["pref_default_mode"] if prefs and prefs["pref_default_mode"] else "Long-Term Momentum Pullback"
    user=html_lib.escape(user)
    return HTMLResponse(f'''<!doctype html><html lang="en" data-theme="{theme}"><head><meta charset="utf-8"><title>QUANTIFY. Terminal</title><script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script><style>
:root{{--bg:#050807;--panel:#030504;--panel2:#060908;--border:#14221b;--border2:#0c1712;--text:#9ab8af;--head:#dff5ed;--dim:#436659;--green:#2ecc71;--red:#e74c3c;--orange:#f39c12;--blue:#3498db;--grid-line:#0c1712}}
html[data-theme="light"]{{--bg:#f4f7f5;--panel:#ffffff;--panel2:#eef2ef;--border:#d7e0da;--border2:#e2e8e4;--text:#2c3e37;--head:#0d2318;--dim:#5c7266;--green:#1e8449;--red:#c0392b;--orange:#b9770e;--blue:#2471a3;--grid-line:#e2e8e4}}
*{{box-sizing:border-box}}body{{background:var(--bg);color:var(--text);font:11px 'Courier New',monospace;margin:0;padding:8px}}header,.panel{{background:var(--panel);border:1px solid var(--border)}}header{{padding:8px 14px;display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}}h1{{font-size:13px;color:var(--head);margin:0}}button,input,select{{background:var(--panel2);border:1px solid var(--border);color:var(--text);padding:5px 8px;font:11px 'Courier New',monospace}}button{{color:var(--green);border-color:var(--green);background:var(--panel2);cursor:pointer}}.grid{{display:grid;grid-template-columns:310px 1fr 340px;gap:8px;height:calc(100vh - 60px)}}.panel{{padding:10px;overflow:hidden;display:flex;flex-direction:column}}h3{{font-size:11px;color:var(--dim);border-bottom:1px solid var(--border);padding-bottom:5px;margin:0 0 8px}}.list,.scroll{{overflow:auto;flex:1}}.item{{padding:7px;border-bottom:1px solid var(--border2);cursor:pointer;display:flex;justify-content:space-between}}.item:hover{{background:var(--panel2)}}.chart{{flex:1;min-height:220px}}.idx-row{{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px}}.idx-box{{background:var(--panel2);border:1px solid var(--border);padding:6px}}.idx-label{{font-size:9px;color:var(--dim);margin-bottom:3px;display:flex;justify-content:space-between}}.idx-chart{{height:60px}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:8px}}.metric{{background:var(--panel2);border:1px solid var(--border);padding:7px;text-align:center}}.val{{color:var(--green);font-weight:bold;margin-top:3px}}.ratio{{height:6px;background:var(--border);margin-top:6px;display:flex}}.notice{{padding:8px;background:var(--panel2);border:1px solid var(--border);margin-bottom:8px;line-height:1.5}}a{{color:var(--blue)}}
.badge{{padding:2px 6px;border-radius:2px;font-weight:bold;display:inline-block}}.badge-ok{{background:var(--panel2);color:var(--green);border:1px solid var(--green)}}.badge-warn{{background:var(--panel2);color:var(--orange);border:1px solid var(--orange)}}.badge-danger{{background:var(--panel2);color:var(--red);border:1px solid var(--red)}}.section{{margin-bottom:10px}}.section b{{color:var(--head);display:block;margin-bottom:2px}}
</style></head><body><header><h1>QUANTIFY. TERMINAL · <span style="color:#2ecc71">{user}</span></h1><div><select id="mode"><option>Long-Term Momentum Pullback</option><option>Short-Term Volatility Breakout</option><option>Institutional Flow Leaders</option></select><button onclick="runBatch()">⚡ RESCAN</button><button onclick="location='/portfolio'">PORTFOLIO</button><button onclick="location='/settings'">SETTINGS</button><button onclick="location='/logout'" style="color:#e74c3c;border-color:#e74c3c">LOGOUT</button></div></header><div class="grid"><section class="panel"><h3>MARKET UNIVERSE SCANNER <span id="ucount"></span></h3><input id="tickerInput" placeholder="Ticker (e.g. TSLA)" onkeydown="if(event.key==='Enter')loadTicker(this.value)"><div class="list" id="list">Preparing constituent list...</div></section><section class="panel"><div style="display:flex;justify-content:space-between;align-items:center"><h3 id="title" style="border:0">AAPL · TECHNICAL CHART</h3><div><input id="target" type="number" placeholder="Target $" style="width:90px"><button onclick="setAlert()">🔔</button><button onclick="savePortfolio()">💾 SAVE</button><button onclick="changeTF('1h')">1H</button><button onclick="changeTF('1d')">1D</button><button onclick="changeTF('1wk')">1W</button><button onclick="changeTF('1mo')">1M</button></div></div><div id="chart" class="chart"></div><div class="idx-row"><div class="idx-box"><div class="idx-label"><span>S&amp;P 500 · 60D</span><span id="idx-sp500-val"></span></div><div id="idx-sp500" class="idx-chart"></div></div><div class="idx-box"><div class="idx-label"><span>NASDAQ-100 · 60D</span><span id="idx-ndx-val"></span></div><div id="idx-ndx" class="idx-chart"></div></div></div><div class="metrics"><div class="metric">RSI / MACD<div id="rsi" class="val">-</div></div><div class="metric">52W HIGH<div id="high52" class="val">-</div></div><div class="metric">52W LOW<div id="low52" class="val">-</div></div><div class="metric">TREND<div id="trend" class="val">-</div></div></div><div class="notice">Short interest: <b id="short">No data</b><div class="ratio"><div id="longbar" style="background:#2ecc71"></div><div id="shortbar" style="background:#e74c3c"></div></div></div></section><section class="panel"><h3>AI QUANT REPORT <small style="color:#436659">(for informational purposes only, not investment advice)</small></h3><div id="verdict" style="display:none;margin-bottom:8px"></div><div id="ai" class="scroll">Loading AI analysis based on real data...</div><h3 style="margin-top:10px">NEWS</h3><div id="news" class="scroll">Waiting for news...</div></section></div><script>
const USER_LANGUAGE='{pref_language}';
document.getElementById('mode').value='{pref_default_mode}';
let ticker='AAPL',tf='1d',chart,candle,volume,smaLines={{}},idxCharts={{}};
function init(){{const c=document.getElementById('chart');chart=LightweightCharts.createChart(c,{{width:c.clientWidth,height:c.clientHeight,layout:{{background:{{type:'solid',color:'#030504'}},textColor:'#9ab8af'}},grid:{{vertLines:{{color:'#0c1712'}},horzLines:{{color:'#0c1712'}}}},timeScale:{{timeVisible:true}}}});candle=chart.addCandlestickSeries({{upColor:'#2ecc71',downColor:'#e74c3c',borderUpColor:'#2ecc71',borderDownColor:'#e74c3c',wickUpColor:'#2ecc71',wickDownColor:'#e74c3c'}});volume=chart.addHistogramSeries({{priceFormat:{{type:'volume'}},priceScaleId:''}});volume.priceScale().applyOptions({{scaleMargins:{{top:.8,bottom:0}}}});smaLines.sma20=chart.addLineSeries({{color:'#3498db',lineWidth:1,priceLineVisible:false,lastValueVisible:false}});smaLines.sma50=chart.addLineSeries({{color:'#f39c12',lineWidth:1,priceLineVisible:false,lastValueVisible:false}});smaLines.sma200=chart.addLineSeries({{color:'#e74c3c',lineWidth:1,priceLineVisible:false,lastValueVisible:false}});window.onresize=()=>{{chart.resize(c.clientWidth,c.clientHeight);Object.entries(idxCharts).forEach(([k,ic])=>{{const el=document.getElementById('idx-'+k);if(el)ic.resize(el.clientWidth,el.clientHeight)}})}};['sp500','ndx'].forEach(k=>{{const el=document.getElementById('idx-'+k);const ic=LightweightCharts.createChart(el,{{width:el.clientWidth,height:el.clientHeight,layout:{{background:{{type:'solid',color:'#030504'}},textColor:'#9ab8af',fontSize:9}},grid:{{vertLines:{{visible:false}},horzLines:{{visible:false}}}},rightPriceScale:{{visible:false}},timeScale:{{visible:false}},handleScroll:false,handleScale:false}});idxCharts[k]=ic;idxCharts[k+'_line']=ic.addLineSeries({{color:'#2ecc71',lineWidth:1.5,priceLineVisible:false,lastValueVisible:false}})}})}}
async function loadIndices(){{try{{const r=await fetch('/api/market-indices');const d=await r.json();const map={{sp500:d.sp500,ndx:d.nasdaq100}};Object.entries(map).forEach(([k,series])=>{{if(!series?.length)return;idxCharts[k+'_line'].setData(series.map(p=>({{time:p.time,value:p.close}})));idxCharts[k].timeScale().fitContent();const first=series[0].close,last=series[series.length-1].close;const chg=((last/first-1)*100).toFixed(2);const el=document.getElementById('idx-'+k+'-val');if(el)el.innerHTML=`${{last}} <span style="color:${{chg>=0?'#2ecc71':'#e74c3c'}}">${{chg>=0?'+':''}}${{chg}}%</span>`}})}}catch(e){{console.warn('index load failed',e)}}}}
function verdictClass(v){{return v==='Favorable'?'badge-ok':v==='Caution'?'badge-warn':v==='Risk'?'badge-danger':''}}
async function autoScanOnOpen(){{try{{await fetch('/api/auto-scan',{{method:'POST'}});}}catch(e){{console.warn('auto-scan trigger failed',e)}};scan()}}
async function scan(){{const r=await fetch('/api/scan');const d=await r.json();document.getElementById('ucount').innerText=d.universe_count?` · ${{d.quant_pass_count??0}} detected / ${{d.universe_count}} symbols`:'';if(!d.signals?.length){{const ready=d.universe_status?.ready;const err=d.universe_status?.error;const scanned=d.scanned_count>0;document.getElementById('list').innerHTML='<div class="notice">'+(scanned?'Scan complete — no tickers cleared the quant threshold today.':(ready?'No scan data saved for today. Run RESCAN.':(err?'Could not prepare constituent data. Try RESCAN again.':'Preparing S&P 500 / Nasdaq-100 constituents...')))+'</div>';return}}document.getElementById('list').innerHTML=d.signals.map(s=>`<div class="item" onclick="loadTicker('${{s.ticker}}')"><b>${{s.ticker}}</b><span>${{s.price}} · ${{s.change_pct}}%<br><small>${{s.universe||''}} · Alpha ${{s.alpha_score}}${{s.timing_verdict?' · <span class="badge '+verdictClass(s.timing_verdict)+'">'+s.timing_verdict+'</span>':''}}</small></span></div>`).join('');loadTicker(d.signals[0].ticker)}}
async function runBatch(){{document.getElementById('list').innerHTML='Checking constituents and preparing real market data...';const r=await fetch('/api/admin/run-batch');const d=await r.json();if(!r.ok){{document.getElementById('list').innerHTML='<div class="notice">'+(d.message||'Could not start the batch.')+'</div>';return}}pollBatch()}} async function pollBatch(){{const r=await fetch('/api/admin/batch-status');const d=await r.json();const a=await fetch('/api/admin/ai-status').then(x=>x.json());document.getElementById('list').innerHTML=`<div class="notice">MARKET: ${{d.processed||0}} / ${{d.total||0}}<br>Saved: ${{d.saved||0}}<br>AI CACHE(QUANT PASS): ${{a.processed||0}} / ${{a.total||0}} · READY ${{a.ready||0}}<br>${{d.running?'Collecting market data...':(a.running&&a.total>0?'Generating AI cache for quant-passing tickers...':'Cache ready')}}</div>`;if(d.running||(a.running&&a.total>0))setTimeout(pollBatch,1500);else if(d.error){{return}}else scan()}}
async function loadTicker(t){{ticker=t.toUpperCase().trim();document.getElementById('title').innerText=ticker+' · TECHNICAL CHART';const r=await fetch(`/api/terminal-data-fast?ticker=${{encodeURIComponent(ticker)}}&timeframe=${{tf}}`);const d=await r.json();if(!d.fast?.data_ok){{document.getElementById('rsi').innerText=d.fast?.error||'No data';return}}const cd=d.fast.chart.map(x=>({{time:x.time,open:x.open,high:x.high,low:x.low,close:x.close}}));const vd=d.fast.chart.map(x=>({{time:x.time,value:x.volume}}));candle.setData(cd);volume.setData(vd);['sma20','sma50','sma200'].forEach(k=>{{const pts=d.fast.chart.filter(x=>x[k]!=null).map(x=>({{time:x.time,value:x[k]}}));smaLines[k].setData(pts)}});chart.timeScale().fitContent();document.getElementById('rsi').innerText=`RSI ${{d.fast.rsi}} / MACD ${{d.fast.macd}}`;document.getElementById('high52').innerText=d.fast.pct_from_52w_high==null?'N/A':d.fast.pct_from_52w_high+'%';document.getElementById('low52').innerText=d.fast.pct_from_52w_low==null?'N/A':d.fast.pct_from_52w_low+'%';document.getElementById('trend').innerText=d.fast.above_200d_sma==null?'N/A':(d.fast.above_200d_sma?'Uptrend':'Downtrend');document.getElementById('short').innerText=d.fast.short_percent==null?'No data':d.fast.short_percent+'%';document.getElementById('longbar').style.width=(d.fast.long_ratio??0)+'%';document.getElementById('shortbar').style.width=(d.fast.short_ratio??0)+'%';const a=await fetch(`/api/terminal-data-ai?ticker=${{encodeURIComponent(ticker)}}&mode=${{encodeURIComponent(document.getElementById('mode').value)}}&language=${{USER_LANGUAGE}}`);const x=await a.json();const vEl=document.getElementById('verdict');if(x.ai?.timing_verdict){{vEl.style.display='block';vEl.innerHTML=`<span class="badge ${{verdictClass(x.ai.timing_verdict)}}">${{x.ai.timing_verdict}}</span> Timing score ${{x.ai.timing_score??'-'}} / 100`}}else{{vEl.style.display='none'}}const sec=x.ai?.report_sections;const aiEl=document.getElementById('ai');if(sec){{const labels={{quant_review:'Quant Review',supply_demand:'Supply/Demand',risk_review:'Risk Review',timing_reason:'Timing Rationale'}};aiEl.innerHTML=Object.keys(labels).filter(k=>sec[k]).map(k=>`<div class="section"><b>${{labels[k]}}</b>${{sec[k]}}</div>`).join('')}}else{{aiEl.innerText=x.ai?.ai_report||(x.ai?.status==='PENDING'||x.ai?.status==='RUNNING'?'Preparing AI analysis cache on the server...':'AI analysis is unavailable.')}}const news=x.ai?.news;if(!news)document.getElementById('news').innerText='Could not fetch a live news feed.';else document.getElementById('news').innerHTML=news.map(n=>`<div style="margin-bottom:8px"><a href="${{n.url}}" target="_blank" rel="noopener">${{n.title}}</a><br><small>${{n.published||''}}</small></div>`).join('')}}
async function setAlert(){{const p=Number(document.getElementById('target').value);if(!(p>0))return alert('Enter a target price.');const f=new FormData();f.append('ticker',ticker);f.append('target_price',p);const r=await fetch('/api/alerts/set',{{method:'POST',body:f}});const d=await r.json();alert(d.message||d.error)}}async function savePortfolio(){{const f=new FormData();f.append('ticker',ticker);const r=await fetch('/api/portfolio/save',{{method:'POST',body:f}});const d=await r.json();alert(d.message||d.error)}}function changeTF(x){{tf=x;loadTicker(ticker)}}window.onload=()=>{{init();autoScanOnOpen();loadIndices()}};
</script></body></html>''')


@app.get("/portfolio", response_class=HTMLResponse)
async def portfolio_page(request: Request):
    user = get_logged_in_user(request)
    if not user: return RedirectResponse("/login", status_code=303)
    user = html_lib.escape(user)
    return HTMLResponse(f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>QUANTIFY. Portfolio</title><style>
*{{box-sizing:border-box}}body{{background:#050807;color:#9ab8af;font:11px 'Courier New',monospace;margin:0;padding:8px}}header{{background:#030504;border:1px solid #14221b;padding:8px 14px;display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}}h1{{font-size:13px;color:#dff5ed;margin:0}}a.back{{color:#3498db;text-decoration:none}}button{{background:#0e241b;border:1px solid #2ecc71;color:#2ecc71;padding:5px 10px;font:11px 'Courier New',monospace;cursor:pointer}}.wrap{{max-width:900px;margin:0 auto}}.item{{background:#030504;border:1px solid #14221b;padding:14px;margin-bottom:10px}}.item-head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}}.item-head b{{color:#dff5ed;font-size:14px}}.meta{{color:#436659;font-size:10px;margin-bottom:8px}}.badge{{padding:2px 7px;border-radius:2px;font-weight:bold;font-size:10px}}.badge-ok{{background:#0e241b;color:#2ecc71;border:1px solid #2ecc71}}.badge-warn{{background:#2a2008;color:#f39c12;border:1px solid #f39c12}}.badge-danger{{background:#2a0e0e;color:#e74c3c;border:1px solid #e74c3c}}.note{{color:#9ab8af;font-size:11px;line-height:1.6}}.remove{{background:transparent;border:1px solid #2a0e0e;color:#e74c3c}}.empty{{color:#436659;padding:40px;text-align:center}}
</style></head><body><header><h1>QUANTIFY. PORTFOLIO · <span style="color:#2ecc71">{user}</span></h1><a class="back" href="/terminal">&larr; Back to Terminal</a></header>
<div class="wrap" id="list">Loading...</div>
<script>
function badgeClass(v){{return v==='Favorable'?'badge-ok':v==='Caution'?'badge-warn':v==='Risk'?'badge-danger':''}}
async function load(){{const r=await fetch('/api/portfolio');const d=await r.json();const el=document.getElementById('list');if(!d.items?.length){{el.innerHTML='<div class="empty">Nothing saved yet. Open a ticker in the Terminal and click 💾 SAVE.</div>';return}}el.innerHTML=d.items.map(it=>{{const sec=it.ai_report&&typeof it.ai_report==='object'?it.ai_report:null;const summary=sec?.quant_review||(typeof it.ai_report==='string'?it.ai_report:'');const date=new Date(it.saved_at*1000).toLocaleString();return `<div class="item"><div class="item-head"><b>${{it.ticker}}</b><div>${{it.timing_verdict?`<span class="badge ${{badgeClass(it.timing_verdict)}}">${{it.timing_verdict}}</span> `:''}}<button class="remove" onclick="remove(${{it.id}})">Remove</button></div></div><div class="meta">Saved ${{date}} · Scan date ${{it.scan_date}} · Price $${{it.price}} (${{it.change_pct}}%) · Alpha ${{it.alpha_score}} · RSI ${{it.rsi}}${{it.timing_score!=null?' · Timing score '+it.timing_score+'/100':''}}</div>${{summary?`<div class="note">${{summary}}</div>`:''}}</div>`}}).join('')}}
async function remove(id){{const f=new FormData();f.append('id',id);await fetch('/api/portfolio/remove',{{method:'POST',body:f}});load()}}
load();
</script></body></html>''')


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    user = get_logged_in_user(request)
    if not user: return RedirectResponse("/login", status_code=303)
    user = html_lib.escape(user)
    return HTMLResponse(f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>QUANTIFY. Settings</title><style>
*{{box-sizing:border-box}}body{{background:#050807;color:#9ab8af;font:11px 'Courier New',monospace;margin:0;padding:8px}}header{{background:#030504;border:1px solid #14221b;padding:8px 14px;display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}}h1{{font-size:13px;color:#dff5ed;margin:0}}a.back{{color:#3498db;text-decoration:none}}.wrap{{max-width:520px;margin:0 auto}}.card{{background:#030504;border:1px solid #14221b;padding:20px;margin-bottom:16px}}.card h2{{color:#dff5ed;font-size:12px;margin:0 0 14px;border-bottom:1px solid #14221b;padding-bottom:8px}}label{{display:block;font-size:10px;color:#436659;margin:10px 0 4px}}select,input{{width:100%;background:#060908;border:1px solid #1a2e25;color:#9ab8af;padding:8px;font:11px 'Courier New',monospace}}button{{margin-top:14px;background:#0e241b;border:1px solid #2ecc71;color:#2ecc71;padding:8px 14px;font:11px 'Courier New',monospace;font-weight:bold;cursor:pointer;width:100%}}.msg{{font-size:11px;min-height:14px;margin-top:8px}}.ok{{color:#2ecc71}}.err{{color:#e74c3c}}
</style></head><body><header><h1>QUANTIFY. SETTINGS · <span style="color:#2ecc71">{user}</span></h1><a class="back" href="/terminal">&larr; Back to Terminal</a></header>
<div class="wrap">
<div class="card"><h2>DISPLAY</h2>
<label>THEME</label><select id="theme"><option value="dark">Dark</option><option value="light">Light</option></select>
<label>AI REPORT LANGUAGE</label><select id="language"><option value="en">English</option><option value="ko">Korean (한국어)</option></select>
<label>DEFAULT STRATEGY MODE</label><select id="default_mode"><option>Long-Term Momentum Pullback</option><option>Short-Term Volatility Breakout</option><option>Institutional Flow Leaders</option></select>
<button onclick="saveSettings()">SAVE SETTINGS</button><div class="msg" id="settings-msg"></div>
</div>
<div class="card"><h2>CHANGE PASSWORD</h2>
<label>CURRENT PASSWORD</label><input type="password" id="current_password">
<label>NEW PASSWORD</label><input type="password" id="new_password">
<p style="font-size:9px;color:#567d6e;margin-top:6px">16+ characters · upper/lowercase · special character · 4+ digits</p>
<button onclick="changePassword()">CHANGE PASSWORD</button><div class="msg" id="password-msg"></div>
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
