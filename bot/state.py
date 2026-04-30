import sqlite3
import logging
from datetime import date
from config import DB_PATH

log = logging.getLogger(__name__)


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                title TEXT,
                player_yes TEXT,
                player_no TEXT,
                side TEXT NOT NULL,
                contracts INTEGER NOT NULL,
                price_cents INTEGER NOT NULL,
                amount_usd REAL NOT NULL,
                model_prob REAL,
                market_prob REAL,
                edge REAL,
                kalshi_order_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS daily_pnl (
                date TEXT PRIMARY KEY,
                realized_pnl REAL DEFAULT 0,
                bets_placed INTEGER DEFAULT 0
            )
        """)


def already_bet(ticker: str) -> bool:
    with _conn() as c:
        row = c.execute("SELECT id FROM bets WHERE ticker=?", (ticker,)).fetchone()
        return row is not None


def record_bet(
    ticker: str,
    title: str,
    player_yes: str,
    player_no: str,
    side: str,
    contracts: int,
    price_cents: int,
    amount_usd: float,
    model_prob: float,
    market_prob: float,
    edge: float,
    kalshi_order_id: str,
):
    today = date.today().isoformat()
    with _conn() as c:
        c.execute("""
            INSERT INTO bets (date, ticker, title, player_yes, player_no, side,
                contracts, price_cents, amount_usd, model_prob, market_prob, edge, kalshi_order_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (today, ticker, title, player_yes, player_no, side,
              contracts, price_cents, amount_usd, model_prob, market_prob, edge, kalshi_order_id))
        c.execute("""
            INSERT INTO daily_pnl (date, realized_pnl, bets_placed) VALUES (?, 0, 1)
            ON CONFLICT(date) DO UPDATE SET bets_placed = bets_placed + 1
        """, (today,))


def get_daily_spent(today: str | None = None) -> float:
    """Sum of amount_usd wagered today (cost basis, not potential loss)."""
    today = today or date.today().isoformat()
    with _conn() as c:
        row = c.execute(
            "SELECT COALESCE(SUM(amount_usd), 0) as total FROM bets WHERE date=?", (today,)
        ).fetchone()
        return row["total"]


def get_bet_summary() -> list[dict]:
    with _conn() as c:
        rows = c.execute("""
            SELECT date, COUNT(*) as bets, SUM(amount_usd) as wagered
            FROM bets GROUP BY date ORDER BY date DESC LIMIT 14
        """).fetchall()
        return [dict(r) for r in rows]
