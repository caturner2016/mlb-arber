"""
MLB Arber — Mobile Trade Server

FastAPI backend for the phone app. Exposes:
  GET  /api/players        — today's players who have Kalshi markets
  GET  /api/balance        — current Kalshi balance
  GET  /api/metrics        — today's P&L split by home vs game profile
  POST /api/trade          — sweep the market for a player + trade type

Profiles (set in config.yaml):
  home  — buy up to 98¢, for watching at home (market partially repriced)
  game  — buy up to 96¢, for being at the stadium (you see it first)

Run:
  py app.py

Then expose via ngrok:
  ngrok http 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from kalshi.client import KalshiClient
from kalshi.props import PropCache


log = logging.getLogger("app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

METRICS_FILE = "metrics.json"


# ── Metrics tracking ──────────────────────────────────────────────────────────

def load_metrics() -> dict:
    today = str(date.today())
    if Path(METRICS_FILE).exists():
        try:
            data = json.loads(Path(METRICS_FILE).read_text())
            if data.get("date") == today:
                return data
        except Exception:
            pass
    return {
        "date": today,
        "home":  {"trades": 0, "cost": 0.0, "expected_pnl": 0.0},
        "game":  {"trades": 0, "cost": 0.0, "expected_pnl": 0.0},
    }

def save_metrics(m: dict) -> None:
    Path(METRICS_FILE).write_text(json.dumps(m, indent=2))

def record_trade(profile: str, cost: float, expected_pnl: float) -> None:
    m = load_metrics()
    m[profile]["trades"]       += 1
    m[profile]["cost"]         += cost
    m[profile]["expected_pnl"] += expected_pnl
    save_metrics(m)


# ── Config + clients ──────────────────────────────────────────────────────────

def load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def profile_cfg(cfg: dict, profile: str) -> dict:
    """Return the home or game sub-config, falling back to top-level keys."""
    p = cfg.get(profile, {})
    return {
        "max_buy_cents": p.get("max_buy_cents", cfg.get("max_buy_cents", 96)),
        "sell_cents":    p.get("sell_cents",    cfg.get("sell_cents",    99)),
    }

cfg: dict = {}
kalshi: KalshiClient | None = None
cache: PropCache | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global cfg, kalshi, cache
    cfg = load_config()
    kalshi = KalshiClient(
        key_id=cfg.get("kalshi_key_id", ""),
        private_key_path=cfg.get("kalshi_private_key_path", ""),
        paper_mode=(cfg.get("mode", "paper") != "live"),
    )
    cache = PropCache()
    await cache.build(kalshi)
    log.info(f"Cache loaded: {len(cache.hr_cache)} HR, {len(cache.hits_cache)} hits markets")

    async def refresh_loop():
        while True:
            await asyncio.sleep(300)
            await cache.build(kalshi)
            log.info("Cache refreshed")
    asyncio.create_task(refresh_loop())

    yield
    await kalshi.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Static ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ── API ────────────────────────────────────────────────────────────────────────

@app.get("/api/players")
async def get_players():
    players: dict[str, dict] = {}

    for name, prop in cache.hr_cache.items():
        if name not in players:
            players[name] = {"name": name.title(), "has_hr": False, "has_hit": False,
                             "hr_ask": None, "hit_ask": None}
        players[name]["has_hr"]  = True
        players[name]["hr_ask"]  = prop.yes_ask

    for key, prop in cache.hits_cache.items():
        name      = key.split(":")[0]
        threshold = int(key.split(":")[1])
        if threshold != 1:
            continue
        if name not in players:
            players[name] = {"name": name.title(), "has_hr": False, "has_hit": False,
                             "hr_ask": None, "hit_ask": None}
        players[name]["has_hit"]  = True
        players[name]["hit_ask"]  = prop.yes_ask

    return {"players": sorted(players.values(), key=lambda x: x["name"]),
            "count": len(players)}


@app.post("/api/refresh")
async def refresh_cache():
    """Force rebuild the player cache."""
    await cache.build(kalshi)
    return {"hr_markets": len(cache.hr_cache), "hits_markets": len(cache.hits_cache)}


@app.get("/api/debug")
async def debug():
    """Shows what's in the cache and tries a raw Kalshi market search."""
    hr_sample   = list(cache.hr_cache.items())[:5]
    hits_sample = list(cache.hits_cache.items())[:5]

    # Try a raw search for any MLB markets
    raw = {}
    try:
        raw = await kalshi.get_markets_search("MLB")
    except Exception as e:
        raw = {"error": str(e)}

    return {
        "hr_markets":   len(cache.hr_cache),
        "hits_markets": len(cache.hits_cache),
        "hr_sample":    [{"name": k, "ticker": v.ticker, "ask": v.yes_ask} for k, v in hr_sample],
        "hits_sample":  [{"key": k, "ticker": v.ticker, "ask": v.yes_ask} for k, v in hits_sample],
        "raw_search":   raw,
    }


@app.get("/api/balance")
async def get_balance():
    bal = await kalshi.get_balance()
    return {"balance_usd": round(bal, 2)}


@app.get("/api/metrics")
async def get_metrics():
    return load_metrics()


class TradeRequest(BaseModel):
    player_name: str
    trade_type:  str        # "home_run" or "hit"
    threshold:   int = 1    # hits: 1, 2, or 3
    profile:     str = "home"  # "home" or "game"


class TradeResult(BaseModel):
    success: bool
    player: str
    trade_type: str
    profile: str
    ticker: str
    contracts: int
    avg_price_cents: int
    cost_usd: float
    expected_pnl_usd: float
    latency_ms: float
    paper: bool
    message: str


@app.post("/api/trade", response_model=TradeResult)
async def place_trade(req: TradeRequest):
    start        = time.monotonic()
    player_lower = req.player_name.lower().strip()
    paper        = cfg.get("mode", "paper") != "live"
    profile      = req.profile if req.profile in ("home", "game") else "home"
    pcfg         = profile_cfg(cfg, profile)

    if req.trade_type == "home_run":
        prop    = cache.get_hr_ticker(player_lower)
        max_buy = pcfg["max_buy_cents"]
        sell_at = pcfg["sell_cents"]
    elif req.trade_type == "hit":
        threshold = max(1, min(3, req.threshold))
        prop      = cache.get_hit_ticker(player_lower, threshold)
        max_buy   = pcfg["max_buy_cents"]
        sell_at   = 99
    else:
        raise HTTPException(400, f"Unknown trade_type: {req.trade_type}")

    if prop is None:
        raise HTTPException(404, f"No Kalshi market for {req.player_name} ({req.trade_type})")

    result = await kalshi.get_yes_ask(prop.ticker, max_cents=max_buy)
    if result is None:
        raise HTTPException(409, f"Market above {max_buy}¢ — no edge remaining")

    yes_cents, qty = result
    balance = await kalshi.get_balance() if not paper else 1000.0
    count   = min(qty, int(balance / (yes_cents / 100)))

    if count < 1:
        raise HTTPException(409, "Insufficient balance or no contracts available")

    cost     = count * yes_cents / 100
    settle   = 100 if profile == "home" else sell_at
    expected = count * (settle - yes_cents) / 100

    log.info(f"[{profile.upper()}] {req.player_name} {req.trade_type} | {prop.ticker} | "
             f"{count}x{yes_cents}¢ | paper={paper}")

    await kalshi.place_order(prop.ticker, "yes", yes_cents, count, "limit")

    # Game profile: flip at sell_cents. Home profile: hold to $1.00 settlement.
    if profile == "game" and req.trade_type == "home_run":
        async def flip():
            deadline = asyncio.get_event_loop().time() + 300
            while asyncio.get_event_loop().time() < deadline:
                try:
                    await kalshi.place_order(prop.ticker, "yes", sell_at, count, "limit")
                    log.info(f"SOLD {prop.ticker} @ {sell_at}¢")
                    return
                except Exception as e:
                    log.warning(f"Sell retry: {e}")
                    await asyncio.sleep(3)
        asyncio.create_task(flip())

    record_trade(profile, cost, expected)
    latency_ms = (time.monotonic() - start) * 1000

    return TradeResult(
        success=True,
        player=req.player_name.title(),
        trade_type=req.trade_type,
        profile=profile,
        ticker=prop.ticker,
        contracts=count,
        avg_price_cents=yes_cents,
        cost_usd=round(cost, 2),
        expected_pnl_usd=round(expected, 2),
        latency_ms=round(latency_ms, 0),
        paper=paper,
        message=f"{'[PAPER] ' if paper else ''}[{profile.upper()}] {count} @ {yes_cents}¢ → {sell_at}¢",
    )


@app.post("/api/test-trade")
async def test_trade(req: TradeRequest):
    """Place exactly 1 contract to verify the full Kalshi pipeline."""
    paper      = cfg.get("mode", "paper") != "live"
    player_lower = req.player_name.lower().strip()

    if req.trade_type == "home_run":
        prop = cache.get_hr_ticker(player_lower)
    elif req.trade_type == "hit":
        prop = cache.get_hit_ticker(player_lower, 1)
    else:
        raise HTTPException(400, f"Unknown trade_type: {req.trade_type}")

    if prop is None:
        raise HTTPException(404, f"No Kalshi market for {req.player_name} ({req.trade_type})")

    result = await kalshi.get_yes_ask(prop.ticker, max_cents=99)
    if result is None:
        raise HTTPException(409, "Market has no ask ≤ 99¢")

    yes_cents, _ = result
    count = 1  # exactly 1 contract regardless of config

    log.info(f"[TEST] {req.player_name} {req.trade_type} | {prop.ticker} | 1x{yes_cents}¢ | paper={paper}")

    await kalshi.place_order(prop.ticker, "yes", yes_cents, count, "limit")

    cost     = yes_cents / 100
    expected = (100 - yes_cents) / 100

    record_trade(req.profile or "game", cost, expected)

    return TradeResult(
        success=True,
        player=req.player_name.title(),
        trade_type=req.trade_type,
        profile="test",
        ticker=prop.ticker,
        contracts=1,
        avg_price_cents=yes_cents,
        cost_usd=round(cost, 2),
        expected_pnl_usd=round(expected, 2),
        latency_ms=0,
        paper=paper,
        message=f"{'[PAPER] ' if paper else ''}[TEST] 1 contract @ {yes_cents}¢ on {prop.ticker}",
    )


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
