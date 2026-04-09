"""
MLB Arber — Mobile Trade Server

FastAPI backend for the phone app. Exposes:
  GET  /api/players   — today's players who have Kalshi markets
  GET  /api/balance   — current Kalshi balance
  POST /api/trade     — sweep the market for a player + trade type

Run:
  py app.py

Then expose via ngrok:
  ngrok http 8000

Open the ngrok URL on your phone.
"""

from __future__ import annotations

import asyncio
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
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from kalshi.client import KalshiClient
from kalshi.props import PropCache


log = logging.getLogger("app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")


# ── Config + clients (loaded at startup) ──────────────────────────────────────

def load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)

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

    # Refresh cache every 30 minutes
    async def refresh_loop():
        while True:
            await asyncio.sleep(1800)
            await cache.build(kalshi)
            log.info("Cache refreshed")
    asyncio.create_task(refresh_loop())

    yield
    await kalshi.close()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Serve the mobile PWA ───────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ── API endpoints ──────────────────────────────────────────────────────────────

@app.get("/api/players")
async def get_players():
    """
    Return players who have active Kalshi markets today.
    Sorted alphabetically. Each entry shows which market types they have.
    """
    players: dict[str, dict] = {}

    for name, prop in cache.hr_cache.items():
        if name not in players:
            players[name] = {"name": name.title(), "has_hr": False, "has_hit": False, "hr_ask": None, "hit_ask": None}
        players[name]["has_hr"] = True
        players[name]["hr_ask"] = prop.yes_ask

    for key, prop in cache.hits_cache.items():
        name = key.split(":")[0]
        threshold = int(key.split(":")[1])
        if threshold != 1:
            continue
        if name not in players:
            players[name] = {"name": name.title(), "has_hr": False, "has_hit": False, "hr_ask": None, "hit_ask": None}
        players[name]["has_hit"] = True
        players[name]["hit_ask"] = prop.yes_ask

    sorted_players = sorted(players.values(), key=lambda x: x["name"])
    return {"players": sorted_players, "count": len(sorted_players)}


@app.get("/api/balance")
async def get_balance():
    bal = await kalshi.get_balance()
    return {"balance_usd": round(bal, 2)}


class TradeRequest(BaseModel):
    player_name: str   # lowercase, as stored in cache
    trade_type: str    # "home_run" or "hit"


class TradeResult(BaseModel):
    success: bool
    player: str
    trade_type: str
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
    start = time.monotonic()
    player_lower = req.player_name.lower().strip()
    mode = cfg.get("mode", "paper")
    paper = (mode != "live")

    # Look up market
    if req.trade_type == "home_run":
        prop = cache.get_hr_ticker(player_lower)
        max_buy = cfg.get("max_buy_cents", 96)
        sell_at = cfg.get("sell_cents", 99)
        max_bet = float(cfg.get("max_hr_bet", 5.0))
    elif req.trade_type == "hit":
        prop = cache.get_hit_ticker(player_lower, 1)
        max_buy = cfg.get("max_buy_cents", 96)
        sell_at = 99
        max_bet = float(cfg.get("max_hits_bet", 5.0))
    else:
        raise HTTPException(400, f"Unknown trade_type: {req.trade_type}")

    if prop is None:
        raise HTTPException(404, f"No Kalshi market found for {req.player_name} ({req.trade_type})")

    # Get current ask
    result = await kalshi.get_yes_ask(prop.ticker, max_cents=max_buy)
    if result is None:
        raise HTTPException(409, f"Market already repriced above {max_buy}¢ — no edge remaining")

    yes_cents, qty = result
    balance = await kalshi.get_balance() if not paper else max_bet
    spend  = min(balance, max_bet)
    count  = min(qty, int(spend / (yes_cents / 100)))

    if count < 1:
        raise HTTPException(409, "Insufficient balance or no contracts available")

    cost     = count * yes_cents / 100
    expected = count * (sell_at - yes_cents) / 100

    log.info(f"TRADE: {req.player_name} {req.trade_type} | {prop.ticker} | {count}x{yes_cents}¢ | paper={paper}")

    # Place buy order
    await kalshi.place_order(prop.ticker, "yes", yes_cents, count, "limit")

    # Queue flip (HR: sell at sell_cents; hits: hold to settlement at 99c IOC)
    if req.trade_type == "home_run":
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

    latency_ms = (time.monotonic() - start) * 1000

    return TradeResult(
        success=True,
        player=req.player_name.title(),
        trade_type=req.trade_type,
        ticker=prop.ticker,
        contracts=count,
        avg_price_cents=yes_cents,
        cost_usd=round(cost, 2),
        expected_pnl_usd=round(expected, 2),
        latency_ms=round(latency_ms, 0),
        paper=paper,
        message=f"{'[PAPER] ' if paper else ''}Bought {count} @ {yes_cents}¢, selling at {sell_at}¢",
    )


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
