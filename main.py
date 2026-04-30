#!/usr/bin/env python3
import logging
import schedule
import time
from datetime import date
from config import CHECK_INTERVAL, LOG_PATH
from bot.state import init_db, get_bet_summary
from bot.trader import TennisTrader

# ------------------------------------------------------------------
# Logging — writes to both file and stdout
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("main")


def daily_summary(trader: TennisTrader):
    log.info("=== DAILY SUMMARY ===")
    rows = get_bet_summary()
    for r in rows[:7]:
        log.info("  %s  bets=%-3d wagered=$%.2f", r["date"], r["bets"], r["wagered"])
    log.info("Kalshi balance: $%.2f", trader.kalshi.get_balance())
    log.info("=====================")


def main():
    log.info("Tennis prediction bot starting up")
    init_db()

    trader = TennisTrader()
    trader.initialize()

    # run once immediately on startup
    trader.run_cycle()

    # check for new markets every CHECK_INTERVAL minutes
    schedule.every(CHECK_INTERVAL).minutes.do(trader.run_cycle)

    # refresh Elo data daily at 6am to pick up overnight results
    schedule.every().day.at("06:00").do(trader.refresh_data)

    # daily summary at 9pm
    schedule.every().day.at("21:00").do(daily_summary, trader)

    log.info("Scheduler running — checking markets every %d minutes", CHECK_INTERVAL)

    while True:
        try:
            schedule.run_pending()
        except KeyboardInterrupt:
            log.info("Shutting down")
            break
        except Exception as e:
            log.error("Unexpected error in main loop: %s", e, exc_info=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
