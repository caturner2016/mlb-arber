#!/usr/bin/env python3
"""
HiBid → eBay Resell Scout

Searches HiBid auctions for clothing items, looks up what the same items
actually sold for on eBay, and ranks opportunities by estimated profit.

Usage:
  python hibid_scout.py "levi 501 jeans"
  python hibid_scout.py "nike vintage jacket" --max 30 --min-profit 10
  python hibid_scout.py "ralph lauren polo" --shipping 5

eBay fee assumed: 13.25% (Clothing & Accessories category final value fee).
Adjust --shipping to match your typical cost for the category.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field


import requests
from bs4 import BeautifulSoup


# ── Constants ──────────────────────────────────────────────────────────────────

EBAY_FEE_RATE    = 0.1325   # 13.25% final value fee for Clothing & Accessories
DEFAULT_SHIPPING = 6.00     # conservative estimate; adjust per item weight

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class HiBidLot:
    title:       str
    current_bid: float
    num_bids:    int
    end_time:    str
    url:         str
    lot_number:  str = ""
    auctioneer:  str = ""
    image_url:   str = ""


@dataclass
class EbaySoldData:
    query:  str
    prices: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.prices)

    @property
    def avg(self) -> float:
        return statistics.mean(self.prices) if self.prices else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.prices) if self.prices else 0.0

    @property
    def low(self) -> float:
        return min(self.prices) if self.prices else 0.0

    @property
    def high(self) -> float:
        return max(self.prices) if self.prices else 0.0


# ── HiBid scraper ──────────────────────────────────────────────────────────────

def _parse_price(text: str) -> float | None:
    """Extract the first dollar amount from a string."""
    text = text.replace(",", "")
    m = re.search(r"\$?([\d]+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else None


def search_hibid(query: str, max_results: int = 20) -> list[HiBidLot]:
    """
    Scrape HiBid for lots matching query.

    HiBid renders search results server-side on the catalog/search page.
    If the page uses heavy JS (results come back empty), we also try their
    internal API endpoint that the search page calls.
    """
    lots = _hibid_html(query, max_results)
    if not lots:
        lots = _hibid_api(query, max_results)
    return lots


def _hibid_html(query: str, max_results: int) -> list[HiBidLot]:
    url = "https://hibid.com/catalog/search"
    try:
        resp = SESSION.get(url, params={"q": query}, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [HiBid HTML] request failed: {e}", file=sys.stderr)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    # First check if there's embedded JSON (Next.js / React hydration data)
    for tag in soup.find_all("script", {"type": "application/json"}):
        try:
            data = json.loads(tag.string or "")
            extracted = _extract_from_json(data, query)
            if extracted:
                return extracted[:max_results]
        except (json.JSONDecodeError, TypeError):
            pass

    # Fall back to HTML selectors — try several patterns
    items = (
        soup.select(".lot-tile")
        or soup.select(".lot-item")
        or soup.select("[data-lot-number]")
        or soup.select("article.lot")
        or soup.select(".search-result-item")
    )

    lots: list[HiBidLot] = []
    for item in items[:max_results]:
        lot = _parse_hibid_item(item)
        if lot:
            lots.append(lot)
    return lots


def _parse_hibid_item(item) -> HiBidLot | None:
    """Parse a single HiBid lot element."""
    try:
        title_el = (
            item.select_one(".lot-title")
            or item.select_one(".title")
            or item.select_one("h3")
            or item.select_one("h2")
            or item.select_one("[class*='title']")
        )
        if not title_el:
            return None
        title = title_el.get_text(strip=True)
        if not title:
            return None

        bid_el = (
            item.select_one(".current-bid")
            or item.select_one("[class*='bid']")
            or item.select_one("[class*='price']")
        )
        bid_text = bid_el.get_text(strip=True) if bid_el else "0"
        current_bid = _parse_price(bid_text) or 0.0

        bids_el = item.select_one("[class*='bid-count'], [class*='num-bid']")
        num_bids_text = bids_el.get_text(strip=True) if bids_el else "0"
        num_bids_match = re.search(r"\d+", num_bids_text)
        num_bids = int(num_bids_match.group()) if num_bids_match else 0

        time_el = (
            item.select_one("[class*='end-time']")
            or item.select_one("[class*='time-remain']")
            or item.select_one("time")
        )
        end_time = time_el.get_text(strip=True) if time_el else ""

        link_el = item.select_one("a[href]")
        href = link_el["href"] if link_el else ""
        if href and not href.startswith("http"):
            href = "https://hibid.com" + href

        auctioneer_el = item.select_one("[class*='auctioneer'], [class*='company']")
        auctioneer = auctioneer_el.get_text(strip=True) if auctioneer_el else ""

        return HiBidLot(
            title=title,
            current_bid=current_bid,
            num_bids=num_bids,
            end_time=end_time,
            url=href,
            auctioneer=auctioneer,
        )
    except Exception:
        return None


def _hibid_api(query: str, max_results: int) -> list[HiBidLot]:
    """
    Try HiBid's internal search API (used by their React frontend).
    Endpoint discovered via browser DevTools network tab.
    """
    endpoints = [
        "https://hibid.com/api/search/lots",
        "https://hibid.com/api/lots/search",
        "https://api.hibid.com/search",
    ]
    for ep in endpoints:
        try:
            resp = SESSION.get(
                ep,
                params={"q": query, "limit": max_results, "category": "clothing"},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                lots = _extract_from_json(data, query)
                if lots:
                    return lots[:max_results]
        except Exception:
            continue
    return []


def _extract_from_json(data, query: str) -> list[HiBidLot]:
    """Walk arbitrary JSON looking for lot objects."""
    lots: list[HiBidLot] = []

    def walk(node):
        if isinstance(node, list):
            for x in node:
                walk(x)
        elif isinstance(node, dict):
            # Looks like a lot record
            if any(k in node for k in ("lotTitle", "lot_title", "title", "description")):
                title = (
                    node.get("lotTitle")
                    or node.get("lot_title")
                    or node.get("title")
                    or node.get("description", "")
                )
                bid = (
                    node.get("currentBid")
                    or node.get("current_bid")
                    or node.get("bidAmount")
                    or node.get("bid", 0)
                )
                if isinstance(bid, str):
                    bid = _parse_price(bid) or 0.0
                url = node.get("url") or node.get("lotUrl") or node.get("lot_url", "")
                if url and not url.startswith("http"):
                    url = "https://hibid.com" + url
                if title and isinstance(bid, (int, float)):
                    lots.append(HiBidLot(
                        title=str(title),
                        current_bid=float(bid),
                        num_bids=int(node.get("bidCount", node.get("bid_count", 0))),
                        end_time=str(node.get("endTime", node.get("end_time", ""))),
                        url=url,
                    ))
                    return   # don't recurse into this node's children
            for v in node.values():
                walk(v)

    walk(data)
    return lots


# ── eBay sold listings scraper ─────────────────────────────────────────────────

def search_ebay_sold(query: str) -> EbaySoldData:
    """
    Scrape eBay completed/sold listings and return price statistics.
    Filters to pre-owned condition and sorts by most recently ended.
    """
    url = "https://www.ebay.com/sch/i.html"
    params = {
        "_nkw":            query,
        "LH_Sold":         "1",     # sold listings only
        "LH_Complete":     "1",     # completed auctions
        "_sop":            "13",    # sort: recently ended first
        "_ipg":            "60",    # items per page
        "LH_ItemCondition":"3000",  # pre-owned
    }

    try:
        resp = SESSION.get(url, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [eBay] request failed: {e}", file=sys.stderr)
        return EbaySoldData(query=query)

    soup = BeautifulSoup(resp.text, "html.parser")
    prices: list[float] = []

    for item in soup.select(".s-item, li.s-item"):
        title_el = item.select_one(".s-item__title")
        price_el = item.select_one(".s-item__price")

        if not title_el or not price_el:
            continue
        if "Shop on eBay" in (title_el.text or ""):
            continue

        price_text = price_el.get_text(strip=True)
        # Price ranges like "$8.00 to $25.00" — take lower bound (conservative)
        price_text = price_text.split(" to ")[0]
        price = _parse_price(price_text)
        if price and 1.0 <= price <= 1000.0:
            prices.append(price)

    return EbaySoldData(query=query, prices=prices)


# ── Profit math ────────────────────────────────────────────────────────────────

def estimate_profit(
    hibid_bid: float,
    ebay_median: float,
    shipping_cost: float = DEFAULT_SHIPPING,
) -> float:
    """
    Net profit after eBay final value fee and shipping.
    Does not include purchase-side shipping (factor that in manually
    if the HiBid lot has a buyer's premium or shipping charge).
    """
    ebay_net = ebay_median * (1 - EBAY_FEE_RATE) - shipping_cost
    return ebay_net - hibid_bid


# ── Display ────────────────────────────────────────────────────────────────────

def print_opportunity(lot: HiBidLot, sold: EbaySoldData, profit: float) -> None:
    roi = (profit / lot.current_bid * 100) if lot.current_bid > 0 else 0
    bar = "█" * min(int(roi / 10), 20)  # visual ROI bar, capped at 200%
    print(f"  {'─'*56}")
    print(f"  {lot.title[:56]}")
    if lot.auctioneer:
        print(f"  Auctioneer : {lot.auctioneer}")
    print(f"  HiBid bid  : ${lot.current_bid:6.2f}   ends: {lot.end_time or 'unknown'}")
    print(f"  eBay sold  : ${sold.median:6.2f} median  "
          f"(${sold.low:.2f}–${sold.high:.2f}, {sold.count} sales)")
    print(f"  Est. profit: ${profit:6.2f}   ROI {roi:.0f}%  {bar}")
    if lot.url:
        print(f"  Link       : {lot.url}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search HiBid for clothing lots worth reselling on eBay"
    )
    parser.add_argument("query",        help='e.g. "levi 501 jeans" or "nike lot"')
    parser.add_argument("--max",        type=int,   default=20,
                        help="Max HiBid lots to scan (default 20)")
    parser.add_argument("--min-profit", type=float, default=5.0,
                        help="Min estimated profit to show (default $5)")
    parser.add_argument("--shipping",   type=float, default=DEFAULT_SHIPPING,
                        help=f"Your shipping cost assumption (default ${DEFAULT_SHIPPING})")
    parser.add_argument("--ebay-words", type=int,   default=6,
                        help="How many words of the lot title to use for eBay search (default 6)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  HiBid → eBay Scout")
    print(f"  Query    : {args.query}")
    print(f"  eBay fee : {EBAY_FEE_RATE*100:.2f}%   Shipping: ${args.shipping:.2f}")
    print(f"  Min profit filter: ${args.min_profit:.2f}")
    print(f"{'='*60}\n")

    print("Searching HiBid...")
    lots = search_hibid(args.query, max_results=args.max)

    if not lots:
        print("No HiBid results returned.")
        print("HiBid may require JavaScript — try their site directly:")
        print(f"  https://hibid.com/catalog/search?q={args.query.replace(' ', '+')}")
        print("\nIf results appear on the site but not here, the page is")
        print("rendered client-side. Open a browser DevTools Network tab,")
        print("search on HiBid, and look for the API call that returns lots —")
        print("you can add that endpoint to _hibid_api() above.")
        return

    print(f"Found {len(lots)} lots. Fetching eBay sold prices...\n")

    opportunities: list[tuple[float, HiBidLot, EbaySoldData]] = []
    for lot in lots:
        # Use the first N words of the lot title as the eBay search query
        ebay_q = " ".join(lot.title.split()[: args.ebay_words])
        sold = search_ebay_sold(ebay_q)
        time.sleep(0.4)   # polite delay between eBay requests

        if sold.count == 0:
            continue

        profit = estimate_profit(lot.current_bid, sold.median, args.shipping)
        opportunities.append((profit, lot, sold))

    opportunities.sort(key=lambda x: x[0], reverse=True)

    shown = 0
    for profit, lot, sold in opportunities:
        if profit >= args.min_profit:
            print_opportunity(lot, sold, profit)
            shown += 1

    print(f"\n  {'─'*56}")
    print(f"  {shown} opportunity(ies) above ${args.min_profit:.2f} profit threshold.")
    skipped = len(opportunities) - shown
    if skipped:
        print(f"  {skipped} lot(s) below threshold (raise --min-profit to see all).")
    no_data = len(lots) - len(opportunities)
    if no_data:
        print(f"  {no_data} lot(s) had no eBay sold data — try broadening search.")
    print()


if __name__ == "__main__":
    main()
