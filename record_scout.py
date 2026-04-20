#!/usr/bin/env python3
"""
Record Collection Scout — HiBid (location-aware)

Finds HiBid totes/collections of records within driving distance of your
ZIP code, scores each lot for potentially valuable records, and checks eBay
sold prices for any notable artists mentioned in the listing.

Usage:
  python record_scout.py                      # 200 mi from 47339, bid ≤ $30
  python record_scout.py --miles 150          # tighter radius
  python record_scout.py --max-bid 25         # cheaper lots only
  python record_scout.py --min-score 5        # show more results

Only shows lots that require local pickup — no point driving for something
that ships. Distance is calculated from the auction's city using approximate
coordinates.
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup


# ── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_ZIP    = "47339"    # Modoc, IN — change or pass --zip
DEFAULT_MILES  = 200
DEFAULT_MAXBID = 30.0
EBAY_FEE_RATE  = 0.1325

# Approximate lat/lon for ZIP 47339 (Modoc, Indiana)
ORIGIN_COORDS: dict[str, tuple[float, float]] = {
    "47339": (39.97, -85.11),
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


# ── Search terms ────────────────────────────────────────────────────────────────

COLLECTION_QUERIES = [
    "vinyl records lot",
    "record collection vinyl",
    "lp albums lot",
    "vinyl tote records",
    "box vinyl records",
    "record collection estate",
]

# Approximate coordinates for common auction states/cities (add more as needed)
# Used to estimate distance when only city/state is known
CITY_COORDS: dict[str, tuple[float, float]] = {
    # Indiana
    "indianapolis, in": (39.77, -86.16),
    "fort wayne, in":   (41.08, -85.14),
    "evansville, in":   (37.97, -87.57),
    "south bend, in":   (41.68, -86.25),
    "muncie, in":       (40.19, -85.39),
    "anderson, in":     (40.10, -85.68),
    "richmond, in":     (39.83, -84.89),
    "kokomo, in":       (40.49, -86.13),
    "terre haute, in":  (39.47, -87.41),
    "bloomington, in":  (39.16, -86.53),
    # Ohio
    "columbus, oh":     (39.96, -82.99),
    "cleveland, oh":    (41.50, -81.69),
    "cincinnati, oh":   (39.10, -84.51),
    "dayton, oh":       (39.76, -84.19),
    "toledo, oh":       (41.66, -83.56),
    "akron, oh":        (41.08, -81.52),
    "youngstown, oh":   (41.10, -80.65),
    "canton, oh":       (40.80, -81.38),
    # Illinois
    "chicago, il":      (41.88, -87.63),
    "springfield, il":  (39.80, -89.65),
    "peoria, il":       (40.69, -89.59),
    "champaign, il":    (40.12, -88.24),
    # Michigan
    "detroit, mi":      (42.33, -83.05),
    "grand rapids, mi": (42.96, -85.66),
    "lansing, mi":      (42.73, -84.56),
    "flint, mi":        (43.01, -83.69),
    "kalamazoo, mi":    (42.29, -85.59),
    # Kentucky
    "louisville, ky":   (38.25, -85.76),
    "lexington, ky":    (38.04, -84.50),
    # Tennessee
    "nashville, tn":    (36.17, -86.78),
    # Missouri
    "st. louis, mo":    (38.63, -90.20),
    "kansas city, mo":  (39.10, -94.58),
    # Wisconsin
    "milwaukee, wi":    (43.04, -87.91),
    "madison, wi":      (43.07, -89.40),
}


# ── Value scoring tables ────────────────────────────────────────────────────────

HIGH_VALUE_ARTISTS = {
    # Rock
    "beatles", "rolling stones", "led zeppelin", "pink floyd", "jimi hendrix",
    "david bowie", "queen", "the who", "doors", "fleetwood mac", "bob dylan",
    "neil young", "bruce springsteen", "velvet underground", "ramones",
    "sex pistols", "clash", "talking heads", "tom waits", "lou reed",
    "grateful dead", "jefferson airplane", "creedence", "ccr",
    # Soul / R&B / Funk
    "marvin gaye", "stevie wonder", "james brown", "aretha franklin",
    "curtis mayfield", "al green", "otis redding", "sam cooke", "temptations",
    "four tops", "supremes", "smokey robinson", "sly stone", "parliament",
    "funkadelic", "george clinton", "isaac hayes", "bill withers",
    # Jazz
    "miles davis", "john coltrane", "charlie parker", "thelonious monk",
    "duke ellington", "louis armstrong", "dizzy gillespie", "bill evans",
    "charles mingus", "chet baker", "art blakey", "ornette coleman",
    "herbie hancock", "wayne shorter", "coltrane",
    # Blues
    "muddy waters", "robert johnson", "bb king", "b.b. king", "howlin wolf",
    "howlin' wolf", "son house", "buddy guy", "john lee hooker",
    "albert king", "freddie king", "lightnin hopkins",
    # Country / Americana
    "johnny cash", "willie nelson", "waylon jennings", "hank williams",
    "merle haggard", "loretta lynn", "patsy cline", "gram parsons",
    # Reggae
    "bob marley", "peter tosh", "toots", "burning spear", "lee perry",
    # Other
    "frank zappa", "captain beefheart",
}

VALUE_KEYWORDS = {
    "jazz":             15,
    "blues":            12,
    "soul":             10,
    "funk":             10,
    "reggae":           12,
    "motown":           15,
    "original press":   20,
    "first press":      20,
    "mono":             12,
    "promo":            10,
    "sealed":           15,
    "mint":              8,
    "nm":                8,
    "vg+":               6,
    "audiophile":       10,
    "uk press":         15,
    "uk pressing":      15,
    "classic rock":      8,
    "hard rock":         5,
    "estate":            5,   # estate sales often have untouched collections
}


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class RecordLot:
    title:         str
    description:   str
    current_bid:   float
    num_bids:      int
    end_time:      str
    url:           str
    auctioneer:    str = ""
    location:      str = ""     # city, state as shown on HiBid
    distance_miles: float | None = None

    # Filled in by scoring
    artists_found: list[str] = field(default_factory=list)
    keyword_hits:  list[str] = field(default_factory=list)
    score:         int       = 0
    record_count:  int       = 0


@dataclass
class ArtistSales:
    artist: str
    prices: list[float] = field(default_factory=list)

    @property
    def median(self) -> float:
        return statistics.median(self.prices) if self.prices else 0.0

    @property
    def count(self) -> int:
        return len(self.prices)


# ── Distance ───────────────────────────────────────────────────────────────────

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line distance in miles between two lat/lon points."""
    R = 3958.8   # Earth radius in miles
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(d_lon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def zip_to_coords(zipcode: str) -> tuple[float, float] | None:
    """
    Returns (lat, lon) for a US ZIP code.
    Checks local table first, then falls back to the free zippopotam.us API.
    """
    if zipcode in ORIGIN_COORDS:
        return ORIGIN_COORDS[zipcode]
    try:
        resp = SESSION.get(
            f"https://api.zippopotam.us/us/{zipcode}", timeout=5
        )
        if resp.status_code == 200:
            data = resp.json()
            place = data["places"][0]
            coords = (float(place["latitude"]), float(place["longitude"]))
            ORIGIN_COORDS[zipcode] = coords   # cache it
            return coords
    except Exception:
        pass
    return None


def estimate_distance(location_str: str, origin: tuple[float, float]) -> float | None:
    """
    Try to estimate distance (miles) from origin to a 'City, ST' location string.
    Checks the CITY_COORDS table; returns None if unknown.
    """
    if not location_str:
        return None
    key = location_str.lower().strip()
    if key in CITY_COORDS:
        return haversine(*origin, *CITY_COORDS[key])
    # Try matching just the city name
    city_part = key.split(",")[0].strip()
    for k, coords in CITY_COORDS.items():
        if k.startswith(city_part + ","):
            return haversine(*origin, *coords)
    return None


# ── HiBid scraping ─────────────────────────────────────────────────────────────

def fetch_hibid_lots(query: str, zipcode: str, radius_miles: int,
                     max_results: int = 30) -> list[RecordLot]:
    """
    Search HiBid, passing zip/radius parameters so the site pre-filters
    by distance. Falls back gracefully if those params aren't supported.
    """
    url = "https://hibid.com/catalog/search"
    params = {
        "q":      query,
        "zip":    zipcode,
        "miles":  str(radius_miles),
    }
    try:
        resp = SESSION.get(url, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  HiBid request failed: {e}", file=sys.stderr)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    items = (
        soup.select(".lot-tile")
        or soup.select(".lot-item")
        or soup.select("[data-lot-number]")
        or soup.select("article.lot")
        or soup.select(".search-result-item")
    )

    lots: list[RecordLot] = []
    for item in items[:max_results]:
        lot = _parse_lot(item)
        if lot:
            lots.append(lot)
    return lots


def _parse_lot(item) -> RecordLot | None:
    try:
        title_el = (
            item.select_one(".lot-title") or item.select_one(".title")
            or item.select_one("h3") or item.select_one("h2")
            or item.select_one("[class*='title']")
        )
        if not title_el:
            return None
        title = title_el.get_text(strip=True)
        if not title:
            return None

        desc_el = (
            item.select_one(".lot-description") or item.select_one(".description")
            or item.select_one("[class*='desc']") or item.select_one("p")
        )
        description = desc_el.get_text(strip=True) if desc_el else ""

        bid_el = (
            item.select_one(".current-bid") or item.select_one("[class*='bid']")
            or item.select_one("[class*='price']")
        )
        bid_text  = bid_el.get_text(strip=True) if bid_el else "0"
        bid_match = re.search(r"[\d,]+\.?\d*", bid_text.replace(",", ""))
        current_bid = float(bid_match.group()) if bid_match else 0.0

        bids_el   = item.select_one("[class*='bid-count'], [class*='num-bid']")
        num_text  = bids_el.get_text(strip=True) if bids_el else "0"
        num_match = re.search(r"\d+", num_text)
        num_bids  = int(num_match.group()) if num_match else 0

        time_el  = (
            item.select_one("[class*='end-time']") or item.select_one("[class*='time']")
            or item.select_one("time")
        )
        end_time = time_el.get_text(strip=True) if time_el else ""

        loc_el   = (
            item.select_one("[class*='location']") or item.select_one("[class*='city']")
            or item.select_one("[class*='address']")
        )
        location = loc_el.get_text(strip=True) if loc_el else ""

        auc_el   = item.select_one("[class*='auctioneer'], [class*='company']")
        auctioneer = auc_el.get_text(strip=True) if auc_el else ""

        link_el  = item.select_one("a[href]")
        href     = link_el["href"] if link_el else ""
        if href and not href.startswith("http"):
            href = "https://hibid.com" + href

        return RecordLot(
            title=title,
            description=description,
            current_bid=current_bid,
            num_bids=num_bids,
            end_time=end_time,
            url=href,
            auctioneer=auctioneer,
            location=location,
        )
    except Exception:
        return None


def fetch_lot_detail(lot: RecordLot) -> None:
    """Follow the lot URL to get the full description and location if missing."""
    if not lot.url or (lot.description and lot.location):
        return
    try:
        resp = SESSION.get(lot.url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        if not lot.description:
            desc_el = (
                soup.select_one(".lot-description")
                or soup.select_one("[class*='description']")
                or soup.select_one(".lot-detail")
                or soup.select_one("main p")
            )
            if desc_el:
                lot.description = desc_el.get_text(" ", strip=True)

        if not lot.location:
            loc_el = (
                soup.select_one("[class*='location']")
                or soup.select_one("[class*='address']")
                or soup.select_one("[class*='city']")
            )
            if loc_el:
                lot.location = loc_el.get_text(strip=True)
    except Exception:
        pass


# ── Collection vs. single-record filter ───────────────────────────────────────

# Strong signals that a lot IS a collection/tote
_COLLECTION_WORDS = re.compile(
    r"\b(lot|collection|tote|box|crate|stack|bundle|bulk|assorted|mixed|misc"
    r"|various|estate|group|set|run|batch|pounds? of|lbs? of)\b",
    re.I,
)

# Strong signals that a lot is a SINGLE record (skip these)
_SINGLE_RECORD_PATTERNS = [
    # "Artist - Title" or "Artist: Title" with no collection words
    re.compile(r"^[A-Za-z\s\.'&,]+\s*[-:]\s*[A-Za-z\s\.'&,\(\)]+$"),
    # Ends with format indicator suggesting one item: "... LP", "... 45", "... 7\""
    re.compile(r'\b(single|45\s*rpm|7["\']|one\s+lp|one\s+album|one\s+record)\b', re.I),
]

# If title contains a number >= 2 followed by a record-type word, it's a lot
_QUANTITY_RE = re.compile(
    r"\b([2-9]\d*|[1-9]\d+)\s*\+?\s*(?:records?|albums?|lps?|vinyls?|45s?|singles?)\b",
    re.I,
)


def is_collection(lot: RecordLot) -> bool:
    """
    Return True if the lot looks like a collection/tote rather than a
    single individual record.

    Keeps anything that has:
      - an explicit collection/tote/lot/box/estate keyword
      - a quantity like "50 records" or "12 LPs"
      - 3+ commas in the title (suggesting a list of items)

    Rejects titles that look like a single "Artist - Album" entry with
    none of the above signals.
    """
    title = lot.title.strip()
    title_lower = title.lower()

    # Explicit collection keyword → keep
    if _COLLECTION_WORDS.search(title_lower):
        return True

    # Explicit quantity → keep
    if _QUANTITY_RE.search(title):
        return True

    # Title looks like a list of artists/albums (many commas) → keep
    if title.count(",") >= 2:
        return True

    # Single-record pattern with no collection signal → skip
    for pat in _SINGLE_RECORD_PATTERNS:
        if pat.search(title):
            return False

    # Default: keep if uncertain (better to show too many than miss a tote)
    return True


# ── Scoring ────────────────────────────────────────────────────────────────────

def _record_count_from_text(text: str) -> int:
    m = re.search(r"\b(\d{1,3})\+?\s*(?:records?|albums?|lps?|vinyls?)\b", text, re.I)
    return int(m.group(1)) if m else 0


def score_lot(lot: RecordLot) -> None:
    combined = (lot.title + " " + lot.description).lower()
    lot.record_count = _record_count_from_text(lot.title + " " + lot.description)

    for artist in HIGH_VALUE_ARTISTS:
        if artist in combined:
            lot.artists_found.append(artist.title())
            lot.score += 20

    for kw, pts in VALUE_KEYWORDS.items():
        if kw in combined:
            lot.keyword_hits.append(kw)
            lot.score += pts

    if lot.record_count >= 100:
        lot.score += 10
    elif lot.record_count >= 50:
        lot.score += 5

    if lot.current_bid > 20:
        lot.score -= int((lot.current_bid - 20) * 2)

    lot.score = max(0, lot.score)


# ── eBay artist pricing ────────────────────────────────────────────────────────

def ebay_artist_median(artist: str) -> ArtistSales:
    url = "https://www.ebay.com/sch/i.html"
    params = {
        "_nkw":             f"{artist} vinyl record",
        "LH_Sold":          "1",
        "LH_Complete":      "1",
        "_sop":             "13",
        "_ipg":             "40",
        "LH_ItemCondition": "3000",
    }
    try:
        resp = SESSION.get(url, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException:
        return ArtistSales(artist=artist)

    soup  = BeautifulSoup(resp.text, "html.parser")
    prices: list[float] = []
    for item in soup.select(".s-item, li.s-item"):
        title_el = item.select_one(".s-item__title")
        price_el = item.select_one(".s-item__price")
        if not title_el or not price_el:
            continue
        if "Shop on eBay" in (title_el.text or ""):
            continue
        price_text = price_el.get_text(strip=True).split(" to ")[0]
        m = re.search(r"[\d,]+\.?\d*", price_text.replace(",", ""))
        if m:
            p = float(m.group())
            if 1.0 <= p <= 500.0:
                prices.append(p)
    return ArtistSales(artist=artist, prices=prices)


# ── Display ────────────────────────────────────────────────────────────────────

def star_rating(score: int) -> str:
    if score >= 100: return "★★★★★"
    if score >= 70:  return "★★★★ "
    if score >= 50:  return "★★★  "
    if score >= 30:  return "★★   "
    return              "★    "


def print_lot(lot: RecordLot, artist_data: dict[str, ArtistSales]) -> None:
    stars      = star_rating(lot.score)
    count_str  = f"{lot.record_count}+ records" if lot.record_count else "collection"
    dist_str   = f"  ~{lot.distance_miles:.0f} mi away" if lot.distance_miles is not None else ""
    loc_str    = f"{lot.location}{dist_str}" if lot.location else dist_str.strip()

    print(f"\n  {'═'*58}")
    print(f"  {stars}  {lot.title[:52]}")
    if loc_str:
        print(f"  Location   : {loc_str}")
    if lot.auctioneer:
        print(f"  Auctioneer : {lot.auctioneer}")
    print(f"  Bid        : ${lot.current_bid:.2f}  ({count_str})   ends: {lot.end_time or 'unknown'}")
    print(f"  Score      : {lot.score}")

    if lot.artists_found:
        print(f"\n  Notable artists in listing:")
        for artist in lot.artists_found:
            sales = artist_data.get(artist.lower())
            if sales and sales.count > 0:
                net = sales.median * (1 - EBAY_FEE_RATE) - 5
                print(f"    • {artist:30s}  eBay median ${sales.median:.2f} "
                      f"({sales.count} sales)  → ~${net:.2f} net each")
            else:
                print(f"    • {artist}")

    if lot.keyword_hits:
        print(f"  Keywords   : {', '.join(lot.keyword_hits)}")

    if lot.description:
        snippet = lot.description[:220].replace("\n", " ")
        print(f"  Description: {snippet}{'…' if len(lot.description) > 220 else ''}")

    if lot.url:
        print(f"  Link       : {lot.url}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find record totes/collections worth driving to on HiBid"
    )
    parser.add_argument("--zip",        default=DEFAULT_ZIP,
                        help=f"Your ZIP code (default {DEFAULT_ZIP})")
    parser.add_argument("--miles",      type=int, default=DEFAULT_MILES,
                        help=f"Max driving distance in miles (default {DEFAULT_MILES})")
    parser.add_argument("--max-bid",    type=float, default=DEFAULT_MAXBID,
                        help=f"Max current bid to consider (default ${DEFAULT_MAXBID:.0f})")
    parser.add_argument("--min-score",  type=int, default=10,
                        help="Min score to display a lot (default 10)")
    parser.add_argument("--max",        type=int, default=30,
                        help="Max lots to fetch per search term (default 30)")
    parser.add_argument("--query", "-q",
                        help="Custom search query (default: tries several collection terms)")
    parser.add_argument("--no-ebay",    action="store_true",
                        help="Skip eBay artist price lookups (faster)")
    args = parser.parse_args()

    print(f"\n{'='*62}")
    print(f"  Record Collection Scout")
    print(f"  ZIP: {args.zip}   Radius: {args.miles} mi   Max bid: ${args.max_bid:.0f}")
    print(f"{'='*62}")

    # Resolve origin coordinates
    origin = zip_to_coords(args.zip)
    if origin is None:
        print(f"Warning: could not resolve coordinates for ZIP {args.zip}. "
              "Distance filtering disabled.")

    queries = [args.query] if args.query else COLLECTION_QUERIES

    seen: set[str] = set()
    all_lots: list[RecordLot] = []

    for q in queries:
        print(f"\nSearching HiBid: {q!r}  (zip={args.zip}, radius={args.miles}mi)...")
        lots = fetch_hibid_lots(q, args.zip, args.miles, max_results=args.max)
        for lot in lots:
            key = lot.title.lower().strip()
            if key not in seen:
                seen.add(key)
                all_lots.append(lot)
        time.sleep(0.5)

    if not all_lots:
        print("\nNo HiBid results returned. A few possibilities:")
        print("  1. HiBid renders results with JavaScript — open the URLs below in a browser")
        print("  2. No matching lots exist in your area right now — check back later")
        print("  3. Try --miles 300 for a wider radius\n")
        for q in COLLECTION_QUERIES:
            q_enc = q.replace(" ", "+")
            print(f"  https://hibid.com/catalog/search?q={q_enc}&zip={args.zip}&miles={args.miles}")
        return

    print(f"\nFound {len(all_lots)} unique lots. Filtering singles...")

    # Drop individual single-record listings before fetching detail pages
    collections = [lot for lot in all_lots if is_collection(lot)]
    singles_removed = len(all_lots) - len(collections)
    if singles_removed:
        print(f"  Removed {singles_removed} single-record listing(s) — kept {len(collections)} collections.")

    if not collections:
        print("All results looked like individual records. Try --query with different terms.")
        return

    print(f"Fetching details and scoring {len(collections)} collection(s)...")

    for lot in collections:
        if not lot.description and lot.url:
            fetch_lot_detail(lot)
            time.sleep(0.25)

        # Attach distance if we have origin + location
        if origin and lot.location:
            lot.distance_miles = estimate_distance(lot.location, origin)

        score_lot(lot)

    # Filter: bid ≤ max, score ≥ min, and within radius (if distance known)
    candidates: list[RecordLot] = []
    for lot in collections:
        if lot.current_bid > args.max_bid:
            continue
        if lot.score < args.min_score:
            continue
        if lot.distance_miles is not None and lot.distance_miles > args.miles:
            continue
        candidates.append(lot)

    candidates.sort(key=lambda l: l.score, reverse=True)

    if not candidates:
        print(f"\nNo lots matched filters (bid ≤${args.max_bid:.0f}, "
              f"score ≥{args.min_score}, ≤{args.miles} mi).")
        print("Try: --max-bid higher, --min-score lower, or --miles wider.")
        return

    print(f"{len(candidates)} lot(s) worth investigating.\n")

    # eBay lookups for artists in top lots
    artist_data: dict[str, ArtistSales] = {}
    if not args.no_ebay:
        top_artists: list[str] = []
        for lot in candidates[:8]:
            for a in lot.artists_found:
                if a.lower() not in artist_data and a.lower() not in top_artists:
                    top_artists.append(a.lower())
        if top_artists:
            print(f"Checking eBay sold prices for {len(top_artists)} artist(s)...\n")
            for artist in top_artists:
                artist_data[artist] = ebay_artist_median(artist)
                time.sleep(0.4)

    for lot in candidates:
        print_lot(lot, artist_data)

    print(f"\n{'─'*62}")
    print(f"  {len(candidates)} lot(s) shown  |  "
          f"{singles_removed} singles removed  |  "
          f"{len(collections) - len(candidates)} below threshold")
    if args.no_ebay:
        print("  Re-run without --no-ebay for per-artist eBay pricing.")
    print()


if __name__ == "__main__":
    main()
