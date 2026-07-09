"""
news_scraper.py
---------------
Fetches financial news headlines for a given company from public RSS feeds.

Freshness guarantee:
  * Every item's pubDate is parsed (robustly, via email.utils).
  * Only headlines published within the last NEWS_WINDOW_DAYS days
    (default 3, configured in src/config.py) are kept.
  * Each returned item carries `age_days`, used downstream by the
    sentiment engine for exponential recency weighting, and the
    scraper reports the exact date window used so the report can state
    "based on news from <start> to <end>".

No API keys required - public RSS feeds only.
"""

import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

from src.config import MAX_HEADLINES, NEWS_WINDOW_DAYS

RSS_FEEDS = [
    {"source": "Reuters Business",
     "url": "https://feeds.reuters.com/reuters/businessNews"},
    {"source": "CNBC Finance",
     "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html"},
    {"source": "MarketWatch",
     "url": "https://feeds.marketwatch.com/marketwatch/topstories/"},
    {"source": "Yahoo Finance",
     "url": "https://finance.yahoo.com/rss/"},
    {"source": "Investing.com",
     "url": "https://www.investing.com/rss/news.rss"},
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

TIMEOUT_SECONDS = 6


class NewsScraper:

    def __init__(self,
                 max_headlines: int = MAX_HEADLINES,
                 window_days: int = NEWS_WINDOW_DAYS):
        self.max_headlines = max_headlines
        self.window_days   = window_days


    def fetch_headlines(self, company_name: str, symbol: str = "") -> dict:
        """
        Fetches company-relevant headlines published within the window.

        Returns
        -------
        dict:
            headlines    list[dict]  each: headline, source, published,
                                     published_dt (ISO), age_days
            window_days  int
            window_start str  (YYYY-MM-DD)
            window_end   str  (YYYY-MM-DD)
        """
        now    = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self.window_days)

        keywords  = self._build_keywords(company_name, symbol)
        collected = []

        for feed in RSS_FEEDS:
            if len(collected) >= self.max_headlines:
                break
            collected.extend(
                self._fetch_feed(feed["source"], feed["url"], keywords, cutoff, now)
            )

        seen, unique = set(), []
        collected.sort(key=lambda x: x["age_days"])
        for item in collected:
            if item["headline"] not in seen:
                seen.add(item["headline"])
                unique.append(item)
            if len(unique) >= self.max_headlines:
                break

        return {
            "headlines":    unique,
            "window_days":  self.window_days,
            "window_start": cutoff.strftime("%Y-%m-%d"),
            "window_end":   now.strftime("%Y-%m-%d"),
        }

    def _build_keywords(self, company_name: str, symbol: str) -> list[str]:
        keywords = [company_name.lower().strip()]
        for word in company_name.split():
            if len(word) >= 4:
                keywords.append(word.lower())
        if symbol:
            keywords.append(symbol.lower())
        return list(set(k for k in keywords if k))

    def _fetch_feed(self, source: str, url: str, keywords: list[str],
                    cutoff: datetime, now: datetime) -> list[dict]:
        """Fetch one RSS feed; keep only fresh, relevant items."""
        try:
            request  = urllib.request.Request(url, headers=HEADERS)
            response = urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS)
            root     = ET.fromstring(response.read())
        except Exception:
            return []

        results = []
        for item in root.iter("item"):
            title    = (item.findtext("title") or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()
            if not title:
                continue

            published_dt = self._parse_date(pub_date)

            if published_dt is None or published_dt < cutoff:
                continue

            if any(kw in title.lower() for kw in keywords):
                age_days = max((now - published_dt).total_seconds() / 86400, 0.0)
                results.append({
                    "headline":     title,
                    "source":       source,
                    "published":    published_dt.strftime("%Y-%m-%d %H:%M"),
                    "published_dt": published_dt.isoformat(),
                    "age_days":     round(age_days, 3),
                })
        return results

    @staticmethod
    def _parse_date(raw_date: str):
        """
        Parses an RSS pubDate into an aware UTC datetime.
        email.utils.parsedate_to_datetime handles the RFC-2822 formats
        used by virtually all RSS feeds, including timezone names.
        Returns None if parsing fails.
        """
        if not raw_date:
            return None
        try:
            dt = parsedate_to_datetime(raw_date)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except (TypeError, ValueError):
            return None



if __name__ == "__main__":
    scraper = NewsScraper()
    for company, symbol in [("Apple", "AAPL"), ("Tesla", "TSLA")]:
        print(f"\nSearching headlines for: {company} ({symbol})")
        result = scraper.fetch_headlines(company, symbol)
        print(f"  Window: {result['window_start']} -> {result['window_end']}")
        if not result["headlines"]:
            print("  No fresh headlines found within the window.")
        for h in result["headlines"]:
            print(f"  [{h['source']}] ({h['age_days']:.1f}d ago) {h['headline']}")
