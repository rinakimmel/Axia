"""
config.py
---------
Central configuration for Axia.

Everything tunable lives here so no magic numbers are scattered
around the codebase:
  - News window (how many days back headlines are considered "fresh")
  - Recency half-life used to weight newer headlines more heavily
  - The credit-style rating scale (S&P-inspired) with its explanation
    dictionary, driven by the CALIBRATED stability probability.
"""

NEWS_WINDOW_DAYS = 3

RECENCY_HALF_LIFE_DAYS = 1.5

MAX_HEADLINES = 12

RANDOM_STATE = 42

TEST_SIZE = 0.2

DEFAULT_RISK_THRESHOLD = 0.20

AUTHORS = "The Axia Team"

COPYRIGHT_HOLDER = "Axia"


RATING_SCALE = [
    (95, "A+"),
    (85, "A"),
    (75, "A-"),
    (60, "B+"),
    (45, "B"),
    (30, "B-"),
    (15, "C"),
    (0,  "D"),
]

RATING_DICTIONARY = {
    "A+": {
        "range":   "95-100%",
        "meaning": "Very high financial stability - minimal distress risk",
        "color":   "#16a34a",
    },
    "A": {
        "range":   "85-95%",
        "meaning": "High financial stability",
        "color":   "#22c55e",
    },
    "A-": {
        "range":   "75-85%",
        "meaning": "Good stability - low risk",
        "color":   "#4ade80",
    },
    "B+": {
        "range":   "60-75%",
        "meaning": "Reasonable stability - growth potential",
        "color":   "#facc15",
    },
    "B": {
        "range":   "45-60%",
        "meaning": "Grey zone - further review recommended",
        "color":   "#f59e0b",
    },
    "B-": {
        "range":   "30-45%",
        "meaning": "Signs of weakness - elevated risk",
        "color":   "#fb923c",
    },
    "C": {
        "range":   "15-30%",
        "meaning": "Significant financial risk",
        "color":   "#f87171",
    },
    "D": {
        "range":   "0-15%",
        "meaning": "Financial distress - very high risk",
        "color":   "#ef4444",
    },
}


def stability_to_rating(stability_pct: float) -> str:
    """Maps a calibrated stability probability (0-100) to a letter grade."""
    for threshold, grade in RATING_SCALE:
        if stability_pct >= threshold:
            return grade
    return "D"
