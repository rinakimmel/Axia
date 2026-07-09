"""
sentiment_engine.py
-------------------
NLP training and inference for financial news sentiment classification,
plus recency-weighted aggregate scoring.

Dataset: Financial PhraseBank v1.0 - Sentences_AllAgree.txt
  (sentences where ALL professional annotators agreed - highest quality)

Pipeline:
  1. TF-IDF (unigrams + bigrams) -> SMOTE class balancing
  2. Logistic Regression classifier
  3. Per-text: label + confidence + full probability breakdown
  4. Aggregate: a single sentiment score in [-1, +1] computed as the
     recency-weighted mean of per-item scores, where each item's score
     is P(positive) - P(negative). Newer headlines weigh more via
     exponential decay (half-life from src/config.py).
"""

import math
import os
import pickle

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, RECENCY_HALF_LIFE_DAYS

# ----------------------------------------------------------------------
# Financial sentiment lexicon (Loughran-McDonald inspired)
# ----------------------------------------------------------------------
# The TF-IDF model is trained on Financial PhraseBank, whose vocabulary
# is report-style ("operating profit rose to EUR..."). Headline/analyst
# language ("downgraded", "liquidity concerns", "warns") is largely
# out-of-vocabulary, causing everything to default to neutral.
# A compact financial lexicon closes that domain gap: the final polarity
# blends the model probability signal with lexicon hits.

LEXICON_POSITIVE = {
    "record", "profit", "profits", "profitable", "profitability", "gain",
    "gains", "growth", "grew", "rose", "rises", "rising", "surge", "surged",
    "beat", "beats", "exceeded", "exceeds", "outperform", "outperformed",
    "upgrade", "upgraded", "raised", "raises", "strong", "strength",
    "robust", "improved", "improvement", "improving", "rally", "rallied",
    "expansion", "expanded", "success", "successful", "milestone",
    "breakthrough", "momentum", "optimistic", "bullish", "dividend",
    "buyback", "innovative", "soared", "soars", "jumped", "climbs",
    "climbed", "recovery", "recovered", "rebound", "rebounded",
}

LEXICON_NEGATIVE = {
    "loss", "losses", "decline", "declined", "declining", "fell", "falls",
    "falling", "drop", "dropped", "plunge", "plunged", "downgrade",
    "downgraded", "warn", "warns", "warning", "risk", "risks", "risky",
    "concern", "concerns", "debt", "default", "defaulted", "bankruptcy",
    "insolvency", "insolvent", "distress", "distressed", "weak", "weakness",
    "weakening", "miss", "missed", "misses", "lawsuit", "litigation",
    "investigation", "fraud", "layoff", "layoffs", "restructuring",
    "impairment", "writedown", "write-off", "deficit", "shortfall",
    "liquidity", "downturn", "recession", "slump", "slumped", "crisis",
    "bearish", "pessimistic", "cut", "cuts", "suspended", "delisted",
    "tumbled", "tumbles", "sank", "crashed", "underperform",
    "underperformed", "pressure", "struggling", "struggles",
}

MODEL_WEIGHT   = 0.55
LEXICON_WEIGHT = 0.45

LABEL_THRESHOLD = 0.20


class SentimentEngine:

    def __init__(self, data_path: str = None, model_dir: str = "models"):
        self.data_path       = data_path or os.path.join("data", "Sentences_AllAgree.txt")
        self.model_dir       = model_dir
        self.model_path      = os.path.join(model_dir, "sentiment_model.pkl")
        self.vectorizer_path = os.path.join(model_dir, "sentiment_vectorizer.pkl")
        self.model: LogisticRegression | None   = None
        self.vectorizer: TfidfVectorizer | None = None


    def train_and_save_model(self) -> bool:
        print("\n[SentimentEngine] Starting NLP training pipeline...")

        if not os.path.exists(self.data_path):
            print(f"[SentimentEngine] Dataset not found: {self.data_path}")
            return False

        df = pd.read_csv(
            self.data_path, sep="@", header=None,
            names=["Sentence", "Sentiment"], encoding="latin1",
        ).dropna()
        df["Sentiment"] = df["Sentiment"].str.strip()

        print(f"[SentimentEngine] Loaded {len(df)} sentences.")
        print(f"  Distribution: {df['Sentiment'].value_counts().to_dict()}")

        X_train, X_test, y_train, y_test = train_test_split(
            df["Sentence"], df["Sentiment"],
            test_size=0.2, random_state=RANDOM_STATE, stratify=df["Sentiment"],
        )

        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_features=10000, ngram_range=(1, 2),
        )
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec  = self.vectorizer.transform(X_test)

        X_train_bal, y_train_bal = SMOTE(random_state=RANDOM_STATE).fit_resample(
            X_train_vec, y_train
        )

        self.model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, C=10)
        self.model.fit(X_train_bal, y_train_bal)

        y_pred = self.model.predict(X_test_vec)
        print(f"[SentimentEngine] Test accuracy: {accuracy_score(y_test, y_pred):.4f} "
              f"| macro-F1: {f1_score(y_test, y_pred, average='macro'):.4f}")
        print(classification_report(y_test, y_pred))

        os.makedirs(self.model_dir, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(self.vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

        print("[SentimentEngine] Model and vectorizer saved.")
        return True

    def load_model(self) -> bool:
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            with open(self.vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            print("[SentimentEngine] Pre-trained NLP model loaded.")
            return True
        print("[SentimentEngine] No saved model found - training now...")
        return self.train_and_save_model()


    @staticmethod
    def _lexicon_polarity(text: str) -> tuple[float, int]:
        """
        Lexicon polarity in [-1, +1] and the number of matched terms.
        Returns (0.0, 0) when no lexicon word appears.
        """
        words = [w.strip(".,!?;:()'\"").lower() for w in text.split()]
        pos = sum(1 for w in words if w in LEXICON_POSITIVE)
        neg = sum(1 for w in words if w in LEXICON_NEGATIVE)
        hits = pos + neg
        if hits == 0:
            return 0.0, 0
        return (pos - neg) / hits, hits

    def analyze_sentiment(self, text: str) -> dict:
        """
        Classifies one text using a hybrid of the trained model and a
        financial lexicon (closes the domain gap between PhraseBank
        training sentences and headline-style English).

        Returns dict:
            label       "positive" | "negative" | "neutral"
            confidence  probability/strength of the final label
            scores      full per-class probability breakdown (model)
            polarity    combined polarity in [-1, +1]
        """
        if self.model is None or self.vectorizer is None:
            self.load_model()

        vec   = self.vectorizer.transform([text])
        label = self.model.predict(vec)[0]
        proba = self.model.predict_proba(vec)[0]

        scores = {c: round(float(p), 4) for c, p in zip(self.model.classes_, proba)}
        model_polarity = scores.get("positive", 0.0) - scores.get("negative", 0.0)

        # Hybrid rule: the lexicon only steps in when the model predicts
        # NEUTRAL - the observed out-of-domain failure mode. Confident
        # in-domain positive/negative predictions are never overridden,
        # which preserves the model's PhraseBank accuracy.
        final_label, polarity, confidence = label, model_polarity, scores[label]

        if label == "neutral":
            lex_polarity, lex_hits = self._lexicon_polarity(text)
            if lex_hits > 0:
                polarity = (MODEL_WEIGHT * model_polarity
                            + LEXICON_WEIGHT * lex_polarity)
                if polarity >= LABEL_THRESHOLD:
                    final_label = "positive"
                elif polarity <= -LABEL_THRESHOLD:
                    final_label = "negative"
                if final_label != label:
                    # Label corrected by combined evidence - report its
                    # strength instead of the (wrong) neutral probability.
                    confidence = round(min(0.5 + abs(polarity) / 2, 0.95), 4)

        return {
            "label":      final_label,
            "confidence": confidence,
            "scores":     scores,
            "polarity":   round(polarity, 4),
        }

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        """Analyzes a list of texts; results in input order."""
        return [self.analyze_sentiment(t) for t in texts]


    @staticmethod
    def aggregate_score(results: list[dict],
                        ages_days: list[float] | None = None) -> dict:
        """
        Combines per-item sentiment into one recency-weighted score.

        Parameters
        ----------
        results   : output of analyze_batch (each item has `polarity`)
        ages_days : optional item ages in days (from the news scraper).
                    When given, weight = 0.5 ** (age / half_life), so a
                    3-hour-old headline counts more than a 2.9-day-old one.
                    When None (e.g. an uploaded opinions file with no
                    dates), all items weigh equally.

        Returns dict:
            score        weighted mean polarity in [-1, +1]
            score_0_100  same score rescaled to 0-100 for display
            label        "positive" | "negative" | "neutral" overall
            distribution {label: count}
            n_items      number of texts aggregated
        """
        if not results:
            return {"score": 0.0, "score_0_100": 50.0, "label": "neutral",
                    "distribution": {}, "n_items": 0}

        if ages_days is None:
            weights = [1.0] * len(results)
        else:
            weights = [
                0.5 ** (max(a, 0.0) / RECENCY_HALF_LIFE_DAYS)
                for a in ages_days
            ]

        total_w = sum(weights) or 1.0
        score = sum(r["polarity"] * w for r, w in zip(results, weights)) / total_w

        distribution: dict[str, int] = {}
        for r in results:
            distribution[r["label"]] = distribution.get(r["label"], 0) + 1

        overall = ("positive" if score > 0.15
                   else "negative" if score < -0.15
                   else "neutral")

        return {
            "score":        round(score, 4),
            "score_0_100":  round((score + 1) * 50, 1),
            "label":        overall,
            "distribution": distribution,
            "n_items":      len(results),
        }



if __name__ == "__main__":
    engine = SentimentEngine()
    engine.train_and_save_model()

    samples = [
        "Company reports record profits for the third consecutive quarter.",
        "Firm faces severe debt crisis and possible bankruptcy filing.",
        "Market remains stable with moderate trading volumes today.",
    ]
    results = engine.analyze_batch(samples)
    for s, r in zip(samples, results):
        print(f"\n{s}\n  -> {r['label'].upper()} "
              f"(confidence {r['confidence']:.0%}, polarity {r['polarity']:+.2f})")

    agg = SentimentEngine.aggregate_score(results, ages_days=[0.1, 1.5, 2.8])
    print(f"\nAggregate: {agg}")