"""
bankruptcy_engine.py
--------------------
Machine-learning core for financial resilience prediction.

Key design decisions (see project review notes):
  * XGBoost + scale_pos_weight instead of a plain Random Forest.
    The dataset is heavily imbalanced (~3.2% at-risk), so raw accuracy
    is meaningless - a dummy "always stable" model scores ~96.8%.
  * Isotonic probability calibration (CalibratedClassifierCV) so the
    predicted probabilities are trustworthy enough to drive the
    credit-style A+/.../D rating scale.
  * Decision threshold tuned automatically on a validation split by
    maximizing F-beta (beta=2, recall-oriented): missing a company
    that is genuinely at risk is far more costly than a false alarm.
  * Honest metrics (recall / precision / F1 / ROC-AUC per class) are
    computed on a held-out test set and persisted to
    models/model_metrics.json - the UI reads them from there instead
    of hard-coding a misleading "97% accuracy" figure.
  * SHAP TreeExplainer is built once at load time and cached.
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.column_mapping import ALL_FEATURES, COLUMN_MAPPING, CRITICAL_FEATURES
from src.config import (
    DEFAULT_RISK_THRESHOLD,
    RANDOM_STATE,
    TEST_SIZE,
    stability_to_rating,
)


class BankruptcyEngine:

    def __init__(self, data_path: str = None, model_dir: str = "models"):
        self.data_path = data_path or os.path.join("data", "bankruptcy_data.csv")
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "bankruptcy_model.pkl")
        self.metrics_path = os.path.join(model_dir, "model_metrics.json")

        self.model: CalibratedClassifierCV | None = None
        self.explainer: shap.TreeExplainer | None = None
        self.threshold: float = DEFAULT_RISK_THRESHOLD
        self.metrics: dict = {}
        self.feature_names: list[str] = ALL_FEATURES

    def train_and_save_model(self) -> bool:
        print("\n[BankruptcyEngine] Starting training pipeline...")

        if not os.path.exists(self.data_path):
            print(f"[BankruptcyEngine] Dataset not found: {self.data_path}")
            return False

        df = pd.read_csv(self.data_path, encoding="latin1")
        df.rename(columns=COLUMN_MAPPING, inplace=True)

        X = df[ALL_FEATURES]
        y = df["bankruptcy_flag"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        print(f"[BankruptcyEngine] scale_pos_weight = {scale_pos_weight:.1f}")

        base = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train, y_train, test_size=0.25,
            random_state=RANDOM_STATE, stratify=y_train,
        )
        print("[BankruptcyEngine] Fitting tuning model on the fit split...")
        
        tuning_model = CalibratedClassifierCV(
            clone(base), method="isotonic", cv=3
        )
        
        tuning_model.fit(X_fit, y_fit)

        val_proba = tuning_model.predict_proba(X_val)[:, 1]
        best_t, best_fbeta = DEFAULT_RISK_THRESHOLD, -1.0
        for t in np.arange(0.05, 0.55, 0.025):
            preds = (val_proba >= t).astype(int)
            score = fbeta_score(y_val, preds, beta=2, zero_division=0)
            if score > best_fbeta:
                best_fbeta, best_t = score, float(t)
        self.threshold = round(best_t, 3)
        print(f"[BankruptcyEngine] Tuned threshold = {self.threshold} "
              f"(F2 = {best_fbeta:.3f} on held-out validation)")

        print("[BankruptcyEngine] Fitting final calibrated model on full train...")
        self.model = CalibratedClassifierCV(base, method="isotonic", cv=3)
        self.model.fit(X_train, y_train)

        test_proba = self.model.predict_proba(X_test)[:, 1]
        test_pred = (test_proba >= self.threshold).astype(int)

        self.metrics = {
            "model": "XGBoost (isotonic-calibrated)",
            "n_samples": int(len(df)),
            "n_features": int(len(ALL_FEATURES)),
            "class_balance": {"stable": int((y == 0).sum()),
                              "at_risk": int((y == 1).sum())},
            "decision_threshold": self.threshold,
            "roc_auc": round(float(roc_auc_score(y_test, test_proba)), 4),
            "recall_at_risk": round(float(recall_score(y_test, test_pred)), 4),
            "precision_at_risk": round(float(precision_score(y_test, test_pred)), 4),
            "f1_at_risk": round(float(fbeta_score(y_test, test_pred, beta=1)), 4),
            "accuracy_note": (
                "Raw accuracy is intentionally not the headline metric: "
                "with a 96.8% majority class, accuracy is misleading."
            ),
        }
        print(classification_report(y_test, test_pred,
                                    target_names=["Stable", "At Risk"]))
        print(f"[BankruptcyEngine] ROC-AUC: {self.metrics['roc_auc']}")

        os.makedirs(self.model_dir, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({"model": self.model, "threshold": self.threshold}, f)
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)

        self._build_explainer()
        print(f"[BankruptcyEngine] Model saved to {self.model_path}")
        return True
    
        """Loads model, threshold and metrics; trains if missing."""
    def load_model(self) -> bool:
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                bundle = pickle.load(f)
            self.model = bundle["model"]
            self.threshold = bundle.get("threshold", DEFAULT_RISK_THRESHOLD)
            if os.path.exists(self.metrics_path):
                with open(self.metrics_path, encoding="utf-8") as f:
                    self.metrics = json.load(f)
            self._build_explainer()
            print("[BankruptcyEngine] Pre-trained model loaded.")
            return True

        print("[BankruptcyEngine] No saved model found - training now...")
        return self.train_and_save_model()

    def _build_explainer(self) -> None:
        try:
            booster = self.model.calibrated_classifiers_[0].estimator
            self.explainer = shap.TreeExplainer(booster)
        except Exception as e:
            print(f"[BankruptcyEngine] SHAP explainer unavailable: {e}")
            self.explainer = None

    def predict(self, company_data: pd.DataFrame) -> dict:
        if self.model is None:
            self.load_model()

        input_df = company_data.reindex(columns=self.feature_names, fill_value=0)

        risk_score = float(self.model.predict_proba(input_df)[0][1])
        stable_score = 1.0 - risk_score
        prediction = int(risk_score >= self.threshold)
        rating = stability_to_rating(stable_score * 100)

        return {
            "prediction": prediction,
            "stability_score": round(stable_score, 4),
            "risk_score": round(risk_score, 4),
            "rating": rating,
            "threshold": self.threshold,
            "top_risk_factors": self._explain_prediction(input_df),
        }

    def _explain_prediction(self, input_df: pd.DataFrame, top_n: int = 10) -> list[dict]:
        if self.explainer is None:
            return []

        shap_values = self.explainer.shap_values(input_df)
        row = shap_values[0] if shap_values.ndim == 2 else shap_values[:, :, 1][0]

        factors = [
            {
                "feature": feat,
                "value": round(float(val), 6),
                "shap_impact": round(float(sv), 6),
            }
            for feat, sv, val in zip(self.feature_names, row, input_df.values[0])
        ]
        factors.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)
        return factors[:top_n]

    def validate_input(self, company_data: dict) -> tuple[bool, list[str]]:
        missing = [
            f for f in CRITICAL_FEATURES
            if f not in company_data or company_data[f] is None
        ]
        return (len(missing) == 0, missing)


if __name__ == "__main__":
    engine = BankruptcyEngine()
    engine.train_and_save_model()