<<<<<<< HEAD
# Axia — Financial Resilience Prediction & Market Sentiment AI

Axia predicts a company's financial resilience from its financial ratios
and combines the prediction with a recency-weighted market-sentiment analysis,
producing a professional, self-contained HTML report with a credit-style
rating (A+ … D).

![Axia Logo](assets/logo.png)

![Axia Dashboard Preview](assets/rating_dashboard.png)
*The financial resilience dashboard showing the calibrated risk probability and credit-style rating (A+ ... D).*

## Architecture

| Layer | Component | Technology |
|---|---|---|
| Financial model | `src/bankruptcy_engine.py` | XGBoost + isotonic calibration (scikit-learn) |
| Sentiment model | `src/sentiment_engine.py` | TF-IDF bigrams + SMOTE + Logistic Regression |
| News ingestion | `src/news_scraper.py` | Public RSS feeds, 3-day freshness window |
| Explainability | SHAP TreeExplainer (cached) | `shap` |
| Report | `src/report_generator.py` | Self-contained HTML + base64 charts |
| Client | `app.py` | Streamlit |
| Configuration | `src/config.py` | Rating scale, news window, thresholds |

## Key design decisions

**Honest metrics, not vanity accuracy.** Only ~3.2% of companies in the
Taiwan Economic Journal dataset are at risk, so a dummy "always stable"
model scores ~96.8% accuracy. Axia therefore optimizes and reports
**recall / precision / ROC-AUC for the at-risk class** (metrics are written
to `models/model_metrics.json` at training time and read dynamically by the
UI — nothing is hard-coded).

Held-out test results:
- ROC-AUC: **0.9575**
- At-risk recall: **0.80** (the original Random Forest caught only 0.25)
- Decision threshold tuned on a leak-free validation split, maximizing
  F2 (recall-weighted), because missing a genuinely distressed company is
  costlier than a false alarm.

**Calibrated probabilities drive the rating.** The A+ … D scale maps the
*calibrated* stability probability (isotonic regression), so "92% stable"
actually means ~92%.

| Grade | Stability | Meaning |
|---|---|---|
| A+ | 95–100% | Very high stability — minimal distress risk |
| A | 85–95% | High financial stability |
| A− | 75–85% | Good stability — low risk |
| B+ | 60–75% | Reasonable stability — growth potential |
| B | 45–60% | Grey zone — further review recommended |
| B− | 30–45% | Signs of weakness — elevated risk |
| C | 15–30% | Significant financial risk |
| D | 0–15% | Financial distress — very high risk |

**Fresh, recency-weighted sentiment.** Headlines older than
`NEWS_WINDOW_DAYS` (default **3 days**, `src/config.py`) are discarded at
the parsing stage. Within the window, items are weighted by exponential
decay (half-life 1.5 days), and the aggregate sentiment score (−1 … +1,
also shown as a 0–100 index) is the weighted mean of per-item polarity
P(positive) − P(negative). Alternatively, upload a `.txt` file of opinions
(one per line) instead of using the news feeds.

## Local Explainability (SHAP)

Axia doesn't just output a score; it explains *why*. The model calculates SHAP (SHapley Additive exPlanations) values to highlight the specific financial indicators driving the company toward risk or stability.

![SHAP Key Risk Drivers](assets/shap_drivers.png)
*Local explainability via SHAP, demonstrating the top features influencing the specific prediction.*

## NLP Market Sentiment

By leveraging TF-IDF bigrams and Logistic Regression (trained with SMOTE for class imbalance), Axia contextualizes hard financial data with real-world market sentiment.

![Media Sentiment Analysis](assets/sentiment_analysis.png)
*Recency-weighted NLP sentiment analysis combining financial news or opinions into an aggregate risk index.*

## Setup

```bash
pip install -r requirements.txt
python main.py            # validates data, trains models if missing, launches UI
# or directly:
streamlit run app.py

## Authors

**Tamar Waiss** & **Rina Kimmel**

[![Tamar Waiss LinkedIn](https://img.shields.io/badge/Tamar_Waiss-Profile-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/tamar-waiss)
[![Rina Kimmel LinkedIn](https://img.shields.io/badge/Rina_Kimmel-Profile-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rina-kimmel-7963ba413)
=======
# Axia
AI-powered financial resilience prediction &amp; market sentiment analysis. Calibrated XGBoost credit-style ratings (A+ to D) combined with recency-weighted NLP sentiment from live financial news.
>>>>>>> d740329d773e9772d8abe28952153eb9cc33fbfe
