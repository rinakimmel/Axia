"""
app.py
------
Streamlit web client for Axia.

Flow (matches the product spec):
  1. Enter company name (+ optional stock symbol)
  2. Upload an Excel file with the company's financial data
     (manual entry was removed by design - Excel is the single source)
  3. Choose the sentiment source:
        - Fetch fresh headlines from financial news feeds (last 3 days), or
        - Upload a text file of opinions (one per line)
  4. Generate: calibrated resilience prediction -> credit-style rating
     (A+ ... D) -> SHAP drivers -> recency-weighted sentiment ->
     full downloadable HTML report.

Run:
    streamlit run app.py
"""

import base64
import os
import tempfile

import pandas as pd
import streamlit as st

from src.bankruptcy_engine import BankruptcyEngine
from src.sentiment_engine  import SentimentEngine
from src.news_scraper      import NewsScraper
from src.report_generator  import ReportGenerator
from src.column_mapping    import ALL_FEATURES, CRITICAL_FEATURES
from src.config            import NEWS_WINDOW_DAYS, RATING_DICTIONARY

LOGO_PATH = os.path.join("images", "logo.png")


st.set_page_config(
    page_title="Axia - Financial Resilience AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] { font-family: "DM Sans", sans-serif; }

  /* --- Sidebar on the RIGHT --- */
  [data-testid="stAppViewContainer"] { flex-direction: row-reverse; }
  [data-testid="stSidebar"] {
    border-left: 1px solid #334155;
    border-right: none;
  }

  /* --- Centered rounded logo --- */
  .logo-wrap { text-align: center; margin: 0.8rem 0 1.6rem 0; }
  .logo-wrap img {
    max-width: 520px; width: 90%;
    border-radius: 24px;
    box-shadow: 0 4px 24px rgba(15, 23, 42, 0.25);
  }

  /* --- AI note in sidebar --- */
  .ai-note {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 0.8rem;
    color: #94a3b8;
    line-height: 1.6;
    margin-top: 1rem;
  }
  .ai-note strong { color: #cbd5e1; }

  .main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 2rem;
    text-align: center;
  }
  .main-header h1 {
    font-family: "DM Serif Display", serif;
    font-size: 2.6rem;
    color: white;
    margin: 0;
    letter-spacing: -1px;
  }
  .main-header p { color: #94a3b8; font-size: 1rem; margin-top: 0.5rem; }

  .note {
    background: #1e293b;
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem;
    font-size: 0.85rem;
    color: #94a3b8;
    margin-bottom: 1.2rem;
  }
  .stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    color: white; border: none; border-radius: 8px;
    font-family: "DM Sans", sans-serif; font-weight: 600;
    font-size: 1rem; padding: 0.7rem 2rem; width: 100%;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading AI engines...")
def load_engines():
    bankruptcy = BankruptcyEngine()
    bankruptcy.load_model()
    sentiment = SentimentEngine()
    sentiment.load_model()
    return {
        "bankruptcy": bankruptcy,
        "sentiment":  sentiment,
        "scraper":    NewsScraper(),
        "report":     ReportGenerator(),
    }


def _normalize(name: str) -> str:
    """
    Normalizes a feature name so users can write headers naturally.
    'ROA Before Tax C', 'roa_before_tax_c' and 'Roa before tax c'
    all map to the same key.
    """
    return (str(name).strip().lower()
            .replace("-", " ").replace("/", " ")
            .replace("  ", " ").replace(" ", "_"))


_NORMALIZED_FEATURES = {_normalize(f): f for f in ALL_FEATURES}


def parse_excel(uploaded_file) -> dict:
    """
    Accepts either layout:
      Wide: Row 1 = feature names, Row 2 = values
      Tall: Column A = feature name, Column B = value
    Feature names are matched case/format-insensitively.
    """
    df = pd.read_excel(uploaded_file)

    if df.shape[0] >= 1 and df.shape[1] >= 2:
        first_col_norm = [_normalize(v) for v in df.iloc[:, 0]]
        matches = sum(1 for v in first_col_norm if v in _NORMALIZED_FEATURES)
        if matches >= 3:  # clearly a tall layout
            data = {}
            for raw_name, value in zip(df.iloc[:, 0], df.iloc[:, 1]):
                key = _NORMALIZED_FEATURES.get(_normalize(raw_name))
                if key is not None and pd.notna(value):
                    data[key] = float(value)
            return data

    data = {}
    for col in df.columns:
        key = _NORMALIZED_FEATURES.get(_normalize(col))
        if key is not None and not df[col].empty and pd.notna(df[col].iloc[0]):
            data[key] = float(df[col].iloc[0])
    return data

def run_app(engines: dict = None):
    if engines is None:
        engines = load_engines()

    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f'<div class="logo-wrap">'
            f'<img src="data:image/png;base64,{logo_b64}" alt="Axia"/></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("""
        <div class="main-header">
          <h1>Axia</h1>
          <p>Financial Resilience Prediction &amp; Market Sentiment AI</p>
        </div>
        """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Analysis Settings")

        company_name = st.text_input(
            "Company Name",
            placeholder="e.g. Apple, Tesla ...",
            help="Used to search for relevant news headlines",
        )
        stock_symbol = st.text_input(
            "Stock Symbol (optional)",
            placeholder="e.g. AAPL, TSLA",
            help="Improves news search accuracy",
        )

        st.markdown("""
        <div class="ai-note">
          <strong>Powered by AI</strong><br/>
          Axia is driven by artificial-intelligence and machine-learning
          algorithms that analyze financial indicators and market sentiment
          to assess a company's financial resilience.
        </div>
        <div style="margin-top: 1.2rem; padding: 0 0.5rem; font-size: 0.8rem; color: #94a3b8; text-align: center; line-height: 1.5;">
          © All rights reserved.<br/>
          Developed by the Axia Team:<br/>
          <strong style="color: #0a1128;">Tamar Waiss & Rina Kimmel</strong>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Rating scale dictionary (A+ ... D)"):
        legend_df = pd.DataFrame(
            [(g, i["range"], i["meaning"]) for g, i in RATING_DICTIONARY.items()],
            columns=["Grade", "Stability Range", "Meaning"],
        )
        st.dataframe(legend_df, use_container_width=True, hide_index=True)

    st.markdown("#### Upload Financial Data (Excel)")
    st.markdown(f"""
    <div class="note">
      Your Excel file should follow one of these layouts:<br/>
      &nbsp;&nbsp;<strong>Wide:</strong> Row 1 = feature names, Row 2 = values<br/>
      &nbsp;&nbsp;<strong>Tall:</strong> Column A = feature name, Column B = value<br/>
      Feature names are matched flexibly ("Debt Ratio" = "debt_ratio").
      {len(CRITICAL_FEATURES)} critical features are required for a valid prediction.
    </div>
    """, unsafe_allow_html=True)

    form_values: dict = {}
    uploaded = st.file_uploader("Choose Excel file", type=["xlsx", "xls"])
    if uploaded:
        try:
            parsed = parse_excel(uploaded)
            if parsed:
                st.success(f"Loaded {len(parsed)} features from file.")
                form_values = parsed
                with st.expander("Preview loaded data"):
                    preview_df = pd.DataFrame(
                        [(k.replace("_", " ").title(), v) for k, v in parsed.items()],
                        columns=["Feature", "Value"],
                    )
                    st.dataframe(preview_df, use_container_width=True, hide_index=True)
            else:
                st.warning(
                    "No matching feature columns found. Check that headers "
                    "use the feature names shown in the rating dictionary "
                    "documentation (flexible formats accepted)."
                )
        except Exception as e:
            st.error(f"Error reading file: {e}")

    st.markdown("#### Sentiment Source")
    sentiment_source = st.radio(
        "How should market sentiment be collected?",
        ["Fetch from news feeds (API)", "Upload opinions text file"],
        help=(f"News feeds return headlines from the last "
              f"{NEWS_WINDOW_DAYS} days only, weighted by recency."),
        horizontal=True,
    )

    opinions_file = None
    if sentiment_source.startswith("Upload"):
        st.markdown(f"""
        <div class="note">
          Upload a plain-text file with <strong>one opinion per line</strong>.
          Each line is classified as positive / negative / neutral and
          combined into an aggregate sentiment score.
        </div>
        """, unsafe_allow_html=True)
        opinions_file = st.file_uploader(
            "Choose opinions file", type=["txt"]
        )
        if opinions_file:
            st.success("Opinions file loaded.")

    st.divider()
    run_col, _ = st.columns([1, 2])
    with run_col:
        run_clicked = st.button("Generate Financial Report")

    if not run_clicked:
        return

    if not company_name.strip():
        st.error("Please enter a company name in the sidebar.")
        st.stop()
    if not form_values:
        st.error("Please upload an Excel file with the company's financial data.")
        st.stop()

    is_valid, missing = engines["bankruptcy"].validate_input(form_values)
    if not is_valid:
        st.error(
            "Missing critical fields: "
            + ", ".join(mfield.replace("_", " ").title() for mfield in missing)
            + ". Please add them to your Excel file."
        )
        st.stop()

    input_df = pd.DataFrame([{
        f: (form_values.get(f) if form_values.get(f) is not None else 0.0)
        for f in ALL_FEATURES
    }])

    with st.spinner("Running financial analysis..."):

        # 1. Financial prediction (calibrated) + rating
        prediction = engines["bankruptcy"].predict(input_df)

        sentiment_results, ages, news_window = [], None, None
        source_key = "none"

        if sentiment_source.startswith("Fetch"):
            source_key = "news_api"
            news_data  = engines["scraper"].fetch_headlines(
                company_name.strip(), stock_symbol.strip()
            )
            news_window = news_data
            headlines   = news_data["headlines"]
            if headlines:
                scores = engines["sentiment"].analyze_batch(
                    [h["headline"] for h in headlines]
                )
                ages = [h["age_days"] for h in headlines]
                for h, s in zip(headlines, scores):
                    sentiment_results.append({**h, **s})

        elif opinions_file is not None:
            source_key = "uploaded_file"
            lines = [
                ln.strip() for ln in
                opinions_file.read().decode("utf-8", errors="ignore").splitlines()
                if ln.strip()
            ]
            if lines:
                scores = engines["sentiment"].analyze_batch(lines)
                for text, s in zip(lines, scores):
                    sentiment_results.append(
                        {"headline": text, "source": "Uploaded file",
                         "published": "-", **s}
                    )

        sentiment_agg = SentimentEngine.aggregate_score(sentiment_results, ages)

        # 4. Generate report
        with tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            report_path = tmp.name

        engines["report"].generate(
            company_name      = company_name.strip(),
            prediction_result = prediction,
            sentiment_results = sentiment_results,
            sentiment_agg     = sentiment_agg,
            news_window       = news_window,
            sentiment_source  = source_key,
            submitted_data    = form_values,
            model_metrics     = engines["bankruptcy"].metrics,
            output_path       = report_path,
        )

    st.success("Analysis complete!")

    rating       = prediction["rating"]
    rating_info  = RATING_DICTIONARY.get(rating, {})
    rating_color = rating_info.get("color", "#6366f1")

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:1rem;margin:0.5rem 0 1rem 0">
      <span style="font-family:'DM Mono',monospace;font-size:2.2rem;
        font-weight:700;color:{rating_color};border:2px solid {rating_color};
        border-radius:12px;padding:0.2rem 1.2rem">{rating}</span>
      <span style="color:#94a3b8">{rating_info.get('meaning','')}
        &middot; stability range {rating_info.get('range','')}</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stability (calibrated)", f"{prediction['stability_score']:.1%}")
    col2.metric("Risk probability",       f"{prediction['risk_score']:.1%}")
    col3.metric("Sentiment index",        f"{sentiment_agg['score_0_100']:.0f}/100"
                if sentiment_agg["n_items"] else "N/A")
    col4.metric("Texts analyzed",         sentiment_agg["n_items"])

    st.markdown("#### Key Risk Drivers (SHAP)")
    factors_df = pd.DataFrame(prediction["top_risk_factors"])
    if not factors_df.empty:
        factors_df["feature"] = (factors_df["feature"]
                                 .str.replace("_", " ").str.title())
        factors_df["direction"] = factors_df["shap_impact"].apply(
            lambda x: "Pushes toward risk" if x > 0 else "Pushes toward stability"
        )
        st.dataframe(
            factors_df.rename(columns={
                "feature": "Feature", "value": "Value",
                "shap_impact": "SHAP Impact", "direction": "Direction",
            }),
            use_container_width=True, hide_index=True,
        )

    if sentiment_results:
        window_note = ""
        if news_window:
            window_note = (f" (news from {news_window['window_start']} to "
                           f"{news_window['window_end']}, recency-weighted)")
        st.markdown(f"#### Sentiment Summary{window_note}")
        sent_df = pd.DataFrame(sentiment_results)[
            ["headline", "label", "confidence", "source", "published"]
        ].rename(columns={
            "headline": "Headline / Opinion", "label": "Sentiment",
            "confidence": "Confidence", "source": "Source",
            "published": "Published",
        })
        st.dataframe(sent_df, use_container_width=True, hide_index=True)
    else:
        st.info(f"No opinions or fresh headlines (last {NEWS_WINDOW_DAYS} days) "
                "found - the sentiment section is skipped in the report.")

    st.divider()
    with open(report_path, "r", encoding="utf-8") as f:
        report_html = f.read()

    st.download_button(
        label     = "Download Full HTML Report",
        data      = report_html,
        file_name = f"Axia_{company_name.strip().replace(' ', '_')}_Report.html",
        mime      = "text/html",
    )

    st.markdown("#### Report Preview")
    st.components.v1.html(report_html, height=900, scrolling=True)

    os.unlink(report_path)


if __name__ == "__main__":
    run_app()