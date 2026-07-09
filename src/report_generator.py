"""
report_generator.py
-------------------
Generates the full financial analysis report as a self-contained HTML file.

Report sections:
  - Company header with the credit-style rating badge (A+ ... D)
  - Financial resilience score: calibrated stability gauge + probabilities
  - Rating scale legend (the "dictionary" explaining every grade)
  - SHAP bar chart: top features driving the prediction
  - Media sentiment: aggregate recency-weighted score, distribution
    donut, per-headline table, and the exact news date window used
  - Full table of submitted financial metrics
  - Methodology with HONEST model metrics (recall / precision / AUC),
    read dynamically from training - never hard-coded

All charts are embedded as base64 PNG - the file has no runtime deps.
"""

import base64
import io
from collections import Counter
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.config import AUTHORS, COPYRIGHT_HOLDER, RATING_DICTIONARY

SENTIMENT_COLORS = {
    "positive": "#22c55e",
    "neutral":  "#6366f1",
    "negative": "#ef4444",
}


class ReportGenerator:


    def generate(
        self,
        company_name:      str,
        prediction_result: dict,
        sentiment_results: list,
        sentiment_agg:     dict,
        news_window:       dict | None,
        sentiment_source:  str,
        submitted_data:    dict,
        model_metrics:     dict,
        output_path:       str = "report.html",
    ) -> str:
        """
        Parameters
        ----------
        news_window      : dict with window_start / window_end / window_days
                           (None when sentiment came from an uploaded file)
        sentiment_source : "news_api" | "uploaded_file" | "none"
        model_metrics    : honest metrics dict from BankruptcyEngine
        """
        stability = prediction_result["stability_score"]
        gauge_img = self._build_gauge(stability, prediction_result["rating"])
        shap_img  = self._build_shap_chart(prediction_result["top_risk_factors"])
        sent_img  = (self._build_sentiment_chart(sentiment_results)
                     if sentiment_results else None)

        html = self._render_html(
            company_name, prediction_result, sentiment_results, sentiment_agg,
            news_window, sentiment_source, submitted_data, model_metrics,
            gauge_img, shap_img, sent_img,
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return output_path


    def _build_gauge(self, stability: float, rating: str) -> str:
        """Half-donut gauge of the calibrated stability probability."""
        fig, ax = plt.subplots(figsize=(5, 2.8), subplot_kw={"aspect": "equal"})
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")

        # Left (0%) = distress red ... right (100%) = stable green
        segments = [
            (0,   27,  "#ef4444"),   # 0-15%  D
            (27,  54,  "#f87171"),   # 15-30% C
            (54,  81,  "#f59e0b"),   # 30-45% B-
            (81,  108, "#facc15"),   # 45-60% B
            (108, 135, "#a3e635"),   # 60-75% B+
            (135, 180, "#22c55e"),   # 75-100% A- .. A+
        ]
        for start, end, color in segments:
            theta = np.linspace(np.radians(180 - end), np.radians(180 - start), 100)
            ax.fill_between(
                np.cos(theta), np.sin(theta),
                0.6 * np.cos(theta), 0.6 * np.sin(theta),
                color=color, alpha=0.85,
            )

        angle = np.radians(180 - stability * 180)
        ax.annotate(
            "", xy=(0.72 * np.cos(angle), 0.72 * np.sin(angle)), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5,
                            mutation_scale=18),
        )
        ax.add_patch(plt.Circle((0, 0), 0.07, color="white", zorder=5))

        rating_color = RATING_DICTIONARY.get(rating, {}).get("color", "#6366f1")
        ax.text(0, -0.20, rating, color=rating_color, fontsize=26,
                fontweight="bold", ha="center", va="center",
                fontfamily="monospace")
        ax.text(0, -0.44, "{:.0%} STABILITY".format(stability), color="#94a3b8",
                fontsize=9, ha="center", va="center", fontfamily="monospace")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.55, 1.1)
        ax.axis("off")
        plt.tight_layout(pad=0)
        return self._fig_to_base64(fig)

    def _build_shap_chart(self, top_factors: list) -> str:
        if not top_factors:
            return ""

        labels = [f["feature"].replace("_", " ").title() for f in top_factors]
        values = [f["shap_impact"] for f in top_factors]
        colors = ["#ef4444" if v > 0 else "#22c55e" for v in values]

        fig, ax = plt.subplots(figsize=(7, max(3.5, len(labels) * 0.42)))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")

        bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1],
                       height=0.6, edgecolor="none")
        ax.axvline(0, color="#475569", linewidth=0.8)
        ax.set_xlabel("SHAP impact toward risk", color="#94a3b8", fontsize=9)
        ax.tick_params(colors="#cbd5e1", labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for bar, val in zip(bars, values[::-1]):
            ax.text(
                val + (0.001 if val >= 0 else -0.001),
                bar.get_y() + bar.get_height() / 2,
                "{:+.4f}".format(val), va="center",
                ha="left" if val >= 0 else "right",
                color="white", fontsize=7, fontfamily="monospace",
            )
        plt.tight_layout(pad=0.5)
        return self._fig_to_base64(fig)

    def _build_sentiment_chart(self, sentiment_results: list) -> str:
        counts = Counter(r["label"] for r in sentiment_results)
        labels = list(counts.keys())
        sizes  = list(counts.values())
        colors = [SENTIMENT_COLORS.get(l, "#6366f1") for l in labels]

        fig, ax = plt.subplots(figsize=(4, 3))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")

        _, _, autotexts = ax.pie(
            sizes, labels=None, colors=colors, autopct="%1.0f%%",
            startangle=90, pctdistance=0.75,
            wedgeprops={"width": 0.55, "edgecolor": "#0f172a", "linewidth": 2},
        )
        for at in autotexts:
            at.set_color("white")
            at.set_fontsize(9)

        patches = [mpatches.Patch(color=c, label=l.capitalize())
                   for l, c in zip(labels, colors)]
        ax.legend(handles=patches, loc="lower center",
                  bbox_to_anchor=(0.5, -0.12), ncol=3,
                  frameon=False, labelcolor="#cbd5e1", fontsize=8)
        plt.tight_layout(pad=0.3)
        return self._fig_to_base64(fig)


    def _rating_legend_rows(self, active_rating: str) -> str:
        rows = []
        for grade, info in RATING_DICTIONARY.items():
            active = ' style="background:#1e293b"' if grade == active_rating else ""
            rows.append(
                "<tr" + active + ">"
                '<td><span style="color:' + info["color"]
                + ';font-family:\'DM Mono\',monospace;font-weight:600">'
                + grade + "</span></td>"
                "<td>" + info["range"] + "</td>"
                "<td>" + info["meaning"] + "</td></tr>"
            )
        return "".join(rows)

    def _sentiment_rows(self, sentiment_results: list) -> str:
        rows = []
        for r in sentiment_results[:10]:
            color = SENTIMENT_COLORS.get(r["label"], "#6366f1")
            when  = r.get("published", "-")
            rows.append(
                "<tr><td>" + r.get("headline", r.get("text", "-")) + "</td>"
                '<td><span style="color:' + color + '">' + r["label"] + "</span></td>"
                "<td>" + "{:.0%}".format(r["confidence"]) + "</td>"
                "<td>" + when + "</td></tr>"
            )
        return "".join(rows)

    def _data_rows(self, submitted_data: dict) -> str:
        rows = []
        for k, v in submitted_data.items():
            if v not in (None, "", "nan"):
                rows.append("<tr><td>" + k.replace("_", " ").title()
                            + "</td><td>" + str(v) + "</td></tr>")
        return "".join(rows)

    def _sentiment_section(self, sentiment_results, sentiment_agg,
                           news_window, sentiment_source, sent_img) -> str:
        if not (sentiment_results and sent_img):
            return (
                '<section class="card muted">'
                "<h2>&#128240; Media Sentiment Analysis</h2>"
                "<p>No recent opinions or headlines were available for this "
                "company within the analysis window. Sentiment analysis was "
                "skipped.</p></section>"
            )

        agg_color = SENTIMENT_COLORS.get(sentiment_agg["label"], "#6366f1")
        n = sentiment_agg["n_items"]

        if sentiment_source == "news_api" and news_window:
            source_line = (
                "Based on " + str(n) + " headlines from financial news feeds "
                "&middot; <strong>News window: " + news_window["window_start"]
                + " to " + news_window["window_end"]
                + " (last " + str(news_window["window_days"]) + " days, "
                "recency-weighted)</strong>"
            )
        else:
            source_line = ("Based on " + str(n)
                           + " opinions from an uploaded text file "
                             "(equal weighting)")

        return (
            '<section class="card">'
            "<h2>&#128240; Media Sentiment Analysis</h2>"
            '<p class="subtitle">' + source_line + "</p>"
            '<div class="scores-row" style="grid-template-columns:1fr 1fr 1fr">'
            '<div class="score-box"><div class="value" style="color:' + agg_color
            + '">' + "{:+.2f}".format(sentiment_agg["score"]) + "</div>"
            '<div class="label">AGGREGATE SCORE (-1 to +1)</div></div>'
            '<div class="score-box"><div class="value">'
            + "{:.0f}".format(sentiment_agg["score_0_100"]) + "/100</div>"
            '<div class="label">SENTIMENT INDEX</div></div>'
            '<div class="score-box"><div class="value" style="color:' + agg_color
            + '">' + sentiment_agg["label"].upper() + "</div>"
            '<div class="label">OVERALL TONE</div></div>'
            "</div>"
            + (
                '<div style="border-left:3px solid #f59e0b;background:#0f172a;'
                'border-radius:0 8px 8px 0;padding:0.6rem 1rem;margin:0.4rem 0 1rem 0;'
                'font-size:0.82rem;color:#fbbf24">&#9888;&#65039; '
                "Low sample &mdash; this analysis is based on only " + str(n)
                + " text" + ("s" if n != 1 else "") + ". "
                "Interpret the overall tone with caution.</div>"
                if n < 3 else ""
            )
            + '<div class="two-col">'
            '<div><img src="data:image/png;base64,' + sent_img
            + '" style="width:100%;max-width:320px"/></div>'
            '<div><table class="data-table">'
            "<thead><tr><th>Headline / Opinion</th><th>Sentiment</th>"
            "<th>Confidence</th><th>Published</th></tr></thead>"
            "<tbody>" + self._sentiment_rows(sentiment_results) + "</tbody>"
            "</table></div></div></section>"
        )


    def _render_html(
        self, company_name, pred, sentiment_results, sentiment_agg,
        news_window, sentiment_source, submitted_data, model_metrics,
        gauge_img, shap_img, sent_img,
    ) -> str:
        rating       = pred["rating"]
        rating_info  = RATING_DICTIONARY.get(rating, {})
        rating_color = rating_info.get("color", "#6366f1")
        risk_score   = pred["risk_score"]
        stable_score = pred["stability_score"]
        now_str      = datetime.now().strftime("%B %d, %Y at %H:%M")
        year_str     = str(datetime.now().year)

        verdict = (
            "The model classifies this company as <strong>financially stable"
            "</strong> (rating " + rating + " &mdash; "
            + rating_info.get("meaning", "") + ")."
            if pred["prediction"] == 0 else
            "The model identifies <strong>elevated financial risk</strong> "
            "(rating " + rating + " &mdash; "
            + rating_info.get("meaning", "") + "). "
            "Key indicators suggest potential vulnerability."
        )

        sentiment_html = self._sentiment_section(
            sentiment_results, sentiment_agg, news_window,
            sentiment_source, sent_img)
        data_rows_html   = self._data_rows(submitted_data)
        legend_rows_html = self._rating_legend_rows(rating)

        return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Axia Report &mdash; """ + company_name + """</title>
  <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg:#0f172a; --surface:#1e293b; --border:#334155;
      --text:#e2e8f0; --muted:#94a3b8; --accent:#6366f1;
      --rating:""" + rating_color + """;
    }
    body { background:var(--bg); color:var(--text);
           font-family:"DM Sans",sans-serif; font-size:15px;
           line-height:1.6; padding:2rem 1rem; }
    .container { max-width:900px; margin:0 auto; }
    .report-header { display:flex; justify-content:space-between;
      align-items:flex-start; border-bottom:1px solid var(--border);
      padding-bottom:1.5rem; margin-bottom:2rem; }
    .report-header h1 { font-family:"DM Serif Display",serif;
      font-size:2.2rem; color:white; letter-spacing:-0.5px; }
    .report-header .meta { color:var(--muted); font-size:0.85rem; margin-top:0.3rem; }
    .badge { display:inline-block; padding:0.35rem 1.1rem; border-radius:999px;
      border:1px solid var(--rating);
      background:color-mix(in srgb, var(--rating) 15%, transparent);
      color:var(--rating); font-family:"DM Mono",monospace;
      font-size:1.05rem; font-weight:600; letter-spacing:0.05em; }
    .card { background:var(--surface); border:1px solid var(--border);
      border-radius:12px; padding:1.8rem; margin-bottom:1.5rem; }
    .card.muted { opacity:0.65; }
    .card h2 { font-family:"DM Serif Display",serif;
      font-size:1.3rem; margin-bottom:0.4rem; color:white; }
    .subtitle { color:var(--muted); font-size:0.85rem; margin-bottom:1.2rem; }
    .scores-row { display:grid; grid-template-columns:1fr 1fr 1fr; gap:1rem; margin:1.2rem 0; }
    .score-box { background:#0f172a; border-radius:8px; padding:1rem;
      text-align:center; border:1px solid var(--border); }
    .score-box .value { font-family:"DM Mono",monospace; font-size:1.7rem;
      font-weight:500; color:white; }
    .score-box .label { color:var(--muted); font-size:0.72rem; margin-top:0.2rem; }
    .two-col { display:grid; grid-template-columns:auto 1fr; gap:2rem; align-items:start; }
    .verdict { border-left:3px solid var(--rating); padding:0.8rem 1rem;
      background:#0f172a; border-radius:0 8px 8px 0; margin:1rem 0; font-size:0.95rem; }
    .data-table { width:100%; border-collapse:collapse; font-size:0.85rem; }
    .data-table th { text-align:left; padding:0.5rem 0.7rem;
      border-bottom:1px solid var(--border); color:var(--muted);
      font-weight:500; font-family:"DM Mono",monospace; font-size:0.75rem; }
    .data-table td { padding:0.45rem 0.7rem; border-bottom:1px solid #1e293b; }
    .data-table tr:hover td { background:#16213a; }
    details > summary { cursor:pointer; list-style:none; }
    details > summary::-webkit-details-marker { display:none; }
    details > summary::after { content:"View Table"; float:right; color:var(--muted);
      font-size:0.8rem; transition:transform 0.2s; }
    details[open] > summary::after { transform:rotate(180deg); }
    footer { text-align:center; color:var(--muted); font-size:0.75rem;
      margin-top:3rem; padding-top:1.5rem; border-top:1px solid var(--border); }
    @media (max-width:600px) {
      .scores-row { grid-template-columns:1fr 1fr; }
      .two-col { grid-template-columns:1fr; }
    }
  </style>
</head>
<body>
<div class="container">

  <div class="report-header">
    <div>
      <div style="color:var(--muted);font-family:'DM Mono',monospace;font-size:0.75rem;margin-bottom:0.4rem">
        Axia &middot; FINANCIAL RESILIENCE REPORT
      </div>
      <h1>""" + company_name + """</h1>
      <div class="meta">Generated """ + now_str + """</div>
    </div>
    <div style="text-align:right">
      <div class="badge">RATING: """ + rating + """</div>
    </div>
  </div>

  <section class="card">
    <h2>&#128202; Financial Resilience Score</h2>
    <p class="subtitle">Calibrated machine-learning prediction based on """ + str(len(submitted_data)) + """ submitted financial indicators</p>
    <div class="two-col">
      <img src="data:image/png;base64,""" + gauge_img + """" style="width:260px"/>
      <div>
        <div class="scores-row">
          <div class="score-box">
            <div class="value" style="color:#22c55e">""" + "{:.1%}".format(stable_score) + """</div>
            <div class="label">STABILITY PROBABILITY (CALIBRATED)</div>
          </div>
          <div class="score-box">
            <div class="value" style="color:#ef4444">""" + "{:.1%}".format(risk_score) + """</div>
            <div class="label">RISK PROBABILITY</div>
          </div>
          <div class="score-box">
            <div class="value" style="color:var(--rating)">""" + rating + """</div>
            <div class="label">RATING</div>
          </div>
        </div>
        <div class="verdict">""" + verdict + """</div>
      </div>
    </div>
  </section>

  <section class="card">
    <details>
      <summary>
        <h2 style="display:inline">&#128218; Rating Scale</h2>
        <span class="subtitle" style="display:block;margin:0.4rem 0 0 0">How to read the grade &mdash; click to expand the full scale. The highlighted row is this company's rating.</span>
      </summary>
      <table class="data-table" style="margin-top:1rem">
        <thead><tr><th>Grade</th><th>Stability Range</th><th>Meaning</th></tr></thead>
        <tbody>""" + legend_rows_html + """</tbody>
      </table>
    </details>
  </section>

  <section class="card">
    <h2>&#128269; Key Risk Drivers</h2>
    <p class="subtitle">The financial indicators with the strongest influence on this company's assessment</p>
    <img src="data:image/png;base64,""" + shap_img + """" style="width:100%;max-width:700px"/>
  </section>

  """ + sentiment_html + """

  <section class="card">
    <details>
      <summary>
        <h2 style="display:inline">&#128203; Submitted Financial Data</h2>
        <span class="subtitle" style="display:block;margin:0.4rem 0 0 0">Click to expand the full list of metrics provided for this analysis</span>
      </summary>
      <table class="data-table" style="margin-top:1rem">
        <thead><tr><th>Metric</th><th>Value</th></tr></thead>
        <tbody>""" + data_rows_html + """</tbody>
      </table>
    </details>
  </section>

  <section class="card muted">
    <p style="font-size:0.85rem;color:var(--muted);line-height:1.8">
      <strong style="color:var(--text)">Disclaimer:</strong>
      This report is generated by an academic machine-learning system and does
      not constitute financial advice.
    </p>
  </section>

  <footer>&copy; """ + year_str + """ """ + COPYRIGHT_HOLDER + """ &middot; All rights reserved &middot; Developed by """ + AUTHORS + """</footer>
</div>
</body>
</html>"""


    def _fig_to_base64(self, fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
