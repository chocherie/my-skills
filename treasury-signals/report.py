"""
report.py — Treasury Signal Dashboard Report Generator
=======================================================
Reads results/ CSV files produced by backtest_runner.py and generates:
  1. Per-signal equity curve chart (strategy vs buy-and-hold)
  2. Rolling 252d Sharpe subplot
  3. Master HTML comparison table (all 25 signals side-by-side)
  4. Signal correlation heatmap
  5. Deflated Sharpe Ratio table (multiple-testing correction)

Usage:
    python report.py                     # generate full HTML report
    python report.py --signal S01        # one signal only (debug)
    python report.py --no-charts         # table only, skip matplotlib

Output: results/report.html
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
REPORT_HTML = RESULTS_DIR / "report.html"
DSR_SCRIPT = Path("/home/user/my-skills/four-eye-backtest-audit/scripts/deflated_sharpe_ratio.py")

# ── colour palette ─────────────────────────────────────────────────────────────
STRATEGY_COLOR = "#1f77b4"  # blue
BAH_COLOR = "#ff7f0e"        # orange
SHARPE_COLOR = "#2ca02c"     # green


# ─────────────────────────────────────────────────────────────────────────────
# 0. DSR helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_dsr_table(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Deflated Sharpe Ratio for each signal using the four-eye script.
    Falls back to raw Sharpe if script unavailable.
    """
    if not DSR_SCRIPT.exists():
        summary = summary.copy()
        summary["dsr"] = float("nan")
        summary["dsr_pass"] = "n/a"
        return summary[["signal_id", "name", "oos_sharpe", "dsr", "dsr_pass"]]

    # Import dynamically
    import importlib.util
    spec = importlib.util.spec_from_file_location("dsr_module", DSR_SCRIPT)
    dsr_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dsr_mod)

    results = []
    n_signals = len(summary)

    for _, row in summary.iterrows():
        equity_path = RESULTS_DIR / f"equity_{row['signal_id']}.csv"
        if not equity_path.exists():
            results.append((row["signal_id"], row["name"], row["oos_sharpe"], float("nan"), "n/a"))
            continue

        eq = pd.read_csv(equity_path, index_col=0, parse_dates=True)
        if "strategy" not in eq.columns:
            results.append((row["signal_id"], row["name"], row["oos_sharpe"], float("nan"), "n/a"))
            continue

        strat_ret = eq["strategy"].pct_change().dropna()
        if len(strat_ret) < 50:
            results.append((row["signal_id"], row["name"], row["oos_sharpe"], float("nan"), "n/a"))
            continue

        try:
            dsr_val = dsr_mod.deflated_sharpe_ratio(
                strat_ret.values,
                n_strategies=n_signals,
            )
            passed = "PASS" if dsr_val > 0.5 else "FAIL"
        except Exception:
            dsr_val = float("nan")
            passed = "n/a"

        results.append((row["signal_id"], row["name"], row["oos_sharpe"], dsr_val, passed))

    return pd.DataFrame(results, columns=["signal_id", "name", "oos_sharpe", "dsr", "dsr_pass"])


# ─────────────────────────────────────────────────────────────────────────────
# 1. Per-signal charts
# ─────────────────────────────────────────────────────────────────────────────

def make_signal_chart(signal_id: str, signal_name: str) -> str:
    """
    Generate a base64-encoded PNG with two subplots:
      top: equity curve (strategy vs B&H)
      bottom: 252d rolling Sharpe
    Returns empty string if data unavailable.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import base64
    from io import BytesIO

    eq_path = RESULTS_DIR / f"equity_{signal_id}.csv"
    rs_path = RESULTS_DIR / "rolling_sharpe.csv"

    if not eq_path.exists():
        return ""

    eq = pd.read_csv(eq_path, parse_dates=["date"]).set_index("date")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"{signal_id} — {signal_name}", fontsize=11, fontweight="bold")

    # ── top: equity curve ──────────────────────────────────────────────────
    ax1 = axes[0]
    if "strategy" in eq.columns:
        ax1.plot(eq.index, eq["strategy"], label="Strategy", color=STRATEGY_COLOR, lw=1.2)
    if "buy_and_hold" in eq.columns:
        ax1.plot(eq.index, eq["buy_and_hold"], label="Buy & Hold", color=BAH_COLOR,
                 lw=1.2, linestyle="--", alpha=0.8)
    ax1.set_ylabel("Cumulative Return", fontsize=9)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="black", lw=0.5, alpha=0.4)

    # ── bottom: rolling Sharpe ────────────────────────────────────────────
    ax2 = axes[1]
    if rs_path.exists():
        rs = pd.read_csv(rs_path, index_col=0, parse_dates=True)
        col = signal_id
        if col in rs.columns:
            ax2.plot(rs.index, rs[col], color=SHARPE_COLOR, lw=1.0)
            ax2.axhline(0, color="black", lw=0.5, alpha=0.4)
            ax2.axhline(0.5, color=SHARPE_COLOR, lw=0.5, alpha=0.3, linestyle=":")
    ax2.set_ylabel("252d Sharpe", fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────

def make_correlation_heatmap(equity_dict: dict) -> str:
    """
    Compute daily strategy return correlation across signals.
    equity_dict: {signal_id: equity_df}
    Returns base64 PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import base64
    from io import BytesIO

    returns = {}
    for sid, eq in equity_dict.items():
        if "strategy" in eq.columns and len(eq) > 50:
            r = eq["strategy"].pct_change().dropna()
            returns[sid] = r

    if len(returns) < 2:
        return ""

    ret_df = pd.DataFrame(returns).dropna(how="all")
    corr = ret_df.corr()

    n = len(corr)
    fig_size = max(8, n * 0.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="RdYlGn",
        center=0,
        vmin=-1, vmax=1,
        annot=(n <= 15),
        fmt=".2f" if n <= 15 else "",
        linewidths=0.3,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Signal Strategy Return Correlation", fontsize=11, fontweight="bold")
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# 3. HTML helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(val, fmt=".3f", fallback="—"):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return fallback
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return fallback


def _color_sharpe(sharpe_str: str) -> str:
    try:
        s = float(sharpe_str)
    except ValueError:
        return sharpe_str
    if s >= 0.5:
        bg = "#c6efce"
    elif s >= 0.0:
        bg = "#ffeb9c"
    else:
        bg = "#ffc7ce"
    return f'<span style="background:{bg};padding:2px 6px;border-radius:3px">{sharpe_str}</span>'


def _color_dsr(dsr_pass: str) -> str:
    if dsr_pass == "PASS":
        return '<span style="color:#006100;font-weight:bold">PASS</span>'
    elif dsr_pass == "FAIL":
        return '<span style="color:#9c0006;font-weight:bold">FAIL</span>'
    return dsr_pass


CATEGORY_NAMES = {
    "A": "Momentum",
    "B": "Macro",
    "C": "Cross-Asset",
    "D": "Positioning",
    "E": "Statistical",
    "F": "Global/NLP",
}


def build_html_report(
    summary: pd.DataFrame,
    dsr_df: pd.DataFrame,
    chart_b64: dict,
    heatmap_b64: str,
    include_charts: bool = True,
) -> str:
    from jinja2 import Environment

    # merge summary + dsr
    df = summary.merge(
        dsr_df[["signal_id", "dsr", "dsr_pass"]],
        on="signal_id",
        how="left",
    )
    df = df.sort_values(["category", "signal_id"])

    # ── master table rows ─────────────────────────────────────────────────
    table_rows = []
    for _, row in df.iterrows():
        sid = row["signal_id"]
        is_sharpe = _color_sharpe(_fmt(row.get("oos_sharpe"), ".3f"))
        bah_sharpe = _color_sharpe(_fmt(row.get("bah_sharpe"), ".3f"))
        dsr_pass_str = _color_dsr(str(row.get("dsr_pass", "n/a")))
        cat = str(row.get("category", ""))
        cat_label = CATEGORY_NAMES.get(cat, cat)

        def _pct(val, dec=1):
            """Format a value already stored as % (e.g. 3.5 → '3.5%')."""
            if val is None:
                return "—"
            try:
                v = float(val)
                if math.isnan(v):
                    return "—"
                return f"{v:.{dec}f}%"
            except (TypeError, ValueError):
                return "—"

        table_rows.append({
            "sid": sid,
            "name": row.get("name", ""),
            "cat": f"{cat} – {cat_label}",
            "is_sharpe": _color_sharpe(_fmt(row.get("is_sharpe"), ".3f")),
            "oos_sharpe": is_sharpe,
            "bah_sharpe": bah_sharpe,
            "overfit": _fmt(row.get("overfit_ratio"), ".2f"),
            "ann_ret": _pct(row.get("oos_annual_return")),
            "total_ret": _pct(row.get("oos_total_ret")),
            "max_dd": _pct(row.get("oos_max_drawdown")),
            "win_rate": _pct(row.get("oos_win_rate")),
            "n_oos_pos": str(int(row["n_oos_positive"])) if not math.isnan(float(row.get("n_oos_positive", float("nan")))) else "—",
            "dsr": _fmt(row.get("dsr"), ".3f"),
            "dsr_pass": dsr_pass_str,
        })

    # ── per-signal chart sections ─────────────────────────────────────────
    chart_sections = []
    if include_charts:
        for _, row in df.iterrows():
            sid = row["signal_id"]
            b64 = chart_b64.get(sid, "")
            if b64:
                chart_sections.append({
                    "sid": sid,
                    "name": row.get("name", ""),
                    "img": b64,
                })

    env = Environment(autoescape=True)
    tmpl = env.from_string(HTML_TEMPLATE)
    return tmpl.render(
        table_rows=table_rows,
        chart_sections=chart_sections,
        heatmap_b64=heatmap_b64,
        include_charts=include_charts,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Jinja2 HTML template
# ─────────────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Treasury Signal Dashboard — Backtest Report</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 20px; background: #fafafa; color: #222; }
  h1 { color: #1a1a2e; border-bottom: 2px solid #1f77b4; padding-bottom: 8px; }
  h2 { color: #16213e; margin-top: 40px; }
  table { border-collapse: collapse; width: 100%; font-size: 12px; background: #fff;
          box-shadow: 0 1px 4px rgba(0,0,0,.12); }
  th { background: #1f77b4; color: #fff; padding: 6px 10px; text-align: left;
       position: sticky; top: 0; }
  td { padding: 5px 10px; border-bottom: 1px solid #eee; white-space: nowrap; }
  tr:hover td { background: #f0f6ff; }
  .signal-card { margin: 30px 0; background: #fff; padding: 16px;
                  box-shadow: 0 1px 4px rgba(0,0,0,.1); border-radius: 4px; }
  .signal-card h3 { margin-top: 0; color: #1f77b4; }
  img { max-width: 100%; height: auto; }
  .toc a { color: #1f77b4; text-decoration: none; margin-right: 12px; }
  .toc a:hover { text-decoration: underline; }
  .note { color: #666; font-size: 11px; margin: 4px 0; }
</style>
</head>
<body>
<h1>10Y US Treasury — Signal Backtest Report</h1>
<p class="note">Walk-forward: IS=504d / OOS=63d / step=63d &nbsp;|&nbsp; Primary instrument: IEF ETF total return</p>

<div class="toc">
  <a href="#master-table">Master Table</a>
  <a href="#dsr-table">DSR Table</a>
  {% if heatmap_b64 %}<a href="#heatmap">Correlation Heatmap</a>{% endif %}
  {% if include_charts %}<a href="#charts">Signal Charts</a>{% endif %}
</div>

<!-- ── Master comparison table ── -->
<h2 id="master-table">Master Comparison Table</h2>
<table>
  <thead>
    <tr>
      <th>ID</th><th>Name</th><th>Category</th>
      <th>IS Sharpe</th><th>OOS Sharpe</th><th>B&amp;H Sharpe</th>
      <th>Overfit Ratio</th><th>OOS Ann. Ret</th><th>OOS Total Ret</th>
      <th>OOS Max DD</th><th>Win Rate</th><th>OOS Wins</th>
      <th>DSR</th><th>DSR Pass</th>
    </tr>
  </thead>
  <tbody>
    {% for r in table_rows %}
    <tr>
      <td><a href="#{{ r.sid }}">{{ r.sid }}</a></td>
      <td>{{ r.name }}</td>
      <td>{{ r.cat }}</td>
      <td>{{ r.is_sharpe | safe }}</td>
      <td>{{ r.oos_sharpe | safe }}</td>
      <td>{{ r.bah_sharpe | safe }}</td>
      <td>{{ r.overfit }}</td>
      <td>{{ r.ann_ret }}</td>
      <td>{{ r.total_ret }}</td>
      <td>{{ r.max_dd }}</td>
      <td>{{ r.win_rate }}</td>
      <td>{{ r.n_oos_pos }}</td>
      <td>{{ r.dsr }}</td>
      <td>{{ r.dsr_pass | safe }}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>

<!-- ── DSR table ── -->
<h2 id="dsr-table">Deflated Sharpe Ratio (Multiple-Testing Correction)</h2>
<p class="note">DSR &gt; 0.5 = statistically significant after correcting for {{ table_rows | length }} strategies tested.</p>
<table>
  <thead>
    <tr><th>ID</th><th>Name</th><th>Raw OOS Sharpe</th><th>DSR</th><th>Decision</th></tr>
  </thead>
  <tbody>
    {% for r in table_rows %}
    <tr>
      <td>{{ r.sid }}</td><td>{{ r.name }}</td>
      <td>{{ r.oos_sharpe | safe }}</td>
      <td>{{ r.dsr }}</td>
      <td>{{ r.dsr_pass | safe }}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>

<!-- ── Correlation heatmap ── -->
{% if heatmap_b64 %}
<h2 id="heatmap">Signal Return Correlation Heatmap</h2>
<div class="signal-card">
  <img src="data:image/png;base64,{{ heatmap_b64 }}" alt="Correlation heatmap">
</div>
{% endif %}

<!-- ── Per-signal charts ── -->
{% if include_charts and chart_sections %}
<h2 id="charts">Per-Signal Equity Curves &amp; Rolling Sharpe</h2>
{% for c in chart_sections %}
<div class="signal-card" id="{{ c.sid }}">
  <h3>{{ c.sid }} — {{ c.name }}</h3>
  <img src="data:image/png;base64,{{ c.img }}" alt="{{ c.sid }} chart">
</div>
{% endfor %}
{% endif %}

</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate treasury signal backtest report")
    parser.add_argument("--signal", default=None, help="Only process this signal ID (e.g. S01)")
    parser.add_argument("--no-charts", action="store_true", help="Skip matplotlib charts")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = RESULTS_DIR / "backtest_summary.csv"
    if not summary_path.exists():
        print(f"[report] ERROR: {summary_path} not found — run backtest_runner.py first")
        sys.exit(1)

    summary = pd.read_csv(summary_path)
    print(f"[report] Loaded summary: {len(summary)} signals")

    # Normalise column names from backtest_runner output → report internal names
    col_map = {
        "ID": "signal_id",
        "Name": "name",
        "Category": "category",
        "IS_Sharpe": "is_sharpe",
        "OOS_Sharpe": "oos_sharpe",
        "OOS_AnnReturn": "oos_annual_return",
        "OOS_TotalRet": "oos_total_ret",
        "OOS_MaxDD": "oos_max_drawdown",
        "OOS_WinRate": "oos_win_rate",
        "BAH_Sharpe": "bah_sharpe",
        "Overfit_Ratio": "overfit_ratio",
        "N_OOS_Positive": "n_oos_positive",
        "sharpe_ratio": "full_sharpe",
        "annual_return": "full_annual_return",
        "max_drawdown": "full_max_drawdown",
        "win_rate": "full_win_rate",
    }
    summary = summary.rename(columns=col_map)
    # Fill any still-missing columns with nan
    for col in ["n_oos_positive", "oos_annual_return", "oos_max_drawdown", "oos_win_rate"]:
        if col not in summary.columns:
            summary[col] = float("nan")

    # Filter to single signal if requested
    if args.signal:
        summary = summary[summary["signal_id"] == args.signal]
        if summary.empty:
            print(f"[report] ERROR: signal {args.signal} not found in summary")
            sys.exit(1)

    # ── DSR table ──────────────────────────────────────────────────────────
    print("[report] Computing DSR table...")
    dsr_df = compute_dsr_table(summary)

    # ── Load equity curves ─────────────────────────────────────────────────
    equity_dict = {}
    for _, row in summary.iterrows():
        sid = row["signal_id"]
        eq_path = RESULTS_DIR / f"equity_{sid}.csv"
        if eq_path.exists():
            eq = pd.read_csv(eq_path, parse_dates=["date"])
            eq = eq.set_index("date")
            equity_dict[sid] = eq

    # ── Per-signal charts ──────────────────────────────────────────────────
    chart_b64 = {}
    if not args.no_charts:
        n = len(summary)
        for i, (_, row) in enumerate(summary.iterrows(), 1):
            sid = row["signal_id"]
            name = row.get("name", sid)
            print(f"[report] Chart {i}/{n}: {sid}...", end="\r")
            chart_b64[sid] = make_signal_chart(sid, name)
        print()

    # ── Correlation heatmap ────────────────────────────────────────────────
    heatmap_b64 = ""
    if not args.no_charts and len(equity_dict) >= 2:
        print("[report] Building correlation heatmap...")
        heatmap_b64 = make_correlation_heatmap(equity_dict)

    # ── Build HTML ─────────────────────────────────────────────────────────
    print("[report] Building HTML report...")
    html = build_html_report(
        summary=summary,
        dsr_df=dsr_df,
        chart_b64=chart_b64,
        heatmap_b64=heatmap_b64,
        include_charts=not args.no_charts,
    )

    REPORT_HTML.write_text(html, encoding="utf-8")
    print(f"[report] Report written to: {REPORT_HTML}")
    print(f"[report] Signals: {len(summary)}")
    print(f"[report] Charts: {len(chart_b64)}")
    print(f"[report] DSR PASS: {(dsr_df['dsr_pass'] == 'PASS').sum()}")


if __name__ == "__main__":
    main()
