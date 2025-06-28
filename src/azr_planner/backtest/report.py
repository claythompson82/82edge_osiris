from __future__ import annotations

import base64
import datetime
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Dict, Any

# Import matplotlib only if available and needed, to keep dependencies minimal for other parts.
# However, for this module, it's a direct dependency.
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for saving to file
import matplotlib.pyplot as plt


def _format_metric(value: Optional[float], precision: int = 2, is_percentage: bool = False, is_large_int: bool = False) -> str:
    """Helper to format metrics for display."""
    if value is None:
        return "N/A"
    if is_large_int: # For trade counts
        return f"{int(value):,}"
    if math.isinf(value):
        return "Infinity" # Or some other symbol like ∞
    if is_percentage:
        return f"{value * 100:.{precision}f}%"
    return f"{value:.{precision}f}"

def generate_html_report(
    walk_forward_report_data: Dict[str, Any],
    strategy_equity_curve: Optional[List[float]],
    strategy_equity_timestamps: Optional[List[datetime.datetime]],
    buy_and_hold_equity_curve: Optional[List[float]],
    buy_and_hold_timestamps: Optional[List[datetime.datetime]], # Should align with strategy_equity_timestamps
    trade_pnls: Optional[List[float]],
    out_path: Path
) -> None:
    """
    Generates an HTML report with metrics, equity curve chart, and P&L histogram.
    """
    # Ensure matplotlib doesn't try to use GUI backends if running in a headless environment
    # matplotlib.use('Agg') # Moved to top for module-level setting

    # --- Prepare Metrics ---
    metrics_html = "<h2>Performance Metrics</h2><table>"
    metrics_map = {
        "Mean Sharpe Ratio": (walk_forward_report_data.get("mean_sharpe"), 2, False, False), # Added False for is_large_int
        "Worst Drawdown": (walk_forward_report_data.get("worst_drawdown"), 2, True, False),  # Added False for is_large_int
        "Total Return": (walk_forward_report_data.get("total_return"), 2, True, False),    # Added False for is_large_int
        "Total Trades": (walk_forward_report_data.get("trades"), 0, False, True),         # Trades is int, this one is correct
        "Win Rate": (walk_forward_report_data.get("win_rate"), 2, True, False),           # Added False for is_large_int
    }
    for name, val_config in metrics_map.items():
        # Now val_config is consistently Tuple[Optional[float], int, bool, bool]
        value, precision, is_percentage, is_large_int = val_config
        metrics_html += f"<tr><td>{name}</td><td>{_format_metric(value, precision, is_percentage, is_large_int)}</td></tr>"
    metrics_html += "</table>"

    # --- Generate Equity Curve Chart ---
    equity_chart_html = "<h2>Equity Curve</h2>"
    if strategy_equity_curve and strategy_equity_timestamps and \
       buy_and_hold_equity_curve and buy_and_hold_timestamps and \
       len(strategy_equity_timestamps) == len(strategy_equity_curve) and \
       len(buy_and_hold_timestamps) == len(buy_and_hold_equity_curve):

        try:
            fig_eq, ax_eq = plt.subplots(figsize=(10, 6))
            ax_eq.plot(strategy_equity_timestamps, strategy_equity_curve, label="Strategy Equity") # type: ignore[arg-type]
            ax_eq.plot(buy_and_hold_timestamps, buy_and_hold_equity_curve, label="Buy & Hold Equity", linestyle="--") # type: ignore[arg-type]

            ax_eq.set_title("Strategy vs. Buy & Hold Equity Curve")
            ax_eq.set_xlabel("Date")
            ax_eq.set_ylabel("Equity")
            ax_eq.legend()
            ax_eq.grid(True)
            fig_eq.autofmt_xdate() # Auto format date labels

            img_eq = BytesIO()
            fig_eq.savefig(img_eq, format='png', bbox_inches='tight')
            plt.close(fig_eq) # Close the figure to free memory
            img_eq.seek(0)
            equity_chart_html += f'<img src="data:image/png;base64,{base64.b64encode(img_eq.getvalue()).decode()}" alt="Equity Curve Chart">'
        except Exception as e:
            equity_chart_html += f"<p>Error generating equity curve chart: {e}</p>"
            plt.close(fig_eq) # Ensure closed even on error
    else:
        equity_chart_html += "<p>Equity curve data not available or mismatched.</p>"


    # --- Generate Trade P&L Histogram ---
    pnl_histogram_html = "<h2>Trade P&L Histogram</h2>"
    if trade_pnls and len(trade_pnls) > 0:
        try:
            fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
            ax_hist.hist(trade_pnls, bins=50, edgecolor='black')
            ax_hist.set_title("Distribution of Trade P&L")
            ax_hist.set_xlabel("Profit/Loss per Trade")
            ax_hist.set_ylabel("Number of Trades")
            ax_hist.grid(axis='y', alpha=0.75)

            img_hist = BytesIO()
            fig_hist.savefig(img_hist, format='png', bbox_inches='tight')
            plt.close(fig_hist) # Close the figure
            img_hist.seek(0)
            pnl_histogram_html += f'<img src="data:image/png;base64,{base64.b64encode(img_hist.getvalue()).decode()}" alt="P&L Histogram">'
        except Exception as e:
            pnl_histogram_html += f"<p>Error generating P&L histogram: {e}</p>"
            plt.close(fig_hist) # Ensure closed even on error
    else:
        pnl_histogram_html += "<p>No trade P&L data available for histogram.</p>"


    # --- Assemble HTML ---
    report_title = "Walk-Forward Backtest Report"
    generated_at = walk_forward_report_data.get('report_generated_at', datetime.datetime.now(datetime.timezone.utc).isoformat())
    if isinstance(generated_at, datetime.datetime): # Ensure it's a datetime object before formatting
        generated_at_str = generated_at.strftime('%Y-%m-%d %H:%M:%S %Z')
    else: # It might already be a string from model_dump
        generated_at_str = str(generated_at)

    from_date_val = walk_forward_report_data.get('from_date')
    to_date_val = walk_forward_report_data.get('to_date')

    from_date_str = from_date_val.strftime('%Y-%m-%d') if isinstance(from_date_val, datetime.datetime) else str(from_date_val)
    to_date_str = to_date_val.strftime('%Y-%m-%d') if isinstance(to_date_val, datetime.datetime) else str(to_date_val)


    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{report_title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            table {{ width: 50%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            img {{ max-width: 100%; height: auto; display: block; margin-bottom: 20px; border: 1px solid #ccc; }}
            .container {{ max-width: 900px; margin: auto; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{report_title}</h1>
            <p>Generated at: {generated_at_str}</p>
            <p>Reporting Period: {from_date_str} to {to_date_str}</p>

            {metrics_html}
            {equity_chart_html}
            {pnl_histogram_html}

        </div>
    </body>
    </html>
    """

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)

# Helper for _format_metric, needs math
import math
