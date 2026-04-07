"""
Generates matplotlib charts at session end, saved to charts/YYYYMMDD_HHMMSS/.

Three charts are produced:
  1. score_over_time.png   — cumulative score line chart
  2. latency_by_sla.png    — avg latency bar chart per SLA × task_type (error bar = P95)
  3. task_breakdown.png    — left: SLA distribution pie chart; right: SLA hit rate bar chart
"""
import os
from datetime import datetime
from pathlib import Path

SLA_ORDER = ["Bronze", "Silver", "Gold", "Platinum", "Diamond", "Stellar", "Glorious", "Supreme"]
TASK_TYPE_SHORT = {
    "generate_until": "gen",
    "loglikelihood": "loglike",
    "loglikelihood_rolling": "rolling",
}
TASK_TYPE_COLORS = {
    "generate_until": "#4C72B0",
    "loglikelihood": "#DD8452",
    "loglikelihood_rolling": "#55A868",
}


def save_charts(state, scheduler, session_start: float) -> str:
    """
    Generate charts and save to charts/YYYYMMDD_HHMMSS/.
    Returns the output directory path as a string.
    Silently skips and returns empty string if matplotlib is not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless environment
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import numpy as np
    except ImportError:
        return ""

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("charts") / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_score_over_time(state, session_start, out_dir, plt, np)
    _plot_latency_by_sla(scheduler, out_dir, plt, np)
    _plot_task_breakdown(state, scheduler, out_dir, plt)

    return str(out_dir)


def _plot_score_over_time(state, session_start, out_dir, plt, np):
    score_ts = list(state.score_ts)
    if not score_ts:
        return

    times = [t - session_start for t, _ in score_ts]
    scores = [s for _, s in score_ts]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, scores, color="#4C72B0", linewidth=2)
    ax.fill_between(times, scores, alpha=0.15, color="#4C72B0")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative Score")
    ax.set_title("Score Over Time")
    ax.grid(True, alpha=0.3)

    # annotate final score
    if scores:
        ax.annotate(
            f"Final: {scores[-1]:.2f}",
            xy=(times[-1], scores[-1]),
            xytext=(-60, 10),
            textcoords="offset points",
            fontsize=10,
            color="#4C72B0",
        )

    fig.tight_layout()
    fig.savefig(out_dir / "score_over_time.png", dpi=150)
    plt.close(fig)


def _plot_latency_by_sla(scheduler, out_dir, plt, np):
    keys = scheduler.latency.all_keys()
    if not keys:
        return

    # collect SLA and task_type combinations that have data
    present_slas = [s for s in SLA_ORDER
                    if any(k[1] == s for k in keys)]
    task_types = sorted(set(k[0] for k in keys))

    x = np.arange(len(present_slas))
    width = 0.8 / max(len(task_types), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(present_slas) * 1.5), 5))

    for i, tt in enumerate(task_types):
        avgs, p95s = [], []
        for sla in present_slas:
            avg = scheduler.latency.avg_latency(tt, sla)
            p95 = scheduler.latency.p95_latency(tt, sla)
            avgs.append(avg if avg is not None else 0)
            p95s.append(p95 if p95 is not None else 0)

        yerr = [max(0, p - a) for a, p in zip(avgs, p95s)]
        offset = (i - len(task_types) / 2 + 0.5) * width
        color = TASK_TYPE_COLORS.get(tt, "#888888")
        bars = ax.bar(x + offset, avgs, width * 0.9,
                      label=TASK_TYPE_SHORT.get(tt, tt),
                      color=color, alpha=0.85,
                      yerr=yerr, capsize=3, error_kw={"ecolor": "gray", "alpha": 0.7})

    ax.set_xticks(x)
    ax.set_xticklabels(present_slas, rotation=15)
    ax.set_ylabel("Latency (s)")
    ax.set_title("Avg Latency by SLA × Task Type  (error bar = P95)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "latency_by_sla.png", dpi=150)
    plt.close(fig)


def _plot_task_breakdown(state, scheduler, out_dir, plt):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # left: SLA distribution pie chart
    sla_counts = {k: v for k, v in state.sla_counts.items() if v > 0}
    if sla_counts:
        labels = list(sla_counts.keys())
        sizes = list(sla_counts.values())
        ax1.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=140)
        ax1.set_title(f"Task Distribution by SLA\n(total accepted: {sum(sizes)})")
    else:
        ax1.text(0.5, 0.5, "No data", ha="center", va="center")
        ax1.set_title("Task Distribution by SLA")

    # right: SLA hit rate bar chart
    keys = scheduler.latency.all_keys()
    hit_labels, hit_values, hit_colors = [], [], []
    for tt, sla in sorted(keys, key=lambda k: (SLA_ORDER.index(k[1]) if k[1] in SLA_ORDER else 99, k[0])):
        hr = scheduler.latency.sla_hit_rate(tt, sla)
        if hr is not None:
            hit_labels.append(f"{sla[:3]}\n{TASK_TYPE_SHORT.get(tt, tt)}")
            hit_values.append(hr * 100)
            hit_colors.append("#55A868" if hr >= 0.9 else "#DD8452" if hr >= 0.7 else "#C44E52")

    if hit_values:
        bars = ax2.bar(range(len(hit_labels)), hit_values, color=hit_colors, alpha=0.85)
        ax2.set_xticks(range(len(hit_labels)))
        ax2.set_xticklabels(hit_labels, fontsize=8)
        ax2.set_ylabel("Hit Rate (%)")
        ax2.set_ylim(0, 105)
        ax2.axhline(100, color="gray", linestyle="--", alpha=0.4)
        ax2.set_title("SLA Hit Rate by Category")
        ax2.grid(True, axis="y", alpha=0.3)
        # annotate bar values
        for bar, val in zip(bars, hit_values):
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 1,
                     f"{val:.0f}%", ha="center", va="bottom", fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No data", ha="center", va="center")
        ax2.set_title("SLA Hit Rate by Category")

    fig.tight_layout()
    fig.savefig(out_dir / "task_breakdown.png", dpi=150)
    plt.close(fig)
