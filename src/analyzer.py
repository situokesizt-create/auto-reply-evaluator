"""Successful cases only enter quality statistics; errors remain visible."""
from statistics import mean, median

from .config import METRICS


def analyze(results: list[dict]) -> dict:
    """Aggregate dynamically and select the lowest three with a stable ID tie-break."""
    valid = [r for r in results if r.get("overall_score") is not None and not r.get("error")]
    summary = {"total": len(results), "successful": len(valid),
               "failed": len(results) - len(valid), "metrics": {}, "lowest": []}
    if not valid:
        summary.update(average=None, median=None, best=None, worst=None, score_100=None)
        return summary
    values = [r["overall_score"] for r in valid]
    summary.update(average=mean(values), median=median(values), best=max(values),
                   worst=min(values), score_100=mean(values) * 20)
    for key in METRICS:
        metric_values = [r["scores"][key] for r in valid]
        summary["metrics"][key] = {"average": mean(metric_values),
                                   "min": min(metric_values), "max": max(metric_values),
                                   "distribution": [sum(i <= v < i + 1 for v in metric_values)
                                                    for i in range(5)] + [sum(v == 5 for v in metric_values)]}
    summary["lowest"] = sorted(valid, key=lambda r: (r["overall_score"], r["id"]))[:3]
    return summary


def terminal_summary(summary: dict, mode: str) -> str:
    def fmt(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.2f}"
    lines = ["=" * 48, f"Auto Reply Evaluation Report [{mode.upper()}]", "=" * 48,
             f"Cases: {summary['total']} | Successful: {summary['successful']} | Failed: {summary['failed']}",
             f"Average: {fmt(summary['average'])} / 5 | Median: {fmt(summary['median'])}",
             f"Score (100): {fmt(summary['score_100'])}", "Metric Averages"]
    lines += [f"  {key:15s} {stats['average']:.2f}" for key, stats in summary["metrics"].items()]
    lines += [f"Best: {fmt(summary['best'])} | Worst: {fmt(summary['worst'])}", "Lowest 3 Cases"]
    lines += [f"  {index}. {r['id']}  {r['overall_score']:.2f}" for index, r in enumerate(summary["lowest"], 1)]
    return "\n".join(lines)
