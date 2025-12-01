"""Simple reporting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict


def save_text_summary(pop_summary: Dict, stats_results: Dict, path: Path) -> None:
    """Save a human-readable summary."""
    lines = []
    for group, summary in pop_summary.items():
        lines.append(f"Group: {group}")
        lines.append(f"  MTBF: {summary.mtbf:.2f}")
        lines.append(f"  MTTR: {summary.mttr:.2f}")
        lines.append(f"  Availability: {summary.availability:.3f}")
        lines.append(f"  Energy: {summary.energy:.2f}")
        lines.append(f"  Defect rate: {summary.defect_rate:.4f}")
        lines.append(f"  Maintenance cost: {summary.maintenance_cost:.2f}")
        lines.append("")
    lines.append("Statistical tests:")
    for metric, res in stats_results.items():
        lines.append(
            f"{metric}: {res['test']} stat={res['statistic']:.3f} p={res['p_value']:.4f} "
            f"CI95=({res['confidence_interval_low']:.3f}, {res['confidence_interval_high']:.3f})"
        )
    path.write_text("\n".join(lines), encoding="utf-8")
