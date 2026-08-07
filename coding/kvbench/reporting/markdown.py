"""Markdown report writer.

Produces a single ``report.md`` with embedded tables and plots so the
results can be inspected without re-running anything.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from ..techniques.base import BenchmarkResult
from .tables import (
    aggregate_per_technique,
    build_results_table,
    table_to_csv,
    table_to_markdown,
)


def write_results_markdown(
    results: Sequence[BenchmarkResult],
    output_dir: str | Path,
    config_dict: Optional[Dict] = None,
    also_write_csv: bool = True,
) -> List[Path]:
    out = []
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_results_table(results)
    agg_rows = aggregate_per_technique(results)

    md_lines = []
    md_lines.append("# KV Cache Optimization Benchmark Report\n")
    md_lines.append(
        "Generated on "
        + dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        + "\n"
    )
    if config_dict:
        md_lines.append("## Configuration\n")
        md_lines.append("```yaml\n" + _yaml_dump(config_dict) + "\n```\n")
    md_lines.append("\n## Per-Technique Summary\n")
    md_lines.append(
        table_to_markdown(
            agg_rows,
            columns=[
                "Technique",
                "Category",
                "Avg Peak GPU MB",
                "Avg CPU RSS MB",
                "Avg KV MB",
                "Avg Latency (s)",
                "Avg Tokens/s",
                "Avg TTFT (s)",
                "Notes",
            ],
        )
    )
    md_lines.append("\n## Per-Cell Results\n")
    md_lines.append(table_to_markdown(rows))
    md_lines.append("\n## Plots\n")
    md_lines.append(
        "- ![Memory vs Context](memory_vs_context.png)\n"
        "- ![Throughput vs Context](throughput_vs_context.png)\n"
        "- ![Latency vs Context](latency_vs_context.png)\n"
        "- ![Memory vs Batch](memory_vs_batch.png)\n"
        "- ![Summary Bar](summary_tokens_per_s.png)\n"
    )

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(md_lines), encoding="utf-8")
    out.append(report_path)

    if also_write_csv:
        csv_path = out_dir / "results.csv"
        csv_path.write_text(table_to_csv(rows), encoding="utf-8")
        out.append(csv_path)
        agg_csv_path = out_dir / "results_aggregated.csv"
        agg_csv_path.write_text(table_to_csv(agg_rows), encoding="utf-8")
        out.append(agg_csv_path)

    return out


def _yaml_dump(d: Dict) -> str:
    """Tiny YAML dumper for our scalar-only dicts."""
    import yaml
    return yaml.safe_dump(d, sort_keys=False)