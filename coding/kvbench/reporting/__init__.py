"""Reporting utilities: tables, plots, markdown."""

from .tables import build_results_table, table_to_markdown, table_to_csv
from .plots import (
    plot_memory_vs_context,
    plot_throughput_vs_context,
    plot_latency_vs_context,
    plot_memory_vs_batch,
    plot_summary_bar,
)
from .markdown import write_results_markdown

__all__ = [
    "build_results_table",
    "table_to_markdown",
    "table_to_csv",
    "plot_memory_vs_context",
    "plot_throughput_vs_context",
    "plot_latency_vs_context",
    "plot_memory_vs_batch",
    "plot_summary_bar",
    "write_results_markdown",
]