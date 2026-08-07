"""Tests for the report-table builders."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvbench.reporting.tables import (
    aggregate_per_technique,
    build_results_table,
    table_to_csv,
    table_to_markdown,
)
from kvbench.techniques.base import BenchmarkResult


def make_result(name, ctx=128, bs=1, kv=1.0, tps=10.0):
    return BenchmarkResult(
        technique=name,
        category="test",
        context_length=ctx,
        batch_size=bs,
        analytic_kv_mb=kv,
        peak_gpu_mb=kv * 2,
        cpu_rss_mb=kv * 3,
        wall_time_s=1.0 / tps,
        tokens_per_s=tps,
        time_to_first_token_s=0.1,
        notes="n",
    )


def test_baseline_comparison_columns():
    rows = [
        make_result("Baseline FP16", kv=100, tps=20),
        make_result("INT8", kv=50, tps=22),
    ]
    table = build_results_table(rows)
    int8_row = next(r for r in table if r["Technique"] == "INT8")
    assert int8_row["Memory Saving (%)"] == 50.0
    assert int8_row["Throughput Δ (%)"] == 10.0


def test_table_to_markdown_has_header():
    rows = build_results_table([make_result("A")])
    md = table_to_markdown(rows)
    assert "Technique" in md
    assert "A" in md


def test_csv_roundtrip_keys():
    rows = build_results_table([make_result("A"), make_result("B")])
    csv = table_to_csv(rows)
    assert "Technique" in csv.splitlines()[0]
    assert "A" in csv
    assert "B" in csv


def test_aggregate_per_technique():
    rows = [
        make_result("A", kv=10, tps=5),
        make_result("A", ctx=256, kv=20, tps=10),
    ]
    agg = aggregate_per_technique(rows)
    assert len(agg) == 1
    assert agg[0]["Technique"] == "A"
    assert agg[0]["Avg KV MB"] == 15.0