"""--check on the normaliser summary tolerates float round-off, not real change."""
from __future__ import annotations

from pathlib import Path

from scripts.summarize_normaliser_comparison import build, matches

REPORTS = Path(__file__).resolve().parent.parent / "reports" / "cross_section"


def test_committed_summary_matches_its_rebuild():
    rebuilt = build(REPORTS).to_csv(index=False).encode()
    assert matches(REPORTS / "normaliser_comparison_summary.csv", rebuilt)


def test_one_ulp_float_difference_is_not_stale(tmp_path):
    # pandas 2.x and 3.x can parse the same decimal string one ulp apart
    target = tmp_path / "summary.csv"
    target.write_text("name,value\na,0.3094762905654838\n")
    assert matches(target, b"name,value\na,0.30947629056548387\n")


def test_real_change_or_missing_file_is_stale(tmp_path):
    target = tmp_path / "summary.csv"
    target.write_text("name,value\na,0.3094762905654838\n")
    assert not matches(target, b"name,value\na,0.3095\n")
    assert not matches(tmp_path / "missing.csv", b"name,value\na,1.0\n")
