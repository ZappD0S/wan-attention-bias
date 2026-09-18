import copy
import json
from pathlib import Path

import pytest

from tools.r3_upstream_provenance import REQUIRED_SURFACE_PATHS, validate_evidence

ROOT = Path(__file__).parents[1]
RECORD = ROOT / "docs/r3_upstream_provenance.json"
SURFACE = ROOT / "docs/r3_evidence/upstream-execution-surface.json"
WHOLE_TREE_DIFF = ROOT / "docs/r3_evidence/upstream-whole-tree.diff"


def _artifacts():
    return (
        json.loads(RECORD.read_text()),
        json.loads(SURFACE.read_text()),
        WHOLE_TREE_DIFF.read_bytes(),
    )


def test_r3_p_artifacts_bind_complete_pristine_route_decision():
    record, surface, diff_bytes = _artifacts()

    assert validate_evidence(record, surface, diff_bytes)
    assert record["official_source"]["repository_url"] == (
        "https://github.com/Wan-Video/Wan2.1.git"
    )
    assert record["official_source"]["commit"] == (
        "7c81b2f27defa56c7e627a4b6717c8f2292eee58"
    )
    assert record["local_source"]["commit"] == (
        "00bde1e719ccb56c66a01a1f18a70c49b278c202"
    )
    assert record["whole_tree"]["counts"] == {
        "added": 11,
        "changed": 10,
        "identical": 49,
        "removed": 0,
    }
    assert set(surface["required_paths"]) == REQUIRED_SURFACE_PATHS
    assert len(record["classifications"]) == 21
    assert {
        item["path"]
        for item in record["classifications"]
        if item["execution_surface"]
    } == {
        item["path"]
        for item in surface["comparison"]
        if item["status"] != "identical"
    }
    assert record["baseline_decision"] == "separate-pristine-route-required"
    assert record["scope"] == {
        "checkpoint_or_model_loaded": False,
        "generation_run": False,
        "gpu_used": False,
        "provenance_only": True,
    }


def test_r3_p_rejects_moved_official_revision():
    record, surface, diff_bytes = _artifacts()
    record["official_source"]["commit"] = "0" * 40

    with pytest.raises(ValueError, match="official execution-surface commit mismatch"):
        validate_evidence(record, surface, diff_bytes)


def test_r3_p_rejects_changed_local_tree():
    record, surface, diff_bytes = _artifacts()
    record["local_source"]["tree"] = "f" * 40

    with pytest.raises(ValueError, match="local execution-surface tree mismatch"):
        validate_evidence(record, surface, diff_bytes)


def test_r3_p_rejects_incomplete_classification():
    record, surface, diff_bytes = _artifacts()
    record["classifications"].pop()

    with pytest.raises(ValueError, match="classification is incomplete"):
        validate_evidence(record, surface, diff_bytes)


def test_r3_p_rejects_broad_or_unhashed_instrumentation_allowlist():
    record, surface, diff_bytes = _artifacts()
    attention = next(
        item
        for item in record["classifications"]
        if item["path"] == "wan/modules/attention.py"
    )
    attention["instrumentation_allowlist"].pop("patch_sha256")

    with pytest.raises(ValueError, match="broad or unhashed instrumentation allowlist"):
        validate_evidence(record, surface, diff_bytes)


def test_r3_p_rejects_inconsistent_baseline_decision():
    record, surface, diff_bytes = _artifacts()
    record["baseline_decision"] = "instrumented-vendor-derived-route"

    with pytest.raises(ValueError, match="baseline decision is inconsistent"):
        validate_evidence(record, surface, diff_bytes)


def test_r3_p_rejects_artifact_tampering():
    record, surface, diff_bytes = _artifacts()

    with pytest.raises(ValueError, match="whole-tree diff hash mismatch"):
        validate_evidence(record, surface, diff_bytes + b"tamper")

    changed_surface = copy.deepcopy(surface)
    changed_surface["entrypoints"].append("wan.moved")
    with pytest.raises(ValueError, match="execution-surface hash mismatch"):
        validate_evidence(record, changed_surface, diff_bytes)
