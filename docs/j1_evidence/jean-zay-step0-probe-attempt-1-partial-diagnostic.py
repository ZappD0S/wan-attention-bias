"""Diagnostic only: compare the partial A100 pristine capture (failed attempt) with Bootes."""
import json
import sys
from pathlib import Path

from multi_sample_inference.j1_step0_probe import _compare_stage
from multi_sample_inference.r3_jz import JZ_CROSS_ARCHITECTURE_DECISION

ref_dir, cand_dir, out = map(Path, sys.argv[1:4])
ref = {r["name"]: r for r in json.loads((ref_dir / "probe-manifest.json").read_text())["records"]}
cand = json.loads((cand_dir / "probe-manifest.json").read_text())
rows = []
for record in cand["records"]:
    reference = ref.get(record["name"])
    if reference is None:
        rows.append({"name": record["name"], "passed": False, "reason": "candidate-only"})
        continue
    rows.append(_compare_stage(ref_dir, cand_dir, reference, record, JZ_CROSS_ARCHITECTURE_DECISION))
report = {"record_kind": "j1-step0-partial-diagnostic", "gate": False,
          "candidate_status": cand["status"], "stages": rows}
assert not out.exists()
out.write_text(json.dumps(report, indent=1, default=str) + "\n")
for row in rows:
    print(row["name"], row.get("rule"), row.get("passed"), row.get("relative_l2", ""), row.get("reason", ""))
