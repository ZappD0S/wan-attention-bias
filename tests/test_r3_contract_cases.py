"""CPU-only regressions for the bounded R3 stage-4 single-rank contract-case runner."""

import copy
import json
from pathlib import Path

import pytest

from multi_sample_inference import r3_contract_cases as cases
from multi_sample_inference.r3_contracts import (
    STAGE4_SOURCES,
    STAGE4_V18_SOURCES,
    STAGE4_V19_SOURCES,
    sha256_file,
    stage_execution_blockers,
    validate_protocol,
)

ROOT = Path(__file__).parents[1]
MATRIX = ROOT / "docs/r3_test_matrix_v3.json"
ROOT_INPUT = "/local_scratch2/gzappavi/r3_stage3"


def _synthetic_v17():
    protocol = json.loads((ROOT / "docs/r3_protocol_v16.json").read_text())
    protocol["schema_version"] = 17
    protocol["protocol_id"] = "r3-gpu-contracts-v17"
    protocol["lineage"] = {"protocol_id": "r3-gpu-contracts-v16",
                           "sha256": sha256_file(ROOT / "docs/r3_protocol_v16.json")}
    amendment = protocol["execution_amendment"]
    amendment["amendment_id"] = "r3-gpu-execution-amendment-v17"
    amendment["source_binding"]["parent_revision_at_freeze"] = "1" * 40
    amendment["source_binding"]["parent_production_content_sha256"] = "a" * 64
    amendment["production_component_hashes"]["r3_contracts.py"] = "c" * 64
    amendment["production_component_hashes"]["r3_contract_cases.py"] = "d" * 64
    for stage in amendment["stage_gates"]:
        if stage["id"] == "single-rank-contract-cases":
            stage["authorization"] = "approved"
    amendment["prerequisite_evidence"].append({
        "stage_id": "single-rank-generator-canary",
        "path": "docs/r3_evidence/bootes-generator-pair-unpatchify-passed-attempt.json",
        "sha256": sha256_file(
            ROOT / "docs/r3_evidence/bootes-generator-pair-unpatchify-passed-attempt.json"),
        "protocol_sha256": sha256_file(ROOT / "docs/r3_protocol_v16.json"),
        "status": "passed",
    })
    previous = amendment["authorization_record"]
    output = f"{ROOT_INPUT}/contract-cases-attempt"
    amendment["authorization_record"] = {
        "authorized_stage": "single-rank-contract-cases",
        "authorization_source": previous["authorization_source"],
        "authorized_operations": [
            "verify-bootes-host-source-environment-checkpoint-and-prerequisites",
            "sequential-isolated-single-rank-contract-case-jobs-and-parity-comparisons",
        ],
        "prohibited_operations": previous["prohibited_operations"],
        "command": (
            f"CUDA_VISIBLE_DEVICES={previous['gpu_uuid']} uv run --no-sync --locked "
            "python -m multi_sample_inference.r3_contract_cases "
            "--protocol docs/r3_protocol_v17.json --matrix docs/r3_test_matrix_v3.json "
            f"--output {output}"
        ),
        "checkpoint_inventory": previous["checkpoint_inventory"],
        "sources": {name: {"path": f"{ROOT_INPUT}/input/source-stage4-{name}.json",
                           "sha256": digest} for name, digest in STAGE4_SOURCES.items()},
        "output": output,
        "gpu_uuid": previous["gpu_uuid"],
        "min_free_gpu_bytes": previous["min_free_gpu_bytes"],
        "job_timeout_seconds": previous["job_timeout_seconds"],
        "assets": previous["assets"],
        "stop_after_stage": True,
        "expected_jobs": 200,
        "stop_policy": "stop-on-first-failure",
        "method_parameters": {"regional_prompting": {"beta": 0.5}, "ediff-i": {"strength": 3.0}},
    }
    return protocol


def test_v17_authorizes_only_contract_cases_and_rejects_scope_changes():
    protocol = _synthetic_v17()
    matrix = json.loads(MATRIX.read_text())
    assert validate_protocol(protocol, matrix, matrix_sha256=sha256_file(MATRIX))
    assert stage_execution_blockers(protocol, cases.STAGE) == []
    assert stage_execution_blockers(protocol, "intended-rank-fsdp")
    for change in (
        lambda p: p["execution_amendment"]["production_component_hashes"].update(
            {"fsdp_worker.py": "0" * 64}),
        lambda p: p["execution_amendment"]["authorization_record"].update(output="/tmp/other"),
        lambda p: p["execution_amendment"]["authorization_record"]["sources"]["unipc-flex"]
        .update(sha256="0" * 64),
        lambda p: p["execution_amendment"]["authorization_record"].update(expected_jobs=201),
        lambda p: p["execution_amendment"]["stage_gates"][4].update(authorization="approved"),
        lambda p: p["execution_amendment"]["prerequisite_evidence"].pop(),
        lambda p: p["execution_amendment"]["source_binding"].update(wan_revision="0" * 40),
    ):
        changed = copy.deepcopy(protocol)
        change(changed)
        with pytest.raises(ValueError):
            validate_protocol(changed, matrix, matrix_sha256=sha256_file(MATRIX))


def test_contract_prerequisites_bind_three_passed_records():
    protocol = _synthetic_v17()
    declarations = protocol["execution_amendment"]["prerequisite_evidence"]
    cases._verify_contract_prerequisites(ROOT, declarations)
    for mutate in (lambda d: d.pop(), lambda d: d[2].update(sha256="0" * 64),
                   lambda d: d[2].update(protocol_sha256="0" * 64)):
        changed = copy.deepcopy(declarations)
        mutate(changed)
        with pytest.raises(ValueError):
            cases._verify_contract_prerequisites(ROOT, changed)


def _plan_fixture(protocol):
    """Synthetic manifests covering exactly the runnable single-rank matrix cases."""
    matrix = json.loads(MATRIX.read_text())
    auth = protocol["execution_amendment"]["authorization_record"]
    routes = protocol["execution_amendment"]["route_process_bindings"]
    output = Path(auth["output"])
    names = {("unipc", "flash_attention_2"): "unipc-flash", ("unipc", "flex_attention"): "unipc-flex",
             ("dpm++", "flash_attention_2"): "dpmpp-flash", ("dpm++", "flex_attention"): "dpmpp-flex"}
    sources, manifests = {}, {name: [] for name in names.values()}
    for (solver, backend), name in names.items():
        source = {"smoke_only": False, "scenes": [{"assignments": [{}]}], "video_seeds": [{}],
                  "inference": {"rank_count": 1, "solver": solver},
                  "r3_evidence": {"attention_backend": backend}}
        if backend == "flash_attention_2":
            source["r3_evidence"]["parity_pair"] = {"pair_id": f"pair-{name}"}
        sources[name] = (Path(auth["sources"][name]["path"]), source)
    for index, case in enumerate(matrix["lineage_cases"]):
        selection = case["selection"]
        if (case["disposition"] != "runnable" or selection.get("rank_mode") != "single"
                or case["family"] not in cases.JOB_FAMILIES):
            continue
        name = names[(selection["solver"], selection["attention_backend"])]
        method = selection.get("method", "upstream")
        route = "official-pristine" if method == "upstream" else "local-custom"
        manifest = {
            "job_id": f"job-{index}", "smoke_only": False,
            "identity": {"condition_id": f"condition-{index}"},
            "intervention": {"method": method},
            "source": copy.deepcopy(auth["sources"][name]),
            "inference": {"rank_count": 1},
            "outputs": {"artifact_dir": f"{output}/{name}/job-{index}"},
            "r3_evidence": {
                "route_process": routes[route],
                "case": {"case_id": case["case_id"],
                         "lineage_disposition": {"disposition": "runnable",
                                                 "selection": selection}},
            },
        }
        pair = sources[name][1]["r3_evidence"].get("parity_pair")
        parity_member = pair and (method == "upstream" or (
            method == "none" and selection["mask_configuration"] == "fixed:fixed"
            and selection["mask_sharing"] == "current"
            and not selection["self_attention_masking"]))
        if parity_member:
            label = "upstream" if method == "upstream" else "custom-none"
            manifest["identity"]["condition_id"] = cases.PARITY_CONDITIONS[label]
            manifest["r3_evidence"]["parity_artifact"] = {"route": label,
                                                          "pair_id": pair["pair_id"]}
        manifests[name].append(manifest)
    return matrix, sources, manifests, output


def test_case_plan_covers_all_runnable_single_rank_cases_and_orders_parity_first(monkeypatch):
    protocol = _synthetic_v17()
    monkeypatch.setattr(cases, "validate_checkout_route_binding", lambda _binding: True)
    matrix, sources, manifests, output = _plan_fixture(protocol)
    plan, pairs = cases.validate_case_plan(protocol, matrix, sources, manifests, output)
    assert len(plan) == 200 and set(pairs) == {"unipc-flash", "dpmpp-flash"}
    ordered = cases.order_jobs(plan, pairs)
    assert [m["r3_evidence"].get("parity_artifact", {}).get("route") for _n, m in ordered[:4]] \
        == ["upstream", "custom-none", "upstream", "custom-none"]
    early = {(m["r3_evidence"]["case"]["lineage_disposition"]["selection"].get("method"),
              m["r3_evidence"]["case"]["lineage_disposition"]["selection"].get(
                  "mask_configuration")) for _n, m in ordered[4:16]}
    assert {method for method, _mask in early} >= {
        "none", "regional_prompting", "concept_weaver", "ediff-i"}
    assert {mask for _method, mask in early} >= {"fixed:fixed", "hard:dynamic", "soft:dynamic"}


@pytest.mark.parametrize("mutation", [
    lambda s, m: m["unipc-flex"].pop(),
    lambda s, m: m["unipc-flex"].append(copy.deepcopy(m["unipc-flex"][0])),
    lambda s, m: m["unipc-flex"][1].update(job_id=m["unipc-flex"][0]["job_id"]),
    lambda s, m: m["unipc-flex"][0]["source"].update(sha256="0" * 64),
    lambda s, m: m["unipc-flex"][0]["outputs"].update(artifact_dir="/tmp/elsewhere"),
    lambda s, m: m["unipc-flex"][0]["inference"].update(rank_count=2),
    lambda s, m: m["unipc-flex"][0]["r3_evidence"].update(
        route_process={**m["unipc-flex"][0]["r3_evidence"]["route_process"], "wan_root": "/tmp/x"}),
    lambda s, m: next(x for x in m["unipc-flash"] if x["r3_evidence"].get("parity_artifact"))
    ["r3_evidence"].pop("parity_artifact"),
    lambda s, m: s["unipc-flash"][1]["r3_evidence"].pop("parity_pair"),
    lambda s, m: s["unipc-flex"][1].update(smoke_only=True),
])
def test_case_plan_rejects_missing_duplicate_rerouted_or_unbound_jobs(monkeypatch, mutation):
    protocol = _synthetic_v17()
    monkeypatch.setattr(cases, "validate_checkout_route_binding", lambda _binding: True)
    matrix, sources, manifests, output = _plan_fixture(protocol)
    mutation(sources, manifests)
    with pytest.raises(ValueError):
        cases.validate_case_plan(protocol, matrix, sources, manifests, output)


def _synthetic_v18():
    protocol = json.loads((ROOT / "docs/r3_protocol_v17.json").read_text())
    protocol["schema_version"] = 18
    protocol["protocol_id"] = "r3-gpu-contracts-v18"
    protocol["lineage"] = {"protocol_id": "r3-gpu-contracts-v17",
                           "sha256": sha256_file(ROOT / "docs/r3_protocol_v17.json")}
    amendment = protocol["execution_amendment"]
    amendment["amendment_id"] = "r3-gpu-execution-amendment-v18"
    amendment["source_binding"]["parent_revision_at_freeze"] = "1" * 40
    amendment["source_binding"]["parent_production_content_sha256"] = "a" * 64
    for index, name in enumerate(("r3_contracts.py", "r3_contract_cases.py", "experiment_pipeline.py")):
        amendment["production_component_hashes"][name] = str(index) * 64
    record = amendment["authorization_record"]
    output = f"{ROOT_INPUT}/contract-cases-attempt-2"
    record["command"] = record["command"].replace("r3_protocol_v17.json", "r3_protocol_v18.json").replace(
        f"--output {record['output']}", f"--output {output}")
    record["output"] = output
    record["sources"] = {name: {"path": f"{ROOT_INPUT}/input/source-stage4-v18-{name}.json",
                                "sha256": digest} for name, digest in STAGE4_V18_SOURCES.items()}
    return protocol


def test_v18_reruns_only_contract_cases_with_the_bound_validator_fix():
    protocol = _synthetic_v18()
    matrix = json.loads(MATRIX.read_text())
    assert validate_protocol(protocol, matrix, matrix_sha256=sha256_file(MATRIX))
    assert stage_execution_blockers(protocol, cases.STAGE) == []
    assert stage_execution_blockers(protocol, "intended-rank-fsdp")
    old = json.loads((ROOT / "docs/r3_protocol_v17.json").read_text())["execution_amendment"]
    for change in (
        lambda p: p["execution_amendment"]["production_component_hashes"].update(
            {"fsdp_worker.py": "0" * 64}),
        lambda p: p["execution_amendment"]["production_component_hashes"].update(
            {"experiment_pipeline.py": old["production_component_hashes"]["experiment_pipeline.py"]}),
        lambda p: p["execution_amendment"]["authorization_record"].update(output=old["authorization_record"]["output"]),
        lambda p: p["execution_amendment"]["authorization_record"]["sources"]["unipc-flex"]
        .update(sha256=STAGE4_SOURCES["unipc-flex"]),
        lambda p: p["execution_amendment"]["authorization_record"].update(expected_jobs=195),
        lambda p: p["execution_amendment"]["stage_gates"][4].update(authorization="approved"),
        lambda p: p["execution_amendment"]["source_binding"].update(wan_revision="0" * 40),
    ):
        changed = copy.deepcopy(protocol)
        change(changed)
        with pytest.raises(ValueError):
            validate_protocol(changed, matrix, matrix_sha256=sha256_file(MATRIX))


WAN_V19 = {"commit": "4b822d5b053e8383a5bd76b25f58b75cb3344602",
           "tree": "bc214c83dcecc0c5342741abce1bca96f119e581"}


def _synthetic_v19():
    protocol = json.loads((ROOT / "docs/r3_protocol_v18.json").read_text())
    protocol["schema_version"] = 19
    protocol["protocol_id"] = "r3-gpu-contracts-v19"
    protocol["lineage"] = {"protocol_id": "r3-gpu-contracts-v18",
                           "sha256": sha256_file(ROOT / "docs/r3_protocol_v18.json")}
    amendment = protocol["execution_amendment"]
    amendment["amendment_id"] = "r3-gpu-execution-amendment-v19"
    amendment["source_binding"]["parent_revision_at_freeze"] = "1" * 40
    amendment["source_binding"]["parent_production_content_sha256"] = "a" * 64
    amendment["source_binding"]["wan_revision"] = WAN_V19["commit"]
    amendment["route_process_bindings"]["local-custom"]["checkout"] = dict(WAN_V19)
    for index, name in enumerate(("r3_contracts.py", "r3_contract_cases.py")):
        amendment["production_component_hashes"][name] = str(index) * 64
    record = amendment["authorization_record"]
    output = f"{ROOT_INPUT}/contract-cases-attempt-3"
    record["command"] = record["command"].replace("r3_protocol_v18.json", "r3_protocol_v19.json").replace(
        f"--output {record['output']}", f"--output {output}")
    record["output"] = output
    record["sources"] = {name: {"path": f"{ROOT_INPUT}/input/source-stage4-v19-{name}.json",
                                "sha256": digest} for name, digest in STAGE4_V19_SOURCES.items()}
    return protocol


def test_v19_reruns_only_contract_cases_with_the_bound_mask_sharing_fix():
    protocol = _synthetic_v19()
    matrix = json.loads(MATRIX.read_text())
    assert validate_protocol(protocol, matrix, matrix_sha256=sha256_file(MATRIX))
    assert stage_execution_blockers(protocol, cases.STAGE) == []
    assert stage_execution_blockers(protocol, "intended-rank-fsdp")
    assert cases.SCHEMA_VERSION == 19
    old = json.loads((ROOT / "docs/r3_protocol_v18.json").read_text())["execution_amendment"]
    for change in (
        lambda p: p["execution_amendment"]["production_component_hashes"].update(
            {"experiment_pipeline.py": "0" * 64}),
        lambda p: p["execution_amendment"]["production_component_hashes"].update(
            {"r3_contract_cases.py": old["production_component_hashes"]["r3_contract_cases.py"]}),
        lambda p: p["execution_amendment"]["source_binding"].update(
            wan_revision=old["source_binding"]["wan_revision"]),
        lambda p: p["execution_amendment"]["route_process_bindings"]["local-custom"].update(
            checkout=old["route_process_bindings"]["local-custom"]["checkout"]),
        lambda p: p["execution_amendment"]["authorization_record"].update(output=old["authorization_record"]["output"]),
        lambda p: p["execution_amendment"]["authorization_record"]["sources"]["dpmpp-flex"]
        .update(sha256=STAGE4_V18_SOURCES["dpmpp-flex"]),
        lambda p: p["execution_amendment"]["authorization_record"].update(expected_jobs=153),
        lambda p: p["execution_amendment"]["stage_gates"][4].update(authorization="approved"),
    ):
        changed = copy.deepcopy(protocol)
        change(changed)
        with pytest.raises(ValueError):
            validate_protocol(changed, matrix, matrix_sha256=sha256_file(MATRIX))
