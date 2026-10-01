"""J1 Jean Zay relocation branch of the R3 protocol lineage.

``r3-gpu-contracts-jz-v*`` protocols keep the frozen v19 structure (schema
version 19) and change only what a host move requires: a declared A100 node
class replaces Bootes' hostname and GPU UUIDs, paths move under the Jean Zay
``$WORK`` tree, and three J1 stages replace the Bootes stage list. Each version
is reconstructed exactly from frozen v19 plus the deltas enumerated here, so any
other change fails closed. The branch never extends the linear Bootes chain.

Node-class hardware identifier: SHA-256 of the canonical JSON of the
``environment_binding`` entries named in ``NODE_CLASS_HARDWARE_KEYS``. The
SLURM-assigned hostname and GPU UUID are observed and recorded as evidence;
they are never predeclared.

Versions: each ``JZ_VERSIONS`` entry names its lineage and the stage prefix it
approves once frozen (``frozen_at`` set). ``jz-v1`` is either the unfrozen draft
(nothing approved, never executable) or frozen with only the backend canary
approved. Later stages add one entry each (lineage to the previous frozen jz
file). Source and Bootes-capture hashes are freeze-time values only for versions
that approve the step-0 probe or later; each version binds its own pair source,
whose ``r3_evidence.protocol`` names that version's protocol file.
"""

from __future__ import annotations

import copy
import hashlib
import re
import subprocess
import threading
from pathlib import Path

from .parity_contracts import canonical_json_bytes
from .r3_contracts import (
    V19_PROTOCOL_SHA256,
    _read_json,
    _require,
    _require_git_revision,
    _require_sha256,
    sha256_file,
)

JZ_WORK_ROOT = "/lustre/fswork/projects/rech/xvh/ukl39yh"
JZ_REPO_ROOT = f"{JZ_WORK_ROOT}/wan_experiments_j1"
JZ_PRISTINE_ROOT = f"{JZ_WORK_ROOT}/j1_pristine-Wan2.1"
JZ_RUN_ROOT = f"{JZ_WORK_ROOT}/j1_runs"
JZ_CHECKPOINT_PATH = (
    f"{JZ_WORK_ROOT}/.cache/huggingface/hub/models--Wan-AI--Wan2.1-I2V-14B-480P/"
    "snapshots/6b73f84e66371cdfe870c72acd6826e1d61cf279"
)
# Observed by the user-approved nvidia-smi-only job 444270 (jean-zay-iam07, 2026-10-01).
JZ_DRIVER_VERSION = "595.71.05"
JZ_GPU_MODEL = "NVIDIA A100-SXM4-80GB"
JZ_COMPUTE_CAPABILITY = "8.0"
JZ_HOSTNAME_PATTERN = "^jean-zay-iam[0-9]{2}$"
JZ_SCHEDULER = {
    "system": "slurm",
    "partition": "gpu_p5",
    "account": "xvh@a100",
    "constraint": "a100",
}
JZ_MATRIX_PATH = "docs/r3_test_matrix_v3.json"

STEP0_STAGE = "step0-cross-architecture-probe"
J1_STAGE_IDS = ("backend-kernel-canary", STEP0_STAGE, "single-rank-generator-canary")
# Modules that J1 stages execute beyond the v19 production component set.
JZ_EXTRA_COMPONENTS = (
    "j1_stage.py",
    "j1_step0_probe.py",
    "r3_backend_canary.py",
    "r3_divergence_probe.py",
    "r3_environment.py",
    "r3_jz.py",
    "r3_preflight.py",
)
NODE_CLASS_HARDWARE_KEYS = (
    "binding_kind",
    "driver_version",
    "gpu_model",
    "gpu_compute_capability",
    "visible_gpu_count",
    "hostname_pattern",
    "scheduler",
)
NODE_CLASS_EXACT_KEYS = (
    "python_version",
    "torch_version",
    "cuda_runtime_version",
    "flash_attention_version",
    "flex_attention_version",
    "sam2_version",
    "driver_version",
    "gpu_model",
    "gpu_compute_capability",
    "pyproject_sha256",
    "uv_lock_sha256",
)
NODE_CLASS_OBSERVED_KEYS = frozenset({*NODE_CLASS_EXACT_KEYS, "hostname", "gpu_uuids"})

JZ_PROVISIONING = {
    "cuda_module": "cuda/12.8.0",
    "cuda_toolkit_path": "/lustre/fshomisc/sys/common/nvidia/cuda/12.8",
    "nvcc_version": "12.8.61",
    "flash_attention_distribution_version": "2.8.3+cu128torch2.10",
    "flash_attention_wheel_sha256": "e3976faa725618328617934fb773cae337408f91569b5ca4aadcef833c523f9d",
    "static_sm80_cubins": ["flash_attention_2", "sam2"],
    "uv_version": "0.12.17",
    "uv_tarball_sha256": "fa82fd8dde8e8eefdecada6aa0889666556cfceb690d06e0c3bca49eb3070a63",
    "build_recipe": {
        "path": "tools/j1_jeanzay_sync_env.sh",
        "sha256": "8f1c5cd33435d47853624bed22fa00453d4b1617bdd417e5eccbdf7fcb88833a",
    },
}
# Decision (d), frozen 2026-10-01 before any J1 measurement.
JZ_CROSS_ARCHITECTURE_DECISION = {
    "gate": "step0-divergence-probe-stages-vs-retained-bootes-pristine-capture",
    "bitwise_stage_patterns": ["encoder/clip/*/input/*", "encoder/vae/*/input/*"],
    "relative_l2_max": 0.02,
    "relative_l2_definition": "float64 ||a-b||_2 / ||b||_2, complex via view_as_real",
    "zero_reference_norm": "bitwise",
    "value_stages": "exact-equality",
    "shapes_dtypes_finiteness": "identical-shape-dtype-all-finite",
    "allowed_candidate_only_stages": {
        "official-pristine": [],
        "local-custom": ["encoder/t5/72f6646e35e0e69d/0"],
    },
    "final_latent_vs_bootes_v16": "report-only-max-abs-relative-l2-cosine",
    "decoded_video_vs_bootes_v16": "report-only-psnr",
    "relaxation_after_measurement": "forbidden",
}
JZ_REFERENCE_ROOT = f"{JZ_RUN_ROOT}/reference"
JZ_BOOTES_V16_REFERENCE = {
    route: {
        "latent": {"path": f"{JZ_REFERENCE_ROOT}/bootes-v16-pair/{job}/r3-parity-{route}.npy",
                   "sha256": "3388e2600e60991b97bb35af1f5328e06c7effba049157a9bf4900ad20f1da4b"},
        "video": {"path": f"{JZ_REFERENCE_ROOT}/bootes-v16-pair/{job}/video.mp4",
                  "sha256": "733e50d1f1bf7d702ada58cbf3b1f64886eea0c68e4b6ae380e49818e2d119c5"},
    }
    for route, job in (("upstream", "job-65efc7fbb5c377ef"), ("custom-none", "job-0a01efd9e11fc017"))
}
JZ_REFERENCE_REPORT = {
    "path": "docs/r3_evidence/bootes-divergence-probe-pristine-vs-local-fixed.json",
    "sha256": "f074512e0a9172f5d779d35399ac1992e8488c745c75a1168218f77a5e73c05c",
}
JZ_CLAIM_BOUNDARY = (
    "J1 qualifies one Jean Zay A100 node class for R3's code-level conclusions. Only the "
    "stages approved in this version may run, one attempt each, on one SLURM-allocated A100; "
    "no scientific claim, distributed execution, contract cases, or bitwise cross-architecture claim."
)
JZ_AMENDMENT_RULE = (
    "Stop after each approved stage attempt, including partial failure. Any retry, later stage, "
    "threshold change or binding change requires fresh user approval and a new jz version."
)
JZ_RANK_BASIS = "Inherited R3 declaration for a future amendment; J1 runs no distributed stage."
JZ_STAGE_GATES = (
    {
        "id": "backend-kernel-canary",
        "prerequisites": [],
        "scope": (
            "One bounded FA2, flex-attention and SAM2 CUDA canary on one SLURM-allocated A100 of "
            "the declared node class; checkpoint identity rehash only, no model load."
        ),
        "stop_on": [
            "node-class-mismatch",
            "environment-identity-mismatch",
            "FA3-availability",
            "kernel-error",
            "nonfinite-output",
        ],
    },
    {
        "id": STEP0_STAGE,
        "prerequisites": ["backend-kernel-canary"],
        "scope": (
            "Two sequential isolated single-rank step-0 probe captures (official-pristine, then "
            "local-custom) on one A100, then the frozen CPU cross-architecture comparison against "
            "the retained Bootes pristine capture."
        ),
        "stop_on": [
            "pristine-route-identity-or-isolation-failure",
            "probe-capture-failure",
            "reference-capture-identity-mismatch",
            "cross-architecture-gate-failure",
        ],
    },
    {
        "id": "single-rank-generator-canary",
        "prerequisites": ["backend-kernel-canary", STEP0_STAGE],
        "scope": (
            "One single-rank checkpoint-bound official-pristine/custom-none pair on one A100 under "
            "the unchanged R3 parity rule; comparison with the Bootes v16 latent is report-only."
        ),
        "stop_on": [
            "pristine-route-identity-or-isolation-failure",
            "parity-failure",
            "finiteness-failure",
            "seed-mismatch",
            "hook-cardinality-failure",
        ],
    },
)
JZ_STAGE_SLURM = {
    "backend-kernel-canary": "",
    STEP0_STAGE: "",
    "single-rank-generator-canary": "--qos=qos_gpu_a100-t3 --time=04:00:00 ",
}
JZ_VERSIONS = {
    "r3-gpu-contracts-jz-v1": {
        "lineage": ("r3-gpu-contracts-v19", V19_PROTOCOL_SHA256),
        "draft_allowed": True,
        "approved_stages": ("backend-kernel-canary",),
    },
}
_FROZEN_AT = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def is_jz_protocol(protocol):
    return isinstance(protocol, dict) and protocol.get("protocol_id") in JZ_VERSIONS


def jz_protocol_relative_path(protocol_id):
    """``r3-gpu-contracts-jz-v1`` -> ``docs/r3_protocol_jz_v1.json``."""
    return f"docs/r3_protocol_{jz_version_suffix(protocol_id)}.json"


def require_jz_protocol_path(repo, protocol_path, matrix_path, protocol):
    """Runners accept only the exact repository path of a known jz protocol file."""
    _require(is_jz_protocol(protocol), "J1 runner requires a jz protocol")
    repo = Path(repo).resolve()
    _require(
        Path(protocol_path).resolve() == repo / jz_protocol_relative_path(protocol["protocol_id"])
        and Path(matrix_path).resolve() == repo / JZ_MATRIX_PATH,
        "J1 runner requires the exact jz protocol and v3 matrix paths",
    )


def jz_version_suffix(protocol_id):
    """``r3-gpu-contracts-jz-v1`` -> ``jz_v1``."""
    _require(protocol_id in JZ_VERSIONS, "unknown J1 jz protocol identity")
    return protocol_id.removeprefix("r3-gpu-contracts-").replace("-", "_")


def jz_source_path(protocol_id):
    """Each version binds its own pair source naming that version's protocol file."""
    return f"{JZ_RUN_ROOT}/inputs/{jz_version_suffix(protocol_id)}/source-j1-pair.json"


def jz_authorization(protocol, stage_id):
    _require(is_jz_protocol(protocol) and stage_id in J1_STAGE_IDS, "unknown J1 stage")
    return protocol["execution_amendment"]["authorization_record"][stage_id]


def node_class_hardware_identifier(environment):
    payload = {key: environment[key] for key in NODE_CLASS_HARDWARE_KEYS}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def validate_node_class_observation(expected, observations, environ):
    """Match observed runtime against a node class; return the SLURM/host evidence."""
    _require(
        isinstance(expected, dict) and expected.get("binding_kind") == "node-class",
        "J1 node-class binding is missing",
    )
    if not isinstance(observations, dict) or set(observations) != NODE_CLASS_OBSERVED_KEYS:
        raise ValueError("J1 node-class runtime observations are incomplete")
    mismatches = [key for key in NODE_CLASS_EXACT_KEYS if observations[key] != expected[key]]
    if mismatches:
        raise RuntimeError(
            "J1 runtime differs from the declared node class: " + ", ".join(sorted(mismatches))
        )
    hostname = observations["hostname"]
    if not isinstance(hostname, str) or not re.fullmatch(expected["hostname_pattern"], hostname):
        raise RuntimeError("J1 hostname is outside the declared node class")
    uuids = observations["gpu_uuids"]
    if (
        not isinstance(uuids, list)
        or len(uuids) != expected["visible_gpu_count"]
        or not all(isinstance(value, str) and value.startswith("GPU-") for value in uuids)
    ):
        raise RuntimeError(
            "J1 requires exactly the declared number of visible GPUs (cgroup-constrained allocation)"
        )
    scheduler = expected["scheduler"]
    job_id = environ.get("SLURM_JOB_ID", "")
    if (
        not job_id.isdigit()
        or environ.get("SLURM_JOB_PARTITION") != scheduler["partition"]
        or environ.get("SLURM_JOB_ACCOUNT") != scheduler["account"]
    ):
        raise RuntimeError("J1 must run inside a SLURM job on the declared partition and account")
    return {
        "hostname": hostname,
        "gpu_uuids": list(uuids),
        "slurm_job_id": job_id,
        "slurm_job_nodelist": environ.get("SLURM_JOB_NODELIST"),
    }


def require_node_class_cuda_device(environment, torch):
    """Exactly one visible CUDA device whose model and capability match the node class."""
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
             "J1 requires exactly one visible CUDA device")
    properties = torch.cuda.get_device_properties(0)
    _require(
        properties.name == environment["gpu_model"]
        and f"{properties.major}.{properties.minor}" == environment["gpu_compute_capability"],
        "J1 visible CUDA device differs from the declared node class",
    )
    return {
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": properties.total_memory,
        "torch_uuid": str(getattr(properties, "uuid", "")),
    }


class GpuMemorySampler:
    """Poll ``nvidia-smi`` for the peak device memory used by child processes."""

    def __init__(self, interval_seconds=5.0):
        self.interval_seconds = interval_seconds
        self.peak_mib = None
        self.samples = 0
        self.errors = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        try:
            output = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True, timeout=30,
            ).stdout
            values = [int(line.strip()) for line in output.splitlines() if line.strip()]
        except (OSError, ValueError, subprocess.SubprocessError):
            self.errors += 1
            return
        if values:
            self.samples += 1
            self.peak_mib = max([value for value in (self.peak_mib,) if value is not None] + values)

    def _run(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_seconds)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        self._thread.join()
        self._sample()

    def record(self):
        return {
            "method": "nvidia-smi-memory.used-polling",
            "interval_seconds": self.interval_seconds,
            "peak_memory_used_mib": self.peak_mib,
            "samples": self.samples,
            "sample_errors": self.errors,
        }


def _freeze(value, required, check, label):
    if required:
        check(value, label)
        return value
    _require(value is None, f"J1 {label} must be null when this version does not bind it")
    return None


def _check_frozen_at(value, label):
    _require(isinstance(value, str) and _FROZEN_AT.fullmatch(value), f"{label} must be a UTC timestamp")


def _stage_authorization(stage_id, protocol_id, assets, frozen_values):
    protocol_file = jz_protocol_relative_path(protocol_id)
    command = (
        f"sbatch {JZ_STAGE_SLURM[stage_id]}tools/j1_slurm_stage.sh {stage_id} {protocol_file}"
    )
    common = {
        "authorized_stage": stage_id,
        "authorization_source": "explicit-current-session-user-approval",
        "command": command,
        "checkpoint_inventory": "docs/u1_checkpoint_inventory.json",
        "stop_after_stage": True,
    }
    source = {"path": jz_source_path(protocol_id), "sha256": frozen_values["source_sha256"]}
    if stage_id == "backend-kernel-canary":
        return common | {
            "authorized_operations": [
                "verify-jean-zay-node-class-source-environment-and-checkpoint",
                "fa2-flex-sam2-cuda-canaries",
            ],
            "prohibited_operations": [
                "checkpoint-or-model-load", "generation", "distributed-execution",
                "repeat-attempt", "later-stage-execution", "scientific-claim",
            ],
            "output": f"{JZ_RUN_ROOT}/backend-canary/attempt-1/backend-kernel-canary.json",
        }
    if stage_id == STEP0_STAGE:
        return common | {
            "authorized_operations": [
                "verify-jean-zay-node-class-source-environment-checkpoint-and-prerequisites",
                "two-sequential-isolated-single-rank-step0-probe-captures",
                "cpu-cross-architecture-comparison",
            ],
            "prohibited_operations": [
                "repeat-attempt", "additional-jobs", "full-generation", "distributed-execution",
                "later-stage-execution", "scientific-claim",
            ],
            "source": source,
            "assets": assets,
            "reference_capture": {
                "path": f"{JZ_REFERENCE_ROOT}/bootes-divergence-probe-pristine",
                "manifest_sha256": frozen_values["reference_manifest_sha256"],
                "report": dict(JZ_REFERENCE_REPORT),
            },
            "output": f"{JZ_RUN_ROOT}/step0-probe/attempt-1",
            "min_free_gpu_bytes": 75161927680,
            "job_timeout_seconds": 1800,
        }
    return common | {
        "authorized_operations": [
            "verify-jean-zay-node-class-source-environment-checkpoint-and-prerequisites",
            "two-sequential-isolated-single-rank-generator-jobs-and-offline-parity",
        ],
        "prohibited_operations": [
            "repeat-attempt", "additional-jobs", "distributed-execution",
            "later-stage-execution", "scientific-claim",
        ],
        "source": source,
        "assets": assets,
        "output": f"{JZ_RUN_ROOT}/generator-pair/attempt-1",
        "min_free_gpu_bytes": 75161927680,
        "job_timeout_seconds": 3600,
        "report_only_bootes_v16_reference": JZ_BOOTES_V16_REFERENCE,
    }


def _prerequisite_evidence(amendment, approved, frozen):
    """Frozen versions bind the passed evidence of the last approved stage's prerequisites."""
    declared = amendment.get("prerequisite_evidence")
    if not frozen or not approved:
        _require(declared == [], "J1 draft or unapproved version must not declare prerequisite evidence")
        return []
    prerequisites = next(
        gate["prerequisites"] for gate in JZ_STAGE_GATES if gate["id"] == approved[-1]
    )
    _require(
        isinstance(declared, list)
        and [entry.get("stage_id") if isinstance(entry, dict) else None for entry in declared]
        == prerequisites,
        "J1 prerequisite evidence differs from the approved stage's prerequisites",
    )
    for entry in declared:
        _require(
            set(entry) == {"stage_id", "path", "sha256", "protocol_sha256", "status"}
            and entry["status"] == "passed"
            and isinstance(entry["path"], str)
            and entry["path"].startswith("docs/j1_evidence/")
            and ".." not in Path(entry["path"]).parts,
            "J1 prerequisite evidence entry is malformed",
        )
        _require_sha256(entry["sha256"], "J1 prerequisite evidence sha256")
        _require_sha256(entry["protocol_sha256"], "J1 prerequisite protocol sha256")
    return declared


def expected_jz_protocol(protocol):
    """Reconstruct the only acceptable jz protocol for ``protocol['protocol_id']``."""
    protocol_id = protocol.get("protocol_id")
    spec = JZ_VERSIONS[protocol_id]
    frozen = protocol.get("frozen_at") is not None
    _require(frozen or spec["draft_allowed"], f"{protocol_id} must be frozen")
    approved = list(spec["approved_stages"]) if frozen else []
    _require(approved == list(J1_STAGE_IDS[: len(approved)]) and (bool(approved) or not frozen),
             "J1 stages must be approved in sequence")
    # Canary-only versions never depend on the pair source or the Bootes capture.
    binds_inputs = STEP0_STAGE in approved
    base = Path(__file__).resolve().parents[1] / "docs/r3_protocol_v19.json"
    _require(base.is_file() and sha256_file(base) == V19_PROTOCOL_SHA256,
             "J1 frozen v19 base is unavailable or changed")
    expected = _read_json(base)
    amendment = protocol.get("execution_amendment")
    _require(isinstance(amendment, dict), "J1 execution amendment is missing")
    old = expected["execution_amendment"]

    source = amendment.get("source_binding")
    components = amendment.get("production_component_hashes")
    records = amendment.get("authorization_record")
    _require(
        isinstance(source, dict) and isinstance(components, dict) and isinstance(records, dict)
        and set(records) == set(J1_STAGE_IDS),
        "J1 amendment bindings are missing or malformed",
    )
    _require(
        set(components) == set(old["production_component_hashes"]) | set(JZ_EXTRA_COMPONENTS),
        "J1 production component set differs",
    )
    pair_source = records.get(STEP0_STAGE, {}).get("source", {})
    reference = records.get(STEP0_STAGE, {}).get("reference_capture", {})
    _require(isinstance(pair_source, dict) and isinstance(reference, dict),
             "J1 step-0 authorization is malformed")
    frozen_values = {
        "frozen_at": _freeze(protocol.get("frozen_at"), frozen, _check_frozen_at, "frozen_at"),
        "revision": _freeze(source.get("parent_revision_at_freeze"), frozen,
                            _require_git_revision, "parent revision"),
        "production": _freeze(source.get("parent_production_content_sha256"), frozen,
                              _require_sha256, "production source"),
        "components": {
            name: _freeze(value, frozen, _require_sha256, f"{name} component")
            for name, value in components.items()
        },
        "source_sha256": _freeze(pair_source.get("sha256"), binds_inputs, _require_sha256,
                                 "pair source sha256"),
        "reference_manifest_sha256": _freeze(reference.get("manifest_sha256"), binds_inputs,
                                             _require_sha256, "reference capture manifest sha256"),
    }
    old_environment = old["environment_binding"]
    node_class = {
        "binding_kind": "node-class",
        **{key: old_environment[key] for key in (
            "python_version", "torch_version", "cuda_runtime_version",
            "flash_attention_version", "flex_attention_version", "sam2_version",
        )},
        "driver_version": JZ_DRIVER_VERSION,
        "gpu_model": JZ_GPU_MODEL,
        "gpu_compute_capability": JZ_COMPUTE_CAPABILITY,
        "visible_gpu_count": 1,
        "hostname_pattern": JZ_HOSTNAME_PATTERN,
        "scheduler": dict(JZ_SCHEDULER),
        "pyproject_sha256": old_environment["pyproject_sha256"],
        "uv_lock_sha256": old_environment["uv_lock_sha256"],
    }
    routes = old["route_process_bindings"]
    routes["official-pristine"]["wan_root"] = JZ_PRISTINE_ROOT
    routes["official-pristine"]["adapter"]["path"] = (
        f"{JZ_REPO_ROOT}/multi_sample_inference/r3_pristine_adapter.py"
    )
    routes["local-custom"]["wan_root"] = f"{JZ_REPO_ROOT}/wan2.1"
    assets = {
        name: {"path": f"{JZ_REFERENCE_ROOT}/bootes-inputs/{name}", "sha256": asset["sha256"]}
        for name, asset in old["authorization_record"]["assets"].items()
    }
    approval = "approved" if approved else "required-not-approved"

    expected["protocol_id"] = protocol_id
    expected["lineage"] = {"protocol_id": spec["lineage"][0], "sha256": spec["lineage"][1]}
    expected["frozen_at"] = frozen_values["frozen_at"]
    expected["claim_boundary"] = JZ_CLAIM_BOUNDARY
    expected["amendment_rule"] = JZ_AMENDMENT_RULE
    expected["approvals"] = {"gpu_execution": approval, "hardware_environment": approval}
    expected["runtime_declarations"]["hardware_identifier"] = node_class_hardware_identifier(
        node_class
    )
    expected["execution_state"] = {
        key: ("prepared" if key == "cpu_schema_and_preflight" else "not-run-on-jean-zay-a100")
        for key in expected["execution_state"]
    }
    expected["execution_amendment"] = {
        "amendment_id": protocol_id.replace("contracts", "execution-amendment"),
        "authorization_state": "approved-bounded-stage" if approved else "not-approved",
        "source_binding": old["source_binding"] | {
            "parent_revision_at_freeze": frozen_values["revision"],
            "parent_production_content_sha256": frozen_values["production"],
        },
        "checkpoint_binding": old["checkpoint_binding"] | {"path": JZ_CHECKPOINT_PATH},
        "environment_binding": node_class,
        "provisioning_binding": JZ_PROVISIONING,
        "rank_decision": {
            "intended_fsdp_rank_count": old["rank_decision"]["intended_fsdp_rank_count"],
            "basis": JZ_RANK_BASIS,
        },
        "tolerance_decision": old["tolerance_decision"],
        "cross_architecture_decision": JZ_CROSS_ARCHITECTURE_DECISION,
        "stage_gates": [
            gate | {"authorization": "approved" if gate["id"] in approved else "not-approved"}
            for gate in JZ_STAGE_GATES
        ],
        "upstream_provenance_binding": old["upstream_provenance_binding"],
        "production_component_hashes": frozen_values["components"],
        "route_process_bindings": routes,
        "authorization_record": {
            stage: _stage_authorization(stage, protocol_id, assets, frozen_values)
            for stage in J1_STAGE_IDS
        },
        "prerequisite_evidence": _prerequisite_evidence(amendment, approved, frozen),
        "hook_canary_contract": old["hook_canary_contract"],
    }
    # Never alias module constants: a caller mutating the result must not change the binding.
    return copy.deepcopy(expected)


def validate_jz_protocol_identity(protocol):
    schema_version = protocol.get("schema_version")
    _require(type(schema_version) is int and schema_version == 19,
             "J1 jz protocols use the structural v19 schema")
    lineage = JZ_VERSIONS[protocol["protocol_id"]]["lineage"]
    _require(
        protocol.get("lineage") == {"protocol_id": lineage[0], "sha256": lineage[1]},
        "J1 jz protocol lineage differs from its frozen predecessor",
    )


def validate_jz_execution_amendment(protocol):
    expected = expected_jz_protocol(protocol)
    _require(protocol == expected, f"{protocol['protocol_id']} differs from its exact J1 binding")


def jz_stage_execution_blockers(protocol, stage_id, base_blockers):
    """Blockers for one J1 stage; any pending freeze value blocks execution."""
    blockers = list(base_blockers)
    amendment = protocol["execution_amendment"]
    if amendment["authorization_state"] != "approved-bounded-stage":
        blockers.append("explicit-user-authorization")
    if protocol.get("frozen_at") is None:
        blockers.append("jz-draft-not-frozen")
    stages = {stage["id"]: stage for stage in amendment["stage_gates"]}
    if stage_id not in stages:
        blockers.append(f"unknown-stage:{stage_id}")
        return blockers
    if stages[stage_id]["authorization"] != "approved":
        blockers.append(f"stage-not-authorized:{stage_id}")
    for prerequisite in stages[stage_id]["prerequisites"]:
        if stages[prerequisite]["authorization"] != "approved":
            blockers.append(f"stage-prerequisite-not-authorized:{prerequisite}")
    return blockers


def verify_jz_prerequisites(repo, protocol, stage_id):
    """Rehash the bound passed J1 evidence for every prerequisite of ``stage_id``."""
    gate = next(stage for stage in protocol["execution_amendment"]["stage_gates"]
                if stage["id"] == stage_id)
    declared = protocol["execution_amendment"]["prerequisite_evidence"]
    _require([entry["stage_id"] for entry in declared] == gate["prerequisites"],
             f"J1 {stage_id} prerequisite evidence is incomplete")
    repo = Path(repo).resolve()
    records = {}
    for entry in declared:
        path = (repo / entry["path"]).resolve()
        _require(path.is_relative_to(repo) and path.is_file()
                 and sha256_file(path) == entry["sha256"],
                 f"J1 prerequisite {entry['stage_id']} changed or is missing")
        record = _read_json(path)
        _require(
            record.get("stage_id") == entry["stage_id"]
            and record.get("status") == "passed"
            and record.get("bindings", {}).get("protocol_sha256") == entry["protocol_sha256"]
            and record.get("scope", {}).get("distributed_execution_performed") is False,
            f"J1 prerequisite {entry['stage_id']} has invalid stage/status/binding/scope",
        )
        records[entry["stage_id"]] = record
    return records
