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

Contract chunks (jz-v5 on): R3 stage 4 runs as sequential chunk stages. Their
Jean Zay sources and the frozen ordered job plan are recomputed at validation from
the committed byte-exact Bootes v19 sources and the v3 matrix, so neither can drift.
"""

from __future__ import annotations

import copy
import hashlib
import json
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
# R3 stage 4 on Jean Zay (jz-v5 on): the v19 contract cases, cut into sequential chunks
# because the ~22 h A100 total exceeds the 20 h qos_gpu_a100-t3 wall limit.
CONTRACT_STAGE = "single-rank-contract-cases"
JZ_CONTRACT_CHUNK_COUNT = 4
JZ_CONTRACT_CHUNK_STAGES = tuple(
    f"{CONTRACT_STAGE}-chunk-{index}" for index in range(1, JZ_CONTRACT_CHUNK_COUNT + 1)
)
JZ_ALL_STAGE_IDS = (*J1_STAGE_IDS, *JZ_CONTRACT_CHUNK_STAGES)
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
# ~50 jobs at ~6.5 min (A100 pair with offload) is ~5.5-7 h; 16 h keeps a >2x margin
# under the 20 h qos_gpu_a100-t3 cap. Each later chunk waits for its predecessor
# (afterok) and is cancelled if that dependency can never be satisfied.
JZ_CONTRACT_CHUNK_SLURM = "--qos=qos_gpu_a100-t3 --time=16:00:00 "
JZ_PREVIOUS_CHUNK_JOB_VARIABLE = "$J1_PREVIOUS_CHUNK_JOB_ID"
# Bootes v19 observed 3-4 min per job (one 10.5 min); A100 with offload is ~2x slower.
JZ_CONTRACT_JOB_TIMEOUT_SECONDS = 2700
# The four frozen Bootes v19 stage-4 sources, committed byte-exact (fetched from Bootes,
# SHA-256 equal to the v19 declaration); Jean Zay sources are derived from them.
JZ_BOOTES_V19_STAGE4_SOURCE_ROOT = "docs/j1_inputs"
JZ_CONTRACT_HOOK_CANARY_WAIVER = {
    "stage_id": "checkpoint-load-hook-canary",
    "rationale": (
        "Waived on Jean Zay; the v19 stage-4 prerequisite never ran on an A100. R3's "
        "code-level hook-neutrality conclusion (Pollux v7: observer/no-observer layer-0 "
        "outputs bitwise identical) carries over through the passed J1 gate (within-A100 "
        "official-pristine/custom-none code equivalence), and the jz-v4 A100 pair validated "
        "source-hook cardinality and finiteness for both routes. The user may reject this "
        "waiver before freezing; a rejection requires an A100 hook-canary stage first."
    ),
}
# Frozen jz files bound as lineage by later versions (verified on disk at validation).
JZ_V1_PROTOCOL_SHA256 = "6ef409e45045a0f5217023c2cbd91611d26dfa8fea4846f9cd1f8897fa8aa5db"
JZ_V2_PROTOCOL_SHA256 = "8e7d7a01b20365d8760dbe76f5a8c4f7707c0da276150ac381db0c4584c3784a"
JZ_V3_PROTOCOL_SHA256 = "8b0765f2a781797ca1e50ece9704bbe0e727fcd079f1c0bce3dbc9a5c071755f"
JZ_V4_PROTOCOL_SHA256 = "d94cd662bc00fd87ff3e23562e62fbdc6c77aaf07617db57479d361c7734867a"
# The jz-v4 pair source that produced the passed A100 pair (job 454384).
JZ_V4_PAIR_SOURCE_SHA256 = "f7e2ddf8e67c05a0a6abba301d3b6bf59f9d23d84bc94e6f1928da8fea3851f3"
JZ_V5_CLAIM_BOUNDARY = (
    "R3 stage 4 on one Jean Zay A100 node class: only the approved single-rank contract-case "
    "chunks may run, once each, in order, each on one SLURM-allocated A100 after its "
    "predecessor passed; the canary and pair keep their passed attempts. No distributed "
    "execution, scientific claim, or bitwise cross-architecture claim."
)
# ``attempts`` numbers a stage's one-shot output directory; it defaults to 1. A retry
# after a consumed approval is a new version with the next attempt number; a stage
# that already passed keeps its used attempt, so its runner refuses a rerun.
# ``prerequisite_protocols`` pins which frozen version produced each prerequisite record.
# ``withdrawn_stages`` leave the approval sequence (never approved again), and
# ``stage_gate_overrides`` replace named gate fields for that version only.
# ``generation_settings`` (jz-v4 on) is the only trusted source of Wan ``offload_model``;
# every version without it, and every Bootes protocol, runs with ``offload_model=False``.
# ``drop_report_only_reference`` removes the non-comparable Bootes v16 latent/video binding.
JZ_VERSIONS = {
    "r3-gpu-contracts-jz-v1": {
        "lineage": ("r3-gpu-contracts-v19", V19_PROTOCOL_SHA256),
        "draft_allowed": True,
        "approved_stages": ("backend-kernel-canary",),
    },
    # Canary retry: jz-v1 attempt 1 failed before any CUDA kernel (no git on compute nodes).
    "r3-gpu-contracts-jz-v2": {
        "lineage": ("r3-gpu-contracts-jz-v1", JZ_V1_PROTOCOL_SHA256),
        "draft_allowed": False,
        "approved_stages": ("backend-kernel-canary",),
        "attempts": {"backend-kernel-canary": 2},
    },
    # Step-0 probe after the passed jz-v2 canary (job 448407).
    "r3-gpu-contracts-jz-v3": {
        "lineage": ("r3-gpu-contracts-jz-v2", JZ_V2_PROTOCOL_SHA256),
        "draft_allowed": False,
        "approved_stages": ("backend-kernel-canary", STEP0_STAGE),
        "attempts": {"backend-kernel-canary": 2},
        "prerequisite_protocols": {"backend-kernel-canary": JZ_V2_PROTOCOL_SHA256},
    },
    # Post-measurement amendment: step-0 probe withdrawn (A100 OOM; CUDA initial noise differs
    # across GPU models); one within-A100 parity pair with offload, after the jz-v2 canary.
    "r3-gpu-contracts-jz-v4": {
        "lineage": ("r3-gpu-contracts-jz-v3", JZ_V3_PROTOCOL_SHA256),
        "draft_allowed": False,
        "approved_stages": ("backend-kernel-canary", "single-rank-generator-canary"),
        "attempts": {"backend-kernel-canary": 2},
        "prerequisite_protocols": {"backend-kernel-canary": JZ_V2_PROTOCOL_SHA256},
        "withdrawn_stages": (STEP0_STAGE,),
        "stage_gate_overrides": {
            STEP0_STAGE: {
                "scope": (
                    "Withdrawn by the 2026-10-01 post-measurement amendment: the A100 capture ran "
                    "out of memory and CUDA initial noise differs across GPU models."
                ),
            },
            "single-rank-generator-canary": {"prerequisites": ["backend-kernel-canary"]},
        },
        "generation_settings": {"offload_model": True},
        "drop_report_only_reference": True,
    },
    # R3 stages 4-5 moved to Jean Zay (user decision 2026-10-01); Bootes v19 never runs.
    # Stage 4 as sequential chunks after the jz-v2 canary and the passed jz-v4 pair.
    "r3-gpu-contracts-jz-v5": {
        "lineage": ("r3-gpu-contracts-jz-v4", JZ_V4_PROTOCOL_SHA256),
        "draft_allowed": False,
        "approved_stages": (
            "backend-kernel-canary", "single-rank-generator-canary", *JZ_CONTRACT_CHUNK_STAGES,
        ),
        "attempts": {"backend-kernel-canary": 2},
        "prerequisite_protocols": {
            "backend-kernel-canary": JZ_V2_PROTOCOL_SHA256,
            "single-rank-generator-canary": JZ_V4_PROTOCOL_SHA256,
        },
        "withdrawn_stages": (STEP0_STAGE,),
        "stage_gate_overrides": {
            STEP0_STAGE: {
                "scope": (
                    "Withdrawn by the 2026-10-01 post-measurement amendment: the A100 capture ran "
                    "out of memory and CUDA initial noise differs across GPU models."
                ),
            },
            "single-rank-generator-canary": {"prerequisites": ["backend-kernel-canary"]},
        },
        "generation_settings": {"offload_model": True},
        "drop_report_only_reference": True,
        # The passed pair keeps the jz-v4 source and output, so its runner refuses a rerun.
        "pinned_pair_source": ("r3-gpu-contracts-jz-v4", JZ_V4_PAIR_SOURCE_SHA256),
        "contract_chunks": True,
        "claim_boundary": JZ_V5_CLAIM_BOUNDARY,
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
    pinned = JZ_VERSIONS[protocol_id].get("pinned_pair_source")
    if pinned is not None:
        protocol_id = pinned[0]
    return f"{JZ_RUN_ROOT}/inputs/{jz_version_suffix(protocol_id)}/source-j1-pair.json"


def jz_contract_source_path(protocol_id, name):
    return f"{JZ_RUN_ROOT}/inputs/{jz_version_suffix(protocol_id)}/source-stage4-{name}.json"


def jz_stage_ids(protocol_id):
    """All stage ids of one version: the J1 stages, plus contract chunks from jz-v5 on."""
    chunks = JZ_CONTRACT_CHUNK_STAGES if JZ_VERSIONS[protocol_id].get("contract_chunks") else ()
    return (*J1_STAGE_IDS, *chunks)


def _contract_chunk_gates():
    gates = []
    for index, stage_id in enumerate(JZ_CONTRACT_CHUNK_STAGES, start=1):
        previous = [JZ_CONTRACT_CHUNK_STAGES[index - 2]] if index > 1 else []
        gates.append({
            "id": stage_id,
            "prerequisites": ["backend-kernel-canary", "single-rank-generator-canary", *previous],
            "waived_prerequisites": [dict(JZ_CONTRACT_HOOK_CANARY_WAIVER)],
            "scope": (
                f"Chunk {index} of {JZ_CONTRACT_CHUNK_COUNT} of the runnable v3 single-rank "
                "contract cases: exactly its frozen ordered job ids, each in an isolated route "
                "process on one SLURM-allocated A100 with offload_model=True, stopping at the "
                "first failure; rejected v3 lineage cases remain pre-launch failures."
            ),
            "stop_on": [
                "node-class-mismatch", "previous-chunk-not-passed", "chunk-plan-mismatch",
                "backend-route-mismatch", "mask-identity-failure", "tracker-contract-failure",
                "artifact-failure", "parity-failure",
            ],
        })
    return gates


def jz_stage_gates(protocol_id, approved):
    """Stage gates of one version, with its overrides and approved prefix."""
    overrides = JZ_VERSIONS[protocol_id].get("stage_gate_overrides", {})
    chunks = _contract_chunk_gates() if JZ_VERSIONS[protocol_id].get("contract_chunks") else []
    return [
        gate | overrides.get(gate["id"], {})
        | {"authorization": "approved" if gate["id"] in approved else "not-approved"}
        for gate in (*JZ_STAGE_GATES, *chunks)
    ]


def committed_prerequisites(prerequisites):
    """Prerequisites with committed evidence; earlier chunks are verified at run time."""
    return [stage for stage in prerequisites if stage not in JZ_CONTRACT_CHUNK_STAGES]


def derive_jz_bootes_source(bootes_bytes, bound_sha256, bootes_protocol, protocol_id):
    """Rewrite one hash-bound Bootes source into its Jean Zay form (enumerated fields only).

    Only checkpoint/inventory/asset paths and the protocol/matrix references change.
    """
    bootes_root = "/local_scratch2/gzappavi"
    bootes_repo = f"{bootes_root}/wan_experiments_r3_cpu_20260926"
    bootes_input = f"{bootes_root}/r3_stage3/input"
    _require(hashlib.sha256(bootes_bytes).hexdigest() == bound_sha256,
             "Bootes source differs from its bound SHA-256")
    source = copy.deepcopy(json.loads(bootes_bytes))

    def rewrite(container, key, old, new):
        _require(container.get(key) == old,
                 f"Bootes source field {key} differs from the bound value")
        container[key] = new

    rewrite(source["checkpoint"], "path",
            f"{bootes_root}/hf/hub/models--Wan-AI--Wan2.1-I2V-14B-480P/"
            "snapshots/6b73f84e66371cdfe870c72acd6826e1d61cf279", JZ_CHECKPOINT_PATH)
    rewrite(source["checkpoint"], "inventory", f"{bootes_repo}/docs/u1_checkpoint_inventory.json",
            f"{JZ_REPO_ROOT}/docs/u1_checkpoint_inventory.json")
    _require(len(source["scenes"]) == 1, "Bootes source must have one scene")
    scene = source["scenes"][0]
    inputs = f"{JZ_REFERENCE_ROOT}/bootes-inputs"
    rewrite(scene, "reference_image", f"{bootes_input}/reference.png", f"{inputs}/reference.png")
    for actor in scene["actors"]:
        rewrite(actor, "isolated_image", f"{bootes_input}/reference.png", f"{inputs}/reference.png")
    for actor_id, name in (("actor-left", "mask-left.png"), ("actor-right", "mask-right.png")):
        rewrite(scene["segmentation_masks"], actor_id, f"{bootes_input}/{name}", f"{inputs}/{name}")
    rewrite(source["r3_evidence"], "protocol", f"{bootes_repo}/docs/{bootes_protocol}",
            f"{JZ_REPO_ROOT}/{jz_protocol_relative_path(protocol_id)}")
    rewrite(source["r3_evidence"], "matrix", f"{bootes_repo}/docs/r3_test_matrix_v3.json",
            f"{JZ_REPO_ROOT}/{JZ_MATRIX_PATH}")
    text = json.dumps(source, indent=2) + "\n"
    _require(bootes_root not in text, "derived Jean Zay source still names a Bootes path")
    return text.encode()


def bootes_v19_stage4_source_path(name):
    return f"{JZ_BOOTES_V19_STAGE4_SOURCE_ROOT}/bootes-v19-source-stage4-{name}.json"


def jz_contract_sources(protocol_id, repo=None):
    """Deterministic Jean Zay contract-case source bytes per name, from committed v19 sources."""
    from .r3_contracts import STAGE4_V19_SOURCES  # noqa: PLC0415

    repo = Path(repo or Path(__file__).resolve().parents[1])
    return {
        name: derive_jz_bootes_source(
            (repo / bootes_v19_stage4_source_path(name)).read_bytes(), digest,
            "r3_protocol_v19.json", protocol_id,
        )
        for name, digest in STAGE4_V19_SOURCES.items()
    }


def jz_contract_chunk_plan(sources, matrix):
    """Frozen order (parity pairs first) of every runnable single-rank case, cut into chunks."""
    from .r3_contract_cases import planned_contract_order  # noqa: PLC0415

    ordered, pairs = planned_contract_order(sources, matrix)
    size = -(-len(ordered) // JZ_CONTRACT_CHUNK_COUNT)
    chunks = [ordered[start:start + size] for start in range(0, len(ordered), size)]
    _require(len(chunks) == JZ_CONTRACT_CHUNK_COUNT and all(chunks),
             "J1 contract-case plan does not split into the declared chunks")
    first = {job["job_id"] for job in chunks[0]}
    _require(all(job_id in first for pair in pairs.values() for job_id in pair.values()),
             "J1 contract-case parity pairs must complete within the first chunk")
    return chunks, pairs


def protocol_offload_model(protocol):
    """Wan ``offload_model`` from a validated protocol: only jz-v4+ bindings may enable it."""
    if not is_jz_protocol(protocol):
        return False
    settings = protocol["execution_amendment"].get("generation_settings")
    if settings is None:
        return False
    _require(isinstance(settings, dict) and set(settings) == {"offload_model"}
             and type(settings["offload_model"]) is bool,
             "J1 generation settings are malformed")
    return settings["offload_model"]


def jz_authorization(protocol, stage_id):
    _require(is_jz_protocol(protocol) and stage_id in jz_stage_ids(protocol["protocol_id"]),
             "unknown J1 stage")
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
    attempt = f"attempt-{JZ_VERSIONS[protocol_id].get('attempts', {}).get(stage_id, 1)}"
    protocol_file = jz_protocol_relative_path(protocol_id)
    command = (
        f"sbatch {JZ_STAGE_SLURM.get(stage_id, '')}tools/j1_slurm_stage.sh {stage_id} {protocol_file}"
    )
    common = {
        "authorized_stage": stage_id,
        "authorization_source": "explicit-current-session-user-approval",
        "command": command,
        "checkpoint_inventory": "docs/u1_checkpoint_inventory.json",
        "stop_after_stage": True,
    }
    source = {"path": jz_source_path(protocol_id), "sha256": frozen_values["source_sha256"]}
    if stage_id in JZ_CONTRACT_CHUNK_STAGES:
        return _contract_chunk_authorization(stage_id, protocol_id, assets, frozen_values, common)
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
            "output": f"{JZ_RUN_ROOT}/backend-canary/{attempt}/backend-kernel-canary.json",
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
            "output": f"{JZ_RUN_ROOT}/step0-probe/{attempt}",
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
        "output": f"{JZ_RUN_ROOT}/generator-pair/{attempt}",
        "min_free_gpu_bytes": 75161927680,
        "job_timeout_seconds": 3600,
    } | ({} if JZ_VERSIONS[protocol_id].get("drop_report_only_reference")
         else {"report_only_bootes_v16_reference": JZ_BOOTES_V16_REFERENCE})


def jz_contract_chunk_output(protocol_id, stage_id):
    attempt = f"attempt-{JZ_VERSIONS[protocol_id].get('attempts', {}).get(stage_id, 1)}"
    index = JZ_CONTRACT_CHUNK_STAGES.index(stage_id) + 1
    return f"{JZ_RUN_ROOT}/contract-cases/chunk-{index}/{attempt}"


def _contract_chunk_authorization(stage_id, protocol_id, assets, frozen_values, common):
    contract = frozen_values["contract"]
    v19 = contract["v19_record"]
    index = JZ_CONTRACT_CHUNK_STAGES.index(stage_id)
    protocol_file = jz_protocol_relative_path(protocol_id)
    dependency = (
        "" if index == 0
        else f"--dependency=afterok:{JZ_PREVIOUS_CHUNK_JOB_VARIABLE} --kill-on-invalid-dep=yes "
    )
    chunk_jobs = contract["chunks"][index]
    previous = None
    if index:
        previous_stage = JZ_CONTRACT_CHUNK_STAGES[index - 1]
        previous = {
            "stage_id": previous_stage,
            "record": f"{jz_contract_chunk_output(protocol_id, previous_stage)}/attempt.json",
        }
    return common | {
        "command": (
            f"sbatch --parsable {dependency}{JZ_CONTRACT_CHUNK_SLURM}tools/j1_slurm_stage.sh "
            f"{stage_id} {protocol_file}"
        ),
        "authorized_operations": [
            "verify-jean-zay-node-class-source-environment-checkpoint-and-prerequisites",
            *(["verify-passed-previous-contract-chunk-record"] if index else []),
            "sequential-isolated-single-rank-contract-case-jobs-of-this-chunk-and-parity-comparisons",
        ],
        "prohibited_operations": [
            "repeat-attempt", "additional-jobs", "jobs-outside-this-chunk",
            "distributed-execution", "later-stage-execution", "scientific-claim",
        ],
        "sources": {
            name: {"path": jz_contract_source_path(protocol_id, name), "sha256": digest}
            for name, digest in contract["sources"].items()
        },
        "assets": assets,
        "output": jz_contract_chunk_output(protocol_id, stage_id),
        "min_free_gpu_bytes": 75161927680,
        "job_timeout_seconds": JZ_CONTRACT_JOB_TIMEOUT_SECONDS,
        "stop_policy": v19["stop_policy"],
        "method_parameters": v19["method_parameters"],
        "expected_jobs": v19["expected_jobs"],
        "chunk": {
            "index": index + 1,
            "count": JZ_CONTRACT_CHUNK_COUNT,
            "plan_sha256": contract["plan_sha256"],
            "jobs": chunk_jobs,
            "parity_pairs": contract["pairs"] if index == 0 else {},
        },
        "previous_chunk": previous,
    }


def contract_plan_sha256(chunks, pairs):
    return hashlib.sha256(canonical_json_bytes({"chunks": chunks, "pairs": pairs})).hexdigest()


def _prerequisite_evidence(amendment, approved, frozen, spec, gates):
    """Frozen versions bind the passed evidence of the last approved stage's prerequisites."""
    declared = amendment.get("prerequisite_evidence")
    if not frozen or not approved:
        _require(declared == [], "J1 draft or unapproved version must not declare prerequisite evidence")
        return []
    prerequisites = committed_prerequisites(next(
        gate["prerequisites"] for gate in gates if gate["id"] == approved[-1]
    ))
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
        _require(entry["protocol_sha256"] == spec.get("prerequisite_protocols", {}).get(entry["stage_id"]),
                 f"J1 prerequisite {entry['stage_id']} was not produced by its pinned frozen version")
    return declared


def _contract_values(protocol_id, v19_record):
    """Contract-case sources and chunk plan, recomputed from committed inputs at validation."""
    repo = Path(__file__).resolve().parents[1]
    source_bytes = jz_contract_sources(protocol_id, repo)
    chunks, pairs = jz_contract_chunk_plan(
        {name: json.loads(data) for name, data in source_bytes.items()},
        _read_json(repo / JZ_MATRIX_PATH),
    )
    return {
        "sources": {name: hashlib.sha256(data).hexdigest() for name, data in source_bytes.items()},
        "chunks": chunks,
        "pairs": pairs,
        "plan_sha256": contract_plan_sha256(chunks, pairs),
        "v19_record": v19_record,
    }


def expected_jz_protocol(protocol):
    """Reconstruct the only acceptable jz protocol for ``protocol['protocol_id']``."""
    protocol_id = protocol.get("protocol_id")
    spec = JZ_VERSIONS[protocol_id]
    frozen = protocol.get("frozen_at") is not None
    _require(frozen or spec["draft_allowed"], f"{protocol_id} must be frozen")
    approved = list(spec["approved_stages"]) if frozen else []
    stage_ids = jz_stage_ids(protocol_id)
    sequence = [stage for stage in stage_ids if stage not in spec.get("withdrawn_stages", ())]
    _require(approved == sequence[: len(approved)] and (bool(approved) or not frozen),
             "J1 stages must be approved in sequence")
    # Canary-only versions never depend on the pair source or the Bootes capture.
    binds_inputs = STEP0_STAGE in approved
    # The pair source is bound once any source-consuming stage is approved.
    binds_source = binds_inputs or "single-rank-generator-canary" in approved
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
        and set(records) == set(stage_ids),
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
        "source_sha256": _freeze(pair_source.get("sha256"), binds_source, _require_sha256,
                                 "pair source sha256"),
        "contract": None,
        "reference_manifest_sha256": _freeze(reference.get("manifest_sha256"), binds_inputs,
                                             _require_sha256, "reference capture manifest sha256"),
    }
    pinned = spec.get("pinned_pair_source")
    _require(pinned is None or frozen_values["source_sha256"] == pinned[1],
             "J1 pinned pair source differs from the passed pair's source")
    if spec.get("contract_chunks"):
        frozen_values["contract"] = _contract_values(protocol_id, old["authorization_record"])
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
    gates = jz_stage_gates(protocol_id, approved)

    expected["protocol_id"] = protocol_id
    expected["lineage"] = {"protocol_id": spec["lineage"][0], "sha256": spec["lineage"][1]}
    expected["frozen_at"] = frozen_values["frozen_at"]
    expected["claim_boundary"] = spec.get("claim_boundary", JZ_CLAIM_BOUNDARY)
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
        "stage_gates": gates,
        "upstream_provenance_binding": old["upstream_provenance_binding"],
        "production_component_hashes": frozen_values["components"],
        "route_process_bindings": routes,
        "authorization_record": {
            stage: _stage_authorization(stage, protocol_id, assets, frozen_values)
            for stage in stage_ids
        },
        "prerequisite_evidence": _prerequisite_evidence(amendment, approved, frozen, spec, gates),
        "hook_canary_contract": old["hook_canary_contract"],
    }
    if "generation_settings" in spec:
        expected["execution_amendment"]["generation_settings"] = spec["generation_settings"]
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
    if lineage[0] in JZ_VERSIONS:
        # A jz predecessor must still exist unchanged at its repository path.
        frozen = Path(__file__).resolve().parents[1] / jz_protocol_relative_path(lineage[0])
        _require(frozen.is_file() and sha256_file(frozen) == lineage[1],
                 f"J1 frozen predecessor {lineage[0]} is unavailable or changed")


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
    _require([entry["stage_id"] for entry in declared]
             == committed_prerequisites(gate["prerequisites"]),
             f"J1 {stage_id} prerequisite evidence is incomplete")
    repo = Path(repo).resolve()
    records = {}
    for entry in declared:
        path = (repo / entry["path"]).resolve()
        _require(path.is_relative_to(repo) and path.is_file()
                 and sha256_file(path) == entry["sha256"],
                 f"J1 prerequisite {entry['stage_id']} changed or is missing")
        record = _read_json(path)
        if entry["stage_id"] == "single-rank-generator-canary":
            # Pair records carry their binding at top level and must show passed parity
            # under this version's generation settings.
            valid = (
                record.get("protocol_sha256") == entry["protocol_sha256"]
                and record.get("comparison", {}).get("measurements", {}).get("passed") is True
                and record.get("r3_acceptance") is False
                and record.get("generation_settings")
                == {"offload_model": protocol_offload_model(protocol)}
            )
        else:
            valid = (
                record.get("bindings", {}).get("protocol_sha256") == entry["protocol_sha256"]
                and record.get("scope", {}).get("distributed_execution_performed") is False
            )
        _require(
            record.get("stage_id") == entry["stage_id"] and record.get("status") == "passed"
            and valid,
            f"J1 prerequisite {entry['stage_id']} has invalid stage/status/binding/scope",
        )
        records[entry["stage_id"]] = record
    return records
