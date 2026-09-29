"""One-shot diagnostic that localizes pristine/custom-none numerical divergence.

This is a diagnostic tool, not an R3 acceptance stage: it binds no protocol,
never promotes evidence and always records ``r3_acceptance: false``. Inside one
isolated route process it wraps Python call boundaries (encoders, the DiT
forward and forward hooks on DiT submodules), saves full tensors for the
conditioning inputs and the first conditional DiT forward, then stops after
the step-0 negative forward. Every wrapper and hook is removed on exit and no
Wan source file is edited.

``--replay`` substitutes a reference capture's DiT inputs into the local
custom-none forward, separating conditioning differences from DiT arithmetic.
``compare`` reports the first stage that is not bitwise identical.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import pickle
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from .parity_contracts import canonical_json_bytes, compare_outputs
from .r3_route_isolation import isolated_module_command
from .r3_runtime import _tensor_identity_and_finiteness

RECORD_KIND = "r3-divergence-probe"
MANIFEST_NAME = "probe-manifest.json"
BRANCHES = ("conditional", "negative")
# First-call outputs of these DiT submodules are captured for the conditional
# forward. Both WanModel and CustomWanModel call them in this order.
DIT_SUBMODULES = (
    "patch_embedding",
    "time_embedding",
    "time_projection",
    "text_embedding",
    "img_emb",
)
DIT_INPUTS = ("x", "t", "context", "seq_len", "clip_fea", "y")
FROZEN_ATOL = 1e-5
FROZEN_RTOL = 0.016
_ROUTE_METHODS = {"official-pristine": "upstream", "local-custom": "custom"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class ProbeComplete(Exception):
    """Raised after the step-0 negative forward to stop generation."""


def _first_tensor(value):
    """Return the primary tensor of a DiT/block output (tuple and list layouts)."""
    while isinstance(value, (list, tuple)):
        _require(value, "probe output is empty")
        value = value[0]
    return value


class _Capture:
    def __init__(self, output_dir):
        self.dir = Path(output_dir)
        self.tensor_dir = self.dir / "tensors"
        self.tensor_dir.mkdir(parents=True, exist_ok=False)
        self.records = []
        self.names = set()
        self.restorations = []
        self.hooks = []

    def save(self, name, value, **metadata):
        import torch  # noqa: PLC0415

        _require(name not in self.names, f"probe stage captured twice: {name}")
        self.names.add(name)
        record = {"name": name, **metadata}
        if isinstance(value, torch.Tensor):
            # Clone so a CPU view never serializes its whole backing storage.
            tensor = value.detach().cpu().contiguous().clone()
            identity, finite = _tensor_identity_and_finiteness(tensor)
            path = self.tensor_dir / f"{len(self.records):04d}.pt"
            torch.save(tensor, path)
            record |= {"file": str(path.relative_to(self.dir)), "finite": finite, **identity}
        else:
            _require(
                value is None or isinstance(value, (bool, int, float, str)),
                f"probe stage {name} has an unsupported value type",
            )
            record["value"] = value
        self.records.append(record)

    def patch(self, owner, name, replacement):
        original = getattr(owner, name)
        setattr(owner, name, replacement)
        self.restorations.append((owner, name, original))
        return original

    def restore(self):
        for handle in reversed(self.hooks):
            handle.remove()
        self.hooks.clear()
        for owner, name, original in reversed(self.restorations):
            setattr(owner, name, original)
        self.restorations.clear()


def _text_key(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _install_encoders(capture, t5_type, clip_type, vae_type):
    counts = {"clip": 0, "vae": 0}
    t5_seen = {}
    t5_call = t5_type.__call__

    def t5_forward(encoder, texts, *args, **kwargs):
        result = t5_call(encoder, texts, *args, **kwargs)
        _require(len(result) == len(texts), "probe T5 output count differs from inputs")
        for text, tensor in zip(texts, result, strict=True):
            key = _text_key(text)
            occurrence = t5_seen.get(key, 0)
            t5_seen[key] = occurrence + 1
            capture.save(f"encoder/t5/{key}/{occurrence}", tensor, text=text)
        return result

    capture.patch(t5_type, "__call__", t5_forward)

    def list_encoder(kind, owner, method):
        original = getattr(owner, method)

        def wrapped(encoder, videos, *args, **kwargs):
            index = counts[kind]
            counts[kind] += 1
            for position, video in enumerate(videos):
                capture.save(f"encoder/{kind}/{index}/input/{position}", video)
            result = original(encoder, videos, *args, **kwargs)
            outputs = result if isinstance(result, (list, tuple)) else [result]
            for position, output in enumerate(outputs):
                capture.save(f"encoder/{kind}/{index}/output/{position}", output)
            return result

        capture.patch(owner, method, wrapped)

    list_encoder("clip", clip_type, "visual")
    list_encoder("vae", vae_type, "encode")


def _load_replay_inputs(replay_dir):
    import torch  # noqa: PLC0415

    replay_dir = Path(replay_dir)
    manifest = json.loads((replay_dir / MANIFEST_NAME).read_text())
    _require(manifest.get("record_kind") == RECORD_KIND, "probe replay manifest is invalid")
    _require(manifest.get("status") == "complete", "probe replay capture is incomplete")
    inputs = {branch: {} for branch in BRANCHES}
    for record in manifest["records"]:
        parts = record["name"].split("/")
        if parts[0] != "dit" or len(parts) < 4 or parts[2] != "input":
            continue
        branch, name = parts[1], parts[3]
        if "file" in record:
            value = torch.load(replay_dir / record["file"], map_location="cpu", weights_only=True)
        else:
            value = record["value"]
        if len(parts) == 5:
            inputs[branch].setdefault(name, []).append((int(parts[4]), value))
        else:
            inputs[branch][name] = value
    for branch in BRANCHES:
        for name, value in list(inputs[branch].items()):
            if isinstance(value, list):
                inputs[branch][name] = [item for _index, item in sorted(value)]
        _require(set(inputs[branch]) >= set(DIT_INPUTS), f"probe replay lacks {branch} inputs")
    digest = hashlib.sha256((replay_dir / MANIFEST_NAME).read_bytes()).hexdigest()
    return inputs, digest


def _to_device(value, device):
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value.to(device) if hasattr(value, "to") else value


def _install_dit(capture, model, replay_inputs):
    import torch  # noqa: PLC0415

    model_type = type(model)
    original = model_type.forward
    signature = inspect.signature(original)
    calls = {"count": 0}

    def save_input(prefix, name, value):
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                capture.save(f"{prefix}/{name}/{index}", item)
        else:
            capture.save(f"{prefix}/{name}", value)

    def register_hooks(prefix):
        fired = set()

        def hook_for(stage):
            def hook(_module, _inputs, output):
                if stage not in fired:
                    fired.add(stage)
                    capture.save(f"{prefix}/{stage}", _first_tensor(output))

            return hook

        for name in DIT_SUBMODULES:
            capture.hooks.append(getattr(model, name).register_forward_hook(hook_for(name)))
        for index, block in enumerate(model.blocks):
            capture.hooks.append(block.register_forward_hook(hook_for(f"block/{index:02d}")))
        capture.hooks.append(model.head.register_forward_hook(hook_for("head")))
        return len(capture.hooks)

    def forward(current, *args, **kwargs):
        if current is not model:
            return original(current, *args, **kwargs)
        call = calls["count"]
        _require(call < len(BRANCHES), "probe DiT forward continued after completion")
        calls["count"] += 1
        branch = BRANCHES[call]
        prefix = f"dit/{branch}"
        bound = signature.bind(current, *args, **kwargs)
        bound.apply_defaults()
        if replay_inputs is not None:
            device = _first_tensor(bound.arguments["x"]).device
            for name in DIT_INPUTS:
                bound.arguments[name] = _to_device(replay_inputs[branch][name], device)
        for name in DIT_INPUTS:
            save_input(f"{prefix}/input", name, bound.arguments[name])
        if call == 0:
            freqs = getattr(current, "freqs", None)
            if isinstance(freqs, torch.Tensor):
                capture.save(f"{prefix}/freqs", freqs)
            hook_count = len(capture.hooks)
            register_hooks(prefix)
        try:
            result = original(*bound.args, **bound.kwargs)
        finally:
            if call == 0:
                for handle in capture.hooks[hook_count:]:
                    handle.remove()
                del capture.hooks[hook_count:]
        capture.save(f"{prefix}/output", _first_tensor(result))
        if call == len(BRANCHES) - 1:
            raise ProbeComplete
        return result

    capture.patch(model_type, "forward", forward)


def _write_manifest(capture, metadata, status, error=None):
    manifest = {
        "record_kind": RECORD_KIND,
        "r3_acceptance": False,
        "status": status,
        **metadata,
        "records": capture.records,
    }
    if error is not None:
        manifest["error"] = error
    path = capture.dir / MANIFEST_NAME
    path.write_bytes(canonical_json_bytes(manifest) + b"\n")


@contextmanager
def install_probe(generator, output_dir, *, encoder_types, replay_dir=None, metadata=None):
    """Capture one DiT step; the body must end with :class:`ProbeComplete`."""
    capture = _Capture(output_dir)
    metadata = dict(metadata or {})
    replay_inputs = None
    if replay_dir is not None:
        replay_inputs, digest = _load_replay_inputs(replay_dir)
        metadata["replay"] = {"path": str(Path(replay_dir).resolve()), "manifest_sha256": digest}
    try:
        _install_encoders(capture, *encoder_types)
        _install_dit(capture, generator.model, replay_inputs)
        try:
            yield capture
        except ProbeComplete:
            _write_manifest(capture, metadata, "complete")
            return
        except BaseException as error:
            _write_manifest(capture, metadata, "failed", f"{type(error).__name__}: {error}")
            raise
        _write_manifest(capture, metadata, "failed", "generation finished without probe stop")
        raise RuntimeError("probe generation finished without reaching the step-0 stop")
    finally:
        capture.restore()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_capture(args):
    """Child entry point: runs inside the isolated route process under torchrun."""
    import torch  # noqa: PLC0415
    import torch.distributed as dist  # noqa: PLC0415

    from .fsdp_worker import _build_generator, run_inference  # noqa: PLC0415
    from .generation_routes import generation_route  # noqa: PLC0415
    from .r3_route_isolation import assert_loaded_wan_modules  # noqa: PLC0415
    from .task_contracts import validate_worker_task  # noqa: PLC0415

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    try:
        with open(args.task_file, "rb") as handle:
            task = pickle.load(handle)
        validate_worker_task(task)
        _require(
            generation_route(task) == _ROUTE_METHODS[args.route],
            "probe task route differs from the requested isolated route",
        )
        generator = _build_generator(
            task, local_rank=local_rank, rank=dist.get_rank(),
            t5_fsdp=False, dit_fsdp=False, t5_cpu=False,
        )
        encoder_types = (
            importlib.import_module("wan.modules.t5").T5EncoderModel,
            importlib.import_module("wan.modules.clip").CLIPModel,
            importlib.import_module("wan.modules.vae").WanVAE,
        )
        loaded = assert_loaded_wan_modules(args.wan_root)
        metadata = {
            "route": args.route,
            "wan_root": str(Path(args.wan_root).resolve()),
            "loaded_wan_module_count": len(loaded),
            "task_file": str(Path(args.task_file).resolve()),
            "task_sha256": _sha256_file(args.task_file),
            "torch_version": torch.__version__,
            "device_name": torch.cuda.get_device_name(local_rank),
            "dit_weight_dtypes": sorted(
                {str(p.dtype) for p in generator.model.parameters() if p.is_floating_point()}
            ),
            "runtime_param_dtype": str(generator.param_dtype),
        }
        with install_probe(
            generator, args.output, encoder_types=encoder_types,
            replay_dir=args.replay, metadata=metadata,
        ):
            run_inference(generator, task)
    finally:
        dist.destroy_process_group()


def build_launch_command(*, python, route, wan_root, project_root, task_file, output, replay=None):
    """Build the isolated single-rank torchrun command for one probe capture."""
    _require(route in _ROUTE_METHODS, "probe route is unsupported")
    _require(replay is None or route == "local-custom", "probe replay requires local-custom")
    child = [
        "--nproc_per_node=1", "-m", "multi_sample_inference.r3_divergence_probe", "capture",
        "--route", route, "--wan-root", str(Path(wan_root).resolve()),
        "--task-file", str(Path(task_file).resolve()), "--output", str(Path(output).resolve()),
    ]
    if replay is not None:
        child += ["--replay", str(Path(replay).resolve())]
    return isolated_module_command(
        python=python, route=route, wan_root=wan_root, project_root=project_root,
        target_module="torch.distributed.run", args=child,
    )


def run_launch(args):
    _require(not Path(args.output).exists(), "probe output already exists")
    project_root = Path(__file__).resolve().parents[1]
    command = build_launch_command(
        python=sys.executable, route=args.route, wan_root=args.wan_root,
        project_root=project_root, task_file=args.task_file, output=args.output,
        replay=args.replay,
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(command, cwd=project_root, env=environment, check=True,
                   timeout=args.timeout_seconds)


def _load_manifest(directory):
    manifest = json.loads((Path(directory) / MANIFEST_NAME).read_text())
    _require(manifest.get("record_kind") == RECORD_KIND, f"{directory} is not a probe capture")
    _require(manifest.get("status") == "complete", f"{directory} capture is incomplete")
    return manifest


def _comparable(tensor):
    import torch  # noqa: PLC0415

    return torch.view_as_real(tensor) if tensor.is_complex() else tensor


def _compare_record(reference_dir, candidate_dir, reference, candidate, atol, rtol):
    import torch  # noqa: PLC0415

    result = {"name": reference["name"]}
    if candidate is None:
        return result | {"status": "missing"}
    if "file" not in reference or "file" not in candidate:
        equal = reference.get("value") == candidate.get("value") and (
            "file" in reference) == ("file" in candidate)
        return result | {
            "status": "equal" if equal else "different",
            "reference_value": reference.get("value"),
            "candidate_value": candidate.get("value"),
        }
    result |= {
        "reference": {key: reference[key] for key in ("shape", "dtype", "sha256")},
        "candidate": {key: candidate[key] for key in ("shape", "dtype", "sha256")},
    }
    if reference["sha256"] == candidate["sha256"]:
        return result | {"status": "equal"}
    def load(root, record):
        return torch.load(Path(root) / record["file"], map_location="cpu",
                          weights_only=True, mmap=True)

    ref_tensor = _comparable(load(reference_dir, reference))
    cand_tensor = _comparable(load(candidate_dir, candidate))
    metrics = compare_outputs(ref_tensor, cand_tensor, atol=atol, rtol=rtol)
    if (metrics["shape_match"] and not metrics["dtype_match"]
            and ref_tensor.is_floating_point() and cand_tensor.is_floating_point()):
        metrics["upcast"] = compare_outputs(
            ref_tensor.double(), cand_tensor.double(), atol=atol, rtol=rtol)
    return result | {"status": "different", "metrics": metrics}


def compare_captures(reference_dir, candidate_dir, *, atol=FROZEN_ATOL, rtol=FROZEN_RTOL):
    """Compare two captures stage by stage in reference capture order."""
    reference = _load_manifest(reference_dir)
    candidate = _load_manifest(candidate_dir)
    candidate_records = {record["name"]: record for record in candidate["records"]}
    reference_names = {record["name"] for record in reference["records"]}
    stages = [
        _compare_record(reference_dir, candidate_dir, record,
                        candidate_records.get(record["name"]), atol, rtol)
        for record in reference["records"]
    ]

    def first(prefix):
        return next((stage["name"] for stage in stages
                     if stage["name"].startswith(prefix) and stage["status"] != "equal"), None)

    return {
        "record_kind": f"{RECORD_KIND}-comparison",
        "r3_acceptance": False,
        "reference": {"path": str(Path(reference_dir).resolve()), "route": reference.get("route")},
        "candidate": {"path": str(Path(candidate_dir).resolve()), "route": candidate.get("route"),
                      "replay": candidate.get("replay")},
        "tolerance": {"atol": atol, "rtol": rtol},
        "first_divergent_encoder_stage": first("encoder/"),
        "first_divergent_dit_stage": first("dit/"),
        "candidate_only_stages": sorted(set(candidate_records) - reference_names),
        "stages": stages,
    }


def run_compare(args):
    report = compare_captures(args.reference, args.candidate)
    Path(args.report).write_bytes(canonical_json_bytes(report) + b"\n")
    print(json.dumps({
        "first_divergent_encoder_stage": report["first_divergent_encoder_stage"],
        "first_divergent_dit_stage": report["first_divergent_dit_stage"],
        "candidate_only_stages": len(report["candidate_only_stages"]),
    }))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("launch", "capture"):
        command = commands.add_parser(name)
        command.add_argument("--route", choices=sorted(_ROUTE_METHODS), required=True)
        command.add_argument("--wan-root", required=True, type=Path)
        command.add_argument("--task-file", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--replay", type=Path)
        if name == "launch":
            command.add_argument("--timeout-seconds", type=int, default=1800)
    compare = commands.add_parser("compare")
    compare.add_argument("--reference", required=True, type=Path)
    compare.add_argument("--candidate", required=True, type=Path)
    compare.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    {"launch": run_launch, "capture": run_capture, "compare": run_compare}[args.command](args)


if __name__ == "__main__":
    main()
