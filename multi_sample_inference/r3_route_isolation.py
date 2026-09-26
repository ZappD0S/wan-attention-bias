"""Exact-path subprocess isolation for pristine-upstream and local-custom Wan."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

_ROUTES = {"official-pristine", "local-custom"}
_BOOTSTRAP = """import json,os,runpy,sys
route_root,project_root,target=json.loads(sys.argv.pop(1))
os.environ['PYTHONPATH']=os.pathsep.join((route_root,project_root))
os.environ['PYTHONSAFEPATH']='1'
sys.path[:0]=[route_root,project_root]
sys.argv=[target,*sys.argv[1:]]
runpy.run_module(target,run_name='__main__')
"""


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_exact_route_binding(binding):
    """Validate exact source/adapter files without importing either Wan route."""
    _require(
        isinstance(binding, dict)
        and set(binding) == {"route", "wan_root", "source_files", "adapter"},
        "R3 isolated route binding is incomplete or unexpected",
    )
    _require(binding["route"] in _ROUTES, "R3 isolated route is unsupported")
    root = Path(binding["wan_root"]).resolve()
    _require(root.is_dir(), "R3 isolated Wan root is missing")
    files = binding["source_files"]
    _require(isinstance(files, dict) and files, "R3 isolated route source binding is empty")
    for relative, digest in files.items():
        _require(isinstance(relative, str) and relative, "R3 isolated route source path is invalid")
        path = (root / relative).resolve()
        _require(path.is_relative_to(root) and path.is_file(), "R3 isolated route source file is missing")
        _require(_sha256(path) == digest, f"R3 isolated route source changed: {relative}")
    adapter = binding["adapter"]
    if binding["route"] == "official-pristine":
        _require(
            isinstance(adapter, dict) and set(adapter) == {"path", "sha256"},
            "R3 pristine observation adapter binding is missing",
        )
        adapter_path = Path(adapter["path"]).resolve()
        _require(adapter_path.is_file(), "R3 pristine observation adapter is missing")
        _require(
            _sha256(adapter_path) == adapter["sha256"],
            "R3 pristine observation adapter changed",
        )
    else:
        _require(adapter is None, "R3 local-custom route must not install the pristine adapter")
    return True


def validate_pristine_route_contract(contract, project_root):
    """Validate the frozen CPU route implementation and its exact source hashes."""
    _require(
        isinstance(contract, dict)
        and set(contract)
        == {
            "schema_version",
            "contract_id",
            "status",
            "claim_boundary",
            "lineage",
            "sources",
            "implementation",
            "cpu_validation",
            "execution_state",
        },
        "R3 pristine route contract is incomplete or unexpected",
    )
    _require(
        contract["schema_version"] == 1
        and contract["contract_id"] == "r3-pristine-route-v1"
        and contract["status"] == "implemented-cpu-validated-unapproved",
        "R3 pristine route contract identity is invalid",
    )
    _require(
        contract["lineage"]
        == {
            "protocol_id": "r3-gpu-contracts-v8",
            "sha256": "f8a9c578f47957e8818f70c42d8bad7bfa8947bbbf7cf5eac1917a7e100126fa",
        },
        "R3 pristine route contract lineage is invalid",
    )
    sources = contract["sources"]
    _require(
        sources
        == {
            "official": {
                "repository_url": "https://github.com/Wan-Video/Wan2.1.git",
                "commit": "7c81b2f27defa56c7e627a4b6717c8f2292eee58",
                "tree": "91b74dfa24e32dc86350fe2ef2b5f2f2e06f6ee9",
                "files": {
                    "wan/image2video.py": "ab3906d53831c2a6717e543a02d66c13377910257ae710705515bd826212b924",
                    "wan/modules/model.py": "c1572ade3bf7345bd4c00d4a19535788fab62f9c08b5aeb15f6d01f4435bb47a",
                    "wan/modules/attention.py": "23fe7c6f6e4065242d95e5e188cb2d1a16bd283f05dca7e7e878977158fcfdbc",
                },
            },
            "custom": {
                "commit": "00bde1e719ccb56c66a01a1f18a70c49b278c202",
                "tree": "ee7dddb233e6acc14a89cf96951cca6536587fee",
            },
        },
        "R3 pristine route source binding is invalid",
    )
    implementation = contract["implementation"]
    _require(
        isinstance(implementation, dict)
        and set(implementation)
        == {"adapter", "launcher", "worker_integration", "process_isolation"},
        "R3 pristine route implementation binding is incomplete",
    )
    root = Path(project_root).resolve()
    component_keys = {
        "adapter": {"path", "sha256", "interface", "scope"},
        "launcher": {"path", "sha256", "interpreter_isolation"},
        "worker_integration": {
            "path",
            "sha256",
            "pre_model_guard",
            "official_adapter_selection",
        },
    }
    for name, expected_keys in component_keys.items():
        component = implementation[name]
        _require(
            isinstance(component, dict) and set(component) == expected_keys,
            f"R3 pristine route {name} binding is malformed",
        )
        path = (root / component["path"]).resolve()
        _require(
            path.is_relative_to(root) and path.is_file(),
            f"R3 pristine route {name} source is missing",
        )
        _require(
            _sha256(path) == component["sha256"],
            f"R3 pristine route {name} source changed",
        )
    _require(
        implementation["adapter"]["interface"]
        == "install_pristine_runtime_observer"
        and implementation["adapter"]["scope"]
        == "single-rank-official-generator"
        and implementation["launcher"]["interpreter_isolation"]
        == "python-I-exact-sys-path-bootstrap"
        and implementation["worker_integration"]["pre_model_guard"] is True
        and implementation["worker_integration"]["official_adapter_selection"]
        == "schema-v9-plus-upstream-only",
        "R3 pristine route implementation semantics are invalid",
    )
    _require(
        implementation["process_isolation"]
        == "separate-python-processes-no-shared-wan-modules",
        "R3 pristine route process isolation is invalid",
    )
    _require(
        contract["cpu_validation"]
        == {
            "adapter_fake_generator": "passed",
            "adapter_failure_and_restoration": "passed",
            "synthetic_distinct_process_imports": "passed",
            "actual_checkout_resolve_only": "passed",
            "real_checkout_import": "cpu-unsafe-hidden-cuda-init-failed",
            "checkpoint_or_model_loaded": False,
            "gpu_visible_allocated_or_used": False,
            "generation_run": False,
        },
        "R3 pristine route CPU validation claims are invalid",
    )
    _require(
        contract["execution_state"]
        == {
            "target_host_checkout": "required-not-revalidated",
            "target_host_locked_environment": "required-not-revalidated",
            "generator_stage_authorization": "required-not-approved",
            "immutable_execution_amendment": "required-not-frozen",
        },
        "R3 pristine route execution state is invalid",
    )
    _require(
        isinstance(contract["claim_boundary"], str) and contract["claim_boundary"],
        "R3 pristine route claim boundary is missing",
    )
    return True


def _origins(module):
    values = []
    path = getattr(module, "__file__", None)
    if path is not None:
        values.append(Path(path).resolve())
    package_paths = getattr(module, "__path__", ())
    values.extend(Path(item).resolve() for item in package_paths)
    return values


def assert_loaded_wan_modules(wan_root):
    """Reject a process containing any ``wan`` module outside one exact root."""
    root = Path(wan_root).resolve()
    loaded = {
        name: module
        for name, module in sys.modules.items()
        if name == "wan" or name.startswith("wan.")
    }
    _require(loaded, "R3 isolated process has no loaded Wan modules")
    evidence = {}
    for name, module in sorted(loaded.items()):
        origins = _origins(module)
        _require(origins, f"R3 loaded Wan module has no filesystem origin: {name}")
        _require(
            all(origin.is_relative_to(root) for origin in origins),
            f"R3 loaded Wan module escaped the isolated root: {name}",
        )
        evidence[name] = [str(origin) for origin in origins]
    return evidence


def isolated_module_command(
    *, python, route, wan_root, project_root, target_module, args=()
):
    """Build a clean-interpreter command with one Wan root ahead of the project."""
    _require(route in _ROUTES, "R3 isolated route is unsupported")
    # Preserve a virtual-environment interpreter symlink; resolving it would
    # silently drop that environment's site-packages in the isolated child.
    python = Path(python).absolute()
    wan_root = Path(wan_root).resolve()
    project_root = Path(project_root).resolve()
    _require(python.is_file(), "R3 isolated Python executable is missing")
    _require(wan_root.is_dir(), "R3 isolated Wan root is missing")
    _require(project_root.is_dir(), "R3 isolated project root is missing")
    _require(
        isinstance(target_module, str) and target_module,
        "R3 isolated target module is invalid",
    )
    payload = json.dumps(
        [str(wan_root), str(project_root), target_module], separators=(",", ":")
    )
    return [str(python), "-I", "-c", _BOOTSTRAP, payload, *map(str, args)]


def run_isolated_module(**kwargs):
    """Run one bounded isolated child; callers remain responsible for GPU scope."""
    command = isolated_module_command(**kwargs)
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    return subprocess.run(command, check=True, capture_output=True, text=True, env=env)


def _write_immutable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise FileExistsError(f"R3 isolation probe output exists: {path}") from error
    with os.fdopen(descriptor, "w") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _probe(route, wan_root, output, expected_marker, resolve_only):
    _require(route in _ROUTES, "R3 isolation probe route is unsupported")
    _require(
        not any(name == "wan" or name.startswith("wan.") for name in sys.modules),
        "R3 isolation probe inherited a loaded Wan module",
    )
    root = Path(wan_root).resolve()
    if resolve_only:
        spec = importlib.util.find_spec("wan")
        origin = Path(spec.origin).resolve() if spec is not None and spec.origin else None
        _require(
            origin is not None and origin.is_relative_to(root),
            "R3 isolation probe resolved the wrong Wan package",
        )
        origins = {}
    else:
        module = importlib.import_module("wan")
        if expected_marker is not None:
            _require(
                getattr(module, "R3_TEST_MARKER", None) == expected_marker,
                "R3 isolation probe imported the wrong Wan package",
            )
        origins = assert_loaded_wan_modules(root)
        origin = Path(module.__file__).resolve()
    _write_immutable_json(
        output,
        {
            "schema_version": 1,
            "record_kind": "r3-cpu-process-isolation-probe",
            "probe_mode": "resolve-only" if resolve_only else "import",
            "route": route,
            "pid": os.getpid(),
            "wan_root": str(root),
            "wan_resolution": str(origin),
            "wan_modules": origins,
            "marker": expected_marker,
            "gpu_or_model_used": False,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--resolve-only", action="store_true")
    parser.add_argument("--route", choices=sorted(_ROUTES))
    parser.add_argument("--wan-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-marker")
    args = parser.parse_args()
    if not args.probe:
        parser.error("only the bounded CPU isolation probe is directly executable")
    if None in (args.route, args.wan_root, args.output):
        parser.error("the isolation probe requires route, root, and output")
    _probe(
        args.route,
        args.wan_root,
        args.output,
        args.expected_marker,
        args.resolve_only,
    )


if __name__ == "__main__":
    main()
