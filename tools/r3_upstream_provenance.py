"""Build and validate the CPU-only R3-P Wan upstream provenance gate."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SCHEMA_VERSION = 1
ENTRYPOINTS = (
    "wan",
    "wan.configs.wan_i2v_14B",
    "wan.image2video",
)
REQUIRED_SURFACE_PATHS = frozenset(
    {
        "wan/__init__.py",
        "wan/configs/__init__.py",
        "wan/configs/shared_config.py",
        "wan/configs/wan_i2v_14B.py",
        "wan/distributed/fsdp.py",
        "wan/image2video.py",
        "wan/modules/__init__.py",
        "wan/modules/attention.py",
        "wan/modules/clip.py",
        "wan/modules/model.py",
        "wan/modules/t5.py",
        "wan/modules/tokenizers.py",
        "wan/modules/vae.py",
        "wan/utils/fm_solvers.py",
        "wan/utils/fm_solvers_unipc.py",
    }
)
DISPOSITIONS = frozenset(
    {
        "observation-only-instrumentation",
        "substantive-baseline-change",
        "outside-upstream-execution-surface",
        "unresolved",
    }
)
INSTRUMENTATION_CONSTRAINTS = frozenset(
    {
        "observer_off_is_no_op",
        "tensor_values_unchanged",
        "seed_and_state_unchanged",
        "loading_and_dispatch_unchanged",
        "scheduler_and_branching_unchanged",
        "observer_off_exceptions_unchanged",
    }
)


def canonical_json_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _git(root, *args, text=False):
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=text,
    )
    return result.stdout


def _git_line(root, *args):
    return _git(root, *args, text=True).strip()


def _clean_identity(root):
    status = _git(root, "status", "--porcelain=v1", "-z")
    _require(not status, f"source checkout is dirty: {root}")
    return {
        "porcelain_v1_sha256": sha256_bytes(status),
        "is_clean": True,
    }


def source_identity(root, *, require_detached=False):
    root = Path(root).resolve()
    commit = _git_line(root, "rev-parse", "HEAD")
    branch = _git_line(root, "branch", "--show-current")
    if require_detached:
        _require(not branch, "pristine checkout must be detached")
    return {
        "path": str(root),
        "repository_url": _git_line(root, "remote", "get-url", "origin"),
        "commit": commit,
        "tree": _git_line(root, "rev-parse", "HEAD^{tree}"),
        "branch": branch or None,
        "detached": not branch,
        "worktree": _clean_identity(root),
    }


def _tree_manifest(root, commit):
    raw = _git(root, "ls-tree", "-r", "-z", "--full-tree", commit)
    result = {}
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        metadata, path_bytes = entry.split(b"\t", 1)
        mode, kind, oid = metadata.decode("ascii").split()
        path = path_bytes.decode("utf-8")
        _require(kind == "blob", f"unsupported tracked object {kind}: {path}")
        content = _git(root, "show", f"{commit}:{path}")
        result[path] = {
            "mode": mode,
            "git_oid": oid,
            "size": len(content),
            "sha256": sha256_bytes(content),
        }
    return result


def _module_map(manifest):
    modules = {}
    for path in manifest:
        if not path.startswith("wan/") or not path.endswith(".py"):
            continue
        if path.endswith("/__init__.py"):
            module = path[: -len("/__init__.py")].replace("/", ".")
        else:
            module = path[:-3].replace("/", ".")
        modules[module] = path
    return modules


def _package_name(module, path):
    return module if path.endswith("/__init__.py") else module.rpartition(".")[0]


def _import_targets(tree_root, commit, module, path, modules):
    source = _git(tree_root, "show", f"{commit}:{path}").decode("utf-8")
    parsed = ast.parse(source, filename=path)
    targets = set()
    current_package = _package_name(module, path)
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            package_parts = current_package.split(".") if current_package else []
            keep = len(package_parts) - node.level + 1
            if keep < 0:
                continue
            base_parts = package_parts[:keep]
            if node.module:
                base_parts.extend(node.module.split("."))
            base = ".".join(base_parts)
        else:
            base = node.module or ""
        if base:
            targets.add(base)
        if not node.module:
            for alias in node.names:
                child = ".".join(part for part in (base, alias.name) if part)
                if child in modules:
                    targets.add(child)
    return targets


def _add_module_and_packages(pending, target, modules):
    if target in modules:
        pending.add(target)
    parts = target.split(".")
    for index in range(1, len(parts)):
        package = ".".join(parts[:index])
        if package in modules:
            pending.add(package)


def execution_surface(root, commit, manifest, entrypoints=ENTRYPOINTS):
    modules = _module_map(manifest)
    pending = set()
    for entrypoint in entrypoints:
        _add_module_and_packages(pending, entrypoint, modules)
        _require(entrypoint in modules, f"missing execution entrypoint: {entrypoint}")
    visited = set()
    while pending:
        module = min(pending)
        pending.remove(module)
        if module in visited:
            continue
        visited.add(module)
        path = modules[module]
        for target in _import_targets(root, commit, module, path, modules):
            _add_module_and_packages(pending, target, modules)
    paths = {modules[module] for module in visited}
    missing = REQUIRED_SURFACE_PATHS - paths
    _require(not missing, f"execution surface misses required paths: {sorted(missing)}")
    return paths


def _latest_common_commit(official_root, official_head, local_root, local_head):
    local_commits = set(_git_line(local_root, "rev-list", local_head).splitlines())
    for commit in _git_line(official_root, "rev-list", official_head).splitlines():
        if commit in local_commits:
            return commit
    raise ValueError("official and local histories have no common commit")


def _change_status(official, local):
    if official is None:
        return "added"
    if local is None:
        return "removed"
    if official == local:
        return "identical"
    return "changed"


def _path_patch(local_root, official_commit, local_commit, path):
    return _git(
        local_root,
        "diff",
        "--binary",
        "--full-index",
        "--no-ext-diff",
        official_commit,
        local_commit,
        "--",
        path,
    )


def _name_status(local_root, official_commit, local_commit):
    output = _git_line(
        local_root,
        "diff",
        "--name-status",
        "--find-renames",
        official_commit,
        local_commit,
    )
    entries = []
    renames = []
    for line in output.splitlines():
        fields = line.split("\t")
        code = fields[0]
        if code.startswith("R"):
            _require(len(fields) == 3, "malformed rename status")
            renames.append(
                {"similarity": int(code[1:]), "from": fields[1], "to": fields[2]}
            )
        else:
            _require(len(fields) == 2, "malformed path status")
            entries.append({"status": code, "path": fields[1]})
    return entries, renames


def _surface_side(commit, tree, paths, manifest):
    return {
        "commit": commit,
        "tree": tree,
        "paths": [
            {"path": path, **manifest[path]}
            for path in sorted(paths)
        ],
    }


def _classification_spec_by_path(specs):
    by_path = {}
    for item in specs:
        _require(isinstance(item, dict), "classification entries must be objects")
        _require(isinstance(item.get("path"), str) and item["path"], "classification path is invalid")
        _require(item["path"] not in by_path, f"duplicate classification: {item['path']}")
        _require(item.get("disposition") in DISPOSITIONS, f"invalid disposition: {item['path']}")
        _require(isinstance(item.get("rationale"), str) and item["rationale"], f"missing rationale: {item['path']}")
        _require(isinstance(item.get("reviewer"), str) and item["reviewer"], f"missing reviewer: {item['path']}")
        by_path[item["path"]] = item
    return by_path


def _build_classifications(
    specs,
    official_manifest,
    local_manifest,
    surface_union,
    local_root,
    official_commit,
    local_commit,
):
    changed = {
        path
        for path in set(official_manifest) | set(local_manifest)
        if official_manifest.get(path) != local_manifest.get(path)
    }
    by_path = _classification_spec_by_path(specs)
    _require(set(by_path) == changed, "classifications do not exactly cover tracked-tree differences")
    result = []
    for path in sorted(changed):
        spec = by_path[path]
        in_surface = path in surface_union
        disposition = spec["disposition"]
        if disposition == "outside-upstream-execution-surface":
            _require(not in_surface, f"surface path classified outside: {path}")
        patch_sha256 = sha256_bytes(
            _path_patch(local_root, official_commit, local_commit, path)
        )
        item = {
            "path": path,
            "status": _change_status(
                official_manifest.get(path), local_manifest.get(path)
            ),
            "execution_surface": in_surface,
            "disposition": disposition,
            "official": official_manifest.get(path),
            "local": local_manifest.get(path),
            "patch_sha256": patch_sha256,
            "rationale": spec["rationale"],
            "reviewer": spec["reviewer"],
        }
        if disposition == "observation-only-instrumentation":
            constraints = spec.get("instrumentation_constraints")
            _require(
                isinstance(constraints, dict)
                and set(constraints) == INSTRUMENTATION_CONSTRAINTS
                and all(value is True for value in constraints.values()),
                f"incomplete instrumentation constraints: {path}",
            )
            item["instrumentation_allowlist"] = {
                "patch_sha256": patch_sha256,
                "official_content_sha256": (
                    official_manifest.get(path) or {}
                ).get("sha256"),
                "local_content_sha256": (local_manifest.get(path) or {}).get("sha256"),
                "constraints": constraints,
            }
        result.append(item)
    return result


def _validate_instrumentation_item(item, path):
    allowlist = item.get("instrumentation_allowlist")
    _require(isinstance(allowlist, dict), f"missing instrumentation allowlist: {path}")
    _require(
        allowlist.get("patch_sha256") == item["patch_sha256"],
        f"broad or unhashed instrumentation allowlist: {path}",
    )
    _require(
        allowlist.get("official_content_sha256")
        == (item.get("official") or {}).get("sha256")
        and allowlist.get("local_content_sha256")
        == (item.get("local") or {}).get("sha256"),
        f"instrumentation content identity mismatch: {path}",
    )
    constraints = allowlist.get("constraints")
    _require(
        isinstance(constraints, dict)
        and set(constraints) == INSTRUMENTATION_CONSTRAINTS
        and all(value is True for value in constraints.values()),
        f"incomplete instrumentation allowlist: {path}",
    )


def _validate_classifications(record, surface):
    classifications = record.get("classifications")
    _require(isinstance(classifications, list), "classifications are missing")
    paths = [item.get("path") for item in classifications]
    _require(len(paths) == len(set(paths)), "classification paths are duplicated")
    surface_entries = {item["path"]: item for item in surface["comparison"]}
    changed_surface = {
        path for path, item in surface_entries.items() if item["status"] != "identical"
    }
    classified = set(paths)
    expected_changed = set(record["whole_tree"]["changed_paths"])
    _require(classified == expected_changed, "tracked-tree classification is incomplete")
    for item in classifications:
        path = item["path"]
        disposition = item.get("disposition")
        _require(disposition in DISPOSITIONS, f"invalid disposition: {path}")
        _require(item.get("status") != "identical", f"identical path was classified: {path}")
        _require(
            item.get("execution_surface") == (path in surface_entries),
            f"surface membership mismatch: {path}",
        )
        if path in surface_entries:
            surface_item = surface_entries[path]
            _require(
                surface_item["status"] == item["status"]
                and surface_item["official"] == item["official"]
                and surface_item["local"] == item["local"]
                and surface_item.get("disposition") == disposition,
                f"surface classification binding mismatch: {path}",
            )
        _require(
            item.get("patch_sha256") and len(item["patch_sha256"]) == 64,
            f"missing path patch hash: {path}",
        )
        if disposition == "outside-upstream-execution-surface":
            _require(path not in surface_entries, f"surface path classified outside: {path}")
        if disposition == "observation-only-instrumentation":
            _validate_instrumentation_item(item, path)
    _require(not (changed_surface - classified), "surface differences are unclassified")
    _require(
        all(item["disposition"] != "unresolved" for item in classifications),
        "unresolved difference blocks the provenance gate",
    )
    return classifications


def _expected_decision(classifications):
    relevant = [item for item in classifications if item["execution_surface"]]
    if any(item["disposition"] == "substantive-baseline-change" for item in relevant):
        return "separate-pristine-route-required"
    if relevant:
        _require(
            all(
                item["disposition"] == "observation-only-instrumentation"
                for item in relevant
            ),
            "relevant differences have inconsistent dispositions",
        )
        return "instrumented-vendor-derived-route"
    return "pristine-current-route"


def validate_evidence(record, surface, diff_bytes):
    _require(record.get("schema_version") == SCHEMA_VERSION, "wrong provenance schema")
    _require(record.get("task") == "R3-P", "wrong provenance task")
    _require(record.get("status") == "complete", "provenance gate is not complete")
    _require(surface.get("schema_version") == SCHEMA_VERSION, "wrong surface schema")
    comparison = record["comparison"]
    for side in ("official", "local"):
        source = record[f"{side}_source"]
        _require(
            surface[side]["commit"] == source["commit"],
            f"{side} execution-surface commit mismatch",
        )
        _require(
            surface[side]["tree"] == source["tree"],
            f"{side} execution-surface tree mismatch",
        )
    _require(
        comparison["whole_tree_diff"]["sha256"] == sha256_bytes(diff_bytes),
        "whole-tree diff hash mismatch",
    )
    surface_bytes = canonical_json_bytes(surface) + b"\n"
    _require(
        comparison["execution_surface"]["sha256"] == sha256_bytes(surface_bytes),
        "execution-surface hash mismatch",
    )
    classifications = _validate_classifications(record, surface)
    _require(
        record.get("baseline_decision") == _expected_decision(classifications),
        "baseline decision is inconsistent with classifications",
    )
    digest_payload = {
        "official": record["official_source"],
        "local": record["local_source"],
        "whole_tree_diff_sha256": comparison["whole_tree_diff"]["sha256"],
        "execution_surface_sha256": comparison["execution_surface"]["sha256"],
        "classifications": classifications,
        "baseline_decision": record["baseline_decision"],
    }
    _require(
        comparison["comparison_digest_sha256"]
        == sha256_bytes(canonical_json_bytes(digest_payload)),
        "comparison digest mismatch",
    )
    return True


def build_evidence(
    official_root,
    local_root,
    classification_specs,
    *,
    evidence_date,
    official_observed_head,
):
    official_root = Path(official_root).resolve()
    local_root = Path(local_root).resolve()
    official = source_identity(official_root, require_detached=True)
    local = source_identity(local_root)
    _require(
        official["repository_url"] == "https://github.com/Wan-Video/Wan2.1.git",
        "unexpected official repository URL",
    )
    observed_tree = _git_line(
        official_root, "rev-parse", f"{official_observed_head}^{{tree}}"
    )
    _require(observed_tree, "official observed head is unavailable")
    latest_common = _latest_common_commit(
        official_root, official_observed_head, local_root, local["commit"]
    )
    _require(
        official["commit"] == latest_common,
        "pristine checkout is not at the latest common official commit",
    )
    _require(
        _git_line(local_root, "rev-parse", f"{official['commit']}^{{tree}}")
        == official["tree"],
        "official commit tree differs between repositories",
    )
    official_manifest = _tree_manifest(official_root, official["commit"])
    local_manifest = _tree_manifest(local_root, local["commit"])
    official_surface = execution_surface(
        official_root, official["commit"], official_manifest
    )
    local_surface = execution_surface(local_root, local["commit"], local_manifest)
    surface_union = official_surface | local_surface
    classifications = _build_classifications(
        classification_specs,
        official_manifest,
        local_manifest,
        surface_union,
        local_root,
        official["commit"],
        local["commit"],
    )
    classification_by_path = {
        item["path"]: item["disposition"] for item in classifications
    }
    surface_comparison = []
    for path in sorted(surface_union):
        status = _change_status(official_manifest.get(path), local_manifest.get(path))
        item = {
            "path": path,
            "status": status,
            "official": official_manifest.get(path),
            "local": local_manifest.get(path),
        }
        if status != "identical":
            item["disposition"] = classification_by_path[path]
        surface_comparison.append(item)
    surface = {
        "schema_version": SCHEMA_VERSION,
        "kind": "r3-upstream-execution-surface",
        "entrypoints": list(ENTRYPOINTS),
        "closure_method": "Python AST local-import closure including package initializers",
        "required_paths": sorted(REQUIRED_SURFACE_PATHS),
        "official": _surface_side(
            official["commit"], official["tree"], official_surface, official_manifest
        ),
        "local": _surface_side(
            local["commit"], local["tree"], local_surface, local_manifest
        ),
        "comparison": surface_comparison,
    }
    surface_bytes = canonical_json_bytes(surface) + b"\n"
    diff_bytes = _git(
        local_root,
        "diff",
        "--binary",
        "--full-index",
        "--find-renames",
        "--no-ext-diff",
        official["commit"],
        local["commit"],
    )
    name_status, renames = _name_status(
        local_root, official["commit"], local["commit"]
    )
    changed_paths = sorted(item["path"] for item in classifications)
    counts = dict.fromkeys(("identical", "added", "removed", "changed"), 0)
    for path in set(official_manifest) | set(local_manifest):
        counts[_change_status(official_manifest.get(path), local_manifest.get(path))] += 1
    official_source = official | {
        "selected_as_latest_common_commit_with_official_default_branch": True,
        "default_branch_observed_commit": official_observed_head,
        "default_branch_observed_tree": observed_tree,
        "annotated_tag": None,
        "model_card_exact_code_revision": None,
        "selection_rationale": (
            "The official repository and model card name the Wan2.1 codebase but do "
            "not pin a code tag. This commit is the latest commit shared by the "
            "official default-branch history observed on the evidence date and the "
            "local custom history, so it is the exact vendor fork point."
        ),
    }
    local_source = local | {
        "official_fork_point_commit": official["commit"],
        "official_fork_point_tree": official["tree"],
    }
    record = {
        "schema_version": SCHEMA_VERSION,
        "task": "R3-P",
        "status": "complete",
        "evidence_date_utc": evidence_date,
        "official_source": official_source,
        "local_source": local_source,
        "tools": {
            "git": _git_line(local_root, "--version"),
            "python": sys.version.split()[0],
            "hash": "SHA-256 over exact Git blob bytes; canonical JSON uses sorted keys, compact separators, UTF-8, and one trailing LF",
            "diff": "git diff --binary --full-index --find-renames --no-ext-diff <official> <local>",
            "surface": "stdlib ast import closure over exact committed Python blobs",
        },
        "whole_tree": {
            "official_tracked_paths": len(official_manifest),
            "local_tracked_paths": len(local_manifest),
            "counts": counts,
            "changed_paths": changed_paths,
            "name_status": name_status,
            "renames": renames,
        },
        "comparison": {
            "whole_tree_diff": {
                "path": "docs/r3_evidence/upstream-whole-tree.diff",
                "sha256": sha256_bytes(diff_bytes),
                "size": len(diff_bytes),
            },
            "execution_surface": {
                "path": "docs/r3_evidence/upstream-execution-surface.json",
                "sha256": sha256_bytes(surface_bytes),
                "size": len(surface_bytes),
            },
        },
        "classifications": classifications,
        "baseline_decision": "separate-pristine-route-required",
        "decision_rationale": (
            "The standard local I2V execution surface contains substantive source "
            "differences from the exact official fork point. Future upstream parity "
            "must therefore bind and execute the separate pristine checkout."
        ),
        "scope": {
            "checkpoint_or_model_loaded": False,
            "gpu_used": False,
            "generation_run": False,
            "provenance_only": True,
        },
    }
    digest_payload = {
        "official": official_source,
        "local": local_source,
        "whole_tree_diff_sha256": record["comparison"]["whole_tree_diff"]["sha256"],
        "execution_surface_sha256": record["comparison"]["execution_surface"]["sha256"],
        "classifications": classifications,
        "baseline_decision": record["baseline_decision"],
    }
    record["comparison"]["comparison_digest_sha256"] = sha256_bytes(
        canonical_json_bytes(digest_payload)
    )
    validate_evidence(record, surface, diff_bytes)
    return record, surface, diff_bytes


def _specs_from_record(record):
    specs = []
    for item in record["classifications"]:
        spec = {
            key: item[key]
            for key in ("path", "disposition", "rationale", "reviewer")
        }
        if item["disposition"] == "observation-only-instrumentation":
            spec["instrumentation_constraints"] = item["instrumentation_allowlist"][
                "constraints"
            ]
        specs.append(spec)
    return specs


def _write_or_verify(path, content):
    path = Path(path)
    if path.exists():
        _require(path.read_bytes() == content, f"immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _load_json(path):
    return json.loads(Path(path).read_text())


def generate(args):
    spec = _load_json(args.classification_spec)
    record, surface, diff_bytes = build_evidence(
        args.official_root,
        args.local_root,
        spec["classifications"],
        evidence_date=spec["evidence_date_utc"],
        official_observed_head=spec["official_default_branch_observed_commit"],
    )
    _write_or_verify(args.diff, diff_bytes)
    _write_or_verify(args.surface, canonical_json_bytes(surface) + b"\n")
    _write_or_verify(args.record, canonical_json_bytes(record) + b"\n")
    return record


def check(args):
    record = _load_json(args.record)
    surface = _load_json(args.surface)
    diff_bytes = Path(args.diff).read_bytes()
    validate_evidence(record, surface, diff_bytes)
    expected_record, expected_surface, expected_diff = build_evidence(
        args.official_root or record["official_source"]["path"],
        args.local_root or record["local_source"]["path"],
        _specs_from_record(record),
        evidence_date=record["evidence_date_utc"],
        official_observed_head=record["official_source"][
            "default_branch_observed_commit"
        ],
    )
    _require(expected_record == record, "provenance record does not reproduce")
    _require(expected_surface == surface, "execution surface does not reproduce")
    _require(expected_diff == diff_bytes, "whole-tree diff does not reproduce")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--official-root", required=True, type=Path)
    generate_parser.add_argument("--local-root", required=True, type=Path)
    generate_parser.add_argument("--classification-spec", required=True, type=Path)
    generate_parser.add_argument("--record", required=True, type=Path)
    generate_parser.add_argument("--surface", required=True, type=Path)
    generate_parser.add_argument("--diff", required=True, type=Path)
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--official-root", type=Path)
    check_parser.add_argument("--local-root", type=Path)
    check_parser.add_argument("--record", required=True, type=Path)
    check_parser.add_argument("--surface", required=True, type=Path)
    check_parser.add_argument("--diff", required=True, type=Path)
    args = parser.parse_args(argv)
    result = generate(args) if args.command == "generate" else check(args)
    print(
        json.dumps(
            {
                "task": result["task"],
                "status": result["status"],
                "baseline_decision": result["baseline_decision"],
                "comparison_digest_sha256": result["comparison"][
                    "comparison_digest_sha256"
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
