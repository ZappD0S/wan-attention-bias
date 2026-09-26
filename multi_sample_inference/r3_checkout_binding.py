"""Whole-checkout identity gate for future isolated R3 process bindings."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .r3_route_isolation import validate_exact_route_binding


def _git(root, *args):
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return result.stdout.decode("utf-8").strip()


def _validate_checkout_clean(root):
    try:
        dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none")
        ignored = _git(root, "ls-files", "--others", "--ignored", "--exclude-standard", "-z")
        indexed = _git(root, "ls-files", "-s")
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired) as error:
        raise ValueError("R3 whole-checkout cleanliness cannot be verified") from error
    if dirty:
        raise ValueError("R3 whole-checkout has tracked, untracked or submodule changes")
    executable_suffixes = (".py", ".pyc", ".so", ".pyd", ".pth")
    if any(path.endswith(executable_suffixes) for path in ignored.split("\0")):
        raise ValueError("R3 checkout has ignored importable files")
    if any(line.startswith("120000 ") for line in indexed.splitlines()):
        raise ValueError("R3 checkout has tracked symlinks")


def validate_checkout_route_binding(binding):
    """Require a clean, pinned whole Wan checkout before importing either route."""
    if not isinstance(binding, dict) or set(binding) != {
        "route", "wan_root", "source_files", "adapter", "checkout"
    }:
        raise ValueError("R3 whole-checkout route binding is incomplete or unexpected")
    checkout = binding["checkout"]
    if not isinstance(checkout, dict) or set(checkout) != {"commit", "tree"}:
        raise ValueError("R3 whole-checkout identity is incomplete or unexpected")
    if any(
        not isinstance(checkout[key], str)
        or len(checkout[key]) != 40
        or any(char not in "0123456789abcdef" for char in checkout[key])
        for key in ("commit", "tree")
    ):
        raise ValueError("R3 whole-checkout commit or tree is malformed")
    root = Path(binding["wan_root"])
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("R3 whole-checkout root must be an existing absolute directory")
    root = root.resolve()
    try:
        if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
            raise ValueError("R3 Wan route must be a complete checkout root")
        if (
            _git(root, "rev-parse", "HEAD") != checkout["commit"]
            or _git(root, "rev-parse", "HEAD^{tree}") != checkout["tree"]
        ):
            raise ValueError("R3 whole-checkout commit or tree differs from binding")
        if binding["route"] == "official-pristine":
            symbolic_ref = subprocess.run(
                ["git", "-C", str(root), "symbolic-ref", "-q", "HEAD"],
                check=False, capture_output=True, timeout=30,
            )
            if symbolic_ref.returncode != 1:
                raise ValueError("R3 pristine checkout must have detached HEAD")
    except subprocess.CalledProcessError as error:
        raise ValueError("R3 whole-checkout identity cannot be verified") from error
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError("R3 whole-checkout identity cannot be verified") from error
    _validate_checkout_clean(root)
    return validate_exact_route_binding({key: value for key, value in binding.items() if key != "checkout"})
