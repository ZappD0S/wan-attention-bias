# Wan2.1 package and exports inventory

## Scope and attribution

This report is limited to the detached `wan2.1` checkout at `f7472d354e0cb46b2853f15bf97b8b283d4780f8` and these five files only:

- `.gitignore`
- `pyproject.toml`
- `wan/__init__.py`
- `wan/modules/__init__.py`
- `wan/regional_prompt/__init__.py`

Git history was filtered to commits whose author is exactly `gzappavi <gianluca.zappavigna@inria.fr>`. Nine matching commits touch this file subset. The current `HEAD` commit (`f7472d3`) has the same author but changes a file outside the assigned subset, so it is not included in the evolution below. No other source or configuration file was read or analyzed.

## Current role and import/data flow

This subset describes package metadata and the public import surface rather than the model's tensor/data pipeline:

1. `pyproject.toml:5-33` defines the `wan` distribution (version `2.1.0`), Python requirement `>=3.10,<4.0`, runtime dependencies, and the `einsum` dependency. `pyproject.toml:53-57` declares setuptools packaging and a recursive Python-file package-data pattern.
2. Importing `wan` (`wan/__init__.py:1-6`) eagerly imports the `configs`, `distributed`, and `modules` namespaces and re-exports `WanFLF2V`, `WanI2V`, `WanT2V`, `WanVace`, and `WanVaceMP`. The custom I2V export remains commented at line 4.
3. Importing `wan.modules` (`wan/modules/__init__.py:1-6`) eagerly imports attention, model, text-encoder/tokenizer, VACE-model, and VAE symbols. Its `__all__` list at lines 8-19 is intended to define wildcard-exported names.
4. `wan/regional_prompt/__init__.py:1` re-exports `WanI2V` from its local `image2video` module.
5. `.gitignore:1-38` excludes dotfiles, Python caches, model/checkpoint formats, media and archive formats, storage/sample directories, named model directories, `poetry.lock`, and egg-info metadata. These rules affect source-control visibility, not Python runtime flow.

No model input, latent, or output flow can be established from the assigned files alone.

## gzappavi commit evolution

| Date | Commit | Assigned-file change |
|---|---|---|
| 2025-09-17 | `4734c3e` | Added `*.egg-info/` to `.gitignore`. |
| 2025-09-18 | `e18e588` | Added `CustomWanModelVaceWanModel` to `wan/modules/__init__.py::__all__`. |
| 2025-09-18 | `bf83df6` | Added Ruff format quote preservation and a lint ignore for `F401`. |
| 2025-09-22 | `9db7c30` | Added Ruff line length `88`. |
| 2025-10-07 | `4ec6025` | Added `einsum>=0.3.0` to project dependencies. |
| 2025-10-09 | `44e09da` | Added the `CustomWanI2V` top-level export. |
| 2025-10-30 | `2d19180` | Commented out that custom top-level export and created `wan/regional_prompt/__init__.py` with a `WanI2V` re-export. |
| 2025-12-04 | `06d345e` | Added `[tool.pyright]` with `typeCheckingMode = "standard"`. |
| 2026-03-23 | `390de16` | Expanded Ruff ignores to `F401` and `E741`; also normalized the `.gitignore` final newline (the egg-info rule itself originated in `4734c3e`). |

The history shows experimentation with a custom model export followed by disabling that export, while the malformed `__all__` entry added in September remains in the current file.

## Static bug and gap review

### Definite issue: invalid `__all__` entry

`wan/modules/__init__.py:11` lists `CustomWanModelVaceWanModel`, but lines 1-6 import no symbol with that name and the file defines no such name. A static comparison of the local imports and `__all__` confirms this mismatch. Consequently, `from wan.modules import *` is expected to fail while resolving the missing attribute, unless another mechanism outside this file injects it. The spelling also looks like two names concatenated, but the intended replacement cannot be determined within scope.

### Packaging declaration is incomplete or ambiguous

`pyproject.toml:54` explicitly sets `packages = ["wan"]`, while `wan/__init__.py:1` relies on `configs`, `distributed`, and `modules` subpackages and `wan/regional_prompt/__init__.py` establishes another subpackage. The recursive package-data rule at lines 56-57 may copy Python files, but it does not make the explicit package list self-documenting and should not be assumed to provide normal subpackage metadata. An installed-wheel import of these subpackages should be validated; no build was run here.

### Eager and unstable public surface

The package initializers eagerly import multiple implementation modules (`wan/__init__.py:1-6` and `wan/modules/__init__.py:1-6`), increasing import-time coupling and making any dependency/import failure affect basic package import. `wan/__init__.py:4` is a commented-out export, and `wan/regional_prompt/__init__.py:1` exposes another `WanI2V` path without an explicit `__all__`; the intended supported distinction between standard, custom, and regional imports is therefore unclear.

### Broad lint suppressions reduce detection

`pyproject.toml:78-82` globally ignores `F401` and `E741`. `F401` is commonly useful for detecting unused imports, although package initializers intentionally use imports as re-exports; suppressing it project-wide hides unrelated cases. Ignoring `E741` likewise removes a readability/error signal everywhere. The Ruff section does not define a project-specific `select` list in the current file, so the effective checks depend on Ruff defaults and version.

## Reproducibility and maintenance risks

- Most runtime dependencies in `pyproject.toml:15-32` are unpinned or only lower-bounded; the development dependencies at lines 35-43 are also unpinned. Only NumPy has an upper bound, and Python is the only explicit interpreter range. This allows resolver drift and makes reproducing the environment difficult.
- `.gitignore:13-17` excludes common model weights and JSON metadata, while lines 33-38 exclude named model directories, `poetry.lock`, and egg-info. This reduces accidental large-file commits but can also hide files needed to reconstruct an environment or diagnose packaging.
- The declared distribution version (`pyproject.toml:7`) remains `2.1.0` despite later custom/regional export changes in the scoped history, so consumers have no metadata-level indication of those interface changes.
- The malformed export is especially likely to remain unnoticed because the global `F401` suppression is designed to silence initializer import diagnostics, and no scoped export test is declared here.

## Validation performed

- Parsed all three assigned Python initializers with Python `ast.parse`; all parsed successfully.
- Parsed `pyproject.toml` with Python `tomllib`; it is syntactically valid TOML.
- Compared `wan/modules/__init__.py` local imported names against its literal `__all__`; `CustomWanModelVaceWanModel` is the sole missing name.
- Ran `git diff --check` over the assigned-file range from the pre-gzappavi parent of `4734c3e` through `HEAD`; it reported no whitespace errors.
- Queried Git history using the exact author identity and reviewed only the assigned-file diffs.

No package import, dependency installation, build/wheel generation, model inference, web app, or network validation was run.

## Limitations and recommended follow-up

This is a cursory static review constrained to five files. It cannot verify whether the missing `__all__` name is injected elsewhere, whether package-data behavior produces a usable installed subpackage layout, or whether dependency versions are mutually compatible. Recommended follow-up is to correct or remove the invalid export, explicitly decide/document the custom and regional public APIs, use explicit subpackage discovery, narrow lint exceptions where possible, and add a locked/tested environment plus import and wheel-install checks.
