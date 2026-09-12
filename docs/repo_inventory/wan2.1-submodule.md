# `wan2.1` submodule inventory

**Scope:** historical local inspection only. This report records the pre-initialization state; the submodule was subsequently initialized at the pinned commit. See [`wan2.1/README.md`](wan2.1/README.md) for the current author-scoped source inventory. No fetch, submodule initialization, checkout, or other submodule modification was performed during this original inspection.

## Evidence: parent repository metadata

- Path: `wan2.1/`.
- `.gitmodules` declares:
  - URL: `https://github.com/ZappD0S/Wan2.1.git`
  - configured update branch: `dev`
  - path: `wan2.1`
- The parent `HEAD` and index both record the gitlink commit
  `f7472d354e0cb46b2853f15bf97b8b283d4780f8` for `wan2.1`.
- The parent-local config has `submodule.wan2.1.active=true` and the same URL.
- `git submodule status --recursive` reports:

  ```text
  -f7472d354e0cb46b2853f15bf97b8b283d4780f8 wan2.1
  ```

  The leading `-` is Git's uninitialized-submodule status marker.
- The parent is on `master` and was not clean during inspection: unrelated tracked deletions and untracked files were already present. No parent gitlink change for `wan2.1` was observed.

## Evidence: current checkout and nested Git metadata

- `wan2.1/` exists as an empty directory (no files, no `.git` marker; mode observed as `drwx------`).
- The nested repository metadata is retained at `.git/modules/wan2.1/`. Its config records a non-bare repository, worktree `../../../wan2.1`, and remote URL `https://github.com/ZappD0S/Wan2.1.git`.
- Nested `HEAD` is detached at `f7472d354e0cb46b2853f15bf97b8b283d4780f8`, whose title is `further changes to reduce VRAM usage`.
- Locally retained nested refs report:
  - `origin/dev` and `origin/split_sentences` at `f7472d3` (the gitlink target).
  - `origin/main` and local `main` at `7c81b2f` (`Update README.md`).
  - `origin/ediffi` at `8180035`.
- The nested object database can enumerate a 66-path tree for the target commit, including `README.md`, `pyproject.toml`, and the `wan/` package, but those files are not checked out under `wan2.1/`.
- Explicit status using the retained nested Git directory and the empty worktree reports the tracked files as deleted. This is consistent with an absent worktree, not evidence that those files were intentionally deleted in the parent repository.

**State assessment (evidence):** the submodule worktree is currently absent/uninitialized, while a prior nested clone and the target commit's Git objects remain locally available. The parent gitlink itself is present and points to an object that is available in the retained nested metadata.

**Inference:** the worktree was likely deinitialized or otherwise removed after the nested repository was cloned and checked out. The available metadata does not establish which operation removed it.

## Evidence: expected role of Wan2.1

- The parent `pyproject.toml` lists `wan2.1` as a UV workspace member, maps the `wan` dependency to that workspace as an editable source, and declares `wan` as a project dependency. `uv.lock` likewise records `wan` as editable from `wan2.1`.
- Parent code imports the package provided by that path, including `wan.configs.wan_i2v_14B`, `wan.regional_prompt.WanI2V`, and `wan.utils.utils.cache_video`.
- `containers/cuda_ubuntu.def` copies `wan2.1` into the image build context and then runs a frozen UV sync.
- Parent inference scripts use Wan I2V code and refer separately to checkpoints such as `weights/Wan2.1-I2V-14B-480P/`; `download_weights.sh` downloads `Wan-AI/Wan2.1-I2V-14B` weights. Thus source code and model weights are separate dependencies.
- The retained nested `pyproject.toml` identifies the package as `wan`, version `2.1.0`; its retained README describes Wan2.1 as a video foundation-model suite supporting (among other tasks) image-to-video.

**Inference:** in this parent project, the submodule is intended to supply the local Python implementation of the Wan2.1 video model, including the project's regional-prompting extensions, while downloaded model checkpoints supply the parameters. This role is evidenced by workspace configuration and imports; it is not runnable from the current empty worktree.

## Availability and reproducibility implications

### Available locally

- Parent-side submodule declaration, active-submodule config, and a pinned gitlink SHA.
- Nested Git metadata, commit object, refs, and source tree objects for the pinned SHA.
- Parent references to the package and expected checkpoint locations.

### Unavailable in the current checkout

- All working-tree source files, package metadata files, README assets, and build inputs below `wan2.1/`.
- A usable local `wan` package at the workspace path unless it has been installed independently outside this checkout.
- Model checkpoints: the repository references/downloads them, but no checkpoint inventory was assumed or inferred from the missing submodule.

**Inference:** a fresh environment using this checkout cannot reliably run the parent code or reproduce the container build as-is: imports from `wan` have no checked-out source, the UV workspace member path is empty, and the container recipe expects `wan2.1` to be populated. The pinned SHA improves provenance once the declared remote or another copy is available, but the current local state does not itself provide a usable source checkout. The `.gitmodules` `branch=dev` is not the reproducibility pin; the parent gitlink SHA is the pin. No remote fetch was performed, so current remote availability or branch movement was not assessed.

## Cursory gap/bug review (no fixes)

- **Missing dependency state (high impact):** `git submodule status` marks `wan2.1` uninitialized and the directory is empty, despite the parent workspace, lockfile, imports, and container recipe requiring it.
- **Dangling filesystem references:** the `wan2.1` workspace member, editable `wan` source, `from wan ...` imports, and container `%files` input all resolve to an absent working tree. No dangling gitlink SHA was found: the target commit is present in the retained nested object database.
- **Provenance/documentation gap:** the parent root `README.md` is empty, and no root-level setup guidance for initializing this submodule was found. The parent URL is a project fork (`ZappD0S/Wan2.1.git`) while the retained nested README links the upstream Wan-Video project; this distinction and the required pinned commit are not explained in parent documentation.
- **Reproducibility gap:** the source commit is pinned, but the current checkout does not contain the source, and the separate model-weight download requirement is not represented by the submodule itself. A clean, runnable reproduction therefore requires restoring the submodule worktree and obtaining the referenced weights in addition to the parent repository.
