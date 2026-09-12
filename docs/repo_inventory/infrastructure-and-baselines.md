# Infrastructure and baseline inventory

Scope: `containers/`, `apptainer_images/`, and `baseline_comparison/` only. This is an inventory of the checkout as inspected; no build or notebook execution was attempted.

## Container definitions and runtime purpose

### `containers/`

- `containers/cuda_ubuntu.def` is an Apptainer definition for a CUDA/Python runtime. It uses a two-stage build: `ghcr.io/astral-sh/uv:latest` supplies `uv`/`uvx`, and `nvidia/cuda:12.8.2-devel-ubuntu24.04` is the runtime base (`cuda_ubuntu.def:1-6,23-25`).
- The definition copies only the root `pyproject.toml`, `uv.lock`, `.python-version`, and the two workspace trees `lama` and `wan2.1` into `/opt/build` (`cuda_ubuntu.def:27-33`). This is a focused image for the Wan/LAMA workspace rather than a copy of the whole experiment repository. The root `pyproject.toml` confirms those two workspace members and maps `wan` and `lama-inpainting` to them; it also requires Python `>=3.11,<3.12`, CUDA-128 PyTorch wheels, and GPU-oriented packages such as `flash-attn`.
- `%post` installs compilers/build tools, Git/LFS, `ffmpeg`, and basic GL libraries, then runs `uv sync --frozen --no-dev --no-editable` (`cuda_ubuntu.def:35-57`). It sets CUDA compilation targets `7.0;7.5;8.0;8.6;8.9;9.0+PTX` and removes the build tree afterward. The runtime exports CUDA paths, `/opt/venv`, and an offline/no-Python-download UV configuration (`cuda_ubuntu.def:8-21`). The runscript simply executes Bash (`cuda_ubuntu.def:59-60`).
- `containers/build_image.sh` is a host-specific wrapper: it hard-codes `APPTAINER_CACHEDIR=/local_scratch/gzappavi/.cache/apptainer` and force-builds `containers/cuda_ubuntu.sif` from the definition (`build_image.sh:1-3`). The output SIF is not present in this checkout and would be ignored by the root `*.sif` rule (`.gitignore`). The wrapper therefore assumes Apptainer, the named local scratch path, and invocation from the repository root (the `%files` paths are repository-relative).

### `apptainer_images/` nested repository

This directory is not a root Git submodule: it appears in the root status as an untracked `apptainer_images/` directory, and root `.gitmodules` declares only `wan2.1` and `lama`. It contains its own `.git` repository and currently has these working-tree files: `.gitattributes`, `.gitignore`, `.gitlab-ci.yml`, and `build.sh`; `arch_ml/` and `cuda_ubuntu/` are empty directories.

- The nested `build.sh` is an interactive `arch_ml` build helper (`apptainer_images/build.sh:1-86`). It requires `jq`, a local `.env`, `GITHUB_TOKEN`, Bitwarden client credentials, the `bw` CLI, an unlocked/usable Bitwarden session, and `$HOME/.config/Bitwarden CLI` (`build.sh:4-23,25-73`). It then passes those values as Apptainer build arguments and builds `arch_ml.sif` from `arch_ml.def` (`build.sh:74-84`). It assumes it is launched from the nested repository root; it does not locate its own directory before reading `.env` or the definition.
- `.gitlab-ci.yml` defines one GitLab `build-sif` job using floating `archlinux:latest`, a runner tagged `linux` and `large_more_disk`, and `pacman -Sy --noconfirm apptainer` (`.gitlab-ci.yml:1-15`). It builds `arch_ml.sif` from `arch_ml.def`, logs into the GitLab registry, pushes `oras://$CI_REGISTRY_IMAGE/arch_ml:latest`, and retains the SIF as a one-week artifact (`.gitlab-ci.yml:17-33`). It runs only when `arch_ml.def` changes or for a web pipeline (`.gitlab-ci.yml:35-39`).
- The nested index records `arch_ml/arch_ml.def`, `arch_ml/arch_ml.sif`, `cuda_ubuntu/cuda_ubuntu.def`, `cuda_ubuntu/cuda_ubuntu.sif`, and `cuda_ubuntu/cuda_ubuntu_arm64.sif` as staged additions followed by working-tree deletions (`AD`). Their content is unavailable in this checkout: the files are absent and `git fsck` reports their blobs missing. Consequently, the CI and helper references to `arch_ml.def` are not runnable from the current working tree. `.gitattributes` intends all SIF files to use Git LFS, while `.gitignore` ignores only `.env` (`.gitattributes:1`, `.gitignore:1`).

The nested repository has no commits on local `master`; its configured `origin` is `git@gitlab.inria.fr:gzappavi/apptainer_images.git`, but the configured remote-tracking `main`/`master` refs point at invalid/missing objects and the branch reports `origin/master [gone]`. This is a repository-state/reproducibility problem independent of the definitions themselves.

## Baseline-comparison workflow

`baseline_comparison/baselines_comparison.ipynb` is a 14-cell Python 3 notebook (kernel metadata says Python 3.11.13). Its workflow is:

1. Import PyTorch, NumPy, `diffusers` image/video helpers, IPython display, and the workspace `wan` package (`cells 0-1`). A commented alternative shows a Diffusers pipeline/checkpoint path, but the executed path uses `WanI2V`.
2. Construct `WanI2V` with `i2v_14B`, `device_id=0`, `t5_cpu=True`, and the relative checkpoint `../weights/Wan2.1-I2V-14B-480P/` (`cell 2`).
3. `generate_video` loads each local image, computes a size near `480*832`, rounds dimensions to the model VAE/patch stride, clears/synchronizes CUDA memory, calls `wan_i2v.generate` with a fixed negative prompt and `sampling_steps=10`, converts tensor layout/range, and returns `export_to_video(..., fps=16)` (`cell 3`). The output path assignment is commented out, so generated videos are displayed inline rather than saved by the notebook.
4. Nine prompt cases call the function (`cells 4-12`): four variants using `three_people_interacting.jpg`, one using `man_at_desk.jpg`, two using `girl_watering_plants_old_man_reading.png`, one using `girl_texting_man_writing.png`, and one using `twin_girls.png`. They display each returned video with IPython embedding.

The five local input assets all exist and were inspected by file headers only (no image decoding):

| Asset | Format / dimensions | Size |
|---|---|---:|
| `three_people_interacting.jpg` | JPEG, 6016x4016 | 2,936,399 bytes |
| `man_at_desk.jpg` | JPEG, 5760x3840 | 881,983 bytes |
| `girl_texting_man_writing.png` | PNG, 1024x1024 | 2,015,448 bytes |
| `girl_watering_plants_old_man_reading.png` | PNG, 1024x1024 | 2,233,361 bytes |
| `twin_girls.png` | PNG, 1024x1024 | 1,553,710 bytes |

The notebook and all five assets are root-tracked. There are no generated video files in this directory; root `.gitignore` excludes `*.mp4` and `*.avi`.

### Backup/configuration comparison

`baselines_comparison.ipynb.bak` is a materially different 12-cell, 5.59 MB saved run, versus 14 cells and 3.23 MB for the active notebook. The backup's function uses `sampling_steps=50`, while the active notebook uses 10. The backup contains progress/video outputs for the earlier prompt cells (and execution counts such as 1, 2, 4, 5, 6, and 7); the active notebook has no execution counts and retained outputs mainly for cells 8-12. Both retain widget/video output metadata. The `.bak` is therefore useful provenance but is a stale/ambiguous baseline configuration, not an interchangeable copy.

## Inputs, outputs, and repository boundaries

- The notebook depends on model checkpoints outside this scope: `../weights/Wan2.1-I2V-14B-480P/` resolves correctly only when the notebook is run with `baseline_comparison/` as its working directory. The repository's `weights/` directory is currently empty, and weights are ignored by the root `.gitignore`; no checkpoint provenance is recorded in this directory.
- The `wan` import and `i2v_14B` config come from the root workspace dependency (`wan2.1`), while the image assets and notebook are local to `baseline_comparison/`. The container definition similarly copies only `wan2.1`, `lama`, and dependency manifests, establishing a clear infrastructure boundary from the rest of the experiment scripts.
- There is no baseline-comparison script, parameter manifest, result file, metric computation, or automated test in this scope. The notebook is an interactive qualitative generation/display workflow; its only durable inputs here are the notebook code, embedded prior outputs, and five images.

## Cursory gap and bug review

- **Portability:** the root container wrapper requires a user-specific `/local_scratch/gzappavi` cache path; the notebook and nested helper rely on the current working directory; the helper additionally assumes a local `.env`, Bitwarden installation/state, and a particular home-directory config path. These assumptions are explicit in `containers/build_image.sh:1`, notebook relative paths, and `apptainer_images/build.sh:10-17,67-73`.
- **Reproducibility:** `ghcr.io/astral-sh/uv:latest` and CI's `archlinux:latest` are floating bases; the CI installs Apptainer with an unpinned system package; and the root definition installs unpinned APT packages (`cuda_ubuntu.def:2,38-40`; `.gitlab-ci.yml:7,15`). The root Python lockfile and frozen sync improve Python dependency repeatability, but model checkpoints are absent/ignored and the active notebook/backup disagree on sampling steps.
- **Stale or incomplete container state:** nested SIF/definition entries are staged-but-deleted with missing Git objects, the nested repository has no usable commit or remote refs, and the current nested CI references the missing `arch_ml.def`. The root CUDA definition is present, but its generated SIF is intentionally ignored and absent.
- **Automation/tests:** only the nested `arch_ml` CI job is defined; the root CUDA image has a local wrapper but no CI job in this scope. No build smoke test, notebook execution check, output validation, or baseline regression test is present.
- **Obvious references to verify before use:** the five notebook image references resolve, but the checkpoint path does not resolve in the current empty `weights/` directory. The nested `arch_ml.def` reference cannot resolve in the current working tree. The commented `video_path` confirms that displayed outputs are not automatically materialized as comparison artifacts.

Checks performed were non-mutating: directory/file inventory, Git status/ref and integrity inspection, notebook JSON parsing and structural cell comparison, shell syntax inspection, and image-header metadata inspection. No image pixels were decoded, no container was built, and no notebook cell was executed.
