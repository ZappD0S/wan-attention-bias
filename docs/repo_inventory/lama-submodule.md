# LaMa submodule inventory

_As inspected in `/home/zapp/Documents/wan_experiments` (2026-06-05 parent checkout; nested repository state recorded below). This document describes the nested `lama/` repository, not a copy of the parent project's code._

## Identity and Git state

- The parent repository declares `lama` in `.gitmodules` as `https://github.com/advimman/lama.git` and records gitlink `469acc7358a1c6828b647b4ee20c93474a2f36b4`.
- `lama/` is checked out detached at that exact commit (`removed obsolete imgaug subpackage from albumentations`). Its working tree is clean and it has no nested submodules.
- The nested remote is `origin=https://github.com/advimman/lama.git`. Local `HEAD` is two commits ahead of the fetched `origin/main` (`786f5936...`), while the local `main` branch points at that origin commit; therefore the parent points at a local, detached commit rather than the current remote tip.
- Parent-level `git status` was not clean, but the visible changes are outside the LaMa submodule (deleted parent parameter presets and untracked parent directories/files such as `apptainer_images/`, `paper/`, and `evaluation/` additions). No nested LaMa modifications or untracked source files were observed. The other parent submodule `wan2.1` is reported with a leading `-` by `git submodule status` (not initialized according to Git); that is unrelated to LaMa.

## What LaMa contributes to the parent project

LaMa is a complete resolution-robust, Fourier-convolution image-inpainting implementation. It consumes an RGB image and a binary mask and returns an inpainted image, with an optional multi-scale refinement pass. It is not part of the parent Wan model implementation.

The parent has one direct adapter: `image_prompt_generation/big_lama.py` imports `saicinpainting.evaluation.utils.move_to_device` and `saicinpainting.training.trainers.load_checkpoint`, loads `<checkpoint>/config.yaml` plus `<checkpoint>/models/best.ckpt`, and exposes `inpaint_image(model, device, PIL_image, PIL_mask)`. `image_prompt_generation/generate_prompt_image.py` combines SAM2 segmentation with that adapter: it loads `./weights/big-lama`, creates removal masks, and uses LaMa to synthesize “single character” images. The parent `pyproject.toml` also includes `lama-inpainting` as an editable UV workspace member/source. These are the integration points; the nested repository itself does not import parent code.

## Nested repository outline

Selected source-bearing structure (binary/model artifacts and generated caches omitted):

```text
lama/
  README.md                     upstream setup, inference, training/evaluation notes
  pyproject.toml                modern project metadata and dependency ranges
  requirements.txt              legacy pip requirements, several old pins
  conda_env.yml                frozen Python 3.6/ CUDA 10.2-era environment
  LaMa_inpainting.ipynb        Google Colab interactive mask/inpaint demo
  bin/                         prediction, training, evaluation, conversion, mask/data tools
    debug/                      one hard-coded analysis helper
    paper_runfiles/             paper-era, machine-specific shell workflows
  configs/                     Hydra/YAML prediction, training, data-gen and metrics configs
  fetch_data/                  Places/CelebA-HQ download, preparation, sampling, mask scripts
  docker/                      Dockerfiles and mask/predict/evaluate wrappers
  saicinpainting/              importable implementation package
  models/                      ADE20K segmentation code/assets and LPIPS model files
```

There are roughly 274 files in this checkout (about 188 source/config/script/notebook files); no downloaded LaMa checkpoint, experiment directory, dataset, or test image bundle is present at the nested repository root. `models/` contains support assets such as `ade20k/` and `lpips_models/*.pth`, not `big-lama/models/*.ckpt`.

## `saicinpainting` package layers

- `saicinpainting.evaluation`: image/mask dataset loaders (`data.py`), online/offline evaluators, SSIM/LPIPS/FID and segmentation-aware losses, mask proposal/segmentation utilities, visualization, and multi-scale refinement. `InpaintingDataset` expects mask names containing `_mask` and derives the source image name from that stem.
- `saicinpainting.training.data`: Albumentations/imgaug transforms, recursive JPG/WebDataset/segmentation datasets, and random irregular/rectangle/segmentation/super-resolution/outpainting/mixed mask generators.
- `saicinpainting.training.modules`: generator/discriminator building blocks: FFC/Fourier units and FFC-ResNet, Pix2PixHD/global and multi-dilated alternatives, spatial transforms, squeeze-excitation, depthwise/multi-dilated convolutions, and fake-fakes support.
- `saicinpainting.training.losses`: adversarial/R1, feature matching, masked L1, perceptual/ResNet perceptual, style, segmentation, and distance-weighting losses.
- `saicinpainting.training.trainers`: PyTorch Lightning training module, optimizer/training loop, checkpoint loading, validation and evaluator wiring.
- `saicinpainting.training.visualizers`: directory/no-op visualizers and color/base helpers.
- `saicinpainting.utils`: ramps, device/shape/config helpers, DDP/debug handling, and checkpoint-adjacent utilities.
- `models/ade20k` and `models/lpips_models`: auxiliary segmentation and perceptual-evaluation implementation/data used by evaluation and perceptual losses; these are nested-repository assets, not parent code.

## Install and run pathways

### Host install

The README documents three historical routes:

1. A virtualenv with Python 3 and `torch==1.8.0`, `torchvision==0.9.0`, then `pip install -r requirements.txt`.
2. `conda env create -f conda_env.yml`, activation of `lama`, then a CUDA 10.2 PyTorch install and `pytorch-lightning==1.2.9`.
3. Docker.

The checked-out `pyproject.toml` is newer and declares Python `>=3.8,<3.14`, broad minimum versions, optional `dev` and `yadisk` extras, and setuptools package data for YAML. The parent invokes it as an editable UV workspace package rather than using the nested legacy environment. Before running the scripts, the README expects `cd lama`, `TORCH_HOME=$(pwd)`, and `PYTHONPATH=$(pwd)`.

### Inference

`bin/predict.py` is the primary CLI. It uses Hydra config `configs/prediction/default.yaml`, reads the training `config.yaml` inside a checkpoint directory, loads `models/{best.ckpt|other checkpoint}`, forces prediction-only/no-op visualization, discovers masks under `indir`, and writes one output per mask to `outdir`. Default input suffix is `.png`, output extension is `.png`, and input images/masks are padded to a multiple of 8. The expected conventional layout is:

```text
<input dir>/image.png
<input dir>/image_mask001.png
<checkpoint dir>/config.yaml
<checkpoint dir>/models/best.ckpt
```

The README recommends downloading `big-lama.zip` from Hugging Face or the all-model Google Drive folder, then running `python3 bin/predict.py model.path=... indir=... outdir=...`. `refine=True` enables the multi-scale feature refinement path and requires the unpadding metadata produced by the dataset loader. Docker wrappers mount project/model/input/output directories and call the published `windj007/lama` image; available wrappers are `docker/1_generate_masks_from_raw_images.sh`, `docker/2_predict_with_gpu.sh`, and `docker/3_evaluate.sh`.

### Training and evaluation

Training is Hydra-composed from `configs/training/<model>.yaml` and grouped defaults under `data/`, `generator/`, `discriminator/`, `optimizers/`, `trainer/`, `visualizer/`, `evaluator/`, `location/`, and `hydra/`. `bin/train.py` builds the Lightning module, saves the resolved `config.yaml`, writes `models/` checkpoints and TensorBoard logs, and supports DDP. `bin/gen_mask_dataset.py` creates fixed cropped image/mask pairs for validation/test; training generally generates masks online. `bin/evaluate_predicts.py` evaluates precomputed outputs with SSIM/LPIPS/FID (and optional ADE20K segmentation metrics).

## Configuration conventions

- Hydra CLI values use dotted overrides such as `model.path=...`, `dataset.img_suffix=.jpg`, `data.batch_size=10`, and `location=my_dataset`; config names omit `.yaml` in the README.
- `configs/prediction/default.yaml` separates model, dataset, output key, device, and refinement settings. The model directory is authoritative: prediction rereads its saved training `config.yaml`.
- Training defaults compose a location (absolute `data_root_dir`, `out_root_dir`, `tb_dir`, `pretrained_models`), data layout, generator/discriminator, loss/optimizer settings, trainer limits/checkpoint monitor, and visualization. Example data paths assume `train`, `val`, and `visual_test` directories; validation files are paired image/mask PNGs.
- Configs use `${location.data_root_dir}`, `${data.*}`, `${dataset.*}`, `${env:TORCH_HOME}`, and Hydra defaults/overrides. Locations are intentionally machine-specific: `places_example.yaml` uses `/home/user/inpainting-lama/...`, while `docker.yaml` uses `/data/...`.

## Dataset, checkpoint, and resource assumptions

- Generic evaluation expects images and masks together, with masks matching `[image stem]_maskXXX[image suffix]`; mask images are grayscale and inputs are RGB. Training’s default dataset recursively finds JPG images, applies crops/augmentations, and creates masks; WebDataset and image-plus-segmentation variants are also available.
- The documented benchmarks assume very large external Places365 (standard/challenge) or CelebA-HQ downloads. `fetch_data/` scripts unpack/split these datasets, sample fixed validation images, and generate masks. Evaluation claims for Places refer to a previously generated 30,000-image set. No such data is checked in here.
- Inference requires a downloaded checkpoint directory containing both the saved Hydra config and Lightning checkpoint. Perceptual/segmentation evaluation may additionally need ADE20K weights and the included LPIPS assets. CUDA/GPU memory is assumed for the advertised models; refinement accepts GPU IDs and has a pixel budget.
- The parent currently has an empty `weights/` directory: `image_prompt_generation/generate_prompt_image.py` requests `./weights/big-lama`, but no `weights/big-lama/config.yaml` exists in this checkout. The model is therefore not runnable from the parent checkout without separately provisioning the artifact.

## Notebook and Docker contents

`LaMa_inpainting.ipynb` is a six-cell Colab-oriented demo. It clones the upstream repo, installs a mixture of legacy and unpinned packages, downloads a Yandex-hosted model, uploads/draws a mask in browser JavaScript, and invokes `bin/predict.py` for JPG/JPEG/PNG. It is an interactive demo, not a parent integration test.

`docker/Dockerfile` targets CUDA 10.2 Ubuntu 18.04 with Python 3.9-era Miniconda and Torch 1.8.1; `Dockerfile-cuda111` targets CUDA 11.1 and Torch 1.8.2/vision 0.9.2. Build scripts use the `windj007/lama` tag. `docker/entrypoint.sh` simply executes the supplied command. The wrappers bind-mount the nested checkout as `/home/user/project` and use fixed container names.

## Cursory gap/bug review (no fixes made)

Evidence in the checked-out files indicates several reproducibility or integration gaps:

1. **Training default is broken as written.** `bin/train.py` has `@hydra.main(..., config_name='tiny_test.yaml')`, but `configs/training/tiny_test.yaml` does not exist. README commands explicitly override the config name (for example `-cn lama-fourier`), so those commands may avoid this default, but launching `bin/train.py` without an override cannot resolve its declared config.
2. **Stale Docker documentation.** README inference step 2 calls `docker/2_predict.sh`, which is absent; only `docker/2_predict_with_gpu.sh` exists. The README also says Yandex distribution links went bad while retaining later Yandex links and the Colab notebook still downloads via Yandex.
3. **Stale/missing location and relative paths.** README training uses `location=places_standard`, but no `configs/training/location/places_standard.yaml` is checked out (only `docker.yaml`, `places_example.yaml`, and `celeba_example.yaml`). The Places Challenge example calls `bash places_challenge_train_download.sh` from the repository root, while the script is under `fetch_data/`. Paper/debug shell files contain machine-specific `/data/inpainting` and `/media/inpainting` paths.
4. **Dependency drift is material.** The legacy requirements pin Albumentations 0.5.2, Hydra 1.1.0, Lightning 1.2.9, scikit-image 0.17.2, and Python 3.6 in `conda_env.yml`; the parent `uv.lock` currently resolves Albumentations 2.0.8, Hydra 1.3.2, Lightning 2.6.1, scikit-image 0.26.0, Python 3.11, and Torch 2.10. The nested code imports Albumentations internals and `imgaug` (`saicinpainting/training/data/aug.py`), so the modern editable workspace is a compatibility risk despite the broad minimum ranges. In the current shell, direct imports also fail immediately where dependencies are not installed (`imgaug`, `albumentations`, `numpy`, and `torch` were each missing); this is an environment observation, not a source fix.
5. **Packaging declaration is narrow.** `lama/pyproject.toml` sets setuptools `packages = ["saicinpainting"]` rather than enumerating/discovering its `evaluation` and `training` subpackages. Editable source use may conceal this, but a regular wheel/install could omit the imported subpackages.
6. **An unused/secondary evaluation path is visibly broken.** `OurPrecomputedInpaintingResultsDataset.__getitem__` calls `self.file_loader(...)`, but `OurInpaintingDataset` does not define `file_loader`; that path will fail if exercised. The main generic `PrecomputedInpaintingResultsDataset` path uses `load_image` instead.
7. **Parent adapter bypasses LaMa’s preprocessing.** `image_prompt_generation/big_lama.py` builds tensors directly from arbitrary PIL dimensions and calls the model, whereas `bin/predict.py` uses the dataset’s `pad_out_to_modulo: 8` and optional unpadding. Non-modulo image sizes are consequently an integration risk. There is no parent-level LaMa smoke/integration test; nested tests are limited to ADE20K utility tests and the `countless` mask test.
8. **Artifacts are absent.** No nested checkpoint, benchmark data, or parent `weights/big-lama` directory is present, so documentation commands and the parent SAM2+LaMa flow cannot be end-to-end verified from this checkout. Parent-level untracked/deleted files noted above are not nested LaMa changes.

## Inventory status

Path: `docs/repo_inventory/lama-submodule.md`
Status: written; documentation-only change. No nested repository code, model, configuration, or generated artifact was modified.
