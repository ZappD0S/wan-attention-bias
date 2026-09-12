# Paper and loose experiments inventory

Snapshot of the currently present working tree. This inventory is limited to `paper/`, `old/`, `plans/`, `preprocessing/`, `prompts/`, `scratch/`, `weights/`, and the clearly relevant root-level note/status files. Directory listings were used for the empty directories; binary/media/checkpoint contents were not opened.

## Status at a glance

| Path | Contents now | Git/status evidence |
| --- | --- | --- |
| `paper/` | Populated with seven files and one `papers/` subdirectory | No indexed files; all listed files are untracked. Not ignored. |
| `old/` | Empty | Ignored by the root `.gitignore` (`/old/`). No tracked files. |
| `plans/` | Empty | Not ignored and has no index entry. Because it contains no files, Git cannot represent the directory itself. |
| `preprocessing/` | Empty | Not ignored and has no index entry; no files to track. |
| `prompts/` | Empty | Not ignored and has no index entry; no files to track. |
| `scratch/` | Empty | Not ignored and has no index entry; no files to track. |
| `weights/` | Empty | Ignored by the root `.gitignore` (`/weights/`). No tracked checkpoint files. |

Thus none of the six loose experimental/support directories contains current code, notes, media, or checkpoints. The two ignored directories can exist locally without appearing as untracked paths; the other empty directories are also absent from a clone unless a placeholder is added. No media or checkpoint placeholder files were present in these directories at the scan.

## `paper/`

`paper/` is a populated but entirely untracked draft area in this checkout. Its local `.gitignore` contains only `/build/`, so it would ignore a build directory if one were created, but the ignore file itself and all source assets are currently untracked.

### Draft manuscript

- **`paper/main.tex`** — the primary LaTeX manuscript. It is a draft about inference-time compositional control for Wan2.1 image-to-video generation. The proposed method uses first-frame entity masks, self-attention-derived spatio-temporal tracking, entity-specific image/text contexts, LaMa inpainting, cross-attention feature gating, and self-attention masking. The experiments are planned around two entities, similar/identical subjects, SAM2 tracking, per-entity cropped videos, Qwen3-VL scoring, and a correct-versus-wrong action-prompt margin. The file has a title/author and four main sections (`Introduction`, `Related Work`, `Method`, and `Experiments`), but it has no reported experiment tables or results yet.
- **`paper/old_notes.tex`** — an earlier standalone set of “attention masking” notes. It sketches combined video/text masks, self-attention-derived Wan2.1 video masks, an outer-product cross-attention mask, and a beta-weighted masked/unmasked attention alternative. It is not included by `main.tex` and reads as an alternate or superseded formulation rather than a finished appendix.

### Bibliographic and paper-reference assets

- **`paper/references.bib`** — five BibTeX entries: Wan, Concept Weaver, MultiTalk/Let Them Talk, LaMa, and Video-Bench. `main.tex` currently cites Concept Weaver, MultiTalk, and LaMa; the Wan and Video-Bench entries are present but unused by the manuscript's citation commands.
- **`paper/ieeenat_fullname.bst`** — local IEEE/natbib-style bibliography format selected by `main.tex`.
- **`paper/papers/concept_weaver.md`** — a local Markdown copy/notes version of *Concept Weaver: Enabling Multi-Concept Fusion in Text-to-Image Models*. It covers the concept bank, template generation, inversion/feature extraction, SAM/grounding masks, feature-space multi-concept fusion, and reported comparisons. It is used as the paper's cited conceptual source, but is not a LaTeX input.
- **`paper/papers/multitalk.md`** — a long Markdown copy/notes version of *Let Them Talk: Audio-Driven Multi-Person Conversational Video Generation*. It describes MultiTalk's adaptive person localization from reference-image-to-video attention and its L-RoPE audio/person binding, training, datasets, metrics, and appendices. `main.tex` cites it for the attention-based tracking idea, but does not include this Markdown file in the build.

The local `papers/` directory contains Markdown only: there are no downloaded PDFs or local figures. The Markdown copies contain remote figure URLs, so those illustrations are not self-contained repository assets.

## Loose-directory purpose/status and evidenced code links

There is no documentation or code inside the six empty directories that establishes a local purpose beyond their names. The following links are evidenced outside those directories:

- `weights/` is the expected local model-cache location for the root `regional_prompting.py`, which sets `checkpoint_dir` to `./weights/Wan2.1-I2V-14B-480P/`. The root `download_weights.sh` downloads `Wan-AI/Wan2.1-I2V-14B-480P` into that same layout. Since `weights/` is empty and ignored, the expected checkpoint is absent from this checkout and is not versioned.
- The manuscript's method is conceptually linked by its explicit citations to the local Concept Weaver and MultiTalk paper notes and to the LaMa bibliography entry. No source link from `main.tex` to the empty `old/`, `plans/`, `preprocessing/`, `prompts/`, or `scratch/` directories is evidenced.

## Root-level research-note check

`README.md` is the only clearly relevant root-level project note found at this scope, and it is a tracked zero-byte file. It supplies no setup, experiment, or paper guidance. The root `.gitignore` is also relevant status evidence: it ignores `/old/` and `/weights/`, generated media such as `*.mp4`, and common output/debug directories. No other nonempty root-level research note was identified; root-level Python and shell files are implementation/support code rather than notes and are outside this inventory.

## Cursory bug/gap review (no fixes made)

- **Documentation is visibly incomplete.** `main.tex` retains many TODOs and editorial questions, including terminology, mask definitions, the I2V motivation, evaluation naming, and the role of soft masks. Related Work and much of Experiments are bullet-point plans. The manuscript still contains a literal `[...]` placeholder for the quality-assessment discussion and TODOs for two result tables. `old_notes.tex` contains informal wording and appears to preserve an unresolved alternative design.
- **Reproducibility path is missing.** The tracked README is empty; the paper has no checked-in build instructions, generated PDF, datasets, first-frame masks, prompt/config bundle, evaluation commands, or result artifacts. The manuscript names SAM2, Qwen3-VL, VBench, Wan2.1, and LaMa without pinning versions or providing a complete execution path. The expected Wan checkpoint path is empty and ignored, so a fresh checkout cannot run the evidenced root script without an external download/configuration step.
- **Reference/dangling risks.** A scan found no missing BibTeX keys for the three citation commands in `main.tex`, and all three `\ref`/`\eqref` targets resolve to labels in that file. However, the manuscript says “VBench” without citing it, while the bibliography contains an unused “Video-Bench” entry; whether these are intended to be the same evaluation is unresolved. Wan is likewise in the bibliography but not cited. Remote figure URLs in the local Markdown paper copies are not archived locally and may not be reproducible.
- **Version-control anomalies.** All paper assets, including the local bibliography style and ignore file, are currently untracked. `old/` and `weights/` are ignored, while the other four empty support directories are neither tracked nor ignored and therefore have no durable Git representation. The current repository status also reports four tracked deletions under `multi_sample_inference/param_configs/` (`all.json`, `massive_run.json`, `massive_run2.json`, and `only_concept_weaver.json`) plus unrelated untracked/generated material elsewhere; these are snapshot anomalies rather than changes made for this inventory.
