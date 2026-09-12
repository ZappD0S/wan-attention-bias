# Pre-repair checkpoint

This checkpoint preserves the existing research before implementation repairs. No experiment results were regenerated or replaced. Commits are local; nothing was pushed.

## Repository boundaries

The main repository's checkpoint commit contains the previously untracked inventory, assessment, paper drafts and local paper excerpts, prompt notes/backups, evaluation scripts, mask-visualization scripts, and tokenizer scratch script. Its parent is `5fa48ba`. The subsequent repair branch is `research/action-binding-repair`.

Existing submodules remain pinned at their original revisions at checkpoint time:

- `wan2.1`: `f7472d354e0cb46b2853f15bf97b8b283d4780f8` (custom fork, previously detached at `origin/dev`). Repairs will use its own `research/action-binding-repair` branch, not an upstream branch.
- `lama`: `469acc7358a1c6828b647b4ee20c93474a2f36b4`.

Three previously untracked directories are **independent Git repositories**, not new submodules. Their source work was checkpointed in their own histories and their directories are now ignored by the parent; cloning the parent does not retrieve them:

| Directory | Local checkpoint | Remote / recovery |
| --- | --- | --- |
| `apptainer_images/` | `e12d2b3bb1e8233fa466a7190f81bc8da529419b` | Existing `git@gitlab.inria.fr:gzappavi/apptainer_images.git`; preserves the user's staged LFS configuration and three SIF pointers. LFS objects remain local; no upload performed. |
| `preprocessing/` | `392b0a224507ecbd467d2ba950612852608bee9f` | Existing `git@gitlab.inria.fr:gzappavi/preprocessing.git`; source was already committed. Added an ignore for generated `repomix-output.xml`. |
| `soft_mask_test/` | `5973bf59d05c32770218238536c05921d68210e9` | Initial local commit of existing source/config/lockfile. No remote configured; preserve this directory or create a separate backup before relocating the project. |

## Deliberately excluded files

- Existing `.env`, model weights, generated videos, run outputs/logs, caches and container files remain excluded by existing rules. Secrets were not added to Git.
- `mask_visualization/output.npz` (~477 MB) and `mask_visualization/video_latents_0.29_noise.pt` (~8 MB) remain untouched on disk and are newly ignored.
- `preprocessing/repomix-output.xml` remains untouched on disk and ignored in its owning repository.
- Test virtual environments and pytest caches are ignored.

This is a **source checkpoint, not a complete dataset/checkpoint/container backup**. Historical media and unpushed companion commits still require independent backup for off-machine recovery. Broad legacy ignore rules such as `**/*output*/` are retained; future experiment manifests must live outside those patterns or be explicitly tracked.
