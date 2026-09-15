# K1 deletion and preservation manifest

## Candidate identity

- Base revision: `962c2b13a4db784ec0262ca64ff014e136c9dd7f`
- Candidate identity before commit: that base revision plus the K1 working-tree diff
- Date recorded: 2026-09-14
- Scope: one tracked deletion, no tracked moves, and no local/ignored mutation

## Sole deletion

| Path | Base Git blob | SHA-256 | Rationale | Recovery |
|---|---|---|---|---|
| `main.py` | `ace69f63f1bd494e84360742db334671bb966861` | `33dff1ce146d3b1f9809b5c3ec97ea2531c9f36fa970efd0f71f0c5f7d2633a9` | Six-line greeting-only initialization scaffold. It has no package script declaration, tracked caller, experiment behavior, configuration contract or unique research evidence. | `git show 962c2b13a4db784ec0262ca64ff014e136c9dd7f:main.py` (or recover its blob from Git history). |

This record applies only to the root path. `evaluation/main.py` and `video_gallery/main.py` are distinct retained historical/compatibility entry points. Comments or historical documents that mention root `main.py` remain intentional provenance, not dangling executable callers.

## Explicit no-change record

- Tracked path moves: **0**.
- Other tracked deletions: **0**.
- Dependency declaration removals/additions/updates: **0**.
- `pyproject.toml` changes: **0**.
- `uv.lock` changes: **0**.
- `.gitmodules` or gitlink changes: **0**.
- Submodule revisions changed: **0**.
- Ignored/local files moved: **0**.
- Ignored/local files deleted: **0**.
- Historical outputs rewritten: **0**.
- P0/U1 evidence files changed: **0**.

## Retained historical path conditions

Legacy inference dispatch/config/launchers, source assets/producers, baseline notebooks, examples, diagnostics, evaluation/gallery code, research notes, prompts, paper material and site-specific runtime helpers remain at their current paths as noncanonical historical/compatibility material. Preservation was selected because external consumers, output attribution, permissions, or recovery are unresolved.

A later move or deletion requires, as applicable:

1. an approved production M1 source and operator/scheduler cutover;
2. reconciliation of evaluation, gallery, visualization and legacy output consumers;
3. a private content-level inventory and verified recovery for affected ignored/LFS material;
4. migration of imports, workspace/submodule/container links and dependency declarations;
5. clean recursive-checkout validation of the resulting candidate.

Retention in place is not proof that the path works, remains scientifically valid, is publishable, or is recoverable elsewhere.
