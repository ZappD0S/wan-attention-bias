# Project workflow

- `docs/experiment_plan.md` is the authoritative roadmap and task register.
- When asked for the “next task,” select the highest-priority **PLANNED** task whose dependencies are completed. Report blockers separately rather than treating them as conclusions.
- When adding a task, record a pre-work difficulty estimate, brief rationale, and recommended OpenAI model/role in `docs/experiment_plan.md`; freeze the estimate and recommendation once work begins. When marking a task **COMPLETED**, preserve them, add the actual difficulty plus a brief retrospective rationale, and note any model substitution.
- Before beginning a task or assigned role, compare the current model with the plan's recommendation for that role. If they differ, stop and ask the user whether to proceed or switch models; do not begin automatically.
- Inspect `git status` and the diff of modified planning documents before answering roadmap or status questions.
- When a task reaches **COMPLETED**, run its required validation, reconcile the roadmap, and create a task-scoped commit unless the user explicitly says not to. Never stage unrelated or pre-existing changes; if clean separation is uncertain, ask before committing. Do not push unless explicitly requested.
- Use the project environment exclusively through `uv`: treat `pyproject.toml` and `uv.lock` as authoritative, use `uv sync` / `uv run`, and never use system Python, system `pip`, or ad hoc virtual environments for project commands.
- Treat `docs/repo_inventory/` as historical context, not authoritative state; verify its claims against the current tree.
- Do not claim GPU readiness, runtime parity, or scientific results without confirming the required environment, compatible checkpoint, and recorded evidence.
