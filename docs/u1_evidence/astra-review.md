## Review

**Recommendation: COMPLETE U1 for the bounded adapter/one-layer claim. Merge verdict: OK with notes. No blocking finding.** This does not complete R3 or establish generator readiness.

### Findings — non-blocking

- **P2 — Binding helper accepts contradictory provenance.** `multi_sample_inference/parity_contracts.py:443–471` compares checkpoint-content identity but does not compare binding snapshot/inventory identities with their corresponding `record["layer"]` fields. Thus those fields can disagree while verification flags become true. Smallest fix: cross-check both fields and add rejection tests. **The recorded run is unaffected:** its fields agree, and the sole production caller performs inventory verification and concrete loading first (`u1_parity_runner.py:244–307`).

  This helper is a **trusted-runner evidence formatter, not an independent security verifier**: its existing test promotes synthetic CPU evidence using an empty directory and loader-name strings (`tests/test_parity_contracts.py:252–272`). Do not accept arbitrary externally supplied records through it as verified evidence.

- **P2 — Roadmap needs completion reconciliation.** `docs/experiment_plan.md:14,55,179` still says checkpoint-backed parity is unrun/unverified. Update current-state text, append the measured result and bounded environment exception, and complete the actual-difficulty/retrospective entry at line 34 without changing the frozen estimate or historical amendments.

### Correct / supported

- **Protocol implementation:** frozen layer, seeds, shapes, masks, disabled routing and tolerance match the runner (`docs/u1_parity_protocol.md:31–56`; `u1_parity_runner.py:23–39,179–229`). The harness checks concrete types, architecture and identical selected-layer state before executing cloned inputs (`upstream_parity.py:185–234`). Comparison implements the declared upstream-relative elementwise inequality (`parity_contracts.py:187–235`).

- **Tolerance:** `atol=1e-5`, `rtol=.016` is defensible for this BF16-compute diagnostic, not a universal generator-equivalence threshold. Both records actually report **FP32 outputs**; BF16 describes inputs/weights and autocast execution. Both cases have matching output hashes, finite values, matching shape/dtype and zero maximum absolute/relative error (`conditional.one-layer-parity.json:1`; `negative.one-layer-parity.json:1`). Acceptance therefore does not depend on exploiting the tolerance allowance.

- **Freeze chronology:** protocol-hash verification precedes model loading in code and log (`u1_parity_runner.py:68–94,244–262`; `bootes-run.log`). Supervisor supplied corroborating file times and matching copied hashes. This supports **operator-controlled pre-execution freezing**, not independent timestamp attestation.

- **Checkpoint/source/environment binding:** inventory structure contains relative paths, sizes and SHA-256 digests, including all seven diffusion shards. The runner uses full-content verification, pinned inventory/protocol/lock hashes, concrete local loaders, CUDA/PyTorch/FA2 checks and immutable record writing. Supervisor reports reconstructing execution fingerprint `c10edd…` from the retained execution patch/status. Those artifacts were reconstructed after execution; they are not contemporaneous attestations.

- **Claim remains narrow:** two deterministic synthetic contexts exercise the same selected block API—not actual positive/negative prompt encoding, CFG orchestration or diffusion (`docs/u1_parity_protocol.md:7,46`). Records and summary appropriately limit their claims. Distinct upstream/custom generator routing is implemented, with joint-prompt, explicit-negative-prompt and frame-count guards; tests exercise routing contracts, not generator execution.

### Completion disposition and validation limits

The **sam2/nvcc exception and absent full generator execution do not block bounded U1 completion**: full generation/distributed contracts belong to R3 (`docs/experiment_plan.md:17,36`; protocol lines 7,27). Preserve those blockers explicitly; do not describe the subset environment as a successful full-project sync.

I made no edits and ran no commands. Supervisor-reported validation: **42 focused tests passed**, targeted Ruff, compileall and whitespace checks passed. Broader Ruff had seven reported pre-existing findings; repository-wide lint/test success is not established. Runner preflight/failure handling lacks dedicated automated tests.