# Research salvage assessment

## Decision

**Prioritize a bounded investigation of fixed subject–action binding. Do not restart general time-varying gaze control yet.** The current evidence supports neither an effective new controller nor a definitive negative result. It does support a useful experimental platform and several concrete, falsifiable explanations for inconclusive results.

A major-CV submission is not supported by the evidence currently recovered. A publishable successor is possible, conditional on new results; an honest experimental blog is a much more defensible near-term outcome. Neither outcome should be presented as established novelty without a current literature search.

### Why action binding remains a legitimate scientific problem

A model partially succeeding does not make the task ambiguous. Define it operationally:

- Identify subjects A and B in the reference image and preserve their identities through the video.
- Request actions a and b, both feasible for either subject under the same initial conditions.
- Test assignments (A:a, B:b) and (A:b, B:a), rather than only whether both action words appear somewhere.
- Measure absolute target-action occurrence, wrong-action occurrence, joint two-subject success, and visual quality separately.
- If claiming independence, additionally change only A's requested action while holding B's target fixed; measure unintended changes to B.

A static control signal does not imply static output. For sustained actions, define required duration/coverage; for events, define the required transition and what constitutes an unwanted competing action. Do not penalize incidental natural behavior, such as blinking, as leakage unless the task explicitly forbids it.

Gaze is a sharper relational goal but introduces eye/head geometry, tiny visual structures, target identity, visibility and timing simultaneously. Attention between two subjects' tokens is information flow, not a physical gaze direction. Suppressing intersubject communication for independent actions is not a natural mechanism for producing coordinated gaze. Action binding is therefore an economical diagnostic task, not a necessary scientific prerequisite for gaze or evidence that a successful controller would transfer.

## Scope and strength of evidence

Inspected the inventory, current author-scoped Wan source, selected historical Git objects, launch/evaluation code, saved configurations, historical tables, manuscript and local paper notes. Parent checkout: `master`, `5fa48ba`; Wan: detached `f7472d354e0cb46b2853f15bf97b8b283d4780f8`, also referenced by `origin/dev` and `origin/split_sentences`. The custom commits are author work, not merely upstream Wan. `origin/ediffi` is an older implementation at `8180035`.

Some inventories are stale: the submodule is initialized and outputs/configuration grids exist. Historical output populations do not all share the current schema. No authoritative manifest linking the user's reported conclusion to particular videos, seeds, revisions and per-example scores was recovered.

Checks were source inspection, JSON/AST validation, exact binomial arithmetic, an analytic tracking example, and mocked execution of the actual public seed-initialization function. **No real generation, tensor-level attention test, distributed execution, human annotation, or visual-quality audit was performed.** Consequently source defects, plausible mechanisms and observed experimental effects are distinguished below.

External novelty verification was blocked: the literature child did not receive its advertised web tools. It stopped web research and completed a local-source-only audit through the same governed workflow. No external retrieval occurred; local bibliography dates are not retrieval dates. Source code was not modified by this assessment.

## What the recovered results actually say

One April 20 table—not proven to be the authoritative experiment described by the user—contains:

| Discrimination auditor | Baseline | Same selected Concept Weaver configuration |
|---|---:|---:|
| Soft direct | 0.65 | 0.80 |
| Soft two-alternative forced choice | 0.75 | 0.70 |

The selected configuration uses soft masks, no mask sharing and self-attention masking. Other configurations vary substantially. These are printed historical numbers, not recomputed generation results (`evaluation/logs/log_20260420_132825.log:86–119`). Selecting this row is illustrative, not a valid model-selection procedure.

The printed rates and binomial p-values are consistent with **20 object trials per group**, likely ten two-subject videos. For example, P[Binomial(20, 0.5) >= 15] = 0.0206947 matches the printed 0.75 / 0.0207. This does not establish ten distinct scenes, independent seeds, or pairing. Repeated tables in later log sections/dates are not independent replications.

Most importantly, `evaluation/main.py:151–169` tests discrimination against 0.5. It does **not** test intervention minus baseline. Its margin bootstrap resamples individual crops, ignoring shared videos and scenes. A significant method-versus-chance result alongside a nonsignificant baseline-versus-chance result is not a significant method difference. Discrete ties also complicate the assumed 0.5 null for a strictly positive margin.

Thus the defensible conclusion is:

> No robust improvement has been demonstrated by the recovered experiments and evaluation protocol.

It is not yet defensible to conclude that the intervention has no useful effect, or that training-free attention manipulation in general cannot work. Unpaired random draws are not intrinsically invalid, but require appropriate analysis and sacrifice matched-comparison efficiency. Missing seeds additionally prevent exact reproduction.

### Do not merge unrelated artifact populations

The older `multi_sample_inference/output/` contains 320 configs and 964 strictly named `video_<integer>.mp4` files across 20 image-path scene identifiers. Regional/EDiff-I variants account for 240 videos; baseline variants account for the remainder with uneven repetitions. These configs lack saved diffusion seeds and the current split-sentence schema. Some baseline variants deliberately ask both subjects to perform the same action: useful potential controls, not independent-action baselines.

The April Concept Weaver tables, dated configuration directories, the old 20-scene population and the current 197-record processed dataset cannot be silently combined into a single experiment. The latter's image-generation seed fields do not document video diffusion seeds.

## Problems that could materially change the conclusion

### 1. The evaluator measures relative description preference, not action success

For each crop, the evaluator compares the assigned description with the other subject's description and counts a positive difference (`evaluation/main.py:88–120`). A score of 2 versus 1 is a success under that sign test even if neither action convincingly happens. Averaging per-object scores does not require both actors to succeed and does not reliably reject mixtures.

Whole-description swaps can also change props or actor descriptors along with the verb. Current dog examples associate different actions with a red ball and a brown toy (`image_prompt_generation/prompts_modified.json:8–11,60–68`). A frozen crop could contain enough information to choose the intended description without showing the event. This is a demonstrable shortcut opportunity, not a measured claim that Qwen necessarily takes it.

In the 2AFC implementation, the assigned action is always Option A, and the judge must choose A or B (`evaluation/auditor.py:112–145`). Instructions to focus on movement do not calibrate option-order bias or provide a neither/both category. Soft digit/letter scores are renormalized token scores, not calibrated probabilities of action occurrence (`evaluation/engine.py:154–194`). Direct and blind-description judges share the same model and are not independent ground truth.

SAM2 crops can lose hands, props, identity or context. Missing-mask frames may become black, and crop filenames are reused across videos/auditors (`evaluation/sam2_pipeline.py:115–187`; `evaluation/main.py:92–103`). Failure rates and actual human agreement remain unmeasured. Cropping is a sensible tool, but needs validation against full-scene judgments.

**Consequence:** both a small gain and a small loss could be misleading. Validate observation before spending heavily on generation.

### 2. Prompt format and control are confounded

The shipped Concept Weaver grid uses `split_sentences`, while its `none` baseline uses `default` (`multi_sample_inference/param_configs/only_concept_weaver.json`). Split-sentence generation explicitly omits spatial and identity distinctions (`image_prompt_generation/metaprompt.md:39–41`). Splitting also changes T5 conditioning; optional single-character images change visual conditioning.

Two distinct controls are required:

1. **Fair task baseline:** vanilla Wan with clear, unambiguous assignment information, such as locatives tied to the first-frame subjects, and sensible prompt engineering.
2. **Mechanism ablation:** custom `none` under the same conditioning representation as the intervention, to isolate the effect of the hook rather than sentence encoding/image changes.

An ambiguous vanilla prompt is not a fair substitute for the first baseline. Conversely, shared split contexts alone do not establish fair information-equivalent task inputs. Report masks, image edits and preprocessing costs explicitly.

The metaprompt also reverse-engineers appearance from the actions. Some action swaps therefore change feasibility, not just binding. Choose counterfactual pairs with symmetric props and compatible initial states; opening versus closing an object may not be a valid symmetric test from one fixed reference state.

### 3. Attention-derived tracking can be confidently uninformative

The current tracker sums first-frame attention over each region and then applies a class softmax at temperature 0.1 (`wan2.1/wan/utils/simil_mask.py:150–169`; `wan2.1/wan/modules/custom_model.py:104–112`). Its partition-sum assertion establishes normalization, not correspondence accuracy.

An exact analytic example: if attention is uniform and background/A/B occupy 80%/10%/10% of reference tokens, class masses are 0.8/0.1/0.1. The soft version produces approximately **0.99818/0.00091/0.00091**. It is almost certain that everything is background despite having no correspondence information. The hard version selects background everywhere. This is not measured failure prevalence; early static-mask interpolation also moderates the effect.

There are further plausible video-specific issues: tracking uses temporally position-encoded features, the representation is bidirectional rather than causal online tracking, and interventions alter the features that drive subsequent localization. A bad mask can suppress the intended local context and then help perpetuate its own error. This hypothesis is worth testing before adding more attention strength.

The older EDiff-I implementation used area-normalized averages rather than the current summed masses. Do not project this exact current mechanism onto every historical experiment.

### 4. Current implementation contracts need repair or verification

| Finding | Evidence and scientific implication |
|---|---|
| Self-attention ignores the computed `bias` enable flag | `wan2.1/wan/modules/custom_model.py:138–145`: the intended `not bias` guard is commented out. Step/block disabling and the negative CFG branch therefore do not disable self-masking when its other settings enable it. This invalidates a presumed ablation contract, though masking both CFG branches could be an intentional separately tested design. |
| Float masks enter Boolean operations | Interpolation at `custom_model.py:112` produces floating masks; regional/EDiff-I construction uses `attn_mask & simil_mask` at `:254–272`. Current Concept Weaver bypasses this path. This is a current compatibility blocker, **not an explanation for successfully completed old runs with Boolean masks**. |
| Mask-sharing labels do not fully describe cross-attention | Each block returns its newly generated masks; cross-attention consumes those while metadata/sharing may retain first/previous masks (`custom_model.py:468–485,864–879`). Exported diagnostics need not show the actual masks controlling regional composition. |
| Diffusion seed provenance is absent | Worker omits `seed` (`multi_sample_inference/fsdp_worker.py:49–57`). Public generation draws it locally and passes a Generator downstream (`wan2.1/wan/regional_prompt/image2video.py:650–671`), bypassing the integer-only broadcast branch at `:492–505`. Mocked control flow confirms this route. Real multi-rank consistency is **unverified and at risk**, not experimentally shown broken. |
| Split singleton controls can all resolve to entity zero | Worker restarts character enumeration in each sentence (`fsdp_worker.py:27–36`). This matters for regional/EDiff-I token gates; Concept Weaver instead relies on sentence-position ordering. Test the actual sentence→entity→image→token mapping per method. |

AST/JSON validation found the main inspected Python syntactically valid but `param_configs/all.json` malformed. Syntax validity does not establish tensor/runtime correctness. The audit also identified padded-token eligibility, small-mask downsampling, sequence-padding and batch-size contracts worth testing; none is established as the historical dominant failure.

### 5. The manuscript claims more mechanism than the evidence supports

`paper/main.tex:39–59` attributes failures to global cross-attention, composition priors and augmentation without establishing these causes. Blocking one attention route does not guarantee independent actions: background tokens remain a communication path, residual states already contain mixed information, and image/text conditioning influence the whole computation.

The implementation is inspired by image methods, not automatically a faithful reproduction of them. Local Concept Weaver notes describe a trained personalized concept bank plus composition machinery; MultiTalk describes trained audio adapters/attention with localization. Borrowing their primitives without those trained representations is a research hypothesis, not an existing guarantee of controllability.

## What is worth keeping

- The ability to construct two-subject I2V scenes, explicit regions, isolated appearance contexts, multiple intervention variants and paired crops.
- Existing videos, especially both-same-action variants, as material for evaluator calibration and failure taxonomy once their provenance is reconstructed.
- The question of **subject–action assignment**, which is more precise than generic prompt–video alignment.
- Internal-mask instrumentation, provided it records the masks actually used and has independent tracking validation.
- A separate gaze-preprocessing prototype using face tracks and Gazelle (`preprocessing/who_looks_who.py`). It is useful scaffolding, not validated gaze ground truth.

The dataset and pipeline are research assets, not yet a benchmark contribution by themselves. Likewise, discovering implementation bugs is useful engineering, but a bug report about one pipeline is not a general scientific explanation of diffusion-model behavior.

## A sequential experiment plan

### Stage 0 — recover provenance and calibrate observation

Before another sweep, recover one authoritative experiment manifest if possible: parent/submodule revision, checkpoint, prompt/reference hashes, masks, actual diffusion seeds, solver/steps/CFG, rank count, per-video outputs, judge version/prompts, and per-example scores. Distinguish image seeds from diffusion seeds. Do not delete or regenerate old artifacts in place.

Use roughly 80–120 existing clips as an initial annotation exercise, balanced across both-correct, swapped, both-A, both-B, mixed and neither. Obtain blinded full-scene human labels for each actor/action, quality and uncertainty; use multiple raters and adjudicate disagreements. Reserve held-out examples for checking any revised automatic score.

Add cheap diagnostic controls:

- Frozen-first-frame clips preserving identities and props.
- Correct/wrong verbs with actor/object descriptions held constant where feasible.
- AB and BA option order, identical descriptions, and neither/both cases.
- Full frame versus SAM2 crop versus manually checked regions on the same video.
- Frame reversal/shuffle only for actions where temporal order really matters.

A frozen clip scoring well is evidence against that score as a motion detector, not against video generation. Until validated, use humans as the primary outcome and the VLM for triage. Save crop provenance and tracking/visibility failures rather than silently dropping them.

### Stage 1 — cheap contract and competence tests

Before expensive inference:

- Compare upstream and custom `none` using identical tensors, contexts and settings; establish a justified numerical tolerance rather than assuming bitwise equality across kernels.
- Test bias/step/block toggles in both CFG branches, mask-sharing identity, mask dtype/partition and token/entity mappings with small synthetic inputs.
- Persist explicit seeds and compare initial-latent hashes across methods and ranks.
- Quantify initial mask survival after resizing; test uniform-attention and unequal-region tracking cases.

Then verify the selected actions singly and with both subjects performing the same action. If Wan cannot render an action alone, failure on the two-action task is not evidence of a binding-specific bottleneck. Keep distinct strata rather than discarding failures post hoc.

### Stage 2 — a small, matched falsification pilot

For example: **eight stationary scenes × two explicit seeds × both AB/BA assignments** gives 32 outputs per condition. Four conditions give 128 videos, before any additional single-action controls. This is a diagnostic budget, not a powered significance study.

Start with:

1. Fair vanilla prompting with explicit assignment.
2. Shared-representation custom `none`.
3. Regional text control using fixed, disjoint subject regions; no image isolation or self-attention blocking.
4. The same controller with attention-derived moving masks.

Hold fixed everything not intentionally varied. Report encoding/conditioning differences between the fair baseline and mechanism ablation. Inspect departures from fixed regions afterward and count them; do not select only videos that respect the mask. Where promising, add image-context isolation and self-attention blocking **one at a time**, not as another undifferentiated grid.

There is no available oracle for the future trajectory of a not-yet-generated video. Fixed regions make a stationary diagnostic executable. Masks from a reference video constitute a task-relaxed upper bound; masks from an unmodified generation can become stale when intervention changes motion. A fixed-region rescue implicates localization/control interaction, not uniquely the tracker as the sole cause.

Primary outcome: human-labeled joint two-target success, with separate target/nontarget occurrence and quality. Use additional one-subject-target changes to test spillover if independence is claimed.

**Interpretation rules:**

- Isolated-action failure → revise/stratify the action set; do not call all failure binding.
- Prompt representation alone wins → attribute that gain to prompting, not attention routing.
- Fixed-region routing wins but dynamic routing does not → investigate localization and its feedback with control.
- Competent single actions plus functioning fixed-region control with no convincing human gain → deprioritize tracker refinement and further strength sweeps.
- Automatic-score gain without human gain → evaluator failure, not method success.

### Stage 3 — confirmation only after a diagnostic signal

Freeze the controller and metric before a held-out, adequately powered comparison. Select a smallest worthwhile effect in advance—for example, +10 percentage points in joint success could be a project decision, not a universal conference threshold. Estimate required independent scenes from pilot variability and seed/scene clustering; increasing repeats on a few scenes does not establish generalization.

Estimate method-minus-baseline differences with paired scene-level analysis or an appropriately specified hierarchical model/bootstrap. Preserve both crops, seeds and assignment variants within their scene clusters. Report uncertainty, quality, runtime and failure denominators. A confidence interval crossing zero remains inconclusive; an upper bound below the worthwhile-gain threshold can rule out that gain on the tested domain. Equivalence requires a predeclared equivalence margin, not merely p > 0.05.

Include stronger prompt templates and, where relevant, compute-matched best-of-N selection using an independently validated selector. Confirm on additional scene/action families and at least one other suitable contemporary backbone before claiming a general failure mode or general controller. Do not tune on the confirmation set or keep expanding a sweep until something is significant.

## Successor directions and publication decisions

### A. Uncertainty-aware localized action control — closest method path

If fixed-region control works but learned localization fails, investigate confidence-aware routing, region-area priors, temporal-position-neutral localization, less feedback-sensitive mask sources, and fallback to ordinary generation where assignment is uncertain. The contribution would need measured localization/control interaction and better held-out video outcomes, not merely a nicer mask visualization or another combination of known hooks. Novelty remains unverified.

### B. A counterfactual compositional-video benchmark — possible diagnostic path

If a validated human-aligned protocol finds a reproducible binding failure across strong models and controllers, a benchmark/analysis paper may be more credible than a weak method paper. It needs meaningful breadth, validated labels, matched feasible action swaps, strong baselines, uncertainty analysis and a generalizable insight. A small synthetic dataset plus a nonsignificant Wan-only result is not enough.

### C. Observable, geometry-based gaze control — a separate staged project

The preprocessing prototype thresholds predicted targets inside face boxes with in/out threshold 0.9 and maps low-confidence cases to False (`preprocessing/who_looks_who.py:38–90`). That conflates uncertainty with not-looking and does not validate mutual eye contact. Its matrix convention is [frame, observed, observer], whereas the old root controller uses [observer, observed, frame]; this is an interface to document, not a proven connected bug.

Begin with large visible faces, a static camera and a static target choice; then one scheduled switch. Distinguish head turn, eye direction, looking at a partner and mutual eye contact. Include offscreen/occluded/unobservable states. Respect first-frame I2V conditioning, feasible motion transitions, the one-target-per-observer constraint and temporal resolution; arbitrary frame-by-frame Boolean graphs need not be physically realizable.

A geometry-driven, face-local latent-guidance or trajectory-control method using pretrained estimators could remain training-free, but it moves beyond attention-only manipulation and needs separate feasibility tests. Estimator reward hacking, tiny-eye resolution, identity preservation and compute are substantial risks. Do not present such a proposal as an implemented solution or promised novelty.

### D. Blog and demos — worthwhile without an efficacy claim

A reproducible account of where entity/action binding breaks, how the evaluation can be fooled, and which interventions fail is useful. Show complete seed grids, matched baselines, failures and mask diagnostics alongside selected demos. Do not showcase only winners as proof of control.

For practical noninteracting demos, separately animating subjects and compositing tracked layers is another option. It relaxes joint full-scene generation and has occlusion/shadow/interaction limitations; label it honestly. It may be a better engineering solution than preserving the original attention-only constraint at any cost.

**Overall stopping rule:** time-box calibration and the matched pilot before committing to a paper. If no reproducible human-validated effect or broadly useful diagnostic result emerges, stop the attention-heuristic search and publish the honest blog. If a clear bottleneck and a meaningful rescue emerge, pursue the corresponding paper path rather than reviving every abandoned idea at once.

## Supporting audits and residual unknowns

The managed workflow completed four read-only child reports. Read the challenge report alongside the methods/literature reports: it qualifies overly strong distributed-runtime wording and the naive use of oracle masks.

Artifact directory:

`/home/gzappavi/.pi/agent/sessions/--home-gzappavi-Documents-wan_experiments--/subagent-artifacts/outputs/605c025b-bc21-45ad-8b07-2a063390937b/audit/`

Actual output references: `methods.md`, `evaluation.md`, `literature.md`, `challenge.md` under that directory. Workflow receipt:

`/tmp/pi-subagents-uid-677450/async-subagent-runs/605c025b-bc21-45ad-8b07-2a063390937b/workflow-receipt.json`

Primary-source identifiers recorded locally, not fetched in this audit: Concept Weaver (https://arxiv.org/abs/2404.03913), MultiTalk (https://arxiv.org/abs/2505.22647), Wan (https://arxiv.org/abs/2503.20314), LaMa (https://arxiv.org/abs/2109.07161), Video-Bench (https://arxiv.org/abs/2504.04907). Video-Bench and VBench are not interchangeable names.

Before committing to a conference plan, the remaining essentials are the authoritative historical run mapping, human-labelled outcome data, actual runtime/parity checks, an up-to-date primary-source novelty search, and an agreed compute budget/interpretation of training-free versus attention-only constraints.
