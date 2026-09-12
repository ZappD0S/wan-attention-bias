# wan2.1 mask utilities — static inventory

## Scope and attribution

- Repository inspected: `wan2.1`, checked out detached at `f7472d354e0cb46b2853f15bf97b8b283d4780f`.
- In-scope files only:
  - `wan/utils/simil_mask.py` (312 lines in the checked-out snapshot)
  - `wan/utils/smooth_mask.py` (62 lines)
  - `wan/utils/subsequence.py` (64 lines)
- Git history was filtered to author **`gzappavi <gianluca.zappavigna@inria.fr>`** (author identity, not merely committer identity). Eleven matching commits touch this subset. The latest subset change is `38b21b3854b11412dab506a2f0935a53f0b39c8c` (`38b21b3`, 2026-04-14); the checked-out commit is unchanged for these three files after that point.
- No other source or configuration file was analyzed. The review is static and cursory.

## Utility behavior and data flow

### `simil_mask.py`

`compute_attn_weights` (lines 12–45) accepts query/key tensors shaped as batch, token, head, embedding. It computes scaled dot-product attention in query chunks, applies softmax over key tokens, and averages heads, producing `(N, L, S)`. The chunk loop reduces peak score-tensor allocation but still materializes the full returned attention matrix.

The mask utilities treat `face_masks` as boolean first-frame spatial masks:

1. `compute_hard_simil_masks` (lines 86–142) reshapes keys into time/spatial form, selects frame-zero keys, optionally applies temporal-neutralized RoPE, computes attention from all query tokens to those keys, prepends a background complement mask, and projects attention into per-class scores with `einsum`.
2. Optional spatial Gaussian smoothing operates on `(B, C, T, H, W)` (lines 70–83), spatially over the final two dimensions, then restores flattened video-token order.
3. `discretize_masks` (lines 58–67) selects the highest-scoring class per token. Only non-background winners strictly above `threshold` remain entity classes; all others become background. The result is one-hot and checked by `check_mece`.
4. `compute_soft_simil_masks` (lines 145–169) uses the same frame-zero attention projection, then applies a class-axis softmax and returns the background-inclusive soft scores. Its `temperature` is applied after mask projection, not to the attention logits.
5. `compute_soft_simil_masks_iterative` (lines 230–312), under `no_grad`, reshapes both query and key sequences by frame. For each later frame it attends jointly to frame-zero keys and previous-frame keys, propagates the frame-zero and current masks, normalizes over classes, and concatenates frame results into `(N, P, T*H*W)`.

The code assumes `grid_sizes` has exactly the nested form `[[T, H, W]]`. The non-iterative paths also encode a batch-size-one key layout in their rearrangement pattern (lines 99–101 and 150–152), while the iterative path expects key batch size `N` (line 247).

### `smooth_mask.py`

`generate_soft_mask` validates a one-dimensional boolean PyTorch tensor (lines 6–16), detects contiguous true runs with padded integer differences (lines 18–29), fills each run with one, and applies a transition ramp before each run and after each run. It supports a linear ramp or a cosine “smooth” ramp (lines 31–43), clips ramps at sequence boundaries (lines 46–59), preserves the input device, and returns float32. A duration below one or of the wrong type returns the hard float mask rather than raising.

### `subsequence.py`

`_find_subsequence_index` (lines 5–25) finds the first exact window match using NumPy’s `sliding_window_view`; empty subsequences return index zero and overlong subsequences return `-1`. `get_nested_subsequence_mask` (lines 28–64) converts the main sequence and each nested subsequence to arrays, rejects an empty subsequence-list, greedily searches each next subsequence only inside the previously selected window, accumulates the original offset, and returns a boolean mask covering the final subsequence.

## gzappavi commit evolution

- **`4ec6025` (2025-10-07, `progress`)** introduced both `simil_mask.py` and `smooth_mask.py`. The original similarity implementation accumulated head-summed attention in chunks, averaged heads, and selected winner-take-all masks; the smoothing utility was already substantially the current run/ramp implementation.
- **`60c6729` (2025-10-21, `progress`)** clamped weighted-mask denominators and switched allocation to `query.new_zeros`.
- **`209e9d9` (2025-10-25, `progress`)** introduced `subsequence.py` with first-match sliding-window search and greedy nested-mask construction.
- **`e702010` (2025-10-28, `progress`)** factored attention calculation into an optional-bias helper.
- **`2d19180` (2025-10-30, `wip`)** replaced that helper usage with the chunked `compute_attn_weights` API and explicit head averaging.
- **`9aa8930` (2025-11-05, `wip`)** reduced the attention accumulator from zero-initialized to `new_empty` and wrote each chunk directly; head reduction changed from sum/division to mean.
- **`77c3e0f` (2026-01-21)** added an output-shape TODO to the then-single similarity API.
- **`c6d5608` (2026-02-24)** added optional RoPE handling, frame-zero key selection, spatial-bias experimentation, and broader grid-size arguments.
- **`390de16` (2026-03-23, `several fixes`)** removed the spatial-bias path, split the API into hard and soft functions, added temperature handling to attention (including a zero-temperature one-hot mode), and changed background ordering to background-first.
- **`2a7238b` (2026-04-07, `progress`)** added the anchored iterative soft propagation implementation while retaining its predecessor as a large commented block.
- **`38b21b3` (2026-04-14, `progress`)** added MECE checking, thresholded hard discretization, optional Gaussian smoothing, and class-temperature scaling for soft masks. This is the current content of the assigned files.

The two utility files other than `simil_mask.py` have no later matching author commits in the path history.

## Static bug/gap review

### Higher-priority correctness and contract gaps

- **Empty-query edge case in attention:** `compute_attn_weights` declares `logits` only inside the chunk loop (lines 26–39), but the zero-temperature path later uses `logits.dtype` (lines 41–43). A direct call with zero query tokens and `T=0.0` therefore reaches an unbound local. There is no explicit validation for empty sequences or positive `chunk_size`.
- **Unvalidated soft temperature:** `compute_soft_simil_masks` divides by `temperature` at line 162. Zero produces division-by-zero behavior and negative values silently invert the class logits; no range check documents whether either is supported.
- **Shape and dtype assumptions are implicit:** `grid_sizes` is destructured without validation (lines 97, 148, and 234), non-iterative key layouts require a literal batch dimension of one (lines 99–101 and 150–152), and `~face_masks` requires a boolean/integer-compatible mask (lines 130 and 156). Malformed inputs fail later with shape or operator errors rather than at the API boundary.
- **Subsequence public input mismatch:** `get_nested_subsequence_mask` claims to accept NumPy arrays, but `if not nested_subsequences` (lines 38–39) raises NumPy’s ambiguous-truth-value error when `nested_subsequences` itself is a multi-element ndarray. The implementation also assumes one-dimensional arrays: `sliding_window_view` and `np.all(..., axis=1)` (lines 18–21) are not guarded against multidimensional main or subarrays.
- **Smoothing validation asymmetry:** an invalid `transition_type` is rejected only when `transition_duration >= 1`; the early return at lines 15–16 bypasses that validation. Wrong-type or nonpositive durations silently change the requested operation to a hard mask, which is easy for callers to overlook.

### Behavior and maintenance risks

- **Attention memory remains quadratic:** chunking limits the temporary score/softmax chunk, but `avg_attn_weights` is still `(N,L,S)` (line 17). The iterative path additionally retains `(N*(T-1), L, 2L)` attention (lines 269–275), so the VRAM risk is reduced rather than eliminated.
- **Tie and threshold policy is implicit:** `torch.max` in `discretize_masks` (line 60) resolves equal scores by its first index, and the strict `>` at line 61 maps a score exactly equal to the threshold to background. Neither policy is described in a docstring or parameter documentation.
- **Assertions are runtime invariants, not enforced API checks:** `check_mece` relies on `assert` (lines 48–55 and 140). Python optimization can remove these checks. The helper also requires values to be close to exactly zero or one, so any future change that returns probabilistic masks at that point will fail abruptly.
- **Device/shape handling around optional RoPE is not explicit:** the metadata tensor constructed at line 120 has no device specification, and `freqs` shape/placement requirements are undocumented. This is a portability risk for GPU or unusual dtype callers even if the downstream helper currently tolerates the metadata.
- **Iterative implementation has duplicated/dead history:** lines 172–227 preserve an obsolete commented implementation beside the active algorithm. This increases maintenance surface and leaves competing normalization/return-shape descriptions in the same file.
- **Iterative output semantics deserve tests:** frame-zero masks are normalized over the class axis (line 282), later propagated masks are likewise normalized over classes (line 302), and the concatenated result retains background (line 310). That may be intended, but there is no docstring or executable check documenting the `(N,P,T*H*W)` contract or the fact that this function is gradient-free.
- **Nested subsequence search is greedy:** each search starts from the first match in the current narrowed view (lines 47–57). If an earlier match cannot contain a later nested subsequence while a later match could, the function raises instead of backtracking. This is a semantic limitation unless “first nested chain” is the explicit contract.
- **Performance is input-dependent:** subsequence matching materializes a sliding-window comparison whose working set grows with sequence length and candidate length (lines 17–22); smoothing uses Python loops and `.item()` per run (lines 46–59), which can add synchronization overhead for accelerator tensors.
- **Documentation/test coverage is sparse in these files:** only the smoothing and subsequence public functions have basic docstrings; tensor shapes, mask overlap policy, background convention, temperature ranges, and failure modes are mostly comments or inferred from `einops` patterns.

## Reproducibility and maintenance risks

The checked-out state is a detached commit and the assigned files last changed before the checkout’s later history. Reproduction therefore requires preserving the exact commit, Python/NumPy/PyTorch/torchvision/einops versions, tensor dtypes/devices, and `grid_sizes` conventions; none are declared in these files. Floating-point softmax, Gaussian filtering, equality/tolerance checks, and argmax ties can vary across hardware or dtype. No random state is used directly by these utilities, but attention results depend on upstream tensors and numerical kernels.

The APIs are positional and largely untyped (apart from the smoothing mask annotation). The substantial rename/split from the original `compute_simil_masks` API, the background-order change, and the added batch/layout variants make undocumented call contracts a primary upgrade risk. The commented legacy implementation and sparse validation further increase the chance that shape or normalization changes will regress silently.

## Validation performed

- Parsed all three assigned files with Python’s AST parser: all three parsed successfully.
- Ran `git diff --check` scoped to the three assigned paths: no whitespace errors reported.
- Confirmed with Git that these paths are unchanged between `38b21b3` and the checked-out `f7472d3`.
- Reviewed author-filtered path history and file-specific diffs for all matching commits.
- Did **not** run imports, unit tests, model inference, builds, web applications, or numerical benchmarks.

## Limitations

This report intentionally excludes call sites, tests, dependency implementations, configuration, generated artifacts, and all files outside the three assigned paths. Findings are source-level observations only; the contract gaps and risks above should be confirmed with focused shape/device/edge-case tests before changing behavior. No claim is made about runtime performance or model-quality impact without executing the surrounding pipeline.
