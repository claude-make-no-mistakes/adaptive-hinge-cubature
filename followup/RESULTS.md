# Follow-up algorithm results

This follow-up measured implementation changes motivated by the first four studies. The cubature and spatial inputs are new. The sparse experiments use new numerical sequences on the same matrix patterns. The binary delta cohort consists of new release projects. All timing comparisons were paired on one Windows workstation. Values in brackets are 95% intervals resampling the independent input units. [Protocol](PROTOCOL.md) · [Reproduction and raw journals](README.md) · [Original report](../paper/REPORT.md)

![Paired speed and patch size results for the four follow-up studies](figures/results.png)

## Adaptive cubature

The active-set implementation was **5.96× faster [5.18–8.50]** than the previous cached fit on the quadratic-background kink conditions that both verified, including the fit, correction, and residual integration. These 124 shared conditions represent 62 independently drawn integrands at two tolerances. Both implementations verified **124/128** conditions. Directional Genz–Malik verified **31/128**, and Cuhre verified **32/128**. On the shared Cuhre successes, the new method was **3.52× faster [3.00–4.21]** across 16 independent integrands. That conditional ratio describes the inputs where both methods returned a verified answer.

Each finite binary64 coefficient has an exact integer numerator over a power of two. Clearing a common denominator leaves integer corner powers. After factoring out the product of the scaled slopes, one denominator scale remains. Reflecting the box simplifies the shorter tail by the identity $z_+=z+(-z)_+$. The implementation divides once at the end. The integer kernel matched the archived fraction calculation on 64 timed coefficients and 99 edge and mixed-exponent checks. The arithmetic replacement alone preserved the estimate, error, status, and work counters in all **1,920** matched integration attempts. [Cube slicing and truncated-power identities](https://arxiv.org/abs/math/0607715)

| Dimension | Independent hinges | Exact-kernel speed over Fraction |
| --- | ---: | ---: |
| 2 | 16 | 7.38× [7.17–7.92] |
| 4 | 16 | 14.37× [13.85–16.52] |
| 6 | 16 | 29.13× [27.99–30.47] |
| 8 | 16 | 38.90× [35.56–42.55] |

With a given active set, the projected hinge fit is a linear least-squares problem for its slopes and intercept. The new path updates the signs and accepts a fit only at the training and held-out roundoff floors. Its nonlinear fallback and the actual callback still supply the residual integral. All 124 successful quadratic conditions used the linear fit. The separately drawn mixed-background conditions verified 32/128 in both modes, with a **0.95× [0.93–0.99]** conditional speed ratio. Narrow peaks caused 57 false convergences in 64 conditions for both implementations. The kernel arithmetic is exact before final rounding for representable results. The adaptive stopping rule remains an estimate of the integration error. [Cubature raw journal](runs/cubature.jsonl)

## Dynamic R*-trees

At the leaf-parent selection step, a candidate's overlap score adds the intersection enlargement with each sibling. On the guarded finite Cartesian rectangles each term is nonnegative. Once the sum exceeds the incumbent, later terms cannot restore the candidate. At equality, the existing area-enlargement and area ties decide whether the candidate can still win. The implementation keeps Boost's candidate order, final tie order, and capacities. [Original R*-tree paper](https://people.csail.mit.edu/tdanford/6830papers/beckmann-r-star-tree.pdf) · [Pinned Boost parameters](https://github.com/boostorg/geometry/blob/boost-1.92.0/include/boost/geometry/index/detail/rtree/rstar/choose_next_node.hpp)

At 30% reinsertion, the pruned path skipped **66.3%** of the candidate-sibling pairs evaluated by the prefix-only path. It reached **1.14× [1.10–1.15]** prefix-only throughput and **1.22× [1.20–1.25]** stock throughput after aggregating the three profiles within each county. With reinsertion disabled it skipped **63.5%**, reaching **1.08× [1.05–1.09]** over prefix-only and **1.18× [1.17–1.21]** over stock.

| Capacity 32 profile | Pruned plus prefix vs stock, no reinsertion | Pruned plus prefix vs stock, 30% reinsertion | Increment over prefix at 30% |
| --- | ---: | ---: | ---: |
| Moving hotspot | 1.07× [1.06–1.11] | 1.08× [1.06–1.12] | 1.06× [1.05–1.08] |
| Write-heavy | 1.17× [1.17–1.21] | 1.22× [1.20–1.25] | 1.13× [1.10–1.15] |
| Insert-only | 1.39× [1.36–1.42] | 1.34× [1.32–1.37] | 1.19× [1.16–1.22] |

All **152** pruned/stock trace pairs matched topology checkpoints and query signatures. Each mode checked 316,717 primary queries against an independent scan. Four larger traces compared signatures for 83,690 queries and scanned 2,265. Those runs suggest comparable gains at capacity 64 and have four independent counties. The timing scope covers replay after the initial build. [Input receipts](inputs-spatial.json) · [Spatial raw journal](runs/spatial.jsonl)

## Sparse factorization

Solving again with changing values offers two kinds of reuse. A fixed CSC pattern can reuse symbolic CHOLMOD state and repeat numeric factorization. A fixed SPD base matrix with shifts on a few fixed diagonal sites can also reuse the original numeric factor. For the latter, let $E$ select those sites, $Y=A^{-1}E$, $G=E^TY$, and $R=\sqrt{\operatorname{diag}(d)}$. With $z=A^{-1}b$, the corrected solution is $z-YR(I+RGR)^{-1}RE^Tz$. This needs a base solve, a small SPD factorization, and an update using the stored responses. The solver checks normwise backward error and falls back to a fresh shifted factor if necessary. The stored response matrix costs eight dense columns in this experiment. [Sparse factorization modifications](https://people.clas.ufl.edu/hager/files/multirank.pdf)

| SPD sequence and candidate | 1 solve | 4 solves | 12 solves |
| --- | ---: | ---: | ---: |
| Full-diagonal changes, symbolic reuse | 0.99× [0.88–1.08] | 1.07× [1.04–1.09] | 1.09× [1.07–1.20] |
| Eight diagonal sites, symbolic reuse | 0.91× [0.86–0.97] | 1.05× [1.01–1.09] | 1.11× [1.07–1.14] |
| Eight diagonal sites, cached low-rank solves | 0.76× [0.66–0.87] | 2.97× [2.54–3.29] | **7.11× [6.27–7.95]** |

The ratios use fresh numerical factorization with the supplied stock AMD permutation. They include preparing the persistent state, assembling the required numeric values, and checking the result. The shared initial ordering charge is available separately in the machine-readable summary. All 19 SPD matrix patterns and all 3,420 timed SPD solves passed the $10^{-11}$ normwise backward-error check. The measured benefit covers multiple queries with a fixed matrix and eight nonnegative shifts at fixed sites. The fixed-pattern refactor path handles changes across the numeric entries. [Sparse raw journal](runs/sparse.jsonl)

The eight-site method took the fresh-factor fallback for all 36 measured requests on each of LF10000, ex15, and bloweybq. The other 16 matrix groups took the fast path throughout their measured sequences. The 7.11× ratio includes all 19 groups and the fallback costs. Charging both pipelines for the shared AMD order gives a 6.93× median at 12 requests.

The nonsymmetric comparison supplied the same stock COLAMD permutation to both UMFPACK pipelines with the unsymmetric strategy. Reusing symbolic state reached **1.06× [1.05–1.08]** over eight solves on 24 groups, including setup. All **1,152** timed LU solves passed the backward-error threshold. Fresh and persistent LU used the same pivoting, scaling, and refinement settings. The ratios isolate symbolic reuse at a supplied ordering. [UMFPACK source and documentation](https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/v7.14.1/UMFPACK) · [Sparse results](summary.json)

The two LU implementations reported identical forward and backward errors in all 576 matched requests. Three matrix groups showed large forward errors despite small backward residuals. The worst relative forward error was $7.71\times10^{17}$ on shyy41. Replaying the most severe request from each of those groups with the optional minimum pivot ratio of $10^{-12}$ rejected all three. Their UMFPACK U-diagonal ratios were $2.58\times10^{-24}$ for shyy41, $2.67\times10^{-16}$ for k3plates, and $5.39\times10^{-16}$ for iprob. This was a retrospective diagnostic and was outside the timed comparison. UMFPACK documents the ratio as a rough indicator. [UMFPACK numeric diagnostics](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/v7.14.1/UMFPACK/Include/umfpack.h)

## Binary delta

### With BSDIFF40 compatibility

Increasing the difference field by $s$ and decreasing the seek by $s$ preserves the next source position. The target length of the edited instruction also stays fixed. Only the $|s|$ bytes that change stream need to be transformed. The local stream rewrite emitted **identical patch bytes in all 96 paired timings** versus the whole-stream reconstruction. The paired postprocessing speed was 1.05× [0.99–1.08]. Including the seed encoder, the ratio was 1.02× [1.01–1.05]. Compression and decoding still account for work after the splice.

The hybrid tried local windows from two compatible encoders with three switching penalties and chose by fully compressed patch size. It saved **12,644 bytes of 5,588,019 (0.226%)** beyond taking the smaller of classic bsdiff and HDiffPatch BSD on each file. It improved 18/32 file pairs, including 12 where the winning parse switched between encoders. The total encoding cost was 1.50× the time to produce both inputs, by median paired project ratio. All 480 timed patches from the five methods reconstructed correctly with the in-memory decoder and stock bspatch. The delta study resumed at its next verified checkpoint after an interrupted process. [Compatible delta journal](runs/delta.jsonl) · [BSDIFF paper](https://www.daemonology.net/papers/bsdiff.pdf)

### With a ZIP-aware decoder

For a whole wheel, the archive path expands reproducible changed DEFLATE members before running the same HDiffPatch BSD encoder. It carries the recompression recipe, runtime version, and SHA-256 digests in a separate wrapper. It preserves archive headers and other bytes as stored. The inner patch was verified with stock bspatch. The outer decoder restored exactly the original published target bytes. This follows established archive-aware patching methods. [Google Archive Patcher](https://github.com/google/archive-patcher) · [zlib manual](https://zlib.net/manual.html)

Across all **16** whole-wheel pairs, delivered patch bytes fell from **5,148,312 to 888,127**, a reduction of **82.7%** including the full wrapper. Fifteen projects selected the wrapped patch and rpds-py selected the raw seed. All **48** timed outputs reconstructed exactly, and every eligible wrapped attempt verified. The raw-to-delivered median size ratio was **2.88× [1.81–4.17]** by project. The aggregate reduction was **58.8%** with lxml excluded.

The encoder ran both the raw encode and the eligible transformed encode so it could keep the smaller result. The total paired encoding time was **2.93×** the raw encoder. Warm in-memory Python decoding took **2.26×** as long. Wrapped patches require the archive decoder and the recorded zlib 1.3.2 runtime. The 82.7% result applies to whole wheels with that decoder. [Archive raw journal](runs/archive.jsonl) · [Archive summary](archive-summary.json)
