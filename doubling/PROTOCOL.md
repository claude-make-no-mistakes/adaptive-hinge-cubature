# Incremental optimization experiments

The control is the fastest verified route in the follow-up for each workload. The speed target is 2× additional throughput. The wheel target is at most 444,063 bytes across the published 16 pairs, including framing, hashes, recipe, and compressed delta. The existing wheel result is 888,127 bytes.

| Workload | Control | Candidate and acceptance |
| --- | --- | --- |
| Adaptive cubature | Cached active-set pilot in blind-linear mode | Recover a planar kink from one-dimensional restrictions. Validate on separate points, integrate the callback residual, and retain the previous pilot as a budgeted fallback. Compare equal truth-based accuracy and report false convergence across all eight families. |
| R*-tree | Prefix/suffix splits with overlap pruning | Specialize the bounded 2D rectangle hot path. Compare the same operation traces at both reinsertion settings. Require identical topology checkpoints, matching query signatures, and independent query scan checks. |
| Sparse SPD | Checked Woodbury solves with fixed sites | Measure direct sparse factor updates, including checked solves and fallback, on the same 19 patterns and all 12 chronological requests. Record the common initial ordering separately. Require finite results and the same independent backward-error threshold. Report forward error and fallbacks. |
| Wheels | Smaller of raw and ZIP-aware BSDIFF40 | Try exact recompression with additional verified codecs, plus compression and parsing alternatives for normalized blobs. A candidate must decode to the published bytes. Report the complete 16-project size and encode/decode costs. Report compatible bytes separately if testing a BSDIFF40-only change. |

Use the frozen [follow-up inputs and settings](../followup/PROTOCOL.md). Exercise the same repeat order across controls and candidates with one warm-up and three measured repeats per input. Rotate or seed the order deterministically and run official timing lanes serially. Use per-input median timing and percentile bootstrap intervals across independent integrands, counties, matrix patterns, or projects. Collapse repeated tolerances and profiles within each independent input. Compare cubature timings on conditions that both verify and report the full denominator for every family.

Development probes and their candidate selections are exploratory on the reused corpus. Changes after a full run require another complete paired run. Record versions, verification outcomes, fallbacks, and measured costs. Report improvements below the target, any regressions, and the applicable input boundaries.

## Additional trials

- On the ray route, cache the quadratic prediction matrix and reuse the residual rule's initial samples. Repeat the complete cubature cohort after changing sample scheduling.
- Keep the full R*-tree comparison at capacities 32 and 64. Probe smaller capacities separately at 30% reinsertion. For the capacity probe require the same verified query signatures and tree invariants, while allowing the layouts to differ.
- Test an equilibrated binary32 Cholesky factor with binary64 iterative refinement. Include scaling, response setup, all refinement solves, and fallback. Use the same AMD permutation, shift sequence, and independent residual check as the binary64 control.
- For the wheel selector try two parameterizations of HDiffPatch's BSDIFF40 encoder and classic bsdiff. Compare the Bzip2 and XZ encodings of each parse. Try miniz only when the recorded stream cannot be reproduced by zlib. Unknown inputs, the Bzip2 route, and the existing raw BSDIFF40 patch remain eligible for selection.
- Independently screen two unwrapped BSDIFF40 candidates on every archived wheel and selected member. Compare exact patch lengths to the previously verified hybrid. Verify inputs by hash and candidates with both decoders. This is a size-only screen.
