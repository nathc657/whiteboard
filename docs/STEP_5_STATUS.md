# Step 5 checkpoint: ARIA diagnostic tests

Snapshot date: 2026-10-14. Branch: `step-5-complete`.

## Completed and checked

- `aria_diagnostics.py` implements kappa, r1, r2_5, and p on segment-by-query matrices.
- ARIA tests: **21 passed, 1 skipped**. The skipped check needs `output/aria_graddot_results.pt`.
- Synthetic LDS checks in `whiteboard.py`: perfect agreement = 1, reversed agreement = -1, partial agreement = 0.8.
- Cached reference shapes: LDS targets `(50, 178)`; subset indices `(50, 2500)`, spanning indices 0 to 4999. The benchmark selects 5000 training segments and 178 queries.
- CPU environment: Python 3.12.14, PyTorch 2.5.1+cpu; dependency check passes.
- GPU environment: Python 3.12.14, PyTorch 2.11.0+cu128, CUDA 12.8; dependency check passes and RTX 5080 Laptop GPU is detected with about 15.92 GiB VRAM.
- This branch includes the Grad-Dot runner and expanded test suite from commits 514bb3e and 1fbaaa2.

## Scope of completion

Step 5 is complete for synthetic diagnostic validation. A full pretrained MAESTRO attribution matrix and reproduction of the paper's numerical results are still pending. The calibration chart uses reported values and illustrative synthetic points, not a locally computed full benchmark matrix.

`RUN_FULL_TRAK = False` in `whiteboard.py`. `run_graddot.py` defaults to an 8-by-8 smoke run. A full GPU run requires explicit user approval. Loading either runner may access the benchmark cache at `C:\Users\nated\.dattri` and download missing benchmark artifacts.

## Preserve the local dattri change

The submodule stays pinned at ef50a35e6cdfa358ca104b7655ef15dcb9650afe. Its local change makes generated-music downloading conditional on `generated_music=True`. The change is saved in `patches/maestro-generated-music.patch`; see `patches/README.md` to apply it in a fresh checkout. The local edit remains in the working submodule and has not been pushed to dattri upstream.

Environments, downloaded datasets, caches and output artifacts remain ignored.

## Known follow-ups before full reproduction

1. Stabilize the TRAK log-odds calculation near probability one; the current expression can produce infinity for a representable negative log-probability.
2. Document and evaluate the Grad-Dot runner's `random_seq=False` setting relative to the legacy random-crop behavior. Changing token contents can change agreement with released LDS targets.
3. Strengthen the optional real-data test to recompute LDS from the saved score matrix and reference tensors rather than trusting only saved LDS values.

## Next

Step 6: compare dattri, ARIA, Choi et al., and TrueMuse; define the shared benchmark protocol, controls, and contribution. Full GPU computation is a later action with separate explicit approval.
