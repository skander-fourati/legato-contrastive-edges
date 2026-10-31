# Contrastive supervision of LEGATO's view graph (work in progress)

**Status.** Temporary public snapshot. It is incomplete and the development history is not included.

## Question

LEGATO (Liu, Berrevoets, Qian, van der Schaar, ICML 2023) is a graph autoencoder for multi-view tabular data. Each feature set ("view") is a node, and a graph learner produces a per-sample weighted adjacency matrix between views. The edges are trained only indirectly, through the reconstruction loss and a sparsity penalty.

I test adding a pairwise contrastive (InfoNCE) loss between view embeddings, so that an edge has a direct meaning (can view i identify view j of the same sample?). Does this recover the true view dependencies better than reconstruction alone?


## Notebooks

| Notebook | What |
| --- | --- |
| `00_simulation_data` | the synthetic data and the true edge matrix |
| `01_baseline_legato_core` | a minimal LEGATO-style model (no pooling, orthogonality or sparsity loss), its edges, and the contrastive version, one seed |
| `02_run_comparison` | the minimal model, baseline vs contrastive, 5 seeds x 3 values of w, with a held-out split |
| `03_upstream_contrastive` | the same loss added to the authors' own code, with a diagnosis of a BatchNorm evaluation problem |
| `04_upstream_sweep` | 60 runs in the authors' code: edge recovery and the paper's downstream metric |

## Results so far

**Minimal model** (notebook 02; edge AUROC on 200 held-out samples, mean of 5 seeds, baseline -> with contrastive loss):

| w | 0.25 | 0.50 | 0.75 |
| --- | --- | --- | --- |
| edge AUROC | 0.72 -> 0.83 | 0.79 -> 0.92 | 0.83 -> 0.99 |

Reconstruction fit drops slightly (R^2 about 0.98 -> 0.96). Unlinked pairs stay at the chance level of the contrastive loss, so there is no sign of memorisation.

**Authors' implementation** (notebook 04; 5 seeds, edge AUROC with BatchNorm in train mode, baseline -> with contrastive loss):

| w | 0.25 | 0.50 | 0.75 |
| --- | --- | --- | --- |
| authors' early-stopping protocol | 0.49 -> 0.82 | 0.62 -> 0.98 | 0.66 -> 0.98 |
| 300 fixed epochs | 0.49 -> 0.83 | 0.50 -> 1.00 | 0.65 -> 1.00 |

The paper's downstream metric (a linear probe predicting the hidden latents) shows no consistent difference between the two arms; the differences are within the spread across seeds.

**Limits**: synthetic benchmark only, one untuned configuration for both arms, 5 seeds. AUROC in the authors’ code depends on the BatchNorm mode at evaluation (train-mode numbers shown; eval-mode results are mixed, see derived/nb04_sweep.csv)

## Reproducing

```bash
uv sync
# notebooks 00-02 run on their own on a CPU (minutes). Run them from notebooks/; 00 writes derived/*.npz.
# notebooks 03-04 need the authors' repository, unmodified, at the pinned commit:
git clone https://github.com/tennisonliu/LEGATO external/LEGATO
git -C external/LEGATO checkout 514224b
# and compiled torch_sparse / torch_scatter matching your torch (not in the lock file):
uv pip install torch-sparse torch-scatter --no-build-isolation
```

## Credit

The method and the benchmark are from the LEGATO paper (arXiv:2305.19726; code: github.com/tennisonliu/LEGATO). This repository contains none of the authors' code. I am not affiliated with the authors.
