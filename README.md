# Calibration of Multi-Class Voice Activity Detection

Code for the paper *Confident but Wrong: Calibration of Multi-Class Voice Activity Detection Under Class Imbalance and Domain Shift* (target venue: ICEET 2026).

**Running the experiments? Read [RUN_GUIDE.md](RUN_GUIDE.md) first.**

## Research question
A voice activity detector (VAD) labels each 10 ms frame of audio and gives a confidence score. Deployed systems act on that score through thresholds ("wake up if speech > 0.9"). The scores are only useful if they are **calibrated**: among frames scored 90%, about 90% should be correct.

Recent VAD papers report accuracy but not calibration, and most are binary (speech / non-speech). This project measures calibration of a **4-class** VAD and asks:

| ID | Question |
|---|---|
| E1 | Is a standard multi-class VAD miscalibrated, and is the rare speech+music class worst? |
| E2 | Do class-imbalance fixes (weighted CE, focal loss, label smoothing) help or hurt calibration? |
| E3 | Do post-hoc fixes (temperature scaling, vector scaling) fitted on familiar audio still work on unseen audio? |
| E4 | Does temporal median smoothing, used by real VADs, change calibration? |
| E5 | With fixed thresholds, does "confidence ≥ τ" give the accuracy it claims? |

## Classes
| Index | Class | Meaning |
|---|---|---|
| 0 | background | silence or noise only |
| 1 | speech | speech, with or without noise |
| 2 | music | music only (singing counts as music) |
| 3 | speech_music | speech and music at the same time |

## Data
Labelled 10 s mixtures are generated from public corpora, so every frame label is exact.

| Split | Clips | Speech | Music | Noise |
|---|---|---|---|---|
| train | 7,200 (20 h) | MUSAN speech | MUSAN music | MUSAN noise |
| val (early stopping + fitting calibrators) | 700 | MUSAN | MUSAN | MUSAN |
| test_in (in-domain) | 1,000 | MUSAN | MUSAN | MUSAN |
| test_shift (unseen sources) | 1,000 | LibriSpeech test-clean | GTZAN (no hip-hop) | ESC-50 (no human-vocal classes) |

- **MUSAN** is split by file: 70% train, 10% val, 20% test_in, separately for speech, music and noise.
- **Each clip** is built from a quiet noise bed plus 1–4 s events, with target shares of 25% background, 50% speech, 20% music and 5% speech+music.
- **Speech+music events** mix the music in at a speech-to-music ratio of −5, 0, 5, 10 or 15 dB.
- **Labels** come from the clean signals before mixing:
  - speech uses an energy detector (within 35 dB of the segment peak, pauses under 150 ms bridged);
  - music is active wherever it was placed and is above −60 dBFS.
- All settings are in [configs/default.yaml](configs/default.yaml).

## Model and training
- **Input:** 64-band log-mel features (25 ms window, 10 ms hop).
- **Model:** a CRNN with about 86K parameters: 3 conv blocks (16/32/64 channels, pooling over frequency only), a 64-unit GRU and a linear layer giving 4 outputs per frame.
- **Training:** Adam (lr 1e-3), batch 32, random 4 s crops, up to 20 epochs, early stopping on validation macro-F1 (patience 5).
- **Runs:** 4 losses (`ce`, `weighted_ce`, `focal` with γ=2, `label_smoothing` 0.1) × 3 seeds = **12 runs**.

## Metrics and statistics
- **Accuracy:** macro-F1, per-class F1, accuracy.
- **Calibration:** ECE (15 equal-width bins, top label), classwise-ECE, NLL, Brier score, reliability diagrams.
- **Uncertainty:** mean ± std over 3 seeds, and 95% bootstrap CIs over **clips** (1,000 resamples), not frames, because neighbouring frames are highly correlated.
- **Comparisons:** paired clip-bootstrap CIs for method differences (each loss vs CE; calibrated vs raw).
- **Caveat on ECE intervals:** ECE is biased upward on finite samples, so its percentile interval can lie above the point estimate. Read it as a measure of spread, and say so in the paper.
- **Absent classes:** per-class F1 is NaN when a class never occurs in the labels; macro-F1 averages only the classes that do occur.

## Results
The full protocol has been run (12 runs, commit `ee80bb1`). Values are mean ± std over 3 seeds, uncalibrated outputs; lower ECE is better.

| Loss | In-domain macro-F1 | In-domain ECE | Shifted macro-F1 | Shifted ECE |
|---|---|---|---|---|
| CE | 0.875 ± 0.003 | 0.010 ± 0.003 | 0.878 ± 0.011 | 0.010 ± 0.004 |
| Weighted CE | 0.862 ± 0.008 | 0.011 ± 0.004 | 0.863 ± 0.005 | 0.009 ± 0.005 |
| Focal (γ=2) | 0.865 ± 0.003 | 0.105 ± 0.008 | 0.873 ± 0.004 | 0.112 ± 0.009 |
| Label smoothing (0.1) | 0.866 ± 0.003 | 0.059 ± 0.007 | 0.874 ± 0.008 | 0.060 ± 0.012 |

What the numbers show:
- **E1:** the CE baseline is already well calibrated (ECE about 0.01 on both test sets). speech+music is by far the hardest class for accuracy (F1 0.71 in-domain, 0.74 shifted, against 0.90–0.95 for the other classes), but its classwise-ECE (0.009 / 0.013) is not clearly worse than the other classes (0.006–0.014).
- **E2:** focal loss and label smoothing raise ECE about ten-fold and six-fold over CE, without improving macro-F1 or speech+music F1. Weighted CE keeps top-label ECE but raises classwise-ECE and lowers macro-F1.
- **E3:** temperature scaling fitted on the validation set brings every loss to ECE below 0.02 on both test sets (focal: 0.105 → 0.007 in-domain, 0.112 → 0.006 shifted), so the fix transfers to the unseen sources. Vector scaling gives the lowest classwise-ECE.
- **E4:** median smoothing up to 21 frames adds about 0.004 macro-F1 and leaves ECE unchanged for CE (0.010 at every window).
- **E5:** for CE, the accuracy of frames accepted at confidence ≥ τ stays within about 0.01 of the confidence claimed.
- **Domain shift:** the shifted test set is not harder than the in-domain one here; macro-F1 and ECE are about equal on both.

Full tables, confidence intervals and paired tests are in [outputs/tables/](outputs/tables); figures and LaTeX tables are in [outputs/figures/](outputs/figures).

### Reference run
| Item | Value |
|---|---|
| Machine | Windows 11, NVIDIA GeForce RTX 4060 Laptop GPU (8 GB) |
| Software | Python 3.12.11, PyTorch 2.14.1 + CUDA 12.6; all versions in [requirements-lock.txt](requirements-lock.txt) |
| Training | all 12 runs on the GPU, about 19 s per epoch, 20 min in total with the four losses running at the same time |
| Records | [outputs/run_records/](outputs/run_records): per-epoch training log and `run_info.json` of each run, class shares of the mixtures, files and hours per split |

GPU training is not bit-for-bit deterministic, so a new run reproduces these numbers to within the seed spread, not exactly.

## Pipeline
| Stage | Script | Output |
|---|---|---|
| 1 download | `scripts/01_download_data.py` | `data/raw/` (MUSAN, LibriSpeech test-clean, ESC-50, GTZAN) |
| 2 index | `scripts/02_index_sources.py` | `data/index/sources.csv` (file-level splits) |
| 3 mix | `scripts/03_build_mixtures.py` | `data/mixtures/<split>/` (wav, `labels.npy`, `meta.csv`) |
| 4 features | `scripts/04_extract_features.py` | `data/features/<split>.npy`, `norm_stats.npz` |
| 5 train | `scripts/05_train.py --all` | `outputs/runs/<loss>_seed<k>/` (model, log, logits) |
| 6 evaluate | `scripts/06_evaluate.py` | `outputs/tables/*.csv` |
| 7 figures | `scripts/07_make_figures.py` | `outputs/figures/*.png/.pdf`, LaTeX tables |

`python run_pipeline.py` runs all stages in order; `python run_pipeline.py --smoke` runs them on synthetic audio in a few minutes.

## Where each paper result comes from
| Paper item | File |
|---|---|
| Fig. 1, reliability diagrams (E1, E3) | `outputs/figures/fig1_reliability.pdf` |
| Table 1, losses vs F1 and ECE (E2) | `outputs/figures/table1_e2.tex`, `outputs/tables/summary.csv` |
| Table 2, calibration before/after (E3) | `outputs/figures/table2_e3.tex`, `outputs/tables/calibrators.csv` |
| Fig. 2, smoothing (E4) | `outputs/figures/fig2_smoothing.pdf`, `outputs/tables/e4_smoothing.csv` |
| Fig. 3, thresholds (E5) | `outputs/figures/fig3_thresholds.pdf`, `outputs/tables/e5a_selective.csv`, `e5b_speech_rule.csv` |
| Significance of differences | `outputs/tables/paired_differences.csv` |
| Per-class numbers | `outputs/tables/per_class.csv` |
| CIs per run | `outputs/tables/metrics_per_run.csv` |

## Repository layout
```
configs/        default.yaml (frozen protocol), smoke.yaml (quick test)
vadcal/         library code: data, labels, mixing, features, model, losses, training,
                metrics, bootstrap, calibration, smoothing, experiments, plots
scripts/        the 7 numbered pipeline stages
tests/          metric correctness, label/mixing logic, split leakage
outputs/        tables/, figures/ and run_records/ of the full run (results for the paper)
run_pipeline.py runs every stage in order
requirements.txt        minimum versions; requirements-lock.txt is the exact environment used
```
Datasets, mixtures, features, model weights and logits are not stored in the repository; the pipeline regenerates them.

## Limitations
- **Synthetic mixtures:** exact labels, but cleaner than real broadcast audio.
- **MUSAN split:** by file, not by speaker; a reader may appear in more than one split.
- **Speech labels:** energy-based, computed on the clean speech; very quiet syllables can be labelled background.
- **Singing:** many MUSAN and GTZAN music tracks contain singing. By design it counts as music, not speech_music.
- **Validation set:** the same set is used for early stopping and for fitting the calibrators. This is standard practice (Guo et al., 2017), but state it in the paper.
- **GTZAN download:** comes from a Hugging Face copy because the original server is offline. The archive has one known corrupt file, which is skipped.

## Data sources and licences
- MUSAN, Snyder et al. 2015, [OpenSLR 17](https://www.openslr.org/17/), CC BY 4.0
- LibriSpeech, Panayotov et al. 2015, [OpenSLR 12](https://www.openslr.org/12/), CC BY 4.0
- ESC-50, Piczak 2015, [GitHub](https://github.com/karolpiczak/ESC-50), CC BY-NC 3.0
- GTZAN, Tzanetakis & Cook 2002, [Hugging Face copy](https://huggingface.co/datasets/marsyas/gtzan), research use
