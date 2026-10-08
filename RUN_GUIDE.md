# Run guide

This guide is for the team member who runs the experiments. Follow the steps in order. The code has **not been run yet**, so the smoke test in step 2 is mandatory: it finds problems in minutes instead of hours.

**Deadline:** results are needed by 11 Oct 2026, because the paper is due 12 Oct.

## 0. What you need
| Item | Requirement |
|---|---|
| Python | 3.12 (3.11.4 or newer works) |
| Disk | **30 GB free** at peak, about 20 GB after the MUSAN archive is deleted |
| Internet | about 13 GB of downloads (MUSAN alone is 11 GB) |
| GPU | optional. Training is about 12× faster with one. CPU-only works but takes most of a night |
| Git | to clone the repo and send results back |

## 1. Set up
Windows PowerShell:
```powershell
git clone https://github.com/Minahil-Rauf/CaliberationIssue_MultiClassVAD.git
cd CaliberationIssue_MultiClassVAD
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
macOS / Linux:
```bash
git clone https://github.com/Minahil-Rauf/CaliberationIssue_MultiClassVAD.git
cd CaliberationIssue_MultiClassVAD
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
```

**NVIDIA GPU:**
1. Before `pip install -r requirements.txt`, install torch from https://pytorch.org using the command for your CUDA version.
2. Check it: `python -c "import torch; print(torch.cuda.is_available())"` must print `True`.

## 2. Smoke test (about 5 minutes, mandatory)
```
python run_pipeline.py --smoke
```
This runs **all 7 stages** on synthetic audio, without downloading anything.

**Success looks like:**
- the last line is `All stages finished.`;
- `outputs_smoke/tables/` contains 9 CSV files;
- `outputs_smoke/figures/` contains 4 figures (PNG + PDF) and 2 `.tex` tables.

The numbers are meaningless; this test only checks that the code runs.

**If a stage fails:**
1. Copy the full error text and send it to Minahil. Don't change code in `vadcal/` yourself.
2. After a fix, continue from the failed stage, e.g. `python run_pipeline.py --smoke --from-stage train`.

## 3. Unit tests (1 minute)
```
pytest
```
**All tests must pass.** `test_real_index` shows as *skipped* until step 4.2 has run, which is expected.

## 4. Real run, stage by stage
Run the stages one at a time so you can check each one. Every stage skips work that is already done, so re-running after a crash or reboot is always safe.

### 4.1 Download (30–120 min, depends on your connection)
```
python scripts/01_download_data.py
```
- Downloads MUSAN, LibriSpeech test-clean, ESC-50 and GTZAN into `data/raw/`.
- A broken connection is fine: run the same command again and it resumes.
- If every mirror for a dataset fails, the message says which file to download by hand and where to put it.
- To fetch one dataset only: `python scripts/01_download_data.py --datasets gtzan`

### 4.2 Index (5 min)
```
python scripts/02_index_sources.py
```
The script prints a table of files and hours per split. Check it:
- `train`, `val` and `test_in` contain only `musan`;
- `test_shift` contains `librispeech`, `esc50` and `gtzan`;
- every split has speech, music and noise.

About one GTZAN file is reported as unreadable and skipped. That is expected.

Then run `pytest` again; `test_real_index` should now pass.

### 4.3 Mixtures (20–60 min)
```
python scripts/03_build_mixtures.py
```
- **Check the printed frame shares.** Expect roughly background 0.25–0.35, speech 0.40–0.50, music 0.18–0.25, speech_music 0.03–0.06. If speech_music is near 0, or one class is above 0.7, stop and report it.
- **Listen to 5 clips** in `data/mixtures/train/audio/` and 5 in `data/mixtures/test_shift/audio/`. Compare with the `events` column of `meta.csv` (class, start, duration in seconds) and confirm you hear what it says.

### 4.4 Features (10–30 min)
```
python scripts/04_extract_features.py
```

### 4.5 Training (GPU about 1–3 h in total; CPU about 6–12 h)
Run all 12 runs (4 losses × 3 seeds):
```
python scripts/05_train.py --all
```
- Each epoch prints `train_loss`, `val_macro_f1`, `val_nll`, `val_ece` and seconds.
- **Time it:** if the first epoch takes more than 5 minutes on CPU, tell Minahil right away.
- **Resuming:** a finished run writes `outputs/runs/<loss>_seed<k>/DONE` and is skipped next time. A run interrupted mid-way resumes from its last completed epoch. Just run the same command again.
- **On a CPU with many cores,** you can split the work across two terminals:
  ```
  # terminal 1
  python scripts/05_train.py --loss ce --seed 0
  python scripts/05_train.py --loss ce --seed 1
  ...
  # terminal 2
  python scripts/05_train.py --loss focal --seed 0
  ...
  ```
  The full list of 12 runs: losses `ce`, `weighted_ce`, `focal`, `label_smoothing`, each with seeds 0, 1, 2.
- **Check:** at the end, `outputs/runs/` has 12 folders, each with a `DONE` file.

### 4.6 Evaluate (10–30 min)
```
python scripts/06_evaluate.py
```
Writes 9 CSV files to `outputs/tables/`.

### 4.7 Figures (1 min)
```
python scripts/07_make_figures.py
```
Writes figures and LaTeX tables to `outputs/figures/`.

## 5. Quick sanity checks before sending results
- `outputs/tables/summary.csv` has `n_seeds = 3` on every row.
- In `summary.csv`, rows with `calib = ts` have the **same** `macro_f1_mean` as `calib = raw` for the same loss and split. Temperature scaling never changes predictions, so any difference means something is wrong.
- In `calibrators.csv`, every `temperature` is between 0.3 and 5.
- `git_commit` in the tables does **not** end in `+dirty`. If it does, protocol files were edited.

## 6. Send results back
You need write access to the repository. If `git push` is refused, ask Minahil to add your GitHub account as a collaborator.
```
git checkout -b results-run1
git add outputs/tables outputs/figures
git commit -m "Results of the full run"
git push -u origin results-run1
```
Then tell Minahil. Also send these, which are not committed:
- `data/mixtures/class_shares.csv`;
- the console output of the training stage, or the `train_log.csv` of each run.

## 7. Rules
1. **Never edit** `configs/default.yaml`, `vadcal/` or `scripts/`. They define the protocol. If something must change, ask Minahil, then re-run every stage the change affects.
2. **Never commit** `data/`, `data_smoke/`, `outputs/runs/` or `outputs_smoke/`. `.gitignore` already blocks them.
3. **Smoke outputs are never used in the paper.** Tables from smoke mode have `smoke = True`.
4. Use **one machine and one environment** for all 12 runs. Each run records its package versions in `run_info.json`.

## 8. Troubleshooting
| Problem | Fix |
|---|---|
| `every mirror failed` | Check your internet. Download the file named in the message by browser into `data/downloads/`, then rerun 4.1 |
| Extraction error or `expected ... after extraction` | Delete the archive in `data/downloads/` and the dataset's folder in `data/raw/`, then rerun 4.1 |
| `no source files for: test_shift/music` | GTZAN missing: `python scripts/01_download_data.py --datasets gtzan` |
| `CUDA out of memory` | Close other GPU programs. If it persists, report it; batch size is part of the protocol |
| Very slow training on CPU | Make sure nothing else is running. Use the two-terminal split in 4.5 |
| `ModuleNotFoundError` | Activate the venv (step 1), then `pip install -r requirements.txt` |
| PowerShell blocks `Activate.ps1` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once |
| A stage was interrupted | Run the same command again; finished work is skipped |
| Need to redo one stage from scratch | Add `--force`, e.g. `python scripts/03_build_mixtures.py --force`, then rerun every later stage with `--force` |
