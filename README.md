# unet_pruning — Pruning + Finetuning pipeline for Carvana U-Net

This repository contains a refactored pruning layer by layer pipeline using l2 norm criterion for a U-Net trained on the Carvana segmentation dataset.

Primary entry point: **`prune.py`**.

---

## 1) Quickstart

### 1.1 Environment setup (Conda + CUDA)

Below is a working setup that matches your usual environment creation workflow (Python 3.11 + CUDA 12.1 wheels).

```bash
# Create env
conda create -n unet -y python=3.11
conda activate unet

# PyTorch (CUDA 12.1)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Common deps
pip install numpy pandas pillow tqdm opencv-python
pip install matplotlib

# Jupyter kernel (optional)
conda install -y ipykernel
python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
jupyter kernelspec list
```

Notes:
- If you prefer the kernel name to match the env name, use `--name unet --display-name "Python (unet)"`.
- CPU-only is possible (use `--device cpu`), but most experiments are intended for CUDA.

### 1.2 Prerequisites
- Python 3.11 environment (conda recommended)
- `torch`/`torchvision` (CUDA wheels if running `--device cuda`)
- Python packages: `numpy`, `pillow`, `tqdm`, `opencv-python`, `matplotlib`
### 1.3 Dataset layout
`--data_root` must point to a directory containing:
- `train/`
- `train_masks/`

Example:
- `/home/jovyan/work/carvana/train/`
- `/home/jovyan/work/carvana/train_masks/`

### 1.4 Minimal smoke run (CUDA)
From repo root:

```bash
python prune.py   --device cuda   --name smoke_cuda   --data_root ~/work/carvana   --taylor_batches 2   --iters 1   --eval_baseline 0
```

Expected behavior:
- Loads split: `pruning/data/trainval.json`
- Builds model from channels file: `configs/model_channels_baseline.txt`
- Loads checkpoint: `weights/unet_carvana_scale0.5_epoch2.pth`
- Computes Taylor ranks, selects channels, prunes, finetunes, evaluates Dice
- Saves outputs under `save_final/<name>/...`

---

## 2) Repository structure (folders & key files)

> Documents the structure as used by the working pipeline.

### Top-level
- **`prune.py`**
  - Entry point. Calls CLI parsing and launches the pipeline (typically `run_pruning(...)`).
- **`prune_cli.py`**
  - Defines CLI arguments and defaults (paths, pruning scope, friendly pruning, finetune, benchmarking).
- **`configs/`**
  - Baseline architecture channel definitions.
  - Key file: `configs/model_channels_baseline.txt`
- **`weights/`**
  - Pretrained checkpoints used as starting point for pruning.
  - Expected file (default): `weights/unet_carvana_scale0.5_epoch2.pth`
- **`save_final/`** *(generated)*
  - Output runs. Each run stores pruned weights, finetuned weights, selected channels, and logs.

### `pruning/` package
- **`pruning/prune_pipeline.py`**
  - Orchestrates end-to-end flow:
    1) load split
    2) build model
    3) load checkpoint
    4) compute Taylor criterion
    5) select channels according to scope/budget
    6) apply pruning
    7) finetune
    8) evaluate & report metrics (Dice / params / FLOPs)
- **`pruning/pruners/`**
  - Pruning logic (budgets, selection, channel removal mechanics).
  - Example file referenced during refactor issues:
    - `pruning/pruners/pruning_budgets.py`
- **`pruning/p_utils/`**
  - Utility layer: data loading, batching, metrics, helpers used by pipeline.
  - Files observed in traces:
    - `pruning/p_utils/utils.py` (batching helper `batch(...)`)
    - `pruning/p_utils/load.py` (split/dataset iteration helpers)
- **`pruning/data/`**
  - Split definitions used by loader.
  - Key file: `pruning/data/trainval.json`

---

## 3) How the pruning pipeline works (high-level)

1. **Parse CLI** (`prune_cli.py`)
2. **Load dataset split** (default `pruning/data/trainval.json`)
3. **Build baseline model** from channels file (default `configs/model_channels_baseline.txt`)
4. **Load checkpoint** (default `weights/unet_carvana_scale0.5_epoch2.pth`)
5. **Compute Taylor criterion** using `--taylor_batches`
6. **Select channels to remove** based on `--scope` and budgets
7. **Apply structured pruning**
8. **Finetune** for `--iters`
9. **Evaluate** (Dice + params/size/FLOPs breakdowns)

---

## 4) Usage guide (CLI arguments)

### 4.1 Paths / identity
- `--name` : run name (used in output folder naming)
- `--save_root` : root output folder (default: `save_final`)
- `--data_root` : dataset root (must contain `train/` and `train_masks/`)
- `--splitfile` : split json (default should be `pruning/data/trainval.json`)
- `--load` : checkpoint path (default should be `weights/unet_carvana_scale0.5_epoch2.pth`)
- `--channel_txt` : baseline channels file (default should be `configs/model_channels_baseline.txt`)

### 4.2 Device / reproducibility
- `--device` : `auto | cpu | cuda`
- `--seed` : random seed

### 4.3 Taylor ranking controls
- `--taylor_batches` : number of minibatches used for ranking
- `--taylor_shuffle` : shuffle batches for ranking (0/1)
- `--freeze_bn_taylor` : freeze BN during ranking (1 recommended)

### 4.4 Finetuning
- `--iters` : finetune iterations
- `--lr` : finetune learning rate

### 4.5 Pruning budget & constraints
- `--scope` : where to prune (see next section)
- `--prune_channels` : channels to remove (global or single-part scopes)
- `--min_keep_ch` : minimum out channels to keep per conv
- `--min_keep_ratio` : minimum keep ratio per conv (0 disables)

### 4.6 Friendly pruning (rounding budgets)
- `--friendly` : 1 enable / 0 disable
- `--friendly_round` : `up` or `down`
- `--friendly_multiple_cpu` : multiple for CPU runs
- `--friendly_multiple_gpu` : multiple for CUDA runs

### 4.7 Baseline evaluation
- `--eval_baseline` : 1 eval baseline, 0 skip
- `--baseline_dice` : cached baseline Dice used when skipping baseline evaluation

### 4.8 Benchmarking (optional)
- `--bench` : enable runtime benchmark
- `--bench_batch`, `--bench_warmup`, `--bench_runs`, `--bench_amp`, `--bench_channels_last`, `--bench_threads`

---

## 5) Pruning modes (scopes) — how to use them

### 5.1 Global pruning
```bash
python prune.py --scope global --prune_channels 300 ...
```

### 5.2 Single-part pruning
```bash
python prune.py --scope encoder    --prune_channels 300 ...
python prune.py --scope bottleneck --prune_channels 300 ...
python prune.py --scope decoder    --prune_channels 300 ...
```

### 5.3 Custom (explicit integer budgets per part)
```bash
python prune.py --scope custom   --prune_encoder 64   --prune_bottleneck 192   --prune_decoder 48   ...
```

### 5.4 pooled_down (pooled encoder+bottleneck budget)
```bash
python prune.py --scope pooled_down --prune_down_total 300 ...
```

### 5.5 custom_seq (percentage shares)
```bash
python prune.py --scope custom_seq --prune_channels 300   --prune_encoder_pct 20   --prune_bottleneck_pct 60   --prune_decoder_pct 20   ...
```

---

## 6) Outputs (what gets saved)

Each run creates a folder like:
`save_final/<name>/test/<name>-XX/`

Common artifacts:
- `Pruned.pth` : weights right after pruning
- `Finetuned.pth` : weights after finetuning
- `pruned_channels.txt` : selected/pruned channels per layer

The console output prints:
- rank stats per layer
- selection distribution (by group + top convs)
- final summary (Dice, Params, Size, FLOPs) + per-part breakdowns

---

## 7) Reproducibility checklist (important)

To ensure “same result” across codebases:
1. **Same checkpoint loaded** (confirm `Loaded checkpoint: ...`)
2. **Same channels file** (confirm `Built model using channel file: ...`)
3. **Same split file** (confirm `Load split dataset -> ...`)
4. Same `seed`, `scale`, `batch_size`, `taylor_batches`, `taylor_shuffle`, `freeze_bn_taylor`
5. Same scope & budgets (`scope`, `prune_channels`, friendly rounding settings)
6. Same device & deterministic settings (if you enforce determinism)

Minor Dice: Dice differences around ~1e-4 to 1e-3 can be normal if nondeterminism exists. Large selection differences usually mean checkpoint/channels/split differ.

---

## 8) Common errors (and fixes)

- **`channel_txt not found`**
  - Fix: pass `--channel_txt configs/model_channels_baseline.txt` or update default.
- **`UNet.__init__ got unexpected keyword`**
  - Fix: align UNet constructor signature with pipeline call (refactor mismatch).
- **Split unpacking errors (`too many values to unpack`)**
  - Fix: ensure `--splitfile` points to the correct JSON format expected by loader.
- **Checkpoint not found**
  - Fix: provide correct `--load` path or place file under `weights/`.

---

## 9) Planned updates
- Add profiling/inference/plotting documentation as those modules are cleaned.
- Add a “recipes” section with standard commands per experiment type.
