# Tiny Recursive Model (TRM) -- Puzzle Solving

Comparing Tiny Recursive Models (~3-5M params) against fine-tuned LLMs (82-1100M params) on structured reasoning tasks (Sudoku-Extreme, Maze-Hard). TRMs use recursive weight-sharing to iteratively refine solutions, dramatically outperforming LLMs while using orders of magnitude less energy.

**Module:** UFCFAS-15-2 Machine Learning | **Team:** Ahmed AlShamy, Armin Ghaseminejad, Nickolas Greiner

## How TRM Works

Unlike standard transformers that stack unique layers, TRM uses a **single shared 2-layer block** applied recursively. The model maintains three embedding streams:

```
x  =  input embedding   (puzzle tokens, frozen)
y  =  answer embedding  (iteratively refined)
z  =  latent reasoning   (scratch space for thinking)
```

**Three levels of recursion:**

```
Deep Supervision (N_sup=16 steps, each a separate optimizer step)
  |
  +-- Deep Recursion (T=3 passes, only last has gradients)
        |
        +-- Latent Recursion (n=6 inner steps)
              |
              +-- z = block(x + y + z)   # refine reasoning
              +-- y = block(y + z)       # update answer
```

**ACT (Adaptive Computation Time):** A halting head predicts confidence. When confidence > 0.5, recursion stops early -- the model learns to use fewer steps for easy puzzles.

### Architecture Variants

| Model | Params | Sequence Processing | Task |
|-------|--------|-------------------|------|
| TRM-MLP | ~3.3M | MLP-Mixer (token mixing) | Sudoku (L=81) |
| TRM-Att | ~5.1M | Self-Attention + RoPE | Maze (L=900) |

### Building Blocks

- **RMSNorm** (post-norm: applied AFTER residual addition)
- **SwiGLU** FFN: `W2(SiLU(W1(x)) * W3(x))`, no bias anywhere
- **Rotary Position Embedding (RoPE)** for maze attention variant
- **Stable-max cross-entropy** for numerical stability

## Project Structure

```
Machine-Learning/
+-- data/
|   +-- common.py                  # PuzzleDatasetMetadata + dihedral_transform
|   +-- build_sudoku_dataset.py    # Downloads & preprocesses Sudoku-Extreme from HF
|   +-- build_maze_dataset.py      # Downloads & preprocesses Maze-Hard from HF
+-- src/
|   +-- models/
|   |   +-- layers.py              # RMSNorm, SwiGLU, RoPE, StableMaxCE, MLPMixer
|   |   +-- trm_block.py           # Shared 2-layer block (attention or mixer)
|   |   +-- recursion.py           # latent_recursion, deep_recursion, deep_supervision
|   |   +-- trm_sudoku.py          # TRM-MLP (sudoku) + TRM-Att (maze)
|   |   +-- baseline_llm.py        # GPT-2 / TinyLlama + LoRA wrapper
|   |   +-- distilled_llm.py       # Student transformer + distillation loss
|   +-- data/
|   |   +-- sudoku_dataset.py      # PyTorch Dataset for sudoku .npy files
|   |   +-- maze_dataset.py        # PyTorch Dataset for maze .npy files
|   +-- training/
|   |   +-- trainer_trm.py         # Deep supervision + ACT + EMA + CodeCarbon
|   |   +-- trainer_llm.py         # LLM fine-tuning loop
|   |   +-- trainer_distill.py     # Knowledge distillation loop
|   |   +-- ema.py                 # Exponential Moving Average
|   |   +-- carbon_tracker.py      # CodeCarbon wrapper
|   +-- evaluation/
|   |   +-- evaluate.py            # Full eval with checkpoint loading + visualization
|   |   +-- metrics.py             # Cell accuracy, puzzle accuracy
|   +-- utils/
|       +-- config.py              # YAML config + Pydantic models
|       +-- seed.py                # Reproducibility
+-- configs/
|   +-- trm_sudoku.yaml            # TRM-MLP hyperparameters
|   +-- trm_maze.yaml              # TRM-Att hyperparameters
|   +-- llm_config.yaml            # LLM baseline config
+-- main.py                        # CLI entrypoint (train / eval / distill)
+-- Makefile                       # Shortcuts for common tasks
+-- models/                        # Saved checkpoints
+-- experiments/                   # CodeCarbon logs
+-- results/                       # Evaluation outputs
```

## Setup

**Requirements:** Python 3.10+, pip

```bash
# Create venv and install dependencies
make setup

# Or manually:
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Preprocess data (downloads from HuggingFace)
make data-sudoku              # Full dataset
make data-sudoku-small        # 100-sample subset for testing

# 2. Train TRM on Sudoku
make train-sudoku

# 3. Evaluate
make eval-sudoku
```

## Running Manually

### Data Preprocessing

Data scripts must be run from the `data/` directory (they use `from common import ...`).
Override `--output-dir` so data lands at `data/<dataset>` relative to project root:

```bash
cd data

# Sudoku-Extreme (full dataset, no augmentation)
python build_sudoku_dataset.py --output-dir sudoku-extreme-full

# Sudoku with 1000x augmentation (for full training)
python build_sudoku_dataset.py --output-dir sudoku-extreme-full --num-aug 1000

# Small subset for quick testing
python build_sudoku_dataset.py --output-dir sudoku-extreme-full --subsample-size 100

# Maze-Hard with dihedral augmentation
python build_maze_dataset.py --output-dir maze-30x30-hard-1k --aug
```

**Output format:** Each dataset produces `train/` and `test/` directories containing:
- `all__inputs.npy` -- input token IDs [N, seq_len]
- `all__labels.npy` -- target labels [N, seq_len]
- `dataset.json` -- metadata (vocab_size, seq_len, etc.)

### Training

```bash
# TRM-MLP on Sudoku (default config)
python main.py --mode train --config configs/trm_sudoku.yaml

# TRM-Att on Maze
python main.py --mode train --config configs/trm_maze.yaml

# LLM baseline (GPT-2 + LoRA)
python main.py --mode train --config configs/llm_config.yaml

# Knowledge distillation (requires trained teacher checkpoint)
python main.py --mode distill --config configs/llm_config.yaml --checkpoint models/llm_latest.pt
```

### Evaluation

```bash
python main.py --mode eval --config configs/trm_sudoku.yaml --checkpoint models/best.pt
```

## Data Encoding

### Sudoku
- **Vocab:** 11 tokens (pad=0, digits 0-9 shifted to tokens 1-10)
- **Grid:** 9x9 flattened to 81 tokens (row-major)
- **Blanks:** digit '0' becomes token 1 after +1 offset
- **Loss mask:** pre-filled cells (input == label) ignored in loss

### Maze
- **Vocab:** 5 tokens (pad=0, '#'=1, ' '=2, 'S'=3, 'G'=4, 'o'=path)
- **Grid:** 30x30 flattened to 900 tokens
- **Augmentation:** 8 dihedral transforms (rotations + flips)

## Training Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Hidden dim (D) | 512 | Shared block width |
| FF hidden | 2048 | SwiGLU intermediate |
| Inner recursions (n) | 6 | Latent refinement steps |
| Outer recursions (T) | 3 | Only last pass has gradients |
| Supervision steps (N_sup) | 16 | Max, ACT can stop early |
| ACT threshold | 0.5 | Halt when confidence > 0.5 |
| Optimizer | AdamW | betas=(0.9, 0.95) |
| Learning rate | 1e-4 | Linear warmup over 2K steps |
| Weight decay | 1.0 | TRM only; LLM uses 0.01 |
| EMA decay | 0.999 | Applied before evaluation |
| Effective batch | 768 | batch_size * grad_accum |

## Models Compared

| Model | Type | Params | Expected Sudoku | Expected Maze |
|-------|------|--------|----------------|---------------|
| TRM-MLP | Recursive (MLP-Mixer) | ~3.3M | ~87% | -- |
| TRM-Att | Recursive (Attention) | ~5.1M | -- | ~85% |
| Fine-tuned LLM | GPT-2 + LoRA | 124M (0.8M trainable) | ~0% | ~0% |
| Distilled LLM | Small transformer | ~2.4M | ~0% | ~0% |

The thesis: TRM with 3-5M params dramatically outperforms LLMs with 20-100x more parameters on structured reasoning, at a fraction of the energy cost.

## Key Implementation Details

1. **Post-norm, not pre-norm:** `output = RMSNorm(sublayer(x) + x)` -- this matters for matching published results
2. **Detach between supervision steps:** `y` and `z` are detached after each deep_recursion call to prevent OOM
3. **EMA before eval:** Always use EMA shadow weights for evaluation (the trainer handles this automatically)
4. **No bias anywhere:** All linear layers in the TRM use `bias=False`
5. **Stable-max loss:** Custom cross-entropy that clips log-sum-exp for numerical stability

## Reference

- [TRM: Less is More](https://arxiv.org/abs/2510.04871) -- Jolicoeur-Martineau et al.
- [ViTRM](https://arxiv.org/abs/2603.19503) -- Akazan et al.
- [Official TRM code](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
