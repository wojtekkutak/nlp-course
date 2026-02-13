# Separated Evaluation Environments

The code has been split into two independent directories to avoid transformers version conflicts:

## Directory Structure

```
src/
├── llada_evaluation/          # LLaDA evaluation (transformers==4.38.2)
│   ├── __init__.py
│   ├── inference.py           # LLaDA inference wrapper
│   ├── adapters.py            # LLaDA adapter for LSB framework
│   ├── run_evaluation.py      # Full evaluation script
│   ├── test_single_sample.py  # Quick test with 1 prompt
│   ├── utils.py               # Shared utilities
│   └── requirements.txt       # transformers==4.38.2
│
└── mmada_evaluation/          # MMaDA evaluation (transformers==4.46.0)
    ├── __init__.py
    ├── inference.py           # MMaDA inference wrapper
    ├── adapters.py            # MMaDA adapter for LSB framework
    ├── run_evaluation.py      # Full evaluation script
    ├── test_single_sample.py  # Quick test with 1 prompt
    ├── utils.py               # Shared utilities
    └── requirements.txt       # transformers==4.46.0
```

## Quick Start

### For LLaDA Evaluation

```bash
# Navigate to LLaDA directory
cd src/llada_evaluation

# Install dependencies
pip install -r requirements.txt

# Test with single sample
python test_single_sample.py --no-judge --steps 64

# Run full evaluation
python run_evaluation.py --output ../../results/llada --domains health
```

### For MMaDA Evaluation

```bash
# Navigate to MMaDA directory
cd src/mmada_evaluation

# Install dependencies (will upgrade transformers to 4.46.0)
pip install -r requirements.txt

# IMPORTANT: Restart Python runtime if in Colab!

# Test with single sample
python test_single_sample.py --no-judge --steps 64

# Run full evaluation
python run_evaluation.py --output ../../results/mmada --domains health
```

## Google Colab Usage

### LLaDA Evaluation

```python
# Setup
!cd /content/nlp-course/src/llada_evaluation && pip install -r requirements.txt

# Test
!cd /content/nlp-course/src/llada_evaluation && python test_single_sample.py --no-judge

# Full evaluation
!cd /content/nlp-course/src/llada_evaluation && python run_evaluation.py \
    --steps 128 \
    --judge Qwen/Qwen2.5-3B-Instruct \
    --output /content/drive/MyDrive/results/llada \
    --domains health misinformation disinformation
```

### MMaDA Evaluation

```python
# Setup
!cd /content/nlp-course/src/mmada_evaluation && pip install -r requirements.txt

# IMPORTANT: Restart runtime after installing!
import os
os.kill(os.getpid(), 9)

# After restart, test
!cd /content/nlp-course/src/mmada_evaluation && python test_single_sample.py --no-judge

# Full evaluation
!cd /content/nlp-course/src/mmada_evaluation && python run_evaluation.py \
    --steps 128 \
    --judge Qwen/Qwen2.5-3B-Instruct \
    --output /content/drive/MyDrive/results/mmada \
    --domains health misinformation disinformation
```

## Why Separate Directories?

**Problem**: LLaDA requires `transformers==4.38.2` while MMaDA requires `transformers==4.46.0`. Having them in the same codebase causes import conflicts.

**Solution**: Completely separate environments with their own dependencies. Each directory is self-contained and can be used independently.

## Command Reference

### Test Single Sample (Fast!)

```bash
# LLaDA
cd src/llada_evaluation
python test_single_sample.py --domain health --steps 64 --no-judge

# MMaDA
cd src/mmada_evaluation
python test_single_sample.py --domain health --steps 64 --no-judge
```

### Full Evaluation

```bash
# LLaDA
cd src/llada_evaluation
python run_evaluation.py \
    --steps 128 \
    --gen-length 128 \
    --block-length 32 \
    --judge Qwen/Qwen2.5-3B-Instruct \
    --domains health misinformation disinformation \
    --output ../../results/llada

# MMaDA
cd src/mmada_evaluation
python run_evaluation.py \
    --steps 128 \
    --gen-length 128 \
    --block-length 32 \
    --judge Qwen/Qwen2.5-3B-Instruct \
    --domains health misinformation disinformation \
    --output ../../results/mmada
```

### Parameters

- `--steps`: Diffusion sampling steps (64 for testing, 128-256 for eval)
- `--gen-length`: Maximum generation length (default: 128)
- `--block-length`: Semi-autoregressive block size (default: 32)
- `--judge`: Judge model for LLM-as-judge (e.g., `Qwen/Qwen2.5-3B-Instruct`)
- `--domains`: Which domains to evaluate (`health`, `misinformation`, `disinformation`)
- `--output`: Output directory for results
- `--no-judge`: Skip judge evaluation (faster for testing)

## Files Explained

- **`inference.py`**: Model-specific inference wrapper with diffusion generation
- **`adapters.py`**: Adapter to make diffusion model work with LSB framework
- **`run_evaluation.py`**: Complete evaluation pipeline (CLI)
- **`test_single_sample.py`**: Quick test with 1 prompt (for debugging)
- **`utils.py`**: Shared utilities (logging, device detection, stats)
- **`requirements.txt`**: Model-specific dependencies with correct transformers version

## Troubleshooting

**Import Error**: Make sure you're in the right directory (llada_evaluation or mmada_evaluation)

**Version Mismatch**: The scripts check transformers version automatically and exit if wrong

**Colab Runtime**: After installing dependencies, you MUST restart the runtime for changes to take effect

**Dataset Not Found**: Make sure you're running from the correct directory - paths are relative to the evaluation folder

## Results

Results are saved to the specified output directory with:
- `evaluation_*.log`: Detailed logs
- `results_*.json`: Per-domain results
- `metrics_*.json`: Aggregated metrics
- `checkpoint_*.json`: Checkpoints (for resuming)

## Benefits of This Structure

✅ **No version conflicts** - Each model has its own environment
✅ **Cleaner imports** - No cross-model dependencies
✅ **Easier to use** - Just cd into the directory and run
✅ **Independent testing** - Test each model separately
✅ **Colab friendly** - Simple to set up and use

## Migration from Old Structure

The old unified structure in `src/` is still available but deprecated. Use the separated directories for all new work.
