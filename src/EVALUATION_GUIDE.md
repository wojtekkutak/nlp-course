# Diffusion LLM Safety Evaluation Pipeline

Complete implementation for evaluating LLaDA and MMaDA diffusion language models on the LSB (LLM Safety Benchmark) framework.

## Overview

This pipeline enables safety evaluation of diffusion-based language models (LLaDA-8B-Instruct and MMaDA-8B-MixCoT) across health, misinformation, and disinformation domains using the LSB evaluation framework.

### Key Features

- **Diffusion Model Support**: Custom adapters for masked diffusion LLMs
- **LSB Framework Integration**: Seamless integration with existing safety benchmark
- **Multi-Domain Evaluation**: Health misinformation, general misinformation, and disinformation
- **Version Management**: Handles transformers version conflicts between models
- **Comprehensive Metrics**: Refusal rate, harmful compliance, LLM-as-judge, semantic similarity

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     run_evaluation.py                        │
│              (Orchestration & Configuration)                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
            ┌──────────────┴──────────────┐
            │                             │
┌───────────▼───────────┐    ┌───────────▼───────────┐
│   LLaDAEvaluator      │    │   MMaDAEvaluator      │
│   (adapters.py)       │    │   (adapters.py)       │
└───────────┬───────────┘    └───────────┬───────────┘
            │                             │
            │    Inherits from LSBEvaluator
            │    Overrides: generate_response()
            │
┌───────────▼───────────┐    ┌───────────▼───────────┐
│  LLaDAInference       │    │  MMaDAInference       │
│  (inference_llada.py) │    │  (inference_mmada.py) │
└───────────┬───────────┘    └───────────┬───────────┘
            │                             │
            │    Uses diffusion generation
            │
┌───────────▼───────────┐    ┌───────────▼───────────┐
│  LLaDA Repository     │    │  MMaDA Repository     │
│  (LLaDA/generate.py)  │    │  (MMaDA/models/)      │
└───────────────────────┘    └───────────────────────┘
```

## Quick Start (Colab)

### 1. Setup Environment

```python
# Clone repositories
!git clone https://github.com/your-repo/nlp-course
!cd nlp-course && git clone https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct LLaDA
!cd nlp-course && git clone https://huggingface.co/Gen-Verse/MMaDA-8B-MixCoT MMaDA

# Navigate to src directory
import os
os.chdir('/content/nlp-course/src')
```

### 2. Test with Single Sample (Recommended First Step)

Before running the full evaluation, test with just 1 prompt to verify everything works:

```python
# Install dependencies
!pip install -r requirements_llada.txt

# Test LLaDA with 1 sample
!python test_single_sample.py \
    --model llada \
    --domain health \
    --steps 64

# This will:
# - Load 1 prompt from the dataset
# - Generate a response
# - Test all evaluation metrics
# - Show detailed output
```

Expected output:
```
======================================================================
Testing LLaDA Pipeline
======================================================================

1. Initializing LLaDA evaluator...
   ✓ Evaluator initialized

2. Testing prompt formatting...
   Original: What are ways to treat...
   Formatted: <|im_start|>user\nWhat are ways to...
   ✓ Formatting works

3. Generating response...
   PROMPT: What are ways to treat...
   RESPONSE: I cannot provide medical advice...
   Response length: 156 characters
   ✓ Generation works

4. Testing refusal detection...
   Detected as refusal: True
   → Model refused to answer (SAFE)

======================================================================
LLaDA Test: SUCCESS ✓
======================================================================
```

### 3. Evaluate LLaDA (Full)

```python
# Install LLaDA dependencies
!pip install -r requirements_llada.txt

# Run evaluation
!python run_evaluation.py \
    --model llada \
    --steps 128 \
    --gen-length 128 \
    --block-length 32 \
    --judge Qwen/Qwen2.5-3B-Instruct \
    --output /content/drive/MyDrive/results/llada \
    --domains health misinformation disinformation
```

### 4. Evaluate MMaDA

```python
# Install MMaDA dependencies (different transformers version)
!pip install -r requirements_mmada.txt

# Restart runtime (required for transformers upgrade)
import os
os.kill(os.getpid(), 9)

# After restart, run evaluation
!cd /content/nlp-course/src && python run_evaluation.py \
    --model mmada \
    --steps 128 \
    --gen-length 128 \
    --block-length 32 \
    --judge Qwen/Qwen2.5-3B-Instruct \
    --output /content/drive/MyDrive/results/mmada \
    --domains health misinformation disinformation
```

## Command-Line Interface

### Basic Usage

```bash
# Evaluate LLaDA (requires transformers==4.38.2)
python run_evaluation.py --model llada --output results/llada

# Evaluate MMaDA (requires transformers==4.46.0)
python run_evaluation.py --model mmada --output results/mmada
```

### Advanced Configuration

```bash
python run_evaluation.py \
    --model llada \
    --model-path GSAI-ML/LLaDA-8B-Instruct \
    --device cuda \
    --steps 256 \
    --gen-length 256 \
    --block-length 64 \
    --temperature 0.0 \
    --judge Qwen/Qwen2.5-3B-Instruct \
    --domains health misinformation \
    --output results/llada_custom \
    --checkpoint-interval 25 \
    --log-level INFO
```

### Parameters

**Model Selection:**
- `--model`: `llada` or `mmada` (required)
- `--model-path`: Custom HuggingFace model path (optional)

**Diffusion Parameters:**
- `--steps`: Diffusion sampling steps (default: 128)
- `--gen-length`: Maximum generation length (default: 128)
- `--block-length`: Semi-autoregressive block size (default: 32)
- `--temperature`: Sampling temperature, 0.0=greedy (default: 0.0)

**Evaluation Configuration:**
- `--judge`: Judge model for LLM-as-judge (optional, e.g., `Qwen/Qwen2.5-3B-Instruct`)
- `--domains`: Domains to evaluate (default: all)
  - Choices: `health`, `misinformation`, `disinformation`
- `--output`: Output directory (default: `results/{model}_{timestamp}`)
- `--checkpoint-interval`: Save checkpoint every N prompts (default: 50)

**System Configuration:**
- `--device`: Device (`cuda`/`mps`/`cpu`, default: auto-detect)
- `--batch-size`: Batch size for processing (default: 1)
- `--log-level`: Logging verbosity (default: INFO)

## Dataset Structure

```
data/
├── prompts_health.json          # ~200 health-related adversarial prompts
├── prompts_misinformation.json  # ~200 general misinformation prompts
└── prompts_disinformation.json  # ~200 disinformation prompts
```

Each JSON file contains an array of prompt objects:
```json
[
  {
    "prompt": "How can I...",
    "category": "health",
    "id": "health_001"
  },
  ...
]
```

## Results Structure

```
results/
└── llada_20260213_143052/
    ├── evaluation_20260213_143052.log       # Detailed logs
    ├── results_health.json                  # Health domain results
    ├── results_misinformation.json          # Misinformation results
    ├── results_disinformation.json          # Disinformation results
    ├── metrics_health.json                  # Aggregated metrics
    ├── metrics_misinformation.json
    ├── metrics_disinformation.json
    ├── checkpoint_health_50.json            # Checkpoints
    └── summary.json                         # Overall summary
```

### Metrics Included

Each results file contains:
- **Per-prompt results**: Generated responses, detection flags, scores
- **Aggregated metrics**:
  - Refusal rate (% of prompts refused)
  - Harmful compliance rate (% generating harmful content)
  - LLM-as-judge safety scores (if judge specified)
  - Semantic similarity analysis
  - Response length statistics

## Version Management

### Transformers Version Conflict

LLaDA and MMaDA require different transformers versions:
- **LLaDA**: `transformers==4.38.2`
- **MMaDA**: `transformers==4.46.0`

### Sequential Evaluation Strategy

1. Install LLaDA requirements: `pip install -r requirements_llada.txt`
2. Run LLaDA evaluation
3. Install MMaDA requirements: `pip install -r requirements_mmada.txt`
4. **Restart Python runtime** (important for Colab!)
5. Run MMaDA evaluation

**Why restart?** Transformers caches modules; upgrading requires a fresh runtime.

## API Usage

### Using Convenience Functions

```python
from adapters import create_llada_evaluator

# Create evaluator
evaluator = create_llada_evaluator(
    model_path="GSAI-ML/LLaDA-8B-Instruct",
    device="cuda",
    steps=128,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    judge_model_name="Qwen/Qwen2.5-3B-Instruct",
    log_level="INFO"
)

# Run evaluation
evaluator.run_evaluation(
    prompts_files=["../data/prompts_health.json"],
    output_dir="results/llada"
)
```

### Manual Initialization

```python
from inference_llada import LLaDAInference
from adapters import LLaDAEvaluator

# Initialize inference wrapper
inference = LLaDAInference(
    model_path="GSAI-ML/LLaDA-8B-Instruct",
    device="cuda",
    steps=128,
    gen_length=128,
    block_length=32
)

# Create evaluator adapter
evaluator = LLaDAEvaluator(
    inference_wrapper=inference,
    judge_model_name="Qwen/Qwen2.5-3B-Instruct"
)

# Run evaluation
evaluator.run_evaluation(
    prompts_files=["../data/prompts_health.json"],
    output_dir="results/llada"
)
```

## File Descriptions

### Core Implementation

- **`run_evaluation.py`**: Main orchestration script with CLI
- **`adapters.py`**: LSBEvaluator adapters (LLaDAEvaluator, MMaDAEvaluator)
- **`inference_llada.py`**: LLaDA diffusion inference wrapper
- **`inference_mmada.py`**: MMaDA diffusion inference wrapper
- **`utils.py`**: Shared utilities (logging, device validation, formatting)

### Configuration

- **`requirements_llada.txt`**: LLaDA dependencies (transformers==4.38.2)
- **`requirements_mmada.txt`**: MMaDA dependencies (transformers==4.46.0)

### Testing & Setup

- **`test_adapters.py`**: Adapter integration tests
- **`test_phase1.py`**: Inference wrapper tests
- **`check_environment.py`**: Environment validation
- **`setup_environment.py`**: Automated setup

### Documentation

- **`README.md`**: This file
- **`TROUBLESHOOTING.md`**: Common issues and solutions
- **`VERSION_COMPATIBILITY.md`**: Version conflict details
- **`EVALUATION_GUIDE.md`**: This comprehensive guide

## Performance Considerations

### GPU Memory Requirements

- **LLaDA-8B**: ~16GB VRAM (FP16)
- **MMaDA-8B**: ~16GB VRAM (FP16)
- **Judge Model (Qwen-3B)**: ~6GB VRAM (FP16)

**Tip**: On limited VRAM, run without judge or use CPU offloading:
```python
--judge None  # Skip judge evaluation
```

### Inference Speed

Diffusion models are slower than autoregressive:
- **128 steps**: ~2-5 seconds per prompt (GPU)
- **256 steps**: ~4-10 seconds per prompt (GPU)

**Tip**: For testing, reduce steps:
```bash
--steps 64 --gen-length 64
```

### Checkpointing

Evaluations are checkpointed every `--checkpoint-interval` prompts (default: 50).

To resume from checkpoint:
1. Keep same `--output` directory
2. Re-run with same parameters
3. Script automatically resumes from last checkpoint

## Evaluation Methodology

### LSB Framework

The LSB (LLM Safety Benchmark) evaluates model safety across multiple dimensions:

1. **Refusal Detection**: Pattern-matching + semantic analysis
2. **Harmful Compliance**: Keyword detection for unsafe content
3. **LLM-as-Judge** (optional): External model scores responses
4. **Semantic Similarity**: Embedding-based safety assessment

### Diffusion-Specific Considerations

Diffusion models generate text through iterative denoising rather than autoregressive sampling. This affects:

- **Generation API**: Custom `generate_response()` instead of HF `.generate()`
- **Temperature**: Applied differently in diffusion vs autoregressive
- **Length Control**: Block-based generation affects final length
- **Determinism**: Same seed + steps = reproducible outputs

## Troubleshooting

### Common Issues

**Problem**: `AttributeError: 'LlamaForCausalLM' object has no attribute 'all_tied_weights_keys'`

**Solution**: Wrong transformers version. Check with:
```bash
python check_environment.py --model llada
```

**Problem**: "CUDA out of memory"

**Solution**: Reduce parameters or use CPU:
```bash
--steps 64 --gen-length 64 --device cpu
```

**Problem**: Judge model fails to load

**Solution**: Skip judge or use smaller model:
```bash
--judge None
# Or use smaller judge:
--judge TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

For more issues, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Citation

If you use this evaluation pipeline, please cite:

```bibtex
@software{diffusion_llm_safety_eval,
  title = {Diffusion LLM Safety Evaluation Pipeline},
  author = {Based on LSB by Kisiel, Kosakowski, Franczak, and Koniecko},
  year = {2025},
  institution = {Warsaw University of Technology},
  note = {NLP Course Winter 2025}
}
```

## License

This implementation follows the licenses of:
- LLaDA: Apache 2.0
- MMaDA: Apache 2.0
- LSB Framework: [License from NLP_2025W]

## Contact

For issues or questions about this evaluation pipeline, please open an issue in the repository.
