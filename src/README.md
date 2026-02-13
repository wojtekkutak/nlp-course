# Diffusion LLM Safety Evaluation - Source Code

This directory contains the implementation for evaluating diffusion-based Large Language Models (LLaDA and MMaDA) using the LSB (LLM Safety Benchmark) framework.

**Authors:** Based on work by Kinga Frańczak, Kamil Kisiel, Wiktoria Koniecko, Piotr Kosakowski
**Institution:** Warsaw University of Technology, NLP Course Winter 2025

---

## Overview

This implementation enables safety evaluation of diffusion language models across three risk domains:
- **Health**: Medical misinformation, self-harm, dangerous remedies
- **Misinformation**: False claims, conspiracy theories
- **Disinformation**: Fake news generation, coordinated behavior

### Models Evaluated
- **LLaDA-8B-Instruct** (`GSAI-ML/LLaDA-8B-Instruct`) - Masked diffusion language model
- **MMaDA-8B-MixCoT** (`Gen-Verse/MMaDA-8B-MixCoT`) - Multimodal diffusion with Chain-of-Thought

---

## Quick Start Guide

### 1. Install Dependencies

**For LLaDA:**
```bash
pip install -r src/requirements_llada.txt
```

**For MMaDA:**
```bash
pip install -r src/requirements_mmada.txt
```

**Switching between models:**
```bash
# After running LLaDA, switch to MMaDA
pip install -r src/requirements_mmada.txt

# To go back to LLaDA
pip install -r src/requirements_llada.txt
```

### 2. Verify Environment
```bash
python src/check_environment.py
```

This will check all dependencies and report any issues.

### 3. Clone Model Repositories
```bash
git clone https://github.com/ML-GSAI/LLaDA.git
git clone https://github.com/Gen-Verse/MMaDA.git
```

### 4. Test with Single Sample (Recommended)

Test the complete evaluation pipeline with just 1 prompt before running full evaluation:

```bash
# Test LLaDA with 1 sample from health domain
python src/test_single_sample.py --model llada --domain health --steps 64

# Test MMaDA with 1 sample
python src/test_single_sample.py --model mmada --domain misinformation --steps 64

# Skip judge model for faster testing
python src/test_single_sample.py --model llada --no-judge

# Use custom prompt
python src/test_single_sample.py --model llada --prompt "What is artificial intelligence?"
```

This will:
- ✓ Initialize the model
- ✓ Format the prompt
- ✓ Generate a response
- ✓ Test refusal detection
- ✓ Test harmful content detection
- ✓ Test LLM-as-judge (if enabled)

### 5. Test Basic Inference (Optional)

Test just the inference wrappers:

```bash
# Test LLaDA
python src/inference_llada.py --prompt "What is 2+2?" --gen-length 64

# Test MMaDA
python src/inference_mmada.py --prompt "What is 2+2?" --gen-length 64
```

---

## Project Structure

```
src/
├── utils.py                    # Shared utilities (logging, device handling, stats)
├── inference_llada.py          # LLaDA inference wrapper
├── inference_mmada.py          # MMaDA inference wrapper
├── adapters.py                 # LSB framework adapters (✅ Complete)
├── run_evaluation.py           # Main evaluation orchestration script (✅ Complete)
├── test_single_sample.py       # Test pipeline with 1 sample (✅ Complete)
├── requirements_llada.txt      # LLaDA dependencies (transformers==4.38.2)
├── requirements_mmada.txt      # MMaDA dependencies (transformers==4.46.0)
├── README.md                   # This file
├── TROUBLESHOOTING.md          # Detailed troubleshooting guide
├── EVALUATION_GUIDE.md         # Comprehensive evaluation guide
├── VERSION_COMPATIBILITY.md    # Version conflict details
├── setup_environment.py        # Automated dependency installation
├── check_environment.py        # Environment verification script
├── test_adapters.py            # Adapter integration tests
└── test_phase1.py              # Basic validation tests
```

---

## Phase 1: Inference Wrappers (✅ Complete)

### 1. `utils.py`

Shared utilities used by all modules:

**Key Functions:**
- `setup_logging()` - Configure logging with console and file output
- `validate_device()` - Auto-detect or validate compute device (CUDA/MPS/CPU)
- `format_chat_prompt()` - Format messages using tokenizer's chat template
- `GenerationStats` - Track generation performance metrics

**Example:**
```python
from utils import setup_logging, validate_device

logger = setup_logging(log_level="INFO", log_file="logs/eval.log")
device = validate_device(None)  # Auto-detect best device
```

### 2. `inference_llada.py`

Standalone inference wrapper for LLaDA-8B-Instruct.

**Key Features:**
- Clean API wrapping LLaDA's custom `generate()` function
- Configurable diffusion parameters (steps, gen_length, block_length)
- Support for chat templates and multi-turn conversations
- Automatic prompt formatting
- Performance statistics tracking

**Usage:**
```python
from inference_llada import LLaDAInference

# Initialize
inference = LLaDAInference(
    model_path="GSAI-ML/LLaDA-8B-Instruct",
    device="cuda",
    steps=128,           # Diffusion sampling steps
    gen_length=128,      # Maximum generation length
    block_length=32,     # Semi-autoregressive block size
    temperature=0.0      # 0 = greedy, >0 = stochastic
)

# Generate single response
response = inference.generate_response(
    "What are the health benefits of regular exercise?"
)

# Generate batch
responses = inference.generate_batch([
    "Prompt 1",
    "Prompt 2",
    "Prompt 3"
])

# Get statistics
stats = inference.get_stats()
print(stats)  # tokens/s, total time, error rate, etc.
```

**Command-line Testing:**
```bash
python src/inference_llada.py \
    --prompt "What is the capital of France?" \
    --steps 128 \
    --gen-length 128 \
    --block-length 32
```

**Key Parameters:**
- `steps`: Number of diffusion sampling steps (default: 128)
- `gen_length`: Maximum tokens to generate (default: 128)
- `block_length`: Block size for semi-autoregressive remasking (default: 32)
  - Must evenly divide `gen_length`
  - Smaller = more autoregressive, larger = more parallel
- `temperature`: Sampling temperature (0.0 = greedy, higher = more random)
- `cfg_scale`: Classifier-free guidance scale (0.0 = disabled)
- `remasking`: Strategy for remasking (`'low_confidence'` or `'random'`)

### 3. `inference_mmada.py`

Standalone inference wrapper for MMaDA-8B-MixCoT (text-only).

**Key Features:**
- Text-only generation from multimodal diffusion model
- Internal implementation of diffusion generation adapted from MMaDA
- Support for Chain-of-Thought reasoning capabilities
- Similar API to LLaDA wrapper for consistency

**Usage:**
```python
from inference_mmada import MMaDAInference

# Initialize
inference = MMaDAInference(
    model_path="Gen-Verse/MMaDA-8B-MixCoT",
    device="cuda",
    steps=128,
    gen_length=128,
    block_length=32
)

# Generate response (same API as LLaDA)
response = inference.generate_response(
    "Explain why vaccines are important."
)

# Get statistics
print(inference.get_stats())
```

**Command-line Testing:**
```bash
python src/inference_mmada.py \
    --prompt "What causes climate change?" \
    --steps 128 \
    --gen-length 128
```

---

## Phase 2: Evaluation Adapters (✅ Complete)

### `adapters.py`

Adapter classes that make diffusion models compatible with the LSB evaluation framework.

**Key Classes:**

**`DiffusionLLMEvaluator(LSBEvaluator)`** - Base adapter class
- Inherits from LSBEvaluator to reuse all evaluation logic
- Overrides only `generate_response()` and `generate_responses_batch()`
- Delegates to inference wrappers for diffusion generation
- Preserves all LSB metrics: refusal detection, harmful compliance, LLM-as-judge

**`LLaDAEvaluator(DiffusionLLMEvaluator)`** - LLaDA-specific adapter
- Type-checks inference wrapper is LLaDAInference
- Provides clean API for LLaDA evaluation

**`MMaDAEvaluator(DiffusionLLMEvaluator)`** - MMaDA-specific adapter
- Type-checks inference wrapper is MMaDAInference
- Provides clean API for MMaDA evaluation

**Usage:**
```python
from adapters import create_llada_evaluator, create_mmada_evaluator

# LLaDA evaluation
evaluator = create_llada_evaluator(
    model_path="GSAI-ML/LLaDA-8B-Instruct",
    device="cuda",
    steps=128,
    gen_length=128,
    block_length=32,
    judge_model_name="Qwen/Qwen2.5-3B-Instruct"  # Optional
)

evaluator.run_evaluation(
    prompts_files=["../data/prompts_health.json"],
    output_dir="results/llada"
)

# MMaDA evaluation
evaluator = create_mmada_evaluator(
    model_path="Gen-Verse/MMaDA-8B-MixCoT",
    device="cuda",
    steps=128,
    gen_length=128,
    judge_model_name="Qwen/Qwen2.5-3B-Instruct"
)

evaluator.run_evaluation(
    prompts_files=["../data/prompts_misinformation.json"],
    output_dir="results/mmada"
)
```

**Design Notes:**
- Only overrides generation methods, not evaluation logic
- Judge model handled by LSBEvaluator's built-in methods
- Maintains full compatibility with LSB framework metrics
- Supports all diffusion parameters (steps, gen_length, block_length)

---

## Phase 3: Evaluation Orchestration (✅ Complete)

### `run_evaluation.py`

Command-line orchestration script for complete evaluation pipeline.

**Features:**
- Automatic dataset loading from `../data/` directory
- Version checking (transformers 4.38.2 for LLaDA, 4.46.0 for MMaDA)
- Flexible domain selection (health, misinformation, disinformation)
- Comprehensive logging to file and console
- Automatic output directory creation with timestamps
- Support for checkpointing and resumption

**Usage:**
```bash
# Evaluate LLaDA on all domains
python src/run_evaluation.py \
    --model llada \
    --steps 128 \
    --gen-length 128 \
    --block-length 32 \
    --judge Qwen/Qwen2.5-3B-Instruct \
    --output results/llada

# Evaluate MMaDA on specific domains
python src/run_evaluation.py \
    --model mmada \
    --domains health misinformation \
    --steps 256 \
    --output results/mmada_custom

# Quick test with reduced parameters
python src/run_evaluation.py \
    --model llada \
    --domains health \
    --steps 64 \
    --gen-length 64 \
    --checkpoint-interval 25
```

**Parameters:**
- `--model`: Model to evaluate (`llada` or `mmada`)
- `--model-path`: Custom model path (optional)
- `--device`: Device (`cuda`/`mps`/`cpu`, auto-detected by default)
- `--steps`: Diffusion sampling steps (default: 128)
- `--gen-length`: Generation length (default: 128)
- `--block-length`: Block size for semi-AR generation (default: 32)
- `--temperature`: Sampling temperature (default: 0.0 for greedy)
- `--judge`: Judge model for LLM-as-judge (optional)
- `--domains`: Domains to evaluate (default: all)
- `--output`: Output directory (default: `results/{model}_{timestamp}`)
- `--checkpoint-interval`: Save checkpoint every N prompts (default: 50)
- `--log-level`: Logging verbosity (default: INFO)

---

## Dependencies

**⚠️ IMPORTANT: Version Compatibility**

**LLaDA and MMaDA require different transformers versions:**
- **LLaDA** requires `transformers==4.38.2`
- **MMaDA** requires `transformers==4.46.0`

**Solution:** Use separate requirements files and switch between them:

```bash
# For LLaDA inference
pip install -r src/requirements_llada.txt

# For MMaDA inference (run this when switching)
pip install -r src/requirements_mmada.txt
```

**Common Requirements:**
```
torch>=2.0.0
accelerate>=0.24.0
tqdm>=4.65.0
pytest>=7.4.0
```

**For Evaluation Framework:**
```
pytest>=7.4.0  # For testing
```

**Installation:**
```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies (IMPORTANT: Use exact transformers version!)
pip install -r src/requirements.txt

# Or install manually:
pip install torch transformers==4.38.2 accelerate tqdm pytest
```

---

## Testing

### Unit Tests (Planned)

```bash
# Test utilities
pytest tests/test_utils.py

# Test LLaDA inference
pytest tests/test_inference_llada.py

# Test MMaDA inference
pytest tests/test_inference_mmada.py

# Test adapters
pytest tests/test_adapters.py
```

### Manual Testing

**Test LLaDA Inference:**
```bash
python src/inference_llada.py --prompt "What is 2+2?"
```

**Test MMaDA Inference:**
```bash
python src/inference_mmada.py --prompt "Explain photosynthesis."
```

---

## Implementation Notes

### Diffusion vs Autoregressive Models

**Key Differences:**
- **Autoregressive**: Sequential left-to-right token generation
  - `model.generate()` standard HuggingFace API

- **Diffusion**: Iterative denoising/unmasking process
  - Custom `generate()` functions with steps, masking, remasking
  - Non-sequential generation allows different robustness properties

### Model Loading

Both models require `trust_remote_code=True` for custom model classes:
```python
model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)
```

### Memory Requirements

- **LLaDA-8B**: ~16GB VRAM (bfloat16), ~32GB (float32)
- **MMaDA-8B**: ~16GB VRAM (bfloat16), ~32GB (float32)

Recommended: Use CUDA GPU with ≥16GB VRAM or quantization for smaller GPUs.

### Generation Parameters

**For Safety Evaluation:**
- Use `steps=128, gen_length=128` for balance of quality and speed
- Use `temperature=0.0` for deterministic responses
- Use `block_length=32` for semi-autoregressive generation

**For Quality/Creativity:**
- Increase `steps` (256-1024) for better quality
- Increase `temperature` (0.5-1.0) for more diverse responses

---

## Troubleshooting

For detailed troubleshooting information, see **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**.

### Quick Fixes

### AttributeError: 'all_tied_weights_keys'
```
AttributeError: 'LLaDAModelLM' object has no attribute 'all_tied_weights_keys'
```

**Cause**: Wrong transformers version. LLaDA requires exactly version 4.38.2.

**Solution**:
```bash
pip install transformers==4.38.2 --force-reinstall
```

### CUDA Out of Memory
```python
# Reduce generation length
inference = LLaDAInference(gen_length=64, block_length=16)

# Or use CPU (slower)
inference = LLaDAInference(device="cpu")
```

### Import Errors
```bash
# Ensure LLaDA and MMaDA repos are cloned in project root
ls ../LLaDA  # Should exist
ls ../MMaDA  # Should exist

# Clone if missing
git clone https://github.com/ML-GSAI/LLaDA.git ../LLaDA
git clone https://github.com/Gen-Verse/MMaDA.git ../MMaDA
```

### Slow Generation
```python
# Reduce steps for faster generation (lower quality)
inference = LLaDAInference(steps=64)

# Use larger block_length for more parallel generation
inference = LLaDAInference(block_length=64)
```

---

## Next Steps

- [x] Complete Phase 2: Implement adapters for LSB framework
- [x] Complete Phase 3: Implement evaluation orchestration script
- [ ] Add comprehensive unit tests
- [ ] Run pilot evaluations on sample prompts
- [ ] Execute full evaluation on all 600 prompts (3 domains × 200 prompts)
- [ ] Analyze results and compare diffusion vs autoregressive safety

---

## References

- **LLaDA Paper**: [arXiv:2502.09992](https://arxiv.org/abs/2502.09992)
- **MMaDA Paper**: [arXiv:2505.15809](https://arxiv.org/abs/2505.15809)
- **LSB Framework**: NLP_2025W/Kisiel_Kosakowski_Franczak_Koniecko/
- **LLaDA Repository**: [github.com/ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA)
- **MMaDA Repository**: [github.com/Gen-Verse/MMaDA](https://github.com/Gen-Verse/MMaDA)

---

## License

This code is provided for academic research purposes as part of the NLP Course at Warsaw University of Technology.
Please refer to the LICENSE files in the respective model repositories for model-specific licenses.
