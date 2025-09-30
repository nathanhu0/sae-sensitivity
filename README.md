# SAE Feature Evaluation Pipeline

This repository implements the evaluation method from the paper "Evaluating Sparse Autoencoder Feature Sensitivity". The core idea is to test whether SAE features actually respond to the concepts they supposedly represent.

The pipeline works by:
1. Finding text examples where each SAE feature naturally activates
2. Using an LLM to generate new text based on those patterns
3. Testing whether the generated text successfully activates the same features
4. Measuring the "sensitivity" - how reliably features activate on relevant text

## Setup

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv venv
source .venv/bin/activate
uv sync

# Configure API keys
cp .env.example .env
# Edit .env and add:
# - OPENAI_API_KEY for text generation (GPT-4)
# - HF_TOKEN for accessing Gemma-2 models
```

This pipeline heavily relies on the `sae_bench` library for:
- Loading and running SAE models from SAEBench
- Dataset utilities for tokenization and processing
- Activation collection infrastructure


## Supported SAEs

The pipeline uses simple string identifiers to specify which SAEs to evaluate. These strings encode the model, architecture, width, and training configuration, making it easy to compare different SAE variants.

### SAEBench SAEs

**Naming Convention**: `{model}_{width}_{architecture}_s{sparsity}`

| Component | Options | Description |
|-----------|---------|-------------|
| model | `py`, `gem` | `py` = pythia-160m-deduped, `gem` = gemma-2-2b |
| width | `4k`, `16k`, `65k` | Number of features (4096, 16384, 65536) |
| architecture | See table below | SAE architecture type |
| sparsity | `s0` to `s5` | Training configuration (affects L0 sparsity) |

**Architecture Types**:
| Code | Architecture |
|------|--------------|
| `btopk` | BatchTopK |
| `matbtopk` | Matryoshka BatchTopK |
| `gated` | Gated SAE |
| `jumprelu` | JumpReLU |
| `topk` | TopK |
| `relu` | Standard |
| `pan` | PAnneal |

**Valid Examples**:
- Pythia: `py_4k_btopk_s0`, `py_16k_gated_s2`, `py_65k_relu_s5`
- Gemma: `gem_4k_jumprelu_s1`, `gem_16k_topk_s3`, `gem_65k_pan_s4`

### GemmaScope SAEs

**Naming Convention**: `gscope_{width}_l0_{sparsity}`

| Component | Options | Description |
|-----------|---------|-------------|
| width | `16k`, `128k`, `1m` | Number of features |
| l0 | Various (e.g., 82, 128, 256) | Target L0 sparsity |

**Examples**: `gscope_16k_l0_44`, `gscope_128k_l0_82`

See available GemmaScope SAEs: https://huggingface.co/google/gemma-scope-2b-pt-res/tree/main/layer_12

## Command-Line Arguments

### Required Arguments
- `--model`: Base model (`pythia-160m-deduped` or `gemma-2-2b`)
- `--sae-names`: One or more SAE configurations to evaluate (space-separated)

### Example Collection
- `--features`: Number of features to randomly sample (default: 50)
- `--use-saebench-examples`: Use pre-computed SAEBench examples instead of computing fresh (flag, default: compute fresh)
- `--dataset-name`: HuggingFace dataset for fresh collection (default: `monology/pile-uncopyrighted`)
- `--total-tokens`: Total tokens to process for fresh collection (default: 2000000)
- `--context-size`: Context window size (default: 128)

### Generation Settings
- `--llm-model`: OpenAI model for text generation (default: `gpt-4o-mini`)
- `--max-workers`: Parallel API requests (default: 10)
- `--n-top-ex-for-generation`: Top activating examples to use per feature (default: 10)
- `--n-iw-ex-for-generation`: Importance-weighted examples per feature (default: 5)

### Output
- `--output-path`: Custom output directory (default: `evals/{timestamp}`)
- `--verbose`: Print detailed progress information (flag)

## Example Commands

### Basic Evaluation
Evaluate multiple SAEs to compare architectures, widths, or sparsities:
```bash
python src/run.py \
    --model pythia-160m-deduped \
    --sae-names py_4k_btopk_s0 py_4k_gated_s0 py_4k_relu_s0 \
    --features 50
```

### Using Pre-computed SAEBench Examples
By default, the pipeline computes examples fresh from the dataset. To use pre-computed examples from SAEBench for faster evaluation:
```bash
python src/run.py \
    --model pythia-160m-deduped \
    --sae-names py_4k_btopk_s0 py_4k_btopk_s1 py_4k_btopk_s2 \
    --use-saebench-examples \
    --features 100
```

### Evaluating GemmaScope SAEs
GemmaScope SAEs use a different naming scheme based on L0 sparsity:
```bash
python src/run.py \
    --model gemma-2-2b \
    --sae-names gscope_16k_l0_44 \
    --features 50
```

## Output Structure

```
evals/{eval_name}/
├── {sae_name}/
│   ├── feature_examples.csv                      # Activation examples for each feature
│   ├── generated_samples.csv                     # LLM-generated text samples
│   ├── generation_queries_with_responses.csv     # Full LLM prompts and responses
│   ├── generated_samples_with_activations.csv    # Generated samples with SAE activations
│   ├── original_examples_with_activations.csv    # Original examples with SAE activations
│   └── feature_stats.csv                         # Sensitivity metrics (key results)
└── ...
```

### Key Metrics

**feature_stats.csv** contains:
- `feature_index`: SAE feature ID
- `generated_sensitivity`: Fraction of generated samples that activate the feature
- `original_sensitivity`: Fraction of original examples that activate the feature


