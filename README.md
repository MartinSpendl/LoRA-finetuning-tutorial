# LoRA Tutorial: Fine-tuning Qwen3 Models for Orange Data Mining QA

A comprehensive tutorial on fine-tuning large language models using **LoRA** (Low-Rank Adaptation) and **DoRA** (Weight-Decomposed Low-Rank Adaptation) techniques. This project demonstrates how to adapt Qwen3 models to answer questions about the Orange Data Mining software.

## Table of Contents

- [Tutorial Overview](#tutorial-overview)
- [What is LoRA/DoRA?](#what-is-loradora)
- [Repository Overview](#repository-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Workflow for full pipeline (not covered in the hands-on part)](#workflow-for-full-pipeline-not-covered-in-the-hands-on-part)

## Tutorial Overview
The tutorial is divided into three parts:

1. **Literature review part**: Covers the LLM finetuning landscape, dives deeper into LoRA and DoRA.

2. **Dataset preparation part**: Covers the first two steps (extracting documentation and formatting widget connections) in a very high level with slides.

3. **Hands-on part**: Covers the third and fourth step (finetuning and evaluation). We will be running the code in the notebooks (separate notebook for demonstration) and scripts provided in the repository. Meanwhile, we will also check the results for other configurations and comment of the results.

## Repository Overview

This tutorial provides a complete pipeline for:

1. **Data Preparation**: Extracting and formatting documentation from Orange Data Mining repositories
2. **Question-Answer Dataset Creation**: Creating training and test datasets with multiple-choice and open-ended questions
3. **Model Fine-tuning**: Applying LoRA/DoRA adapters to Qwen3 models
4. **Evaluation**: Comparing base and fine-tuned model performance

The project uses **Qwen3-0.6B** and **Qwen3-1.7B** as base models and fine-tunes them on Orange Data Mining documentation to improve their ability to answer domain-specific questions.

Hands-on part only covers the third and fourth step (finetuning and evaluation), and describes the first two steps in a very high level with slides.

## What is LoRA/DoRA?

### LoRA (Low-Rank Adaptation)

LoRA is a parameter-efficient fine-tuning technique that:
- Freezes the original model weights
- Adds trainable low-rank matrices to specific layers (attention projections, MLP layers)
- Reduces trainable parameters by 100-1000x compared to full fine-tuning
- Maintains model performance while being memory-efficient

### DoRA (Weight-Decomposed Low-Rank Adaptation)

DoRA is an enhanced version of LoRA that:
- Decomposes weight updates into magnitude and direction components
- Provides better learning dynamics than standard LoRA
- Often achieves better performance with the same number of parameters
- Is particularly effective for smaller models

## Project Structure

```
LoRA-tutorial/
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── data/                          # Data files
│   ├── train_test_dataset/        # Training and test datasets
│   │   ├── orange_qa_train.jsonl
│   │   ├── orange_qa_MCQ_test.jsonl
│   │   └── orange_qa_MCQ-con_test.jsonl
│   ├── orange_qa_full.json        # Full QA dataset
│   ├── injected_tokens.json       # Custom tokens (optional)
│   └── ...
├── notebooks/                     # Jupyter notebooks for workflow
│   ├── 1.0-ms-extract-documentation.ipynb
│   ├── 2.0-ms-format-widget-connections.ipynb
│   ├── 2.1-ms-format_QA_data.ipynb
│   ├── 3.0-ms-evaluate-models.ipynb
│   ├── 4.0-ms-finetune-Qwen3.ipynb
│   ├── 4.1-ms-finetune-Qwen3+token-injection.ipynb
│   └── 5.0-ms-evaluate-finetuned-models.ipynb
├── src/                          # Python scripts
│   ├── finetune_qwen3.py         # Main fine-tuning script
│   ├── evaluation_function.py    # Model evaluation utilities
│   └── store_load_results.py     # Results management
├── scripts/                      # Batch processing scripts
│   ├── batch_model_FT.py         # Batch fine-tuning
│   └── finetune_qwen3.sbatch     # SLURM batch script
├── models/                       # Trained model adapters
│   └── orange_qa_finetuned_*/    # Fine-tuned model directories
├── results/                      # Evaluation results
│   └── *_results.json            # JSON files with accuracy metrics
└── repositories/                 # Orange Data Mining source code (not included in the repository)
    ├── orange-canvas-core/
    ├── orange-widget-base/
    └── ...
```

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended)
- Colab GPU instance is enough

## Installation

1. **Clone the repository** (if not already done):
```bash
git clone <repository-url>
cd LoRA-tutorial
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**:
```bash
pip install -r requirements.txt
```

   Or install manually:
```bash
pip install torch transformers datasets peft trl accelerate
pip install jupyter notebook ipywidgets  # For running notebooks
pip install numpy pandas tqdm
pip install wandb  # For experiment tracking (optional)
```

## Workflow for full pipeline (not covered in the hands-on part)

The tutorial follows a step-by-step workflow using Jupyter notebooks. Each notebook builds upon the previous one.

### Step 1: Extract Documentation

**Notebook**: `notebooks/1.0-ms-extract-documentation.ipynb`

This notebook extracts documentation from Orange Data Mining repositories:
- Parses Python docstrings from source files
- Extracts markdown and RST documentation files
- Combines all documentation into a single text file

**Output**: `data/all_documentation.txt`

**To run**:
```bash
jupyter notebook notebooks/1.0-ms-extract-documentation.ipynb
```

### Step 2: Format Widget Connections

**Notebook**: `notebooks/2.0-ms-format-widget-connections.ipynb`

This notebook processes widget connection information:
- Extracts widget input/output relationships
- Formats connection data for QA generation

**Output**: `data/orange_widgets_connections.json`

### Step 3: Format QA Data

**Notebook**: `notebooks/2.1-ms-format_QA_data.ipynb`

This notebook creates the training and evaluation datasets:
- Combines QA questions from multiple sources
- Formats questions into chat template format
- Splits data into training and test sets
- Creates multiple-choice (MCQ) and open-ended (QA) question formats

**Outputs**:
- `data/train_test_dataset/orange_qa_train.jsonl`
- `data/train_test_dataset/orange_qa_MCQ_test.jsonl`
- `data/train_test_dataset/orange_qa_MCQ-con_test.jsonl`

**Key Features**:
- System prompts for consistent formatting
- Chat template formatting compatible with Qwen3
- Support for both MCQ and text-based questions

### Step 4: Evaluate Base Models

**Notebook**: `notebooks/3.0-ms-evaluate-models.ipynb`

Before fine-tuning, evaluate the base models to establish a baseline:
- Loads Qwen3-0.6B or Qwen3-1.7B
- Evaluates on test datasets
- Calculates accuracy metrics

**Usage**:
1. Set `MODEL_CONFIG` in the notebook
2. Run all cells
3. Review baseline accuracy scores

### Step 5: Fine-tune Models

**Notebooks**: 
- `notebooks/4.0-ms-finetune-Qwen3.ipynb` - Standard fine-tuning
- `notebooks/4.1-ms-finetune-Qwen3+token-injection.ipynb` - Fine-tuning with custom tokens

These notebooks demonstrate fine-tuning with LoRA/DoRA:

**Key Configuration Options**:
- **Base Model**: `Qwen/Qwen3-0.6B` or `Qwen/Qwen3-1.7B`
- **Method**: LoRA or DoRA
- **Target Modules**: Which layers to adapt (q_proj, k_proj, v_proj, o_proj, gate_proj, down_proj, up_proj)
- **Rank (r)**: Low-rank dimension (typically 2-32)
- **Alpha**: Scaling factor (typically 2× rank)
- **Dropout**: LoRA dropout rate (typically 0.05-0.1)
- **Learning Rate**: Training learning rate (typically 1e-4 to 5e-4)
- **Batch Size**: Training batch size
- **Epochs**: Number of training epochs

**Example Configuration**:
```python
MODEL_ID = "Qwen/Qwen3-0.6B"
peft_config = LoraConfig(
    r=8,                    # Rank
    lora_alpha=16,          # Alpha (typically 2× rank)
    lora_dropout=0.05,      # Dropout
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    use_dora=False,         # Set to True for DoRA
)
```

**Output**: Trained adapter saved in `models/orange_qa_finetuned_*/`

### Step 6: Evaluate Fine-tuned Models

**Notebook**: `notebooks/5.0-ms-evaluate-finetuned-models.ipynb`

Compare fine-tuned models against baselines:
- Loads fine-tuned adapters
- Evaluates on test sets
- Compares accuracy improvements
- Visualizes results

## Using the Scripts

### Fine-tuning Script

The `src/finetune_qwen3.py` script provides a command-line interface for fine-tuning:

**Basic Usage**:
```bash
python src/finetune_qwen3.py \
    --base-model Qwen/Qwen3-0.6B \
    --finetuning \
    --use-dora \
    --lora-r 8 \
    --lora-alpha 16 \
    --lr 5e-4 \
    --batch-size 16 \
    --n-epochs 5 \
    --lora-projections qkvo \
    --lora-dropout 0.05
```

**Arguments**:
- `--base-model`: Base model identifier (default: `Qwen/Qwen3-0.6B`)
- `--finetuning`: Enable fine-tuning (flag)
- `--use-dora`: Use DoRA instead of LoRA (flag)
- `--lora-r`: LoRA rank (default: 8)
- `--lora-alpha`: LoRA alpha (default: 16)
- `--lr`: Learning rate (default: 2e-4)
- `--batch-size`: Batch size (default: 4)
- `--n-epochs`: Number of epochs (default: 3)
- `--lora-projections`: Target modules as string (e.g., `qkvo`, `qvkogdu`)
- `--lora-dropout`: Dropout rate (default: 0.05)
- `--new-tokens-path`: Path to custom tokens JSON file (optional)
- `--new-tokens-init`: Initialization method: `random`, `average`, or `zero` (default: `random`)
- `--new-tokens-train`: Enable training of new token embeddings (flag)
- `--wandb-project`: Weights & Biases project name (optional)

**Projection Codes**:
- `q` = `q_proj` (query projection)
- `k` = `k_proj` (key projection)
- `v` = `v_proj` (value projection)
- `o` = `o_proj` (output projection)
- `g` = `gate_proj` (gate projection in MLP)
- `d` = `down_proj` (down projection in MLP)
- `u` = `up_proj` (up projection in MLP)

**Example**: `qkvo` applies LoRA to all attention projections.

### Batch Fine-tuning

The `scripts/batch_model_FT.py` script runs multiple configurations sequentially:

```bash
python scripts/batch_model_FT.py
```

This script:
- Trains baseline models (no fine-tuning)
- Trains LoRA models with different projection combinations
- Trains DoRA models with different projection combinations
- Trains models with different rank values
- Automatically evaluates each model
- Stores results in `results/`

**Modify Configurations**: Edit `CONFIGURATIONS` list in `scripts/batch_model_FT.py` to customize which models to train.

## Understanding Results

Results are stored in `results/` as JSON files named `{model_name}_results.json`.

**Result Structure**:
```json
{
    "base_model": "Qwen/Qwen3-0.6B",
    "finetuning": true,
    "use_dora": false,
    "lora_r": 8,
    "lora_alpha": 16,
    "accuracy_mcq": 45.2,
    "se_mcq": 1.5,
    "accuracy_mcq_con": 38.7,
    "se_mcq_con": 1.4,
    ...
}
```

**Metrics**:
- `accuracy_mcq`: Accuracy on multiple-choice questions (%)
- `se_mcq`: Standard error for MCQ accuracy
- `accuracy_mcq_con`: Accuracy on connection-based MCQ questions (%)
- `se_mcq_con`: Standard error for connection MCQ accuracy

**Model Naming Convention**:
```
{base_model}_{method}_{projections}_r{rank}_alpha{alpha}_drop{dropout}_proj({projections})_bs{batch_size}_lr{lr}_ep{epochs}
```

Example: `Qwen3-0.6B_LoRA_qkvo_r8_alpha16_drop0.05_proj(qkvo)_bs16_lr0.0005_ep5`

## Model Configurations

### Recommended Configurations

**For Quick Experiments**:
- Base Model: `Qwen/Qwen3-0.6B`
- Method: LoRA
- Projections: `qkvo` (attention layers)
- Rank: 8, Alpha: 16
- Learning Rate: 5e-4
- Batch Size: 16
- Epochs: 5

**For Better Performance**:
- Base Model: `Qwen/Qwen3-1.7B`
- Method: DoRA
- Projections: `qvkogdu` (attention + MLP layers)
- Rank: 16, Alpha: 32
- Learning Rate: 2e-4
- Batch Size: 8
- Epochs: 5

**For Maximum Efficiency**:
- Base Model: `Qwen/Qwen3-0.6B`
- Method: LoRA
- Projections: `qk` (minimal attention layers)
- Rank: 4, Alpha: 8
- Learning Rate: 5e-4
- Batch Size: 32
- Epochs: 3

### Projection Selection Guide

- **`qk`**: Minimal, fastest training, lower performance
- **`qkvo`**: Attention layers only, good balance
- **`gdu`**: MLP layers only, different learning dynamics
- **`qvkogdu`**: All layers, best performance, slower training

## Advanced Features

### Custom Token Injection

You can inject domain-specific tokens to improve model understanding:

1. **Create tokens file** (`data/injected_tokens.json`):
```json
["<widget_name>", "<orange_feature>", ...]
```

2. **Fine-tune with tokens**:
```bash
python src/finetune_qwen3.py \
    --finetuning \
    --new-tokens-path data/injected_tokens.json \
    --new-tokens-init average \
    --new-tokens-train
```

**Initialization Methods**:
- `random`: Random initialization (default)
- `average`: Initialize with average of existing embeddings
- `zero`: Initialize with zeros

### Weights & Biases Integration

Track experiments with W&B:

```bash
export WANDB_PROJECT="orange-lora-tutorial"
python src/finetune_qwen3.py --finetuning --wandb-project orange-lora-tutorial
```

### SLURM Batch Jobs

For cluster environments, use `scripts/finetune_qwen3.sbatch`:

```bash
sbatch scripts/finetune_qwen3.sbatch
```

Modify the script to set your desired configuration.

## Troubleshooting

### Out of Memory Errors

- Reduce `--batch-size`
- Use a smaller base model (`Qwen3-0.6B` instead of `Qwen3-1.7B`)
- Reduce `--lora-r` (e.g., from 8 to 4)
- Use fewer projection layers (e.g., `qk` instead of `qvkogdu`)

### Low Accuracy

- Increase training epochs (`--n-epochs`)
- Try DoRA instead of LoRA (`--use-dora`)
- Increase rank (`--lora-r`) and alpha (`--lora-alpha`)
- Use more projection layers (`qvkogdu` instead of `qk`)

### Slow Training

- Increase `--batch-size` (if memory allows)
- Use gradient accumulation
- Reduce number of epochs
- Use fewer projection layers

### Model Not Loading

- Ensure adapter files exist in `models/`
- Check that `adapter_config.json` is present
- Verify base model name matches configuration
