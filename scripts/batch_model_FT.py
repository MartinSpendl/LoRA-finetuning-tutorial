#!/usr/bin/env python3
"""
Batch fine-tuning script for Qwen3-0.6B models with different configurations.
This script runs multiple fine-tuning jobs sequentially.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

# Get the script directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
FINETUNE_SCRIPT = PROJECT_DIR / "src" / "finetune_qwen3.py"

# Define 3 different configurations
BASE_MODEL = "Qwen/Qwen3-0.6B"
N_EPOCHS = 5 # 5
LR = 5e-4
LORA_R = 8
LORA_ALPHA = 16
BATCH_SIZE = 16
LORA_PROJECTIONS = "qvko"
LORA_DROPOUT = 0.05
NEW_TOKENS_PATH = None # if None, not used, else "data/injected_tokens.json"
NEW_TOKENS_INIT = "random"
NEW_TOKENS_TRAIN = True

BASE_CONFIG = {
    "base_model": BASE_MODEL,
    "finetuning": True,
    "use_dora": True,
    "n_epochs": N_EPOCHS,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "lr": LR,
    "batch_size": BATCH_SIZE,
    "lora_projections": LORA_PROJECTIONS,
    "lora_dropout": LORA_DROPOUT,
    "new_tokens_path": NEW_TOKENS_PATH,
    "new_tokens_init": NEW_TOKENS_INIT,
    "new_tokens_train": NEW_TOKENS_TRAIN,
}

CONFIGURATIONS = [
    ## Baseline Configuration for Qwen3-0.6B and Qwen3-1.7B
    {
        "base_model": BASE_MODEL,
        "finetuning": False,
    },
    {
        "base_model": "Qwen/Qwen3-1.7B",
        "finetuning": False,
    },
    ## Configuration for Qwen3-0.6B with LoRA (qk, qkvo, gdu, qvkogdu) - Lora 8/16
    {
        "use_dora": False,
        "lora_projections": "qk",
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
    },
    {
        "use_dora": False,
        "lora_projections": "qkvo",
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
    },
    {
        "use_dora": False,
        "lora_projections": "gdu",
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
    },
    {
        "use_dora": False,
        "lora_projections": "qvkogdu",
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
    },
    ## Configuration for Qwen3-0.6B with DoRA (qk, qkvo, gdu, qvkogdu) - DoRA 8/16
    {
        "use_dora": True,
        "lora_projections": "qk",
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
    },
    {
        "use_dora": True,
        "lora_projections": "qkvo",
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
    },
    {
        "use_dora": True,
        "lora_projections": "gdu",
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
    },
    {
        "use_dora": True,
        "lora_projections": "qvkogdu",
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
    },
    ## Configuration for Qwen3-0.6B with LoRA (qvkogdu) - Lora 2/4, 4/8, 8/16, 16/32, 32/64
    {
        "use_dora": False,
        "lora_projections": "qvkogdu",
        "lora_r": 2,
        "lora_alpha": 4,
    },
    {
        "use_dora": False,
        "lora_projections": "qvkogdu",
        "lora_r": 4,
        "lora_alpha": 8,
    },
    {
        "use_dora": False,
        "lora_projections": "qvkogdu",
        "lora_r": 8,
        "lora_alpha": 16,
    },
    {
        "use_dora": False,
        "lora_projections": "qvkogdu",
        "lora_r": 16,
        "lora_alpha": 32,
    },
    {
        "use_dora": False,
        "lora_projections": "qvkogdu",
        "lora_r": 32,
        "lora_alpha": 64,
    },
    ## Configuration for Qwen3-0.6B with DoRA (qvkogdu) - DoRA 2/4, 4/8, 8/16, 16/32, 32/64
    {
        "use_dora": True,
        "lora_projections": "qvkogdu",
        "lora_r": 2,
        "lora_alpha": 4,
    },
    {
        "use_dora": True,
        "lora_projections": "qvkogdu",
        "lora_r": 4,
        "lora_alpha": 8,
    },
    {
        "use_dora": True,
        "lora_projections": "qvkogdu",
        "lora_r": 8,
        "lora_alpha": 16,
    },
    {
        "use_dora": True,
        "lora_projections": "qvkogdu",
        "lora_r": 16,
        "lora_alpha": 32,
    },
    {
        "use_dora": True,
        "lora_projections": "qvkogdu",
        "lora_r": 32,
        "lora_alpha": 64,
    },
]
CONFIGURATIONS = [BASE_CONFIG | config for config in CONFIGURATIONS]

def generate_model_name(MODEL_CONFIG):
    model_name = MODEL_CONFIG['base_model'].split('/')[-1]
    if MODEL_CONFIG['finetuning']:
        model_name += "_" + "_".join([
        "DoRA" if MODEL_CONFIG['use_dora'] else "LoRA",
        "".join([x[0] for x in MODEL_CONFIG['lora_projections']]),
        f"r{MODEL_CONFIG['lora_r']}",
        f"alpha{MODEL_CONFIG['lora_alpha']}",
        f"drop{MODEL_CONFIG['lora_dropout']}",
        f"proj({''.join([x[0] for x in MODEL_CONFIG['lora_projections']])})",
        f"bs{MODEL_CONFIG['batch_size']}",
        f"lr{MODEL_CONFIG['lr']}",
        f"ep{MODEL_CONFIG['n_epochs']}",
        f"ntinit({MODEL_CONFIG['new_tokens_init']})" if MODEL_CONFIG['new_tokens_path'] is not None else "",
        f"nttrain" if MODEL_CONFIG['new_tokens_path'] is not None and MODEL_CONFIG['new_tokens_train'] else ""
    ])
    return model_name

def build_command(config):
    """Build the command to run fine-tuning."""
    cmd = ["python", str(FINETUNE_SCRIPT)]
    
    for key, value in config["args"].items():
        arg_name = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(arg_name)
        elif value is None:
            pass
        else:
            cmd.extend([arg_name, str(value)])
    
    return cmd


def run_configuration(config, config_num, total_configs):
    """Run a single configuration."""
    print(f"\n{'='*80}")
    print(f"Running Configuration {config_num}/{total_configs}: {config['name']}")
    print(f"{'='*80}")
    print(f"Arguments: {json.dumps(config['args'], indent=2)}")
    print(f"{'='*80}\n")
    
    cmd = build_command(config)
    print(f"Command: {' '.join(cmd)}\n")
    
    # Change to project directory
    os.chdir(PROJECT_DIR)
    
    # Run the command
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Let output go to stdout/stderr for SLURM
            text=True
        )
        print(f"\n✓ Configuration {config_num} ({config['name']}) completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Configuration {config_num} ({config['name']}) failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ Configuration {config_num} ({config['name']}) failed with exception: {e}")
        return False


def main():
    """Main function to run all configurations."""
    print("="*80)
    print("Batch Fine-tuning Script for Qwen3-0.6B")
    print(f"Total configurations: {len(CONFIGURATIONS)}")
    print("="*80)
    
    # Verify the fine-tuning script exists
    if not FINETUNE_SCRIPT.exists():
        print(f"Error: Fine-tuning script not found at {FINETUNE_SCRIPT}")
        sys.exit(1)
    
    # Track results
    results = []
    
    # Run each configuration
    configuration = [
        {"name": generate_model_name(config), "args": config}
        for config in CONFIGURATIONS
    ]
    for idx, config in enumerate(configuration, start=1):
        success = run_configuration(config, idx, len(configuration))
        results.append({
            "config": config["name"],
            "success": success
        })
    
    # Print summary
    print("\n" + "="*80)
    print("BATCH FINE-TUNING SUMMARY")
    print("="*80)
    for result in results:
        status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
        print(f"{result['config']}: {status}")
    print("="*80)
    
    # Exit with error if any configuration failed
    if not all(r["success"] for r in results):
        print("\nWarning: Some configurations failed!")
        sys.exit(1)
    else:
        print("\nAll configurations completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
