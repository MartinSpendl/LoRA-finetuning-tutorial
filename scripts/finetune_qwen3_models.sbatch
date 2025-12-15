#!/bin/bash
#SBATCH --job-name=orange_qwen3_ft
#SBATCH --output=logs/qwen3_finetune_%j.out
#SBATCH --error=logs/qwen3_finetune_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=frida
#SBATCH --gres=gpu:L4
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

srun --container-image=/shared/home/martin.spendl/dev/Dockerfiles/files/llm-finetuning.sqfs \
  --container-mount-home \
  --job-name=orange-ft \
  sh -c 'export DEBIAN_FRONTEND=noninteractive; \
         echo "Where are you now: $(pwd)"; \
         cd ../root/dev/Tutorials/LoRA-finetuning-tutorial && \
         python3 scripts/batch_model_FT.py'