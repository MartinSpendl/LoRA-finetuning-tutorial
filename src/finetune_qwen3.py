import os
import torch
import json
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer

from argparse import ArgumentParser
from store_load_results import store_results, load_results
from evaluation_function import evaluate_model
from utils import define_model_name

def main(MODEL_CONFIG:dict):

    TRAINDATA_FILE = os.path.join(os.getcwd(), 'data', 'train_test_dataset', 'orange_qa_train.jsonl')
    TESTDATA_MCQ_FILE = os.path.join(os.getcwd(), 'data', 'train_test_dataset', 'orange_qa_MCQ_test.jsonl')
    TESTDATA_MCQ_CON_FILE = os.path.join(os.getcwd(), 'data', 'train_test_dataset', 'orange_qa_MCQ-con_test.jsonl')

    # 1. Configuration Setup
    model_name, OUTPUT_DIR = define_model_name(MODEL_CONFIG)

    # 2. Check if model already trained
    alread_trained = False
    try:
        loaded_results = load_results(MODEL_CONFIG)
        if "accuracy_mcq" in loaded_results and "accuracy_mcq_con" in loaded_results:
            alread_trained = True
        if alread_trained:
            print("Model already trained, skipping training...")
            return
    except Exception as e:
        print(f"Error loading results: {e}")
        pass

    if not os.path.exists(OUTPUT_DIR) and MODEL_CONFIG['finetuning']:

        # 3. Load Train Dataset
        print("Loading training dataset from:", TRAINDATA_FILE)
        dataset = load_dataset("json", data_files=TRAINDATA_FILE, split="train")

        # 4. Load Model & Tokenizer
        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['base_model'])
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIG['base_model'],
            dtype=torch.float16,       # Use float16 to save memory
            device_map="auto",          # Auto-selects GPU or CPU
            do_sample=False,
        )

        from transformers import AddedToken

        # 5. Extend token library (tokenizer + model)
        if MODEL_CONFIG['new_tokens_path'] is not None:
            print("Adding new tokens from:", MODEL_CONFIG['new_tokens_path'])
            DEFAULT_TOKEN_NUM = len(tokenizer)

            with open(MODEL_CONFIG['new_tokens_path'], 'r') as f:
                new_tokens_text = json.load(f)
            new_tokens = [
                AddedToken(token, lstrip=True, rstrip=True)
                for token in new_tokens_text
            ]

            ## extend tokenizer and model
            print("Number of default tokenizer tokens:", DEFAULT_TOKEN_NUM)
            tokenizer.add_tokens(new_tokens)
            model.resize_token_embeddings(len(tokenizer))
            print("Number of new tokenizer tokens:", len(tokenizer))

            new_tokens_ids = [
                tokenizer.encode(token)[0]
                for token in new_tokens_text
            ]
            old_tokens_ids = [id for id in range(DEFAULT_TOKEN_NUM) if id not in new_tokens_ids]

            ## initialize new token embeddings
            model.model.embed_tokens = model.model.embed_tokens.float()

            if MODEL_CONFIG['new_tokens_init'] == "average":
                with torch.no_grad():
                    existing_embeddings = model.model.embed_tokens.weight[:DEFAULT_TOKEN_NUM, :]
                    average_embedding = existing_embeddings.mean(dim=0)
                    for i in range(DEFAULT_TOKEN_NUM, len(tokenizer)):
                        model.model.embed_tokens.weight[i, :] = average_embedding
            elif MODEL_CONFIG['new_tokens_init'] == "zero":
                with torch.no_grad():
                    model.model.embed_tokens[DEFAULT_TOKEN_NUM:, :].fill_(0.0)
            elif MODEL_CONFIG['new_tokens_init'] == "random":
                pass  # already quazi randomly initialized
            else:
                raise ValueError(f"Unknown new_tokens_init method: {MODEL_CONFIG['new_tokens_init']}")

            ## setup trainable weights for new tokens
            if MODEL_CONFIG['new_tokens_train']:
                def zero_out_old_token_grads(grad):
                    new_grad = grad.clone()
                    new_grad[old_tokens_ids, :] = 0.0
                    return new_grad
                model.model.embed_tokens.weight.requires_grad = True
                model.model.embed_tokens.weight.register_hook(zero_out_old_token_grads)
            model.lm_head.weight.requires_grad = True
        
        # 6. Setup LoRA/DoRA and merge with base model
        if MODEL_CONFIG['new_tokens_path'] is not None:
            modules_to_save = ["lm_head"] + (["embed_tokens"] if MODEL_CONFIG['new_tokens_train'] else [])
        else:
            modules_to_save = None
        peft_config = LoraConfig(
            bias="none",
            task_type="CAUSAL_LM",
            # from config
            target_modules=MODEL_CONFIG['lora_projections'],
            use_dora=MODEL_CONFIG['use_dora'],
            r=MODEL_CONFIG['lora_r'],
            lora_alpha=MODEL_CONFIG['lora_alpha'],
            lora_dropout=MODEL_CONFIG['lora_dropout'],
            modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, peft_config)

        # 7. Train arguments
        # Setup wandb project name (priority: command line arg > environment variable > default)
        wandb_project = MODEL_CONFIG.get('wandb_project') or os.environ.get("WANDB_PROJECT", "qwen3-lora-finetuning")
        wandb_run_name = MODEL_CONFIG['model_name']
        
        # Set wandb project via environment variable
        os.environ["WANDB_PROJECT"] = wandb_project
        
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=MODEL_CONFIG['n_epochs'],          # How many times to read the docs
            per_device_train_batch_size=MODEL_CONFIG['batch_size'], 
            gradient_accumulation_steps=1,
            learning_rate=MODEL_CONFIG['lr'],
            fp16=True,                   # Use mixed precision
            logging_steps=10,
            optim="adamw_torch",   
            save_strategy="epoch",       # Save a checkpoint every epoch
            report_to=["wandb"],         # Enable wandb logging
            run_name=wandb_run_name,     # Set run name to model name
        )

        # 8. Initialize Trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
            processing_class=tokenizer,
        )

        # 9. Start Training
        print("Starting training...")
        trainer.train()

        # 10. Save model and tokenizer
        print(f"Saving model to {OUTPUT_DIR}...")
        trainer.model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 11. Load model and tokenizer
    if MODEL_CONFIG['finetuning']:
        tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
        # Load base model first, then load PEFT adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIG['base_model'], 
            device_map="auto", 
            torch_dtype=torch.float16
        )
        base_model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_CONFIG['base_model'], device_map="auto", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['base_model'], trust_remote_code=True)

    # 12. Evaluate and store results
    with open(TESTDATA_MCQ_FILE, "r") as f:
        test_mcq_dataset = json.load(f)
    accuracy_mcq, se_mcq = evaluate_model(model, tokenizer, test_mcq_dataset, batch_size=MODEL_CONFIG['batch_size'])

    with open(TESTDATA_MCQ_CON_FILE, "r") as f:
        test_mcq_con_dataset = json.load(f)
    accuracy_mcq_con, se_mcq_con = evaluate_model(model, tokenizer, test_mcq_con_dataset, batch_size=MODEL_CONFIG['batch_size'])
    results = {
        "accuracy_mcq": accuracy_mcq,
        "se_mcq": se_mcq,
        "accuracy_mcq_con": accuracy_mcq_con,
        "se_mcq_con": se_mcq_con,
    }
    print("Evaluation Results:", results)
    store_results(results, MODEL_CONFIG)
    print("Results stored successfully.")

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--base-model', type=str, default="Qwen/Qwen3-0.6B", help='base model name or path')
    parser.add_argument('--finetuning', action='store_true', help='whether to finetune the model')
    parser.add_argument('--use-dora', action='store_true', help='whether to use DoRA (else LoRA)')
    parser.add_argument('--n-epochs', type=int, default=3, help='number of training epochs')
    parser.add_argument('--lora-r', type=int, default=8, help='LoRA/DoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=16, help='LoRA/DoRA alpha')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--lora-projections', type=str, default="qv", help='list of projection layers to apply LoRA/DoRA')
    parser.add_argument('--lora-dropout', type=float, default=0.05, help='LoRA/DoRA dropout rate')
    parser.add_argument('--new-tokens-path', type=str, default=None, help='path to new tokens file')
    parser.add_argument('--new-tokens-init', type=str, default="random", help='initialization method for new tokens (random, average, zero)')
    parser.add_argument('--new-tokens-train', action='store_true', help='whether to train new tokens or not')
    parser.add_argument('--wandb-project', type=str, default="Orange-LoRA-tutorial", help='wandb project name (default: qwen3-lora-finetuning)')
    args = parser.parse_args()

    PROJECTIONS = {
        "q": "q_proj",
        "k": "k_proj",
        "v": "v_proj",
        "o": "o_proj",
        "g": "gate_proj",
        "d": "down_proj",
        "u": "up_proj"
    }
    projections = [PROJECTIONS[p] for p in list(args.lora_projections)]
    
    MODEL_CONFIG = {
        "base_model": args.base_model,
        "finetuning": args.finetuning,  # True if fine-tuning, False if base_model
        "use_dora": args.use_dora, # True if DoRA, False if LoRA
        "n_epochs": args.n_epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "lora_projections": projections,
        "lora_dropout": args.lora_dropout,
        "new_tokens_path":  args.new_tokens_path,
        "new_tokens_init": args.new_tokens_init,
        "new_tokens_train": args.new_tokens_train,
        "wandb_project": args.wandb_project,  # wandb project name
    }
    print("Model Configuration:", MODEL_CONFIG)

    main(MODEL_CONFIG)
