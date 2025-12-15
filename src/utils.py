import os

def define_model_name(MODEL_CONFIG) -> tuple[str, str]:
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
    MODEL_CONFIG['model_name'] = model_name
    print("Model Configuration:", MODEL_CONFIG['model_name'])
    OUTPUT_DIR = os.path.join(os.getcwd(), 'models', MODEL_CONFIG['model_name'])

    return MODEL_CONFIG['model_name'], OUTPUT_DIR