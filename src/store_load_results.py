import os
import json

def store_results(results, model_config):
    results_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, f'{model_config["model_name"]}_results.json'), 'w') as f:
        json.dump(model_config | results, f, indent=4)

def load_results(model_config):
    results_dir = os.path.join(os.getcwd(), 'results')
    with open(os.path.join(results_dir, f'{model_config["model_name"]}_results.json'), 'r') as f:
        return json.load(f)
    