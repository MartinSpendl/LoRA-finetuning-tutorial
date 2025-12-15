import numpy as np
import torch
from tqdm import tqdm

def evaluate_model(model, tokenizer, dataset, batch_size=32) -> tuple[float, float]:
    correct = []
    batches = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]

    model.eval()

    with torch.no_grad():
        for batch in tqdm(batches, desc="Evaluating model"):
            batch_texts = []
            batch_assistant_messages = []
            
            for item in batch:
                system_user_messages = [
                    {"role": item["messages"][0]["role"], "content": item["messages"][0]["content"]},
                    {"role": item["messages"][1]["role"], "content": item["messages"][1]["content"] + "\n /no_think"}
                ]
                assistant_messages = item["messages"][2] # assistant messages
                batch_assistant_messages.append(assistant_messages)
                
                text = tokenizer.apply_chat_template(
                    system_user_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_tracking=False
                )
                batch_texts.append(text)
            
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=1000,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            for i, (output, assistant_messages) in enumerate(zip(outputs, batch_assistant_messages)):
                input_length = inputs.input_ids[i].shape[0]
                generated_token_ids = output[input_length:]
                decoded_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()
                
                prediction = decoded_text.split("</think>")[1].strip() if "</think>" in decoded_text else decoded_text.strip()
                correct.append(prediction == assistant_messages["content"])

    correct = np.array(correct) * 100
    accuracy = np.mean(correct)
    se = np.std(correct) / np.sqrt(len(correct))
    return accuracy, se
