import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from tqdm import tqdm
import json
from peft import prepare_model_for_kbit_training, PeftConfig, PeftModel

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the dataset
dataset = load_dataset("google/IFEval")

model_name = 'mistralai/Mistral-7B-v0.3'
checkpoint_path = '../models/model10/output_lima/checkpoint-100'
model_name = "Qwen/Qwen2.5-7B"
checkpoint_path = "../models/model11/output_lima/checkpoint-1000"

# Step 1: Load the tokenizer and model with quantization
# model_name = "models/model1"  # Near 3B model (smallest available Qwen model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_fast=True,
    padding_side='left'
)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Enable loading the model in 4-bit precision
    bnb_4bit_quant_type="nf4",            # Specify quantization type as Normal Float 4
    bnb_4bit_compute_dtype=getattr(torch, "bfloat16"), # Set computation data type
    bnb_4bit_use_double_quant=True,       # Use double quantization for better accuracy
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model = prepare_model_for_kbit_training(model)

peft_config = PeftConfig.from_pretrained(checkpoint_path)
model = PeftModel.from_pretrained(model, checkpoint_path)
model.config.pad_token_id = tokenizer.eos_token_id

# Disable gradients to save memory and computation
model.eval()
torch.set_grad_enabled(False)  # Disable gradient computation globally

# Prepare the output file
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)
num_model = checkpoint_path.split('/')[-3].split("model")[-1]
output_file = os.path.join(output_dir, f"input_response_data{num_model}.jsonl")

# Batch processing
batch_size = 8  # Adjust based on your GPU memory capacity
max_length = 128  # Limit output length to avoid excessive memory usage


def tokenize_batch_prompt(batch_prompts):
    # We add at the beginning of each prompt the token for the end of the previous prompt "<Prompt>:"
    # and at the end " \n<Response>:"
    batch_prompts = [f"<Prompt>: {prompt} \n<Response>:" for prompt in batch_prompts]
    return tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)


with open(output_file, 'w') as f:
    # Process in batches
    for i in tqdm(range(0, len(dataset['train']), batch_size), desc="Processing Batches", unit="batch"):
        try:
            if (i + batch_size) > len(dataset['train']):
                batch_prompts = dataset['train']['prompt'][i:len(dataset['train'])]
            else:
                batch_prompts = dataset['train']['prompt'][i:i + batch_size]
            # Tokenize inputs
            # inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            inputs = tokenize_batch_prompt(batch_prompts)
            
            # Generate responses
            with torch.no_grad():  # Ensure gradients are disabled during generation
                if max_length:
                    outputs = model.generate(**inputs, max_new_tokens=max_length)
                else:
                    outputs = model.generate(**inputs)
            # Decode responses and remove the prompt
            responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            for j in range(len(responses)):
                responses[j] = responses[j][len(f"<Prompt>: {batch_prompts[j]} \n<Response>:"):]
            # Write each response directly to the file
            for prompt, response in zip(batch_prompts, responses):
                f.write(json.dumps({"prompt": prompt, "response": response}) + '\n')
        except Exception as e:
            print(f"Error processing batch {i}: {e}")

print(f"Responses saved to {output_file}")
