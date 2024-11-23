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

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the dataset
dataset = load_dataset("google/IFEval")

# Model name
model_name = "models/model6/output_lima/checkpoint-300"

# Load the tokenizer for Mistral
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    add_eos_token=True,
    use_fast=True,
    padding_side='right',
)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to EOS token

# Quantization configuration using bitsandbytes library
compute_dtype = getattr(torch, "bfloat16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

# Load the pre-trained model with the specified quantization configuration
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
model.config.pad_token_id = tokenizer.pad_token_id  # Set the model's padding token ID

# Disable gradients to save memory and computation
model.eval()
torch.set_grad_enabled(False)  # Disable gradient computation globally

# Prepare the output file
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "input_response_data.jsonl")

# Batch processing
batch_size = 8  # Adjust based on your GPU memory capacity
max_length = 128  # Limit output length to avoid excessive memory usage

with open(output_file, 'w') as f:
    # Process in batches
    for i in tqdm(range(0, len(dataset['train']), batch_size), desc="Processing Batches", unit="batch"):
        try:
            if (i + batch_size) > len(dataset['train']):
                batch_prompts = dataset['train']['prompt'][i:len(dataset['train'])]
            else:
                batch_prompts = dataset['train']['prompt'][i:i + batch_size]
            # Tokenize inputs
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            
            # Generate responses
            with torch.no_grad():  # Ensure gradients are disabled during generation
                if max_length:
                    outputs = model.generate(**inputs, max_new_tokens=max_length)
                else:
                    outputs = model.generate(**inputs)
            # Decode responses
            responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

            # Write each response directly to the file
            for prompt, response in zip(batch_prompts, responses):
                f.write(json.dumps({"prompt": prompt, "response": response}) + '\n')
        except Exception as e:
            print(f"Error processing batch {i}: {e}")

print(f"Responses saved to {output_file}")