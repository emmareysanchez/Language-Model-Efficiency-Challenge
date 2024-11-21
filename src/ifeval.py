from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import os


model_name = './models/model6/output_lima/checkpoint-300'

# Step 1: Load the tokenizer and model with quantization
# model_name = "models/model1"  # Near 3B model (smallest available Qwen model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device,
    trust_remote_code=True
)
model.config.pad_token_id = tokenizer.eos_token_id

# Step 2: Load the google/IFEval dataset
ifeval = load_dataset("google/IFEval")
# Select only the first 5 samples for testing
tests = ifeval["train"].select(list(range(5)))

# Step 3: Generate predictions on the dataset
os.makedirs('responses/tests', exist_ok=True)
num_model = model_name.split('/')[-3][-1]
output_file = f"responses/tests/model_{num_model}_responses_test4.json"
# output_file = f"responses/dataset/model_responses{num_model}.json"
with open(output_file, 'w', encoding='utf-8') as f_out:
    for sample in tqdm(tests):
    # for sample in tqdm(dataset['train']):   # Use 'validation' or 'train' split if 'test' is not available
        input_text = sample['prompt']  # Adjust the field name based on the dataset's structure

        # Prepare the input prompt
        prompt = input_text

        # Tokenize input
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate output
        outputs = model.generate(
            inputs,
            max_length=256,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Since the model may include the prompt in its output, we extract the generated response
        response = generated_text[len(prompt):]

        # Prepare the JSON object
        json_obj = {
            "prompt": prompt,
            "response": response
        }

        # Write the JSON object to file
        f_out.write(json.dumps(json_obj) + '\n')

        if len(open(output_file).readlines()) == 5:
            break
