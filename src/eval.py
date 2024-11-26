from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json
from tqdm import tqdm
from peft import prepare_model_for_kbit_training, PeftConfig, PeftModel
import os


model_name = 'mistralai/Mistral-7B-v0.3'
checkpoint_path = './models/model10/output_lima/checkpoint-100'

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
model.config.pad_token_id = tokenizer.eos_token_id

peft_config = PeftConfig.from_pretrained(checkpoint_path)
model = PeftModel.from_pretrained(model, checkpoint_path)

# Step 2: Load the google/IFEval dataset
tests = [
    {'prompt': 'What is the capital of France?', 'response': 'Paris'}, 
    # {'prompt': 'Write the days of the week in English.', 'response': 'Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday'},
    # {'prompt': '¿Cuál es la diferencia entre echo, print, print_r, var_dump y var_export en PHP?', 'response': 'echo: Muestra una o más cadenas de texto. print: Muestra una cadena de texto. print_r: Muestra información sobre una variable de forma legible. var_dump: Muestra información sobre una variable de forma legible. var_export: Muestra una representación de una variable que puede ser utilizada como código PHP.'},
    # {'prompt': 'Tell me a story in English about a boy and a girl whoe went on a date.', 'response': 'Once upon a time'},
    # {'prompt': 'Cuales son los planetas del sistema solar en español?', 'response': 'Mercurio, Venus, Tierra, Marte, Júpiter, Saturno, Urano, Neptuno'}
]

# Step 3: Generate predictions on the dataset
os.makedirs('responses/tests', exist_ok=True)
num_model = checkpoint_path.split('/')[-3].split("model")[-1]
output_file = f"responses/tests/model_{num_model}_responses_test3.json"
# output_file = f"responses/dataset/model_responses{num_model}.json"
with open(output_file, 'w', encoding='utf-8') as f_out:
    for sample in tqdm(tests):
    # for sample in tqdm(dataset['train']):   # Use 'validation' or 'train' split if 'test' is not available
        input_text = sample['prompt']  # Adjust the field name based on the dataset's structure

        # Prepare the input prompt
        prompt = "Prompt:" + input_text + " \nResponse:"

        # Tokenize input
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate output
        outputs = model.generate(
            inputs,
            max_length=512,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Since the model may include the prompt in its output, we extract the generated response
        response = generated_text[len(prompt):]
        response = generated_text

        # Prepare the JSON object
        json_obj = {
            "prompt": prompt,
            "response": response
        }

        # Write the JSON object to file
        f_out.write(json.dumps(json_obj) + '\n')

        if len(open(output_file).readlines()) == 5:
            break
