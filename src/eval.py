from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import load_datasets
import torch
import json
from tqdm import tqdm
import os


model_name = './models/model2/output_lima/checkpoint-100'

# Step 1: Load the tokenizer and model with quantization
# model_name = "models/model1"  # Near 3B model (smallest available Qwen model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device,
    trust_remote_code=True
)

# Step 2: Load the google/IFEval dataset
# _, _, oasst1_test, _, _, lima_test = load_datasets('./data')


tests = [{'prompt': 'What is the capital of France?', 'response': 'Paris'}, 
         {'prompt': 'Write the days of the week.', 'response': 'Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday'},
         {'prompt': '¿Cuál es la diferencia entre echo, print, print_r, var_dump y var_export en PHP?', 'response': 'echo: Muestra una o más cadenas de texto. print: Muestra una cadena de texto. print_r: Muestra información sobre una variable de forma legible. var_dump: Muestra información sobre una variable de forma legible. var_export: Muestra una representación de una variable que puede ser utilizada como código PHP.'},
         {'prompt': 'Dime tres asignaturas que se imparten en la carrera de Ingeniería Informática.', 'response': 'Programación, Matemáticas, Física'},
         {'prompt': 'Cuales son los planetas del sistema solar?', 'response': 'Mercurio, Venus, Tierra, Marte, Júpiter, Saturno, Urano, Neptuno'}]

# Step 3: Generate predictions on the dataset
os.makedirs('responses/tests', exist_ok=True)
num_model = model_name.split('/')[-3][-1]
output_file = f"responses/tests/model_{num_model}_responses_test.json"
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
