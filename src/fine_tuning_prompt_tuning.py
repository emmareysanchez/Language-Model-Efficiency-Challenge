import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model, PromptTuningConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from trl import SFTTrainer, SFTConfig
from evaluate import load
import time
import os
from src.utils import load_datasets


# HACK: Hyperparameters of the training
# 1. Quantization configuration hyperparameters
load_in_8bit = True
llm_int8_threshold = 6.0
llm_int8_skip_modules = None
quant_type = "nf4"

# 2. Promp tuning hyperparameters
num_virtual_tokens = 50

# 3. Training hyperparameters
overwrite_output_dir = True
seed = 42
save_total_limit = 3
logging_strategy = 'steps'
try:
    num_model = len(os.listdir("./models")) + 1
except FileNotFoundError:
    os.makedirs("./models")
    num_model = 1

# 3.1 OASST1 hyperparameters
train_batch_size_oasst1 = 8
eval_batch_size_oasst1 = 8
num_train_epochs_oasst1 = 4
logging_steps_oasst1 = 100
save_steps_oasst1 = 25
output_dir_oasst1 = f"./models/model{num_model}/output_oasst1"
per_device_train_batch_size_oasst1 = 4 # Try 8
per_device_eval_batch_size_oasst1 = 4 # Try 8
gradient_accumulation_steps_oasst1 = 4
warmup_steps_oasst1 = 0
weight_decay_oasst1 = 0.01
learning_rate_oasst1 = 1e-4 # Try 1e-4
max_steps_oasst1 = 100
adam_epsilon_oasst1 = 1e-8
max_grad_norm_oasst1 = 1.0
logging_dir_oasst1 = './logs_oasst1'

# 3.2 LIMA hyperparameters

train_batch_size_lima = 8
eval_batch_size_lima = 8
num_train_epochs_lima = 4
logging_steps_lima = 100
save_steps_lima = 25
output_dir_lima = f"./models/model{num_model}/output_lima"
per_device_train_batch_size_lima = 4 # Try 8
per_device_eval_batch_size_lima = 4 # Try 8
gradient_accumulation_steps_lima = 4
warmup_steps_lima = 0
weight_decay_lima = 0.01
learning_rate_lima = 1e-4
max_steps_lima = 100
adam_epsilon_lima = 1e-8
max_grad_norm_lima = 1.0
logging_dir_lima = './logs_lima'

# 4. Evaluation hyperparameters
eval_steps = 1000
eval_logging_steps = 1000
eval_output_dir = "eval_output"
eval_overwrite_output_dir = True
eval_per_device_eval_batch_size = 8

# 5. Model configuration hyperparameters
model_name = "mistralai/Mistral-7B-v0.3"

# 6. Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#######################################################################################


# Step 1: Configure quantization with BitsAndBytes
tokenizer = AutoTokenizer.from_pretrained(  # por que el tokenizador depende del modelo a cargar? 
    model_name,
    add_eos_token=True,
    use_fast=True,
    padding_side="right",
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token

# Step 2: Load model with quantization
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
)
model = prepare_model_for_kbit_training(model)
model.config.pad_token_id = tokenizer.pad_token_id

# Step 3: Load datasets
data_path = "./data"  # Path where datasets are stored or will be downloaded
lima_train, lima_val, oasst1_train, oasst1_val = load_datasets(data_path)

# Tokenize datasets
def tokenize_function(example):
    conversation = list()
    for prompt, response in zip(example["prompt"], example["response"]):
        conversation.append(prompt + " " + response)
    return tokenizer(
        conversation,
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="pt"
    )

# Tokenize OASST1 and LIMA datasets
tokenized_oasst1_train = oasst1_train.map(tokenize_function, batched=True, remove_columns=oasst1_train.column_names)
tokenized_oasst1_val = oasst1_val.map(tokenize_function, batched=True, remove_columns=oasst1_val.column_names)

tokenized_lima_train = lima_train.map(tokenize_function, batched=True, remove_columns=lima_train.column_names)
tokenized_lima_val = lima_val.map(tokenize_function, batched=True, remove_columns=lima_val.column_names)


# Set format to torch
tokenized_oasst1_train.set_format(type="torch", columns=["input_ids", "attention_mask"])
tokenized_oasst1_val.set_format(type="torch", columns=["input_ids", "attention_mask"])

tokenized_lima_train.set_format(type="torch", columns=["input_ids", "attention_mask"])
tokenized_lima_val.set_format(type="torch", columns=["input_ids", "attention_mask"])


# Step 4: Configure LoRA
prompt_config = PromptTuningConfig(
    num_virtual_tokens=num_virtual_tokens,
    task_type="CAUSAL_LM"
)


# Step 5: Train the model

# 5.1 Training with OASST1
# Training arguments for OASST1
oasst1_training_args = TrainingArguments(
    output_dir=output_dir_oasst1,
    eval_strategy="steps",
    do_eval=True,
    optim="paged_adamw_8bit",
    per_device_train_batch_size=per_device_train_batch_size_oasst1,
    gradient_accumulation_steps=gradient_accumulation_steps_oasst1,
    per_device_eval_batch_size=per_device_eval_batch_size_oasst1,
    log_level="debug",
    logging_steps=logging_steps_oasst1,
    learning_rate=learning_rate_oasst1,
    eval_steps=eval_steps,
    max_steps=max_steps_oasst1,
    save_steps=save_steps_oasst1,
    warmup_steps=warmup_steps_oasst1,
    lr_scheduler_type="linear",
    num_train_epochs=num_train_epochs_oasst1,
    save_total_limit=save_total_limit,
    seed=seed,
    logging_dir=logging_dir_oasst1,
    logging_strategy=logging_strategy,
    # weight_decay=0.01,
    # fp16=True,
    # load_best_model_at_end=True,
)


# Fine-tuning on OASST1
oasst1_trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_oasst1_train,
    eval_dataset=tokenized_oasst1_val,
    peft_config=prompt_config,
    max_seq_length=256,
    tokenizer=tokenizer,
    args=oasst1_training_args,
)
print("Starting fine-tuning on OASST1...")
oasst1_trainer.train()

# Save the model fine-tuned on OASST1
print("Saving fine-tuned model on OASST1 without gradients...")
model.eval()  # Asegúrate de que el modelo esté en modo evaluación
model.save_pretrained(output_dir_oasst1 + "/final", safe_serialization=True)
tokenizer.save_pretrained(output_dir_oasst1 + "/final")
print("Model fine-tuned on OASST1 saved without gradients.")


# Training with LIMA
lima_training_args = TrainingArguments(
    output_dir=output_dir_lima,
    eval_strategy="steps",
    do_eval=True,
    optim="paged_adamw_8bit",
    per_device_train_batch_size=per_device_train_batch_size_lima,
    gradient_accumulation_steps=gradient_accumulation_steps_lima,
    per_device_eval_batch_size=per_device_eval_batch_size_lima,
    log_level="debug",
    logging_steps=logging_steps_lima,
    learning_rate=learning_rate_lima,
    eval_steps=eval_steps,
    max_steps=max_steps_lima,
    save_steps=save_steps_lima,
    warmup_steps=warmup_steps_lima,
    lr_scheduler_type="linear",
    num_train_epochs=num_train_epochs_lima,
    save_total_limit=save_total_limit,
    seed=seed,
    logging_dir=logging_dir_lima,
    logging_strategy=logging_strategy,
    # weight_decay=0.01,
    # fp16=True,
    # load_best_model_at_end=True,
)


# Fine-tuning on LIMA
lima_trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_lima_train,
    eval_dataset=tokenized_lima_val,
    peft_config=prompt_config,
    max_seq_length=256,
    tokenizer=tokenizer,
    args=lima_training_args,
)
print("Starting fine-tuning on LIMA...")
lima_trainer.train()

# Save the model fine-tuned on LIMA
print("Saving fine-tuned model on LIMA without gradients...")
model.eval()  # Asegúrate de que el modelo esté en modo evaluación
model.save_pretrained(output_dir_lima + "/final", safe_serialization=True)
tokenizer.save_pretrained(output_dir_lima + "/final")
print("Model fine-tuned on LIMA saved without gradients.")
