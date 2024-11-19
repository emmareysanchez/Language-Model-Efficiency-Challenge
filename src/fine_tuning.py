import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from trl import SFTTrainer
from evaluate import load
import time
# from utils import 
from src.utils import load_datasets


# HACK: Hyperparameters of the training
# 1. Quantization configuration hyperparameters
load_in_8bit = True
llm_int8_threshold = 6.0
llm_int8_skip_modules = None
quant_type = "nf4"

# 2. LoRA configuration hyperparameters
r = 16               # Try 8
scaling_factor = 16  # Try 32
lora_dropout = 0.05  # Try 0.1
bias = "none"
task_type = "CAUSAL_LM"

# 3. Training hyperparameters
train_batch_size = 8
eval_batch_size = 8
num_train_epochs = 1
logging_steps = 1000
save_steps = 1000
save_total_limit = 1
output_dir = "output"
overwrite_output_dir = True
per_device_train_batch_size = 4 # Try 8
per_device_eval_batch_size = 8
warmup_steps = 0
weight_decay = 0.01
learning_rate = 5e-5
adam_epsilon = 1e-8
max_grad_norm = 1.0
seed = 42

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
    load_in_8bit=load_in_8bit,  # Use 8-bit quantization
    llm_int8_threshold=llm_int8_threshold,
    llm_int8_skip_modules=llm_int8_skip_modules,
    quant_type=quant_type,  # NormalFloat4 is more stable for NLP
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device,
    quantization=bnb_config,
    trust_remote_code=True
)

# Step 3: Configure LoRA
lora_config = LoraConfig(
    r=r,                        # Rank of the LoRA decomposition
    lora_alpha=scaling_factor,  # Scaling factor for LoRA updates
    lora_dropout=lora_dropout,  # Dropout rate applied to LoRA layers
    bias=bias,                  # No bias is added to the LoRA layers
    task_type="CAUSAL_LM",      # Specify the task as causal language modeling
    target_modules=[            # Modules to apply LoRA to
        'k_proj', 'q_proj', 'v_proj', 'o_proj',
        'gate_proj', 'down_proj', 'up_proj'
    ]
)

# Add LoRA to the model
model = get_peft_model(model, lora_config)

# Step 3: Load datasets
data_path = "./data"  # Path where datasets are stored or will be downloaded
lima_train, lima_val, lima_test, oasst1_train, oasst1_val, oasst1_test = load_datasets(data_path)

# Tokenize datasets
def tokenize_function(example):
    return tokenizer(
        example["prompt"],
        text_target=example["response"],
        truncation=True,
        max_length=256,
        padding="max_length"
    )

# Tokenize OASST1 dataset
oasst1_train = oasst1_train.map(tokenize_function, batched=True)
oasst1_val = oasst1_val.map(tokenize_function, batched=True) 

oasst1_train.set_format(type="torch")
oasst1_val.set_format(type="torch")

# Training arguments for OASST1
oasst1_training_args = TrainingArguments(
    output_dir="output_oasst1",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_steps=100,
    save_steps=1000,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_total_limit=1,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True,
    load_best_model_at_end=True,
    seed=42,
)

# Fine-tuning on OASST1
oasst1_trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=oasst1_training_args,
    train_dataset=oasst1_train,
    eval_dataset=oasst1_val,
)

print("Starting fine-tuning on OASST1...")
oasst1_trainer.train()

# Save the model fine-tuned on OASST1
model.save_pretrained("output_oasst1")
tokenizer.save_pretrained("output_oasst1")
print("Fine-tuning on OASST1 complete. Model saved.")



# # Step 5: Train the model


# training_args = TrainingArguments(
#     output_dir="./fine_tuned_lora_model",
#     per_device_train_batch_size=8,
#     learning_rate=3e-4,
#     num_train_epochs=3,
#     logging_dir="./logs",
#     save_strategy="epoch",
#     evaluation_strategy="epoch",
#     logging_steps=100,
#     save_total_limit=2,
#     load_best_model_at_end=True,
#     fp16=True,  # Mixed precision training
# )
# training_arguments = TrainingArguments(
#     output_dir="./models",  # Directory for saving model checkpoints and logs
#     per_device_train_batch_size=4,        # Batch size per device during training
#     eval_strategy="steps",                # Evaluation strategy: evaluate every few steps
#     do_eval=True,                         # Enable evaluation during training
#     optim="paged_adamw_8bit",             # Use 8-bit AdamW optimizer for memory efficiency
#     gradient_accumulation_steps=2,        # Accumulate gradients over multiple steps
#     per_device_eval_batch_size=2,         # Batch size per device during evaluation
#     log_level="debug",                    # Set logging level to debug for detailed logs
#     logging_steps=10,                     # Log metrics every 10 steps
#     learning_rate=1e-4,                   # Initial learning rate
#     eval_steps=25,                        # Evaluate the model every 25 steps
#     max_steps=100,                        # Total number of training steps
#     save_steps=25,                        # Save checkpoints every 25 steps
#     warmup_steps=25,                      # Number of warmup steps for learning rate scheduler
#     lr_scheduler_type="linear",           # Use a linear learning rate scheduler
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=lima_train,
#     tokenizer=tokenizer,
# )

# trainer.train()

# # Step 6: Save the model
# model.save_pretrained("./fine_tuned_lora_model")
# tokenizer.save_pretrained("./fine_tuned_lora_model")
