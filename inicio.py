from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
from huggingface_hub import snapshot_download
from pathlib import Path
import pandas as pd


# Step 1: Load the tokenizer and model with quantization
model_name = "mistralai/Mistral-7B-v0.3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device,
    trust_remote_code=True
)

# Step 2: Load and preprocess the datasets
def load_datasets(data_path):
    """
    Load the datasets from the given path
    
    Args:
        data_path: str, path to the dataset"""
    dataset = load_dataset("csv", data_files=data_path)
    return dataset