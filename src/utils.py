from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
from huggingface_hub import snapshot_download
from pathlib import Path
import pandas as pd
import os
from sklearn.model_selection import train_test_split


# Step 1: Load the tokenizer and model with quantization
def download_model(model_name):
    """
    Download the model from the Hugging Face Hub.
    
    Args:
        model_name: str, name of the model to download
    """
    # TODO: Check this function
    # We create the directory if it does not exist
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_eos_token=True,
        use_fast=True,
        padding_side="left",
        trust_remote_code=True
    )

    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        trust_remote_code=True
    )
    return tokenizer, model


# Step 2: Load and preprocess the datasets
def download_lima_dataset(data_path):
    """
    Download the LIMA dataset and save it in the given data path.
    
    Args:
        data_path: str, path to save the dataset
    """
    lima_path = os.path.join(data_path, "lima")
    # We create the directory if it does not exist
    os.makedirs(lima_path, exist_ok=True)
    # Download the dataset
    dataset_lima = load_dataset("GAIR/lima")["train"]

    processed_records = []
    # We preprocess the dataset by dividing the conversations into prompts and responses
    for record in dataset_lima:
        prompt = record["conversations"][0]
        response = record["conversations"][1]
        processed_records.append({"prompt": prompt, "response": response})

    # We transform the list of dictionaries into a pandas DataFrame
    df_lima = pd.DataFrame(processed_records)

    # We separate the dataset into train, validation and test
    # The seed is set to 42 for reproducibility
    # We use 80% of the data for training, 10% for validation and 10% for testing
    train, temp = train_test_split(df_lima, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    
    # We save the datasets in the lima folder
    train.to_json(os.path.join(lima_path, "lima_processed_train.json"), orient="records", lines=True, force_ascii=False)
    val.to_json(os.path.join(lima_path, "lima_processed_val.json"), orient="records", lines=True, force_ascii=False)
    test.to_json(os.path.join(lima_path, "lima_processed_test.json"), orient="records", lines=True, force_ascii=False)
    print(f"LIMA dataset downloaded and saved in the data path {data_path}/lima.")


def download_and_preprocess_oasst1_split(data_path, split, test_size=None):
    """
    Download the OASST1 dataset split given and save it in the data path.
    
    Args:
        data_path: str, path to save the dataset.
        split: str, split of the dataset to download.
        test_size: float|None, size of the test dataset. If None, there is no split.
    """
    # Load the dataset
    df = pd.read_parquet("hf://datasets/OpenAssistant/oasst1/" + split)
    # Filter for assistant messages and select relevant columns
    cleaned_df = df[df["role"] == "assistant"][["message_id", "parent_id", "text"]].copy()
    # Rename the text column to response
    cleaned_df.rename(columns={"text": "response"}, inplace=True)
    # Add a new column with the prompt for each assistant message
    cleaned_df["prompt"] = cleaned_df["parent_id"].apply(
        lambda x: df[df["message_id"] == x]["text"].values[0]
        if x in df["message_id"].values else None
    )
    # Remove rows with missing prompts
    cleaned_df.dropna(subset=["prompt"], inplace=True)
    # Remove the message_id and parent_id columns
    cleaned_df.drop(columns=["message_id", "parent_id"], inplace=True)
    # Reorder the columns to have the prompt column first and the response column second
    cleaned_df = cleaned_df.sort_index(axis=1)

    if test_size:
        # Split the dataset into train and test
        train, test = train_test_split(cleaned_df, test_size=test_size, random_state=42)
        # Reset the index
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)

        # Save the train and test datasets
        train.to_json(os.path.join(data_path, "oasst1_processed_train.json"), orient="records", lines=True, force_ascii=False)
        test.to_json(os.path.join(data_path, "oasst1_processed_val.json"), orient="records", lines=True, force_ascii=False)
    else:
        # Reset the index
        cleaned_df.reset_index(drop=True, inplace=True)
        # Save the test dataset
        cleaned_df.to_json(os.path.join(data_path, "oasst1_processed_test.json"), orient="records", lines=True, force_ascii=False)


def download_oasst1_dataset(data_path):
    """
    Download the OASST1 dataset and save it in the given data path.
    
    Args:
        data_path: str, path to save the dataset
    """
    splits = {
    'train': 'data/train-00000-of-00001-b42a775f407cee45.parquet', 
    'validation': 'data/validation-00000-of-00001-134b8fd0c89408b6.parquet'
    }
    # We create the directory if it does not exist
    os.makedirs(os.path.join(data_path, "oasst1"), exist_ok=True)
    oasst1_path = os.path.join(data_path, "oasst1")
    # Download the dataset
    download_and_preprocess_oasst1_split(oasst1_path, splits["train"], test_size=0.1)
    download_and_preprocess_oasst1_split(oasst1_path, splits["validation"], test_size=None)

    print(f"OASST1 dataset downloaded and saved in the data path {data_path}/oasst1.")


def load_datasets(data_path):
    """
    Load the datasets from the given path. If the dataset is not found, dowload it
    from the download_dataset function and save it in the given path.
    
    Args:
        data_path: str, path to the dataset
    
    Returns:
        lima_train: pd.DataFrame, LIMA training dataset
        lima_val: pd.DataFrame, LIMA validation dataset
        lima_test: pd.DataFrame, LIMA test dataset
        oasst1_train: pd.DataFrame, OASST1 training dataset
        oasst1_val: pd.DataFrame, OASST1 validation dataset
        oasst1_test: pd.DataFrame, OASST1 test dataset
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        download_lima_dataset(data_path)
        download_oasst1_dataset(data_path)
    
    # Load the datasets
    lima_path = os.path.join(data_path, "lima")
    oasst1_path = os.path.join(data_path, "oasst1")
    lima_train = pd.read_json(os.path.join(lima_path, "lima_processed_train.json"), lines=True)
    lima_val = pd.read_json(os.path.join(lima_path, "lima_processed_val.json"), lines=True)
    lima_test = pd.read_json(os.path.join(lima_path, "lima_processed_test.json"), lines=True)
    oasst1_train = pd.read_json(os.path.join(oasst1_path, "oasst1_processed_train.json"), lines=True)
    oasst1_val = pd.read_json(os.path.join(oasst1_path, "oasst1_processed_val.json"), lines=True)
    oasst1_test = pd.read_json(os.path.join(oasst1_path, "oasst1_processed_test.json"), lines=True)

    return lima_train, lima_val, lima_test, oasst1_train, oasst1_val, oasst1_test


if __name__ == "__main__":
    # model_name = "mistralai/Mistral-7B-v0.3"
    # tokenizer, model = download_model(model_name)
    data_path = "./data"
    lima_train, lima_val, lima_test, oasst1_train, oasst1_val, oasst1_test = load_datasets(data_path)
    print(lima_train.head(5), "\n")
    print(lima_val.head(5), "\n")
    print(lima_test.head(5), "\n")

    print(oasst1_train.head(5), "\n")
    print(oasst1_val.head(5), "\n")
    print(oasst1_test.head(5), "\n")

    print("All datasets loaded successfully.")
