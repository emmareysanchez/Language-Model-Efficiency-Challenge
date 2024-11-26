from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
import torch
import json
from tqdm import tqdm
from huggingface_hub import snapshot_download
from pathlib import Path
import pandas as pd
import os
from sklearn.model_selection import train_test_split


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
    train, val = train_test_split(df_lima, test_size=0.05, random_state=42)

    # We save the datasets in the lima folder
    Dataset.from_pandas(train).save_to_disk(os.path.join(lima_path, "lima_processed_train"))
    Dataset.from_pandas(val).save_to_disk(os.path.join(lima_path, "lima_processed_val"))
    print(f"LIMA dataset downloaded and saved in the data path {data_path}/lima.")


def download_and_preprocess_oasst1_split(data_path, split):
    """
    Download the OASST1 dataset split given and save it in the data path.

    Args:
        data_path: str, path to save the dataset.
        split: str, split of the dataset to download.
        test_size: float|None, size of the test dataset. If None, there is no split.
    """
    # Load the dataset
    df = Dataset.from_parquet("hf://datasets/OpenAssistant/oasst1/" + split)

    # Build a mapping from message_id to text for 'prompter' messages
    message_id_to_text = {
        row["message_id"]: row["text"] for row in df if row["role"] == "prompter"
    }

    # Filter for assistant messages
    assistant_dataset = df.filter(lambda x: x["role"] == "assistant")

    # Filter for messages in english or spanish
    assistant_dataset = assistant_dataset.filter(lambda x: x["lang"] in ["en", "es"])

    # Add a new column with the prompt for each assistant message
    def add_prompt(example):
        parent_id = example["parent_id"]
        if parent_id in message_id_to_text:
            return {"prompt": message_id_to_text[parent_id], "response": example["text"]}
        return None

    assistant_dataset = assistant_dataset.map(add_prompt, remove_columns=["message_id", "parent_id", "role", "lang"])
    assistant_dataset = assistant_dataset.filter(lambda x: x is not None)

    # Split into train and test if test_size is provided
    if "train" in split:
        assistant_dataset.save_to_disk(os.path.join(data_path, "oasst1_processed_train"))

    else:
        assistant_dataset.save_to_disk(os.path.join(data_path, "oasst1_processed_val"))


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
    download_and_preprocess_oasst1_split(oasst1_path, splits["train"])
    download_and_preprocess_oasst1_split(oasst1_path, splits["validation"])

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
        oasst1_train: pd.DataFrame, OASST1 training dataset
        oasst1_val: pd.DataFrame, OASST1 validation dataset
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        download_lima_dataset(data_path)
        download_oasst1_dataset(data_path)

    # Load the datasets
    lima_path = os.path.join(data_path, "lima")
    oasst1_path = os.path.join(data_path, "oasst1")
    lima_train = Dataset.load_from_disk(os.path.join(lima_path, "lima_processed_train"))
    lima_val = Dataset.load_from_disk(os.path.join(lima_path, "lima_processed_val"))
    oasst1_train = Dataset.load_from_disk(os.path.join(oasst1_path, "oasst1_processed_train"))
    oasst1_val = Dataset.load_from_disk(os.path.join(oasst1_path, "oasst1_processed_val"))

    return lima_train, lima_val, oasst1_train, oasst1_val


if __name__ == "__main__":
    data_path = "./data"
    lima_train, lima_val, oasst1_train, oasst1_val = load_datasets(data_path)

    print("All datasets loaded successfully.")
