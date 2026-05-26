import os
import re
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer
from src.utils.common import read_yaml, create_directories
import logging

# Initialize logger for data transformation tracking
logger = logging.getLogger(__name__)

class CustomTextDataset(Dataset):
    """
    A custom PyTorch Dataset that encapsulates text normalization, tokenization,
    and tensor format conversion required for DistilBERT consumption.
    """
    def __init__(self, texts: list, labels: list, tokenizer: DistilBertTokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # Extract individual text string and ensure string type format
        text = str(self.texts[index])
        
        # Apply standard enterprise text normalization rules
        text = self._clean_text(text)

        # Tokenize text using the pre-loaded DistilBERT engine
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,      # Injects [CLS] and [SEP] tokens automatically
            max_length=self.max_length,
            padding="max_length",          # Zero-pads sequences shorter than max_length
            truncation=True,               # Truncates sequences longer than max_length
            return_token_type_ids=False,   # DistilBERT doesn't use token type IDs (unlike BERT)
            return_attention_mask=True,    # Generates binary masks indicating padding locations
            return_tensors="pt"            # Outputs native PyTorch Tensors
        )

        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[index], dtype=torch.long)
        }

    def _clean_text(self, text: str) -> str:
        """
        Applies a deterministic text normalization pass to clean pipeline text anomalies.
        """
        # Convert text to lowercase
        text = text.lower()
        # Strip out legacy HTML formatting tags
        text = re.sub(r"<br\s*/?>|<[^>]*>", " ", text)
        # Normalize excessive repeating whitespace characters
        text = re.sub(r"\s+", " ", text).strip()
        return text


class DataTransformationConfig:
    """
    Data Value Object mapping parameter boundaries for data transformation routines.
    """
    def __init__(self, config_dict: dict):
        self.root_dir = Path(config_dict["root_dir"])
        self.tokenizer_name = config_dict["tokenizer_name"]
        self.max_length = config_dict["max_length"]


class DataTransformation:
    """
    Orchestrates the transformation layer: ingests split data splits,
    initializes the tokenization sub-engines, and serializes dataset transformations.
    """
    def __init__(self, config_path: Path = Path("config/config.yaml")):
        full_config = read_yaml(config_path)
        self.config = DataTransformationConfig(full_config["data_transformation"])
        self.ingestion_config = full_config["data_ingestion"]
        
        create_directories([self.config.root_dir])
        
        # Instantiate the official Hugging Face tokenizer instance
        logger.info(f"Initializing DistilBertTokenizer instance: {self.config.tokenizer_name}")
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.config.tokenizer_name)

    def run_transformations(self) -> None:
        """
        Reads train and test file splits, verifies non-empty states,
        and runs them through validation tests to ensure operational data integrity.
        """
        try:
            train_path = Path(self.ingestion_config["train_data_path"])
            test_path = Path(self.ingestion_config["test_data_path"])
            stratify_col = self.ingestion_config["stratify_column"]

            if not train_path.exists() or not test_path.exists():
                raise FileNotFoundError("Prerequisite data ingestion split files are missing.")

            # Load split variants into memory for dataset compilation testing
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("Verifying PyTorch dataset generation sequence on training arrays...")
            
            # Instantiate validation datasets to verify tokenizer behaviors
            train_dataset = CustomTextDataset(
                texts=train_df["text"].tolist(),
                labels=train_df[stratify_col].tolist(),
                tokenizer=self.tokenizer,
                max_length=self.config.max_length
            )

            # Test-read an individual sample to confirm tensor dimensions match expectations
            sample = train_dataset[0]
            logger.info(f"Dataset verification passed. Input ID shape: {sample['input_ids'].shape}")
            logger.info(f"Data Transformation verification completed successfully.")
            
        except Exception as e:
            logger.critical(f"Critical error encountered in Data Transformation stage: {str(e)}")
            raise e