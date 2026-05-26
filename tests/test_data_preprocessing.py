import pytest
import torch
import pandas as pd
from transformers import DistilBertTokenizer
from src.components.data_transformation import CustomTextDataset

@pytest.fixture
def mock_tokenizer():
    """Fixture to load a lightweight tokenizer instance for testing."""
    return DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

@pytest.fixture
def sample_dataframe():
    """Fixture providing a mock dataframe containing dirty text anomalies."""
    data = {
        "text": [
            "This is a regular sentence.",
            "Text with <br> HTML tags embedded inside.",
            "Excessive    whitespace    normalization check."
        ],
        "label": [0, 1, 0]
    }
    return pd.DataFrame(data)

def test_text_normalization_logic(mock_tokenizer):
    """
    Verifies that the private text cleaning helper strips HTML tags,
    forces lowercase, and collapses excessive whitespaces safely.
    """
    # Instantiate dataset with a single sample to test internal helper methods
    dataset = CustomTextDataset(
        texts=["<br>  HELLO    world  </br>"], 
        labels=[0], 
        tokenizer=mock_tokenizer, 
        max_length=16
    )
    
    # Run text cleaning routine directly
    cleaned_text = dataset._clean_text(dataset.texts[0])
    
    # Assert structural integrity transformations
    assert cleaned_text == "hello world"
    assert "<br>" not in cleaned_text
    assert "   " not in cleaned_text

def test_pytorch_dataset_output_schema(mock_tokenizer, sample_dataframe):
    """
    Ensures that the CustomTextDataset correctly tokenizes sequences and
    outputs flat PyTorch Tensors matching explicit structural shapes.
    """
    max_len = 32
    dataset = CustomTextDataset(
        texts=sample_dataframe["text"].tolist(),
        labels=sample_dataframe["label"].tolist(),
        tokenizer=mock_tokenizer,
        max_length=max_len
    )
    
    # Extract the first processed sample matrix
    sample = dataset[0]
    
    # Assert structural keys exist
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample
    
    # Assert accurate Tensor tracking metrics and dimensions
    assert isinstance(sample["input_ids"], torch.Tensor)
    assert isinstance(sample["attention_mask"], torch.Tensor)
    assert sample["input_ids"].shape == torch.Size([max_len])
    assert sample["attention_mask"].shape == torch.Size([max_len])
    assert sample["labels"].item() == 0