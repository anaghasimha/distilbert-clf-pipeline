import pytest
from unittest.mock import MagicMock, patch
import torch
from src.pipeline.prediction_pipeline import PredictionPipeline

@pytest.fixture
def mock_prediction_pipeline():
    """
    Fixture that instantiates the PredictionPipeline but mocks out 
    the heavy weight loading step to allow lightning-fast unit tests.
    """
    with patch('src.pipeline.prediction_pipeline.DistilBertForSequenceClassification.from_pretrained'), \
         patch('src.pipeline.prediction_pipeline.DistilBertTokenizer.from_pretrained'):
        
        predictor = PredictionPipeline()
        
        # Manually seed mock elements into cached properties to simulate a loaded state
        predictor.model = MagicMock()
        predictor.tokenizer = MagicMock()
        predictor.device = torch.device("cpu")
        
        # Mock tokenizer encoding behaviors
        mock_inputs = {
            "input_ids": torch.ones((1, 16), dtype=torch.long),
            "attention_mask": torch.ones((1, 16), dtype=torch.long)
        }
        predictor.tokenizer.encode_plus.return_with = mock_inputs
        predictor.tokenizer.encode_plus.return_value = mock_inputs
        
        # Mock model forward pass returning logits matching 2 prediction target classes
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[0.1, 2.5]]) # Class 1 has significantly higher logit
        predictor.model.return_value = mock_outputs
        
        return predictor

def test_successful_inference_payload(mock_prediction_pipeline):
    """Verifies that a valid input text string returns accurate parsed prediction maps."""
    result = mock_prediction_pipeline.predict("Excellent sample input text string.")
    
    assert result["status"] == "Success"
    assert result["predicted_class_id"] == 1
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0

def test_empty_input_payload_graceful_handling(mock_prediction_pipeline):
    """Verifies that empty strings don't crash the pipeline and return a protective response map."""
    bad_inputs = ["", "   ", None]
    
    for empty_input in bad_inputs:
        result = mock_prediction_pipeline.predict(empty_input)
        
        assert result["predicted_class_id"] == -1
        assert result["confidence"] == 0.0
        assert "Empty Input Payload Passed" in result["status"]

def test_inference_exception_catch_blocks(mock_prediction_pipeline):
    """Ensures that unexpected structural crashes internal to the model return an informative error state."""
    # Force the internal model call to throw a critical error runtime exception
    mock_prediction_pipeline.model.side_effect = RuntimeError("Simulated Hardware Matrix Fault")
    
    result = mock_prediction_pipeline.predict("Valid input text payload.")
    
    assert result["predicted_class_id"] is None
    assert result["confidence"] is None
    assert "Error" in result["status"]