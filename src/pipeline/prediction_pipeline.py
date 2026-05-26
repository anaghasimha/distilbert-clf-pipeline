import os
import torch
import logging
from pathlib import Path
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from src.utils.common import read_yaml

# Initialize inference logger
logger = logging.getLogger(__name__)

class PredictionPipeline:
    """
    Optimized Inference Pipeline.
    Loads fine-tuned model artifacts from local storage and handles real-time,
    low-latency prediction on individual text payloads.
    """
    def __init__(self, config_path: Path = Path("config/config.yaml")):
        # Read the configuration to map where model weights reside
        full_config = read_yaml(config_path)
        self.model_config = full_config["model_trainer"]
        self.transform_config = full_config["data_transformation"]
        
        # Route path directly to the locally compiled checkpoint directory
        self.model_dir = Path(self.model_config["trained_model_path"])
        
        # Determine target compute device (CPU optimization prioritized for inference cost-efficiencies)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        # Class attributes placeholder for lazy-loading memory safety
        self.model = None
        self.tokenizer = None

    def _load_artifacts(self) -> None:
        """
        Private method to lazily load and cache weights and tokenizers.
        Ensures files are read from disk exactly once, preventing memory leakage.
        """
        if self.model is not None and self.tokenizer is not None:
            return # Artifacts are already securely initialized in RAM cache

        try:
            logger.info(f"Loading local compiled model & tokenizer from: {self.model_dir}")
            
            if not self.model_dir.exists():
                raise FileNotFoundError(
                    f"Model directory missing at: {self.model_dir}. Please run the training pipeline first."
                )

            # Load matching components directly from the local workspace folder
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_dir)
            self.model = DistilBertForSequenceClassification.from_pretrained(self.model_dir)
            
            # Switch structural configuration explicitly to evaluation state
            self.model.to(self.device)
            self.model.eval() 
            
            logger.info("Model and tokenizer successfully initialized into memory caches.")
            
        except Exception as e:
            logger.critical(f"Failed to load local execution assets: {str(e)}")
            raise e

    def predict(self, text_payload: str) -> dict:
        """
        Executes low-latency real-time inference on a single string input.

        Args:
            text_payload (str): Raw unstructured text payload submitted by a client.

        Returns:
            dict: Parsed outputs containing the predicted class index and confidence score.
        """
        try:
            # Enforce lazy loading checklist validation
            self._load_artifacts()
            
            if not text_payload or str(text_payload).strip() == "":
                return {"prediction": -1, "confidence": 0.0, "status": "Empty Input Payload Passed"}

            # Tokenize incoming input text payload
            inputs = self.tokenizer.encode_plus(
                str(text_payload),
                None,
                add_special_tokens=True,
                max_length=self.transform_config["max_length"],
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            # Ship inputs to active compute target
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            # Context Manager: Deactivate autograd engine to optimize memory and execution speeds
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Extract logits from the final classification head layer
                logits = outputs.logits
                
                # Apply Softmax activation function to convert raw score outputs to mathematical probabilities
                probabilities = torch.softmax(logits, dim=1)
                
                # Extract tracking attributes
                confidence_score, predicted_class_idx = torch.max(probabilities, dim=1)

            return {
                "predicted_class_id": int(predicted_class_idx.item()),
                "confidence": float(confidence_score.item()),
                "status": "Success"
            }

        except Exception as e:
            logger.error(f"Inference processing failed on text payload: {str(e)}")
            return {"prediction": None, "confidence": None, "status": f"Error: {str(e)}"}

if __name__ == "__main__":
    # Smoke-test validation script execution
    predictor = PredictionPipeline()
    sample_prediction = predictor.predict("This is an exceptional production-ready platform pipeline.")
    print(f"Sample Validation Result: {sample_prediction}")