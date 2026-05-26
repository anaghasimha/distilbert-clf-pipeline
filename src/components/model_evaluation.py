import os
import json
import torch
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification
from src.utils.common import read_yaml, create_directories
from src.components.data_transformation import CustomTextDataset, DataTransformation

# Initialize logging instance
logger = logging.getLogger(__name__)

class ModelEvaluationConfig:
    """
    Data Value Object mapping tracking files and storage routes for model evaluation.
    """
    def __init__(self, config_dict: dict):
        self.root_dir = Path(config_dict["root_dir"])
        self.metrics_file = Path(config_dict["metrics_file"])
        self.confusion_matrix_img = Path(config_dict["confusion_matrix_img"])


class ModelEvaluation:
    """
    Executes deep evaluation protocols against fine-tuned model checkpoints.
    Computes industry-standard classification metrics and serializes reports.
    """
    def __init__(self, config_path: Path = Path("config/config.yaml")):
        full_config = read_yaml(config_path)
        self.config = ModelEvaluationConfig(full_config["model_evaluation"])
        self.ingestion_config = full_config["data_ingestion"]
        self.transformation_config = full_config["data_transformation"]
        self.trainer_config = full_config["model_trainer"]
        
        # Safely prepare metrics tracking directory
        create_directories([self.config.root_dir])
        
        # Hardware target verification
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def evaluate_checkpoint(self) -> dict:
        """
        Runs batch evaluation over holdout datasets, computes metrics, 
        and writes immutable JSON/PNG report assets to disk.
        
        Returns:
            dict: High-level macro performance overview.
        """
        logger.info("Initializing offline performance validation metrics pass...")
        
        try:
            # 1. Load holdout test split and initialize tokenization pipelines
            test_df = pd.read_csv(Path(self.ingestion_config["test_data_path"]))
            transform_component = DataTransformation()
            
            test_dataset = CustomTextDataset(
                texts=test_df["text"].tolist(),
                labels=test_df[self.ingestion_config["stratify_column"]].tolist(),
                tokenizer=transform_component.tokenizer,
                max_length=self.transformation_config["max_length"]
            )
            
            test_loader = DataLoader(
                test_dataset, 
                batch_size=self.trainer_config["batch_size"], 
                shuffle=False
            )

            # 2. Load the freshly trained local model checkpoint weights
            model_checkpoint_path = Path(self.trainer_config["trained_model_path"])
            logger.info(f"Loading model state weights from disk: {model_checkpoint_path}")
            
            model = DistilBertForSequenceClassification.from_pretrained(model_checkpoint_path)
            model.to(self.device)
            model.eval()

            true_labels = []
            predicted_labels = []

            # 3. Process batch evaluations without tracking autograd gradients
            logger.info("Streaming dataset slices through evaluation loops...")
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    
                    predictions = torch.argmax(logits, dim=1)
                    
                    # Accumulate raw numpy representations for metrics computation
                    true_labels.extend(labels.cpu().numpy())
                    predicted_labels.extend(predictions.cpu().numpy())

            # 4. Compute comprehensive mathematical report frameworks
            true_labels = np.array(true_labels)
            predicted_labels = np.array(predicted_labels)
            
            report_dict = classification_report(true_labels, predicted_labels, output_dict=True)
            logger.info(f"Evaluation Complete. Overall Macro F1-Score: {report_dict['macro avg']['f1-score']:.4f}")

            # 5. Persist raw metrics report structured data to an immutable JSON asset
            metrics_payload = {
                "accuracy": report_dict["accuracy"],
                "macro_precision": report_dict["macro avg"]["precision"],
                "macro_recall": report_dict["macro avg"]["recall"],
                "macro_f1_score": report_dict["macro avg"]["f1-score"]
            }
            
            with open(self.config.metrics_file, "w") as json_file:
                json.dump(metrics_payload, json_file, indent=4)
            logger.info(f"Performance tracking parameters exported to JSON: {self.config.metrics_file}")

            # 6. Generate and save a highly polished Confusion Matrix PNG visual asset
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(true_labels, predicted_labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap=plt.cm.Blues, values_format="d")
            plt.title("DistilBERT Fine-Tuned Model Performance Evaluation")
            plt.savefig(self.config.confusion_matrix_img, bbox_inches="tight", dpi=300)
            plt.close()
            logger.info(f"Confusion matrix visualizations rendered cleanly to file: {self.config.confusion_matrix_img}")

            return metrics_payload

        except Exception as e:
            logger.critical(f"Evaluation layer halted unexpectedly: {str(e)}")
            raise e