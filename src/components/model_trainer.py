import os
import torch
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from src.utils.common import read_yaml, create_directories
from src.components.data_transformation import CustomTextDataset, DataTransformation
import pandas as pd

# Initialize module logger
logger = logging.getLogger(__name__)

class ModelTrainerConfig:
    """
    Data Value Object mapping hyperparameter configurations for the model training loop.
    """
    def __init__(self, config_dict: dict):
        self.root_dir = Path(config_dict["root_dir"])
        self.model_checkpoint = config_dict["model_checkpoint"]
        self.trained_model_path = Path(config_dict["trained_model_path"])
        self.epochs = config_dict["epochs"]
        self.batch_size = config_dict["batch_size"]
        self.learning_rate = float(config_dict["learning_rate"])
        self.weight_decay = float(config_dict["weight_decay"])
        self.warmup_steps = config_dict["warmup_steps"]
        self.fp16_precision = config_dict["fp16_precision"]
        self.logging_steps = config_dict["logging_steps"]


class ModelTrainer:
    """
    Orchestrates a robust, native PyTorch training loop for DistilBERT fine-tuning.
    Integrates Automatic Mixed Precision (AMP) and linear learning rate scheduling.
    """
    def __init__(self, config_path: Path = Path("config/config.yaml")):
        full_config = read_yaml(config_path)
        self.config = ModelTrainerConfig(full_config["model_trainer"])
        self.ingestion_config = full_config["data_ingestion"]
        self.transformation_config = full_config["data_transformation"]
        
        create_directories([self.config.root_dir])
        
        # Device Check: Automatically utilize CUDA, MPS (Apple Silicon), or fall back to CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        logger.info(f"Model Training execution bound to compute device: {self.device}")

    def initiate_model_trainer(self) -> None:
        """
        Executes the full PyTorch deep learning training loop.
        """
        try:
            # 1. Reconstruct PyTorch Datasets from transformed splits
            logger.info("Loading train/test dataset splits into memory tensors...")
            train_df = pd.read_csv(Path(self.ingestion_config["train_data_path"]))
            test_df = pd.read_csv(Path(self.ingestion_config["test_data_path"]))
            
            # Re-initialize the tokenizer transformer engine
            transform_component = DataTransformation()
            
            train_dataset = CustomTextDataset(
                texts=train_df["text"].tolist(),
                labels=train_df[self.ingestion_config["stratify_column"]].tolist(),
                tokenizer=transform_component.tokenizer,
                max_length=self.transformation_config["max_length"]
            )
            
            # 2. Construct DataLoaders to manage minibatch streaming
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.config.batch_size, 
                shuffle=True,
                drop_last=False
            )
            
            # Calculate number of classes from data
            num_labels = len(train_df[self.ingestion_config["stratify_column"]].unique())
            
            # 3. Instantiate the specialized Transformer Model layer
            logger.info(f"Downloading pre-trained weights for: {self.config.model_checkpoint}")
            model = DistilBertForSequenceClassification.from_pretrained(
                self.config.model_checkpoint,
                num_labels=num_labels
            )
            model.to(self.device)
            
            # 4. Initialize Optimization Engines (AdamW & Linear Warmup Scheduler)
            optimizer = AdamW(
                model.parameters(), 
                lr=self.config.learning_rate, 
                weight_decay=self.config.weight_decay
            )
            
            total_training_steps = len(train_loader) * self.config.epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=total_training_steps
            )
            
            # 5. Initialize Scaler for FP16 Mixed Precision training (if enabled and running on CUDA)
            use_amp = self.config.fp16_precision and self.device.type == "cuda"
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            
            # 6. Execute Training Loop
            logger.info("Beginning model fine-tuning process loop...")
            model.train()
            
            for epoch in range(1, self.config.epochs + 1):
                running_loss = 0.0
                
                for step, batch in enumerate(train_loader, 1):
                    optimizer.zero_grad()
                    
                    # Ship current batch tensors to target hardware device
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    
                    # Execute Forward Pass wrapped in Mixed Precision Context Manager
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss
                    
                    # Backward Pass using Scaled Gradients to mitigate underflow errors
                    scaler.scale(loss).backward()
                    
                    # Unscale gradients and step the optimizer/scheduler
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    
                    running_loss += loss.item()
                    
                    if step % self.config.logging_steps == 0:
                        average_step_loss = running_loss / self.config.logging_steps
                        logger.info(f"Epoch [{epoch}/{self.config.epochs}] | Step [{step}/{len(train_loader)}] | Target Loss: {average_step_loss:.4f}")
                        running_loss = 0.0
                        
            # 7. Persist fine-tuned model and state tracking assets
            logger.info(f"Saving fine-tuned model checkpoint state to: {self.config.trained_model_path}")
            self.config.trained_model_path.mkdir(parents=True, exist_ok=True)
            
            # Save the raw PyTorch model weights and structural configurations
            model.save_pretrained(self.config.trained_model_path)
            # Save the companion tokenizers so deployment runs isolated from the web later
            transform_component.tokenizer.save_pretrained(self.config.trained_model_path)
            
            logger.info("Model Training phase completed successfully.")
            
        except Exception as e:
            logger.critical(f"Critical error failure within the Model Training process layer: {str(e)}")
            raise e