import sys
import logging
from pathlib import Path
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Configure centralized logging output format for execution tracking
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(levelname)s - %(module)s]: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training_pipeline_execution.log")
    ]
)
logger = logging.getLogger(__name__)

class TrainingPipeline:
    """
    Enterprise-grade Orchestrator Pipeline.
    Responsible for executing the end-to-end training life cycle sequentially
    without exposing lower-level technical component mechanics.
    """
    def __init__(self, config_path: Path = Path("config/config.yaml")):
        self.config_path = config_path

    def run_pipeline(self) -> None:
        """
        Triggers execution steps across all underlying MLOps architectural modules.
        Handles catch-all structural errors cleanly to isolate crashing layers.
        """
        try:
            logger.info("==================================================")
            logger.info("TRAINING PIPELINE SEQUENCE INITIALIZATION STARTED")
            logger.info("==================================================")

            # Stage 1: Execute Raw Data Ingestion and Stratified Train/Test Splits
            logger.info(">>> Launching Pipeline Stage 1: Data Ingestion <<<")
            ingestion = DataIngestion(config_path=self.config_path)
            train_path, test_path = ingestion.initiate_data_ingestion()
            logger.info(f"Stage 1 Complete. Assets saved: \n - Train: {train_path}\n - Test: {test_path}")

            # Stage 2: Execute Text Normalization Validation Tests
            logger.info(">>> Launching Pipeline Stage 2: Data Transformation Validation <<<")
            transformation = DataTransformation(config_path=self.config_path)
            transformation.run_transformations()
            logger.info("Stage 2 Complete. Tokenizer schemas verified against data matrices.")

            # Stage 3: Trigger Core PyTorch Deep Learning Fine-Tuning Execution Loop
            logger.info(">>> Launching Pipeline Stage 3: Deep Learning Model Training <<<")
            trainer = ModelTrainer(config_path=self.config_path)
            trainer.initiate_model_trainer()
            logger.info(f"Stage 3 Complete. Fine-tuned model checkpoints compiled.")

            logger.info("==================================================")
            logger.info("TRAINING PIPELINE SEQUENCE EXECUTED SUCCESSFULLY")
            logger.info("==================================================")

        except Exception as e:
            logger.critical(f"FATAL: Pipeline execution halted prematurely due to unhandled error: {str(e)}")
            raise e

if __name__ == "__main__":
    # Allows developers to trigger the training loop sequence directly from CLI
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()