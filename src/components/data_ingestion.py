import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.utils.common import read_yaml, create_directories
import logging

# Initialize logger for tracking ingestion pipeline milestones
logger = logging.getLogger(__name__)

class DataIngestionConfig:
    """
    Data Value Object to strictly hold data ingestion configurations.
    Ensures type safety and clear attribute access.
    """
    def __init__(self, config_dict: dict):
        self.root_dir = Path(config_dict["root_dir"])
        self.raw_data_path = Path(config_dict["raw_data_path"])
        self.train_data_path = Path(config_dict["train_data_path"])
        self.test_data_path = Path(config_dict["test_data_path"])
        self.stratify_column = config_dict["stratify_column"]
        self.test_size = config_dict["test_size"]
        self.random_state = config_dict["random_state"]


class DataIngestion:
    """
    Orchestrates the ingestion phase of the ML pipeline:
    Loads raw data, validates structural integrity, executes stratified splits,
    and exports train/test partitions as immutable artifacts.
    """
    def __init__(self, config_path: Path = Path("config/config.yaml")):
        # Load the centralized configuration file
        full_config = read_yaml(config_path)
        # Extract data_ingestion specific parameters
        self.config = DataIngestionConfig(full_config["data_ingestion"])
        
        # Ensure the output directory structure exists safely
        create_directories([self.config.root_dir])

    def initiate_data_ingestion(self) -> tuple[Path, Path]:
        """
        Executes end-to-end data ingestion.
        
        Returns:
            tuple[Path, Path]: File paths to the generated train and test CSV artifacts.
        """
        logger.info("Initiating Data Ingestion component processing...")
        
        try:
            # 1. Load the raw text dataset
            if not self.config.raw_data_path.exists():
                raise FileNotFoundError(f"Raw data file missing at: {self.config.raw_data_path}")
                
            df = pd.read_csv(self.config.raw_data_path)
            logger.info(f"Successfully loaded raw dataset. Shape: {df.shape}")

            # 2. Production Defensive Check: Validate tracking columns
            if self.config.stratify_column not in df.columns:
                raise KeyError(
                    f"Target column '{self.config.stratify_column}' not found in source dataset. "
                    f"Available columns: {list(df.columns)}"
                )

            # 3. Perform Stratified Splitting to mitigate class imbalance issues
            logger.info(f"Executing stratified split (Test Size: {self.config.test_size * 100}%)")
            
            train_set, test_set = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=df[self.config.stratify_column] # The golden key for class distribution
            )
            
            logger.info(f"Split complete. Train shape: {train_set.shape} | Test shape: {test_set.shape}")

            # 4. Save splits to the isolated artifacts directory
            train_set.to_csv(self.config.train_data_path, index=False)
            test_set.to_csv(self.config.test_data_path, index=False)
            
            logger.info(f"Data ingestion artifacts successfully exported to: {self.config.root_dir}")
            
            return (
                self.config.train_data_path,
                self.config.test_data_path
            )

        except Exception as e:
            logger.critical(f"Pipeline failure during data ingestion execution: {str(e)}")
            raise e