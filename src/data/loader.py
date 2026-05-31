"""
Data Loading Module.

Defines the DataLoader class responsible for reading application data
and logging descriptive metadata.
"""

import os
import pandas as pd
from typing import Dict, Any
from src.utils.config import DATA_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DataLoader:
    """
    DataLoader handles the ingestion and analysis of credit risk platform datasets.
    """

    def __init__(self) -> None:
        """Initialize the DataLoader with an empty DataFrame."""
        self.df: pd.DataFrame = pd.DataFrame()

    def load_application_train(self, nrows: int | None = None) -> pd.DataFrame:
        """
        Load the application_train.csv dataset from the configured DATA_PATH.

        Args:
            nrows (int | None): Number of rows to read.

        Returns:
            pd.DataFrame: The loaded training dataset.
        """
        file_path = os.path.join(DATA_PATH, "application_train.csv")
        logger.info(f"Attempting to load data from {file_path} (nrows={nrows})")
        
        if not os.path.exists(file_path):
            error_msg = f"Dataset file not found at: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            self.df = pd.read_csv(file_path, nrows=nrows)
            logger.info("Dataset loaded successfully.")
            
            # Log shape and basic dataset details
            logger.info(f"Dataset Shape: {self.df.shape}")
            
            if "TARGET" in self.df.columns:
                dist = self.get_target_distribution()
                logger.info(f"TARGET Distribution: {dist}")
                
            return self.df
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise e

    def get_target_distribution(self) -> Dict[str, Any]:
        """
        Calculate and return counts and percentages for each target class.

        Returns:
            Dict[str, Any]: Target class counts and percentages.
        """
        if self.df.empty or "TARGET" not in self.df.columns:
            logger.warning("Dataset not loaded or TARGET column missing.")
            return {}
            
        counts = self.df["TARGET"].value_counts().to_dict()
        total = len(self.df)
        
        distribution = {}
        for target_val in [0, 1]:
            count = counts.get(target_val, 0)
            pct = (count / total) * 100 if total > 0 else 0.0
            distribution[str(target_val)] = {
                "count": int(count),
                "percentage": round(pct, 4)
            }
        return distribution

    def get_dataset_summary(self) -> Dict[str, Any]:
        """
        Gather dataset shape, missing percentage, and type breakdowns.

        Returns:
            Dict[str, Any]: Summarized metrics of the dataset.
        """
        if self.df.empty:
            logger.warning("Dataset is empty. Returning empty summary.")
            return {"rows": 0, "columns": 0, "missing_percent": 0.0, "dtypes_breakdown": {}}
            
        rows, cols = self.df.shape
        total_elements = rows * cols
        missing_count = int(self.df.isnull().sum().sum())
        missing_pct = (missing_count / total_elements) * 100 if total_elements > 0 else 0.0
        
        dtypes_counts = self.df.dtypes.value_counts().to_dict()
        dtypes_breakdown = {str(k): int(v) for k, v in dtypes_counts.items()}
        
        return {
            "rows": rows,
            "columns": cols,
            "missing_percent": round(missing_pct, 4),
            "dtypes_breakdown": dtypes_breakdown
        }
