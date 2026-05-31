"""
Configuration Module.

Loads environment variables using python-dotenv, validates setup, and
exposes central configuration variables as constants.
"""

import os
from dotenv import load_dotenv
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Load local environment variables from .env if present
load_dotenv()

# Central configurations exposed as constants
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")

try:
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))
except ValueError:
    logger.warning("Failed to parse LLM_TEMPERATURE. Defaulting to 0.0.")
    LLM_TEMPERATURE = 0.0

DATA_PATH: str = os.getenv("DATA_PATH", "./data")
DB_PATH: str = os.getenv("DB_PATH", "./sql/credit_risk.db")
MODEL_PATH: str = os.getenv("MODEL_PATH", "./models/lgbm_model.joblib")

def validate_config() -> None:
    """
    Validate that all required configurations are defined.

    Raises:
        ValueError: If a required configuration variable is missing.
    """
    logger.info("Validating configurations...")
    
    missing_vars = []
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        missing_vars.append("GROQ_API_KEY")
    if not DATA_PATH:
        missing_vars.append("DATA_PATH")
    if not DB_PATH:
        missing_vars.append("DB_PATH")
    if not MODEL_PATH:
        missing_vars.append("MODEL_PATH")
        
    if missing_vars:
        error_msg = f"Missing required environment configurations: {', '.join(missing_vars)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    logger.info("Configuration validated successfully.")
