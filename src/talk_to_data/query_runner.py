"""
SQLite Query Runner Module.

Defines the QueryRunner class which manages connections to the SQLite database,
seeds database records from raw csv data, validates SELECT queries for security,
and executes SQL statements.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from src.utils.config import DB_PATH, DATA_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)

class QueryRunner:
    """
    QueryRunner handles SQLite database operations, including schema setup,
    data seeding, query verification, and execution.
    """

    def __init__(self) -> None:
        """Initialize connection and set up schema and initial seed data."""
        logger.info("Initializing QueryRunner...")
        
        # Ensure directory for DB exists
        db_dir = os.path.dirname(DB_PATH)
        os.makedirs(db_dir, exist_ok=True)
        
        try:
            self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            logger.info(f"Connected to SQLite database at {DB_PATH}")
            self._init_schema()
            self._seed_if_empty()
            self._update_risk_scores_if_stale()
        except Exception as e:
            logger.error(f"Error initializing SQLite database: {str(e)}")
            raise e

    def validate_sql(self, sql_string: str) -> bool:
        """
        Validate that the query starts with SELECT and is free of destructive keywords.

        Args:
            sql_string (str): The SQL query string.

        Returns:
            bool: True if safe, False otherwise.
        """
        clean_sql = sql_string.strip().upper()
        
        # Must start with SELECT
        if not clean_sql.startswith("SELECT"):
            return False
            
        # Forbidden keywords that indicate write, drop, comment tricks or multiple queries
        forbidden = ["DROP", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "TRUNCATE", "EXEC", "--"]
        for keyword in forbidden:
            if keyword in clean_sql:
                return False
                
        return True

    def execute(self, sql_string: str) -> Dict[str, Any]:
        """
        Safely execute a SQL SELECT statement and return the rows and columns.

        Args:
            sql_string (str): SQL statement to execute.

        Returns:
            Dict[str, Any]: Columns, rows, row count, and error strings.
        """
        if not self.validate_sql(sql_string):
            logger.warning(f"Blocked unsafe SQL query: {sql_string}")
            return {
                "columns": [],
                "rows": [],
                "row_count": 0,
                "error": "unsafe query blocked"
            }
            
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql_string)
            
            # Fetch table column descriptions
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            
            # Retrieve max 100 rows
            rows = cursor.fetchmany(100)
            
            return {
                "columns": columns,
                "rows": [list(row) for row in rows],
                "row_count": len(rows),
                "error": None
            }
        except Exception as e:
            logger.error(f"SQLite syntax/execution error: {str(e)} for query: {sql_string}")
            return {
                "columns": [],
                "rows": [],
                "row_count": 0,
                "error": str(e)
            }

    def get_schema_string(self) -> str:
        """
        Return the database schema in a readable markdown-like layout.

        Returns:
            str: Schema columns description.
        """
        return """
Table: applications
Columns:
- SK_ID_CURR (INTEGER, PRIMARY KEY)
- TARGET (INTEGER) -- 0 = repaid, 1 = defaulted
- NAME_CONTRACT_TYPE (TEXT)
- AMT_CREDIT (REAL)
- AMT_ANNUITY (REAL)
- AMT_INCOME_TOTAL (REAL)
- AMT_GOODS_PRICE (REAL)
- CODE_GENDER (TEXT)
- AGE_YEARS (REAL)
- CNT_CHILDREN (INTEGER)
- CNT_FAM_MEMBERS (REAL)
- NAME_FAMILY_STATUS (TEXT)
- NAME_EDUCATION_TYPE (TEXT)
- NAME_INCOME_TYPE (TEXT)
- EMPLOYMENT_YEARS (REAL)
- NAME_OCCUPATION_TYPE (TEXT)
- NAME_HOUSING_TYPE (TEXT)
- REGION_POPULATION_RELATIVE (REAL)
- REGION_RATING_CLIENT (INTEGER)
- EXT_SOURCE_1 (REAL)
- EXT_SOURCE_2 (REAL)
- EXT_SOURCE_3 (REAL)
- EXT_SOURCE_MEAN (REAL)
- CREDIT_INCOME_RATIO (REAL)
- ANNUITY_INCOME_RATIO (REAL)
- CREDIT_TERM_MONTHS (REAL)
- RISK_SCORE (REAL)
- RISK_BAND (TEXT) -- 'Low', 'Medium', 'High'
"""

    def _init_schema(self) -> None:
        """Create the applications table if it does not exist using sql/schema.sql."""
        schema_path = "./sql/schema.sql"
        if not os.path.exists(schema_path):
            schema_path = os.path.join(os.path.dirname(__file__), "../../sql/schema.sql")
            
        with open(schema_path, "r") as f:
            schema_sql = f.read()
            
        self.conn.executescript(schema_sql)
        self.conn.commit()
        logger.info("Database schema initialized.")

    def _seed_if_empty(self) -> None:
        """Checks if the table is empty, and if so, seeds it with 100,000 rows.
        After seeding, _update_risk_scores_if_stale() will patch predictions if needed."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM applications")
        count = cursor.fetchone()[0]
        
        if count > 0:
            logger.info(f"Database already populated with {count} rows. Seeding skipped.")
            return
            
        csv_path = os.path.join(DATA_PATH, "application_train.csv")
        if not os.path.exists(csv_path):
            logger.warning(f"Seeding skipped: {csv_path} not found. Please run training pipeline first.")
            return
            
        logger.info("Database is empty. Preparing to seed 100,000 records...")
        
        try:
            # Read first 100k rows
            df = pd.read_csv(csv_path, nrows=100000)
            
            # Map dataset column name to schema column name
            if "OCCUPATION_TYPE" in df.columns:
                df = df.rename(columns={"OCCUPATION_TYPE": "NAME_OCCUPATION_TYPE"})
            
            # Apply engineered features needed for schema
            df["AGE_YEARS"] = df["DAYS_BIRTH"] / -365.0
            
            df["DAYS_EMPLOYED_CLEAN"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
            df["EMPLOYMENT_YEARS"] = df["DAYS_EMPLOYED_CLEAN"] / -365.0
            
            df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
            df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
            df["CREDIT_TERM_MONTHS"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
            
            ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
            df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
            
            # Batch score risk if model is trained
            df = self._add_predictions(df)
            
            # Select schema columns
            schema_cols = [
                "SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "AMT_CREDIT", "AMT_ANNUITY",
                "AMT_INCOME_TOTAL", "AMT_GOODS_PRICE", "CODE_GENDER", "AGE_YEARS", "CNT_CHILDREN",
                "CNT_FAM_MEMBERS", "NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE",
                "EMPLOYMENT_YEARS", "NAME_OCCUPATION_TYPE", "NAME_HOUSING_TYPE",
                "REGION_POPULATION_RELATIVE", "REGION_RATING_CLIENT", "EXT_SOURCE_1",
                "EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_SOURCE_MEAN", "CREDIT_INCOME_RATIO",
                "ANNUITY_INCOME_RATIO", "CREDIT_TERM_MONTHS", "RISK_SCORE", "RISK_BAND"
            ]
            
            df_to_insert = df[schema_cols].copy()
            
            # Insert into database
            df_to_insert.to_sql("applications", self.conn, if_exists="append", index=False)
            logger.info("Successfully seeded database with 100,000 records.")
            
        except Exception as e:
            logger.error(f"Error seeding database: {str(e)}")

    def _add_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies model predictions if trained, otherwise populates dummy predictions."""
        try:
            from src.ml.predict import RiskPredictor
            predictor = RiskPredictor()
            df = predictor.batch_predict(df)
            logger.info("Risk scores successfully generated for database seeding.")
        except Exception as e:
            logger.warning(f"Could not load RiskPredictor for DB seeding ({str(e)}). Using default risk columns.")
            df["RISK_SCORE"] = 0.0
            df["RISK_BAND"] = "Low"
        return df

    def _update_risk_scores_if_stale(self) -> None:
        """
        Detect and repair stale (dummy) RISK_SCORE / RISK_BAND values.

        When the database is seeded before the ML model is available, all rows get
        RISK_SCORE=0.0 and RISK_BAND='Low'. This method checks for that condition and,
        if the model artifact now exists, re-reads the source CSV, generates real batch
        predictions, and updates every row in the database in-place.
        """
        try:
            cursor = self.conn.cursor()

            # Check for stale predictions under two conditions:
            # 1. All RISK_SCOREs are 0.0 (seeded before model was available)
            # 2. All RISK_BANDs are 'Low' (seeded with old fixed 0.3/0.6 thresholds that
            #    didn't match this model's calibrated probability range)
            cursor.execute("SELECT COUNT(*) FROM applications WHERE RISK_SCORE = 0.0")
            all_dummy = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT RISK_BAND) FROM applications")
            distinct_bands = cursor.fetchone()[0]

            if all_dummy == 0 and distinct_bands > 1:
                logger.info(
                    f"Risk scores already populated with {distinct_bands} distinct bands. Skipping update."
                )
                return

            # Verify model artifact exists before attempting to load
            from src.utils.config import MODEL_PATH
            if not os.path.exists(MODEL_PATH):
                logger.warning(
                    "Risk scores are stale but model artifact not found. Run training pipeline first."
                )
                return

            # Source CSV must still be available to re-generate features
            csv_path = os.path.join(DATA_PATH, "application_train.csv")
            if not os.path.exists(csv_path):
                logger.warning(
                    "Risk scores are stale but source CSV not found. Cannot update risk scores."
                )
                return

            logger.info(
                "Stale risk scores detected (all rows have dummy defaults). "
                "Re-scoring database with trained model..."
            )

            # Rebuild the feature DataFrame from the CSV (same logic as _seed_if_empty)
            df = pd.read_csv(csv_path, nrows=100000)

            if "OCCUPATION_TYPE" in df.columns:
                df = df.rename(columns={"OCCUPATION_TYPE": "NAME_OCCUPATION_TYPE"})

            df["AGE_YEARS"] = df["DAYS_BIRTH"] / -365.0
            df["DAYS_EMPLOYED_CLEAN"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
            df["EMPLOYMENT_YEARS"] = df["DAYS_EMPLOYED_CLEAN"] / -365.0
            df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
            df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
            df["CREDIT_TERM_MONTHS"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]

            ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
            df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)

            # Generate real predictions using the trained model
            df = self._add_predictions(df)

            if df["RISK_BAND"].eq("Low").all() and df["RISK_SCORE"].eq(0.0).all():
                logger.warning("Prediction still returned dummy values. Model may have failed to load.")
                return

            # Batch UPDATE the SQLite rows (in chunks to avoid memory issues)
            update_records = df[["SK_ID_CURR", "RISK_SCORE", "RISK_BAND"]].values.tolist()
            chunk_size = 5000
            for i in range(0, len(update_records), chunk_size):
                chunk = update_records[i : i + chunk_size]
                cursor.executemany(
                    "UPDATE applications SET RISK_SCORE = ?, RISK_BAND = ? WHERE SK_ID_CURR = ?",
                    [(row[1], row[2], row[0]) for row in chunk],
                )
                self.conn.commit()
                logger.info(f"Updated risk scores for rows {i} to {min(i + chunk_size, len(update_records))}.")

            logger.info("Successfully updated all risk scores in the database.")

        except Exception as e:
            logger.error(f"Error updating stale risk scores: {str(e)}")
