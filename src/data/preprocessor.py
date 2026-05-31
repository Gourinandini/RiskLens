"""
Data Preprocessing Module.

Defines the DataPreprocessor class that prepares the raw Credit Risk dataset
for training and inference through cleaning, feature engineering, imputation,
and categorical encoding.
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DataPreprocessor:
    """
    DataPreprocessor handles the end-to-end cleaning, imputation,
    feature engineering, and encoding of the input dataset.
    """

    def __init__(self) -> None:
        """Initialize the DataPreprocessor with empty state variables."""
        self.cols_to_drop: List[str] = []
        self.numerical_cols: List[str] = []
        self.all_cat_cols: List[str] = []
        self.binary_cols: List[str] = []
        self.low_card_cols: List[str] = []
        self.high_card_cols: List[str] = []
        
        self.medians: Dict[str, float] = {}
        self.binary_mappings: Dict[str, Dict[Any, int]] = {}
        self.ohe: OneHotEncoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.freq_maps: Dict[str, Dict[str, float]] = {}
        self.final_feature_names: List[str] = []

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Fits the preprocessor state on the training split of df and transforms both splits.

        Args:
            df (pd.DataFrame): The raw dataset containing features and the TARGET column.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_val, y_train, y_val.
        """
        logger.info("Starting fit_transform on dataset...")
        
        if "TARGET" not in df.columns:
            error_msg = "TARGET column missing from input data during fit_transform."
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Separate features and target
        y = df["TARGET"]
        X_raw = df.drop(columns=["TARGET"])
        
        # Stratified train-val split
        logger.info("Performing 80/20 train-validation split...")
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X_raw, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 1. Identify columns with >60% missing in training features (ignoring ID columns)
        missing_ratios = X_train_raw.isnull().mean()
        self.cols_to_drop = missing_ratios[missing_ratios > 0.60].index.tolist()
        if "SK_ID_CURR" in self.cols_to_drop:
            self.cols_to_drop.remove("SK_ID_CURR")
        logger.info(f"Dropping {len(self.cols_to_drop)} columns with >60% missing values.")
        
        # Apply drop
        X_train_clean = X_train_raw.drop(columns=self.cols_to_drop, errors="ignore")
        
        # 2 & 3. Feature engineering & fix DAYS_EMPLOYED anomaly
        X_train_eng = self._engineer_features(X_train_clean)
        
        # Separate numeric and categorical
        # Exclude SK_ID_CURR from feature columns
        feat_cols = [c for c in X_train_eng.columns if c != "SK_ID_CURR"]
        
        for col in feat_cols:
            if X_train_eng[col].dtype == "object" or X_train_eng[col].dtype.name == "category":
                self.all_cat_cols.append(col)
            else:
                self.numerical_cols.append(col)
                
        # 4. Fit Imputation
        # Numerical: median
        for col in self.numerical_cols:
            median_val = X_train_eng[col].median()
            # If all are NaN, default to 0.0
            if pd.isna(median_val):
                median_val = 0.0
            self.medians[col] = float(median_val)
            X_train_eng[col] = X_train_eng[col].fillna(self.medians[col])
            
        # Categorical: fill with "Unknown"
        for col in self.all_cat_cols:
            X_train_eng[col] = X_train_eng[col].fillna("Unknown").astype(str)
            
        # 5. Fit Encoding
        self._fit_encoders(X_train_eng)
        
        # Transform train features
        X_train = self._transform_features_only(X_train_eng)
        self.final_feature_names = X_train.columns.tolist()
        logger.info(f"Preprocessed dataset has {len(self.final_feature_names)} features.")
        
        # Transform val features
        X_val = self.transform(X_val_raw)
        
        return X_train, X_val, y_train, y_val

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw data for inference or validation.

        Args:
            df (pd.DataFrame): Raw input data (single row or batch).

        Returns:
            pd.DataFrame: Preprocessed feature matrix matching fitted schema.
        """
        # Ensure TARGET is not in df
        df_clean = df.drop(columns=["TARGET"], errors="ignore")
        
        # Drop columns with >60% missing values
        df_clean = df_clean.drop(columns=self.cols_to_drop, errors="ignore")
        
        # Apply engineering
        df_eng = self._engineer_features(df_clean)
        
        # Apply transformations
        return self._transform_features_only(df_eng)

    def get_feature_names(self) -> List[str]:
        """
        Get the list of final feature names.

        Returns:
            List[str]: Preprocessed feature names.
        """
        return self.final_feature_names

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Internal method to engineer features and fix the DAYS_EMPLOYED anomaly.
        """
        df = df.copy()
        
        # STEP 2 — Fix DAYS_EMPLOYED anomaly
        if "DAYS_EMPLOYED" in df.columns:
            df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
            
        # STEP 3 — Create engineered features
        # AGE_YEARS = DAYS_BIRTH / -365
        if "DAYS_BIRTH" in df.columns:
            df["AGE_YEARS"] = df["DAYS_BIRTH"] / -365.0
        else:
            df["AGE_YEARS"] = np.nan
            
        # EMPLOYMENT_YEARS = DAYS_EMPLOYED / -365
        if "DAYS_EMPLOYED" in df.columns:
            df["EMPLOYMENT_YEARS"] = df["DAYS_EMPLOYED"] / -365.0
        else:
            df["EMPLOYMENT_YEARS"] = np.nan
            
        # RATIOS
        if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
            df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
        else:
            df["CREDIT_INCOME_RATIO"] = np.nan
            
        if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
            df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
        else:
            df["ANNUITY_INCOME_RATIO"] = np.nan
            
        if "AMT_CREDIT" in df.columns and "AMT_ANNUITY" in df.columns:
            df["CREDIT_TERM_MONTHS"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
        else:
            df["CREDIT_TERM_MONTHS"] = np.nan
            
        if "AMT_INCOME_TOTAL" in df.columns:
            if "CNT_FAM_MEMBERS" in df.columns:
                fam_mem = df["CNT_FAM_MEMBERS"].fillna(1.0)
            else:
                fam_mem = 1.0
            df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (fam_mem + 1.0)
        else:
            df["INCOME_PER_PERSON"] = np.nan
            
        # EXT_SOURCE stats
        ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
        if ext_cols:
            df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
            df["EXT_SOURCE_MIN"] = df[ext_cols].min(axis=1)
        else:
            df["EXT_SOURCE_MEAN"] = np.nan
            df["EXT_SOURCE_MIN"] = np.nan
            
        # DOCUMENT_COUNT
        doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
        if doc_cols:
            df["DOCUMENT_COUNT"] = df[doc_cols].sum(axis=1)
        else:
            df["DOCUMENT_COUNT"] = 0
            
        # STEP 6 — Drop original DAYS_* columns
        days_cols = [c for c in df.columns if c.startswith("DAYS_")]
        df = df.drop(columns=days_cols, errors="ignore")
        
        return df

    def _fit_encoders(self, df: pd.DataFrame) -> None:
        """
        Internal method to fit encoders (binary, one-hot, and frequency) on features.
        """
        for col in self.all_cat_cols:
            unique_cats = df[col].unique().tolist()
            num_unique = len(unique_cats)
            
            if num_unique <= 2:
                self.binary_cols.append(col)
                self.binary_mappings[col] = {cat: idx for idx, cat in enumerate(unique_cats)}
            elif num_unique <= 10:
                self.low_card_cols.append(col)
            else:
                self.high_card_cols.append(col)
                # Compute frequency map
                self.freq_maps[col] = df[col].value_counts(normalize=True).to_dict()
                
        # Fit One-Hot Encoder on low-cardinality features
        if self.low_card_cols:
            self.ohe.fit(df[self.low_card_cols])

    def _transform_features_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply imputation and encoding transformations.

        Memory-efficient implementation: every block is built as a float32
        numpy array, all blocks are concatenated in a single np.concatenate
        call (no intermediate pandas copies / block consolidation), and only
        ONE DataFrame is created at the very end.
        """
        n = len(df)
        all_arrays: list[np.ndarray] = []   # shape (n,) or (n, k)
        all_col_names: list[str] = []

        # ── Numerical block ─────────────────────────────────────────────────
        for col in self.numerical_cols:
            fill_val = np.float32(self.medians.get(col, 0.0))
            if col in df.columns:
                arr = df[col].to_numpy(dtype=np.float32, na_value=fill_val)
            else:
                arr = np.full(n, fill_val, dtype=np.float32)
            all_arrays.append(arr)
            all_col_names.append(col)

        # ── Binary block ─────────────────────────────────────────────────────
        for col in self.binary_cols:
            mapping = self.binary_mappings.get(col, {})
            if col in df.columns:
                raw = df[col].fillna("Unknown").astype(str)
                arr = raw.map(lambda v, m=mapping: m.get(v, 0)).to_numpy(dtype=np.float32)
            else:
                arr = np.zeros(n, dtype=np.float32)
            all_arrays.append(arr)
            all_col_names.append(col)

        # ── One-Hot block ────────────────────────────────────────────────────
        if self.low_card_cols:
            ohe_in_data = {}
            for col in self.low_card_cols:
                if col in df.columns:
                    ohe_in_data[col] = df[col].fillna("Unknown").astype(str).values
                else:
                    ohe_in_data[col] = np.full(n, "Unknown")
            ohe_in = pd.DataFrame(ohe_in_data)
            ohe_out = self.ohe.transform(ohe_in).astype(np.float32)  # (n, k)
            ohe_col_names = self.ohe.get_feature_names_out(self.low_card_cols).tolist()
            # Append columns individually so all_arrays stays 1-D per entry
            for i, cname in enumerate(ohe_col_names):
                all_arrays.append(ohe_out[:, i])
                all_col_names.append(cname)

        # ── Frequency block ──────────────────────────────────────────────────
        for col in self.high_card_cols:
            freq_map = self.freq_maps.get(col, {})
            if col in df.columns:
                raw = df[col].fillna("Unknown").astype(str)
                arr = raw.map(freq_map).fillna(0.0).to_numpy(dtype=np.float32)
            else:
                arr = np.zeros(n, dtype=np.float32)
            all_arrays.append(arr)
            all_col_names.append(col)

        # ── Single numpy concatenation → one DataFrame ───────────────────────
        matrix = np.column_stack(all_arrays)   # (n, total_features) — one allocation
        X = pd.DataFrame(matrix, columns=all_col_names)

        # ── Sanitise column names (LightGBM JSON compatibility) ──────────────
        cleaned_cols: list[str] = []
        seen_cols: dict[str, int] = {}
        for col in X.columns:
            cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', str(col))
            cleaned = re.sub(r'_+', '_', cleaned).strip('_') or 'col'
            orig = cleaned
            counter = 1
            while cleaned in seen_cols:
                cleaned = f"{orig}_{counter}"
                counter += 1
            seen_cols[cleaned] = 0
            cleaned_cols.append(cleaned)
        X.columns = cleaned_cols

        # ── Enforce training feature schema (order + missing cols) ───────────
        if self.final_feature_names:
            for c in set(self.final_feature_names) - set(X.columns):
                X[c] = np.float32(0.0)
            X = X[self.final_feature_names]

        return X

