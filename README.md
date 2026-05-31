# 🏦 RiskLens — AI-Powered Credit Risk Intelligence Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3.0-7CB9E8?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLaMA3.3-F97316?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-00C853?style=flat-square)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.7683-00F2FE?style=flat-square)

**An enterprise-grade credit risk intelligence system combining LightGBM, SHAP explainability,
surrogate decision rules, and a natural language SQL chatbot in a premium Streamlit dashboard.**

</div>

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Feature Highlights](#3-feature-highlights)
4. [Dataset Overview](#4-dataset-overview)
5. [Preprocessing Strategy](#5-preprocessing-strategy)
6. [Model Selection & Training](#6-model-selection--training)
7. [Class Imbalance Handling](#7-class-imbalance-handling)
8. [Model Evaluation Results](#8-model-evaluation-results)
9. [SHAP Explainability](#9-shap-explainability)
10. [Decision Rules Engine](#10-decision-rules-engine)
11. [NL-to-SQL Chatbot](#11-nl-to-sql-chatbot)
12. [Prompt Engineering Strategy](#12-prompt-engineering-strategy)
13. [Hallucination Prevention](#13-hallucination-prevention)
14. [Dashboard UI](#14-dashboard-ui)
15. [Setup & Deployment](#15-setup--deployment)
16. [Docker Deployment](#16-docker-deployment)
17. [Project Structure](#17-project-structure)
18. [Known Limitations](#18-known-limitations)
19. [Future Improvements](#19-future-improvements)

---

## 1. Project Overview

The **RiskLens Credit Risk Intelligence Platform** is a production-ready AI system designed to
help financial institutions:

- **Assess** individual applicant default risk in real-time using a trained LightGBM model
- **Explain** model decisions through SHAP feature attributions (local + global)
- **Audit** model behavior via human-readable if/then decision rules
- **Explore** portfolio data interactively through a premium EDA dashboard
- **Query** the credit database in plain English via an NL-to-SQL AI chatbot

The platform is built on the **Home Credit Default Risk** dataset (307,511 applicants, 122 features).

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  RiskLens Platform Architecture                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐    ┌────────────────┐    ┌─────────────────────┐  │
│  │ Raw CSV  │───>│ DataLoader     │───>│ DataPreprocessor    │  │
│  │ (307k)   │    │ (src/data/)    │    │ Impute+Encode+Eng.  │  │
│  └──────────┘    └────────────────┘    └─────────┬───────────┘  │
│                                                   │              │
│                            ┌──────────────────────▼────────────┐│
│                            │   ModelTrainer (LightGBM + AUC)   ││
│                            │   scale_pos_weight + early stop   ││
│                            └──────────┬────────────────────────┘│
│                                       │                          │
│              ┌────────────────────────┼───────────────────┐     │
│              │                        │                   │     │
│   ┌──────────▼──────┐   ┌────────────▼────────┐  ┌──────▼───┐ │
│   │ ModelEvaluator  │   │  RiskPredictor       │  │ Rules    │ │
│   │ ROC/PR/F1/Conf. │   │  SHAP TreeExplainer  │  │ Extractor│ │
│   └─────────────────┘   └──────────┬───────────┘  └──────────┘ │
│                                    │                             │
│   ┌────────────────────────────────▼──────────────────────────┐ │
│   │           Streamlit Dashboard (app.py)                     │ │
│   │  Tab 1: EDA  │ Tab 2: Predictor │ Tab 3: Explainability   │ │
│   │  Tab 4: Rules │ Tab 5: NL-SQL Chatbot (Groq + LangChain) │ │
│   └───────────────────────────────────────────────────────────┘ │
│                                                                   │
│   ┌────────────────────────────────────────────────────────────┐ │
│   │                  SQLite Database                            │ │
│   │  applications table: 100k rows + RISK_SCORE + RISK_BAND   │ │
│   └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│   Deployment: Docker + docker-compose + .env secrets            │
└─────────────────────────────────────────────────────────────────┘
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the detailed Mermaid component diagram.

---

## 3. Feature Highlights

| Feature | Description |
|---------|-------------|
| 📊 **EDA Dashboard** | Interactive Plotly visualizations: distributions, correlations, heatmaps, business insights |
| 🔍 **Risk Predictor** | Real-time LightGBM scoring with probability gauge and SHAP explanations |
| 🧠 **Explainability** | Global feature importance, confusion matrix, PR-AUC radar chart, methodology |
| 📋 **Decision Rules** | Surrogate tree: filterable, sortable if/then rules with confidence & support |
| 💬 **NL-to-SQL Chatbot** | Groq LLM converts plain English to safe SQLite queries with business summaries |
| 🐳 **Docker Ready** | Multi-stage Dockerfile + docker-compose with volume mounts and health checks |
| 🧪 **Unit Tested** | 40+ unit tests covering preprocessor, predictor, and rules extractor |

---

## 4. Dataset Overview

**Source**: [Kaggle — Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)

| Attribute | Value |
|-----------|-------|
| Training rows | 307,511 |
| Test rows | 48,744 |
| Features | 122 (raw) |
| Target | `TARGET` (0 = repaid, 1 = defaulted) |
| Class Distribution | ~91.9% repaid / ~8.1% defaulted |
| Missing Values | ~36% of cells have missing values (sparse data) |

### Key Feature Groups

| Group | Examples |
|-------|---------|
| Financial | `AMT_CREDIT`, `AMT_INCOME_TOTAL`, `AMT_ANNUITY`, `AMT_GOODS_PRICE` |
| Demographics | `CODE_GENDER`, `DAYS_BIRTH`, `CNT_CHILDREN`, `NAME_FAMILY_STATUS` |
| Employment | `DAYS_EMPLOYED`, `NAME_INCOME_TYPE`, `NAME_OCCUPATION_TYPE` |
| External Scores | `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` |
| Geographic | `REGION_RATING_CLIENT`, `REGION_POPULATION_RELATIVE` |
| Documents | `FLAG_DOCUMENT_2` through `FLAG_DOCUMENT_21` |

---

## 5. Preprocessing Strategy

The `DataPreprocessor` (`src/data/preprocessor.py`) implements a complete ML pipeline:

### Step 1: Missing Value Removal
- Columns with **> 60% missing values** are dropped (identified on training split only)
- Protects against leaking test-set information into imputation statistics

### Step 2: Feature Engineering (7 derived features)
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `AGE_YEARS` | `DAYS_BIRTH / -365` | Human-readable age |
| `EMPLOYMENT_YEARS` | `DAYS_EMPLOYED / -365` | Fix anomaly (365243 → NaN) |
| `CREDIT_INCOME_RATIO` | `AMT_CREDIT / AMT_INCOME_TOTAL` | Leverage ratio |
| `ANNUITY_INCOME_RATIO` | `AMT_ANNUITY / AMT_INCOME_TOTAL` | Debt burden ratio |
| `CREDIT_TERM_MONTHS` | `AMT_CREDIT / AMT_ANNUITY` | Loan duration proxy |
| `INCOME_PER_PERSON` | `AMT_INCOME_TOTAL / (CNT_FAM_MEMBERS + 1)` | Disposable income |
| `EXT_SOURCE_MEAN` / `EXT_SOURCE_MIN` | Aggregate of EXT_SOURCE_1/2/3 | Composite credit signal |

### Step 3: Imputation
- **Numerical columns**: Median imputation (fit on train split only)
- **Categorical columns**: Fill with `"Unknown"` string

### Step 4: Encoding
| Cardinality | Strategy |
|------------|---------|
| Binary (≤2 unique) | Integer mapping (0/1) |
| Low cardinality (3–10) | One-Hot Encoding (OHE with `handle_unknown='ignore'`) |
| High cardinality (> 10) | Frequency encoding (relative frequency mapping) |

### Step 5: Schema Enforcement
- All column names sanitized for LightGBM JSON compatibility (no special chars)
- Missing inference-time columns filled with `0.0` to match training schema exactly

---

## 6. Model Selection & Training

### Why LightGBM?

| Criterion | Reason |
|-----------|--------|
| **Speed** | Histogram-based gradient boosting is 10–20× faster than XGBoost on large tabular data |
| **Imbalance handling** | Native `scale_pos_weight` parameter directly adjusts the loss gradient |
| **SHAP compatibility** | `shap.TreeExplainer` works natively with LightGBM for exact SHAP values |
| **Tabular performance** | Consistently top performer on Kaggle tabular classification benchmarks |

### Hyperparameters

```python
lgb.LGBMClassifier(
    n_estimators      = 1000,   # with early stopping
    learning_rate     = 0.05,
    num_leaves        = 63,
    max_depth         = 7,
    min_child_samples = 100,
    subsample         = 0.8,    # row subsampling
    subsample_freq    = 1,
    colsample_bytree  = 0.8,    # feature subsampling
    reg_alpha         = 0.1,    # L1 regularization
    reg_lambda        = 1.0,    # L2 regularization
    metric            = 'auc',  # single-metric early stopping
    scale_pos_weight  = ~11.3,  # neg/pos class ratio
    random_state      = 42,
)
```

### Training Protocol
1. **80/20 stratified split** preserves the 8.1% class ratio in both splits
2. **Early stopping** (50 rounds of no AUC improvement) prevents overfitting
3. **Single metric** (`auc`) avoids conflicting signals from multiple metrics

---

## 7. Class Imbalance Handling

The dataset has severe class imbalance: **91.9% negative (repaid) vs 8.1% positive (defaulted)**.

### Approach 1: `scale_pos_weight`
```python
scale_pos_weight = num_negative / num_positive  # ≈ 11.3
```
LightGBM multiplies the gradient for positive-class samples by this factor, forcing the model to
weight incorrect default predictions ~11× more heavily than incorrect repaid predictions.

### Approach 2: Optimal Threshold Selection
Rather than using the default 0.5 classification threshold, we find the **F1-optimal threshold**
via the precision-recall curve:
```python
precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
optimal_threshold = thresholds[np.argmax(f1_scores)]  # → 0.6815
```

### Approach 3: PR-AUC as Primary Metric
We track **PR-AUC** (area under the precision-recall curve) as the primary evaluation metric.
PR-AUC is invariant to class ratio and directly measures how well the model balances precision
and recall for the minority (default) class.

---

## 8. Model Evaluation Results

> All metrics computed on a held-out 20% validation split (stratified).

| Metric | Value |
|--------|-------|
| **ROC-AUC** | **0.7683** |
| **PR-AUC** | **0.2591** |
| **F1-Score (optimal)** | **0.3202** |
| Optimal Threshold | 0.6815 |
| Overall Accuracy | 86% |

### Classification Report (at optimal threshold 0.6815)

```
              precision    recall  f1-score   support

           0       0.94      0.91      0.92     56,538
           1       0.27      0.39      0.32      4,965

    accuracy                           0.86     61,503
   macro avg       0.61      0.65      0.62     61,503
weighted avg       0.89      0.86      0.88     61,503
```

### Confusion Matrix

|  | Predicted: Repaid | Predicted: Default |
|--|--|--|
| **Actual: Repaid** | 51,231 (TN) | 5,307 (FP) |
| **Actual: Default** | 3,007 (FN) | 1,958 (TP) |

**Recall for defaults = 39.4%** — the model captures 2 in 5 actual defaulters at this threshold.
Raising the threshold reduces false positives at the cost of missed defaults.

---

## 9. SHAP Explainability

We use **`shap.TreeExplainer`** for exact, deterministic Shapley value computation:

### How SHAP Works
SHAP assigns each feature a contribution value that represents how much that feature shifted the
prediction away from the model's average prediction (the base value):

```
prediction = base_value + SHAP(EXT_SOURCE_2) + SHAP(AGE_YEARS) + SHAP(CREDIT_INCOME_RATIO) + ...
```

### Global Explanations
The **Top 20 Feature Importance** chart in the Explainability tab shows the model's global
`split_gain` importances — how much each feature reduces impurity across all trees.

**Top 5 globally important features:**
1. `EXT_SOURCE_2` — External credit bureau score (strongest predictor)
2. `EXT_SOURCE_3` — Second external score
3. `EXT_SOURCE_MEAN` — Composite of all three external scores
4. `AGE_YEARS` — Client age
5. `CREDIT_INCOME_RATIO` — Loan amount / annual income ratio

### Local Explanations
In the Risk Predictor tab, SHAP values for a specific applicant show which features pushed
their predicted probability **up** (increases_risk, shown in 🔴 red) or **down** (decreases_risk,
shown in 🟢 green) relative to the base value.

---

## 10. Decision Rules Engine

**File**: `src/ml/rules.py` — `RulesExtractor` class

### Methodology
1. Load the trained LightGBM model and run it on 30,000 training samples → get probabilities
2. Binarize probabilities at 0.5 → `y_surrogate`
3. Fit a **shallow DecisionTreeClassifier** (`max_depth=4`) to predict `y_surrogate` from features
4. Walk every leaf node of the surrogate tree → extract the path conditions as an if/then rule
5. Compute **confidence** (% of samples in the leaf assigned to the predicted class) and
   **support** (% of total portfolio covered by this rule)
6. Translate internal feature names (e.g., `EXT_SOURCE_2`) to human-readable labels

### Sample Rules Generated

**Rule #1 (Low Risk, 34.2% support, 91.3% confidence)**
```
External Credit Score 2 > 0.5391
AND Avg External Credit Score > 0.4876
→ REPAID (avg default prob: 12.4%)
```

**Rule #5 (High Risk, 8.7% support, 78.1% confidence)**
```
External Credit Score 2 ≤ 0.3102
AND Credit-to-Income Ratio > 4.21
→ DEFAULT (avg default prob: 68.9%)
```

### Why Surrogate Trees?
- LightGBM itself has thousands of trees — not human-readable
- A single shallow decision tree mimics the LightGBM boundary with ~85% fidelity
- Rules are auditable, explainable to regulators, and actionable for underwriters

---

## 11. NL-to-SQL Chatbot

**Files**: `src/talk_to_data/`

### Architecture
```
User Question
     │
     ▼
TalkToDataAgent.ask()
     │
     ├─ Step 1: Load conversation history (LangChain WindowMemory, k=5)
     ├─ Step 2: Call Groq LLM (LLaMA-3.3-70b) with schema + NL_TO_SQL_TEMPLATE
     ├─ Step 3: Validate generated SQL (SELECT-only, no forbidden keywords)
     ├─ Step 4: Execute SQL on SQLite via QueryRunner
     ├─ Step 5: Call Groq LLM with ANSWER_TEMPLATE → plain English summary
     └─ Step 6: Save (question, answer) to ConversationBufferWindowMemory
```

### SQLite Database
The `applications` table contains 100,000 enriched records:
- Raw features from `application_train.csv`
- Engineered features: `AGE_YEARS`, `EMPLOYMENT_YEARS`, `CREDIT_INCOME_RATIO`, etc.
- Model predictions: `RISK_SCORE`, `RISK_BAND`

---

## 12. Prompt Engineering Strategy

### NL-to-SQL Prompt Design Principles

1. **Exact Schema Injection**: The system prompt includes the complete `CREATE TABLE` DDL — column
   names, types, and semantic annotations (e.g., `TARGET = 1 means defaulted`). This eliminates
   hallucination of column names.

2. **Strict Output Constraints**: The LLM is instructed to return *only* the raw SQL string —
   no markdown code blocks, no explanations, no prefixes.

3. **Fallback Signal**: If the question cannot be answered from the schema, the model returns
   the exact string `QUERY_ERROR` which the agent catches and converts to a user-friendly message.

4. **Conversation Context**: LangChain `ConversationBufferWindowMemory` (k=5) is injected into
   each prompt, allowing multi-turn follow-up questions.

### Answer Summarization Prompt
A second LLM call uses `ANSWER_TEMPLATE` to convert the SQL result table into a 2–3 sentence
plain English business summary — avoiding database jargon and focusing on actionable insights.

---

## 13. Hallucination Prevention & SQL Validation

The `QueryRunner.validate_sql()` method implements a two-layer defense:

### Layer 1: SQL Structural Validation
```python
# Must start with SELECT
if not clean_sql.strip().upper().startswith("SELECT"):
    return False

# Block destructive keywords
forbidden = ["DROP", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "TRUNCATE", "EXEC", "--"]
for keyword in forbidden:
    if keyword in clean_sql:
        return False
```

### Layer 2: Schema-Anchored Generation
- The system prompt provides the **exact column names** with types
- The LLM is told to return `QUERY_ERROR` for out-of-schema questions
- Only `SELECT` queries reach `cursor.execute()` — no write path exists

### Layer 3: Row Limit
- `cursor.fetchmany(100)` caps result size, preventing resource exhaustion from full-table scans

---

## 14. Dashboard UI

The premium Streamlit UI features:

| Design Element | Implementation |
|----------------|---------------|
| **Dark Glassmorphism** | `backdrop-filter: blur(16px)` + semi-transparent cards |
| **Color System** | Cyan `#00F2FE`, Purple `#7F00FF`, Green `#00E676`, Red `#FF1744` |
| **Typography** | Outfit (headings, 700–900wt) + Inter (body) + JetBrains Mono (code) |
| **Animations** | CSS `@keyframes float`, `fadeInUp`, `pulseIcon` |
| **Plotly Theme** | Custom `style_fig()` with transparent backgrounds and muted grid |
| **KPI Cards** | CSS custom properties + radial gradient glows |
| **Risk Cards** | Color-coded decision badges with gradient top borders |

---

## 15. Setup & Deployment

### Prerequisites
- Python 3.11+
- 8 GB RAM minimum (16 GB recommended for full 307k dataset)
- [Groq API Key](https://console.groq.com) (free tier available)
- `application_train.csv` placed in `./data/`

### Local Setup

```bash
# 1. Clone repository
git clone https://github.com/Gourinandini/AI-Powered-Credit-Risk-Intelligence-Platform.git
cd credit_risk_platform

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Run training pipeline (trains model + extracts rules)
python train_pipeline.py

# 6. Launch dashboard
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### Run Tests

```bash
python -m pytest tests/ -v
```

### Training Pipeline Options

```bash
# Full dataset training (307k rows, ~15-20 min)
python train_pipeline.py

# Quick test with 50k rows
python train_pipeline.py --nrows 50000

# Training only (skip rules extraction)
python train_pipeline.py --skip-rules
```

---

## 16. Docker Deployment

### Quick Start

```bash
# 1. Configure secrets
copy .env.example .env
# Edit .env and add your GROQ_API_KEY

# 2. Build and run
docker-compose up --build

# Access at http://localhost:8501
```

### Architecture

```yaml
services:
  app:
    build: .              # Multi-stage Dockerfile
    ports: ["8501:8501"]
    volumes:
      - ./data:/app/data    # Dataset CSVs (read-only)
      - ./models:/app/models  # Model artifacts (persisted)
      - ./sql:/app/sql        # SQLite database (persisted)
    healthcheck:
      test: curl http://localhost:8501/_stcore/health
      interval: 30s
      retries: 3
```

### Pre-train Before Running Docker

The Docker container **does not auto-train** the model. Run the training pipeline locally first:
```bash
python train_pipeline.py
```
This creates `models/lgbm_model.joblib`, `models/preprocessor.joblib`,
`models/metrics.json`, `models/feature_importances.csv`, and `models/rules.json`.

Then launch Docker — the `./models/` volume mount provides artifacts to the container.

### Resource Requirements

| Resource | Minimum | Recommended |
|---------|---------|-------------|
| CPU | 2 cores | 4 cores |
| RAM | 4 GB | 8 GB |
| Disk | 3 GB | 5 GB |

---

## 17. Project Structure

```
credit_risk_platform/
├── app.py                          # Streamlit dashboard (5-tab UI)
├── train_pipeline.py               # Unified training + rule extraction CLI
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Multi-stage production image
├── docker-compose.yml              # Service orchestration
├── .env.example                    # Environment variable template
├── .env                            # Secrets (not committed to git)
├── .gitignore
├── .dockerignore
├── .streamlit/
│   └── config.toml                 # Streamlit theme configuration
├── data/
│   ├── application_train.csv       # Training data (307k rows)
│   └── application_test.csv        # Kaggle test data
├── models/
│   ├── lgbm_model.joblib           # Trained LightGBM model
│   ├── preprocessor.joblib         # Fitted DataPreprocessor
│   ├── metrics.json                # Validation performance metrics
│   ├── feature_importances.csv     # Top 20 feature importance scores
│   └── rules.json                  # Extracted decision rules
├── sql/
│   ├── schema.sql                  # SQLite table DDL
│   └── credit_risk.db              # SQLite database (100k records)
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py               # CSV ingestion + metadata logging
│   │   └── preprocessor.py         # Full ML preprocessing pipeline
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── train.py                # ModelTrainer (LightGBM + early stop)
│   │   ├── evaluate.py             # ModelEvaluator (ROC/PR/F1/CM)
│   │   ├── predict.py              # RiskPredictor + SHAP TreeExplainer
│   │   └── rules.py                # RulesExtractor (surrogate tree)
│   ├── talk_to_data/
│   │   ├── __init__.py
│   │   ├── nl_to_sql.py            # TalkToDataAgent (Groq + LangChain)
│   │   ├── prompt_templates.py     # System/NL-to-SQL/Answer prompts
│   │   └── query_runner.py         # SQLite executor + SQL validation
│   └── utils/
│       ├── __init__.py
│       ├── config.py               # Environment config constants
│       └── logger.py               # Structured logging setup
├── tests/
│   ├── test_preprocessor.py        # 18 DataPreprocessor unit tests
│   ├── test_predictor.py           # 16 RiskPredictor unit tests
│   └── test_rules.py               # 22 RulesExtractor unit tests
└── docs/
    ├── architecture_diagram.png    # System architecture diagram
    ├── ARCHITECTURE.md             # Detailed component documentation
    └── MODEL_REPORT.md             # Complete model card
```

---

## 18. Known Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| **Training on 100k rows** (not full 307k) | Model sees ~1/3 of available data | Use full dataset with sufficient RAM; adjust `nrows` in config |
| **No Platt/isotonic calibration** | Raw probabilities may not be perfectly calibrated | Apply `CalibratedClassifierCV` post-training for deployment |
| **EXT_SOURCE_1 sparsely available** | ~44% missing; imputed with median | Collect EXT_SOURCE_1 from bureau at application time |
| **NL-to-SQL limited to `applications` table** | Cannot join supplementary bureau tables | Extend schema + prompts for multi-table joins |
| **Surrogate tree fidelity ~85%** | 15% of applicants may get different rules vs model | Use for explanation only, not as primary scoring engine |
| **Groq rate limits** | Free tier: 30 requests/minute | Implement caching or retry logic for production |

---

## 19. Future Improvements

- **Feature Store Integration**: Connect to Feast or Hopsworks for real-time feature retrieval
- **Supplementary Tables**: Merge bureau, previous applications, installments datasets (Kaggle)
- **Model Calibration**: Add `CalibratedClassifierCV(cv=5, method='isotonic')` for calibrated probabilities
- **MLflow Tracking**: Add experiment logging and model registry
- **LIME Integration**: Add local linear approximation as complementary explanation to SHAP
- **API Layer**: Wrap `RiskPredictor.predict()` in a FastAPI REST endpoint for system integration
- **Batch Scoring**: Add scheduled batch prediction job for portfolio re-scoring
- **Monitoring**: Add data drift detection (Evidently AI) and model performance degradation alerts
- **Multi-language Support**: Extend NL-to-SQL to support Hindi/regional languages via translation layer
- **Regulatory Compliance**: Add SR 11-7 model documentation and adverse action reason codes

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

**Gourinandini** — AI-Powered Credit Risk Intelligence Platform  
GitHub: [Gourinandini/AI-Powered-Credit-Risk-Intelligence-Platform](https://github.com/Gourinandini/AI-Powered-Credit-Risk-Intelligence-Platform)
