# Model Report — LightGBM Credit Default Classifier

## Model Card Summary

| Attribute | Value |
|-----------|-------|
| **Model Type** | LightGBM Gradient Boosted Decision Tree (GBDT) |
| **Task** | Binary Classification (default prediction) |
| **Training Dataset** | Home Credit Default Risk — application_train.csv (100,000 rows) |
| **Validation Dataset** | 20% stratified split (61,503 rows) |
| **Training Date** | 2026-05 |
| **Artifacts** | `lgbm_model.joblib`, `preprocessor.joblib` |
| **Primary Metric** | ROC-AUC |
| **Threshold** | 0.6815 (F1-optimal) |

---

## Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|---------------|
| **ROC-AUC** | **0.7683** | Model correctly ranks 76.8% of (default, repaid) pairs |
| **PR-AUC** | **0.2591** | Strong for heavily imbalanced data (baseline = 0.081) |
| **F1-Score** | **0.3202** | At optimal threshold; balances precision/recall |
| Accuracy | 86.0% | High, but misleading due to imbalance |
| Recall (default) | 39.4% | Captures ~2 in 5 actual defaulters |
| Precision (default) | 26.9% | Of flagged defaulters, 27% are true defaults |

### Why ROC-AUC of 0.77 is Competitive

The Home Credit dataset is notoriously challenging:
- Severe class imbalance (8.1% positive rate)
- ~36% missing values across features
- Many features are proxies (e.g., DAYS_EMPLOYED, document flags)
- Top Kaggle submissions achieve ROC-AUC ~0.80 using feature-rich ensembles

Our model uses only the `application_train.csv` file (no supplementary tables) and achieves
ROC-AUC 0.7683 — a strong baseline that improves significantly when bureau and installment
tables are incorporated.

---

## Confusion Matrix (Validation Set, threshold=0.6815)

```
                Predicted Repaid    Predicted Default
Actual Repaid       51,231 (TN)          5,307 (FP)
Actual Default       3,007 (FN)          1,958 (TP)
```

- **True Negatives (51,231)**: Correctly identified safe loans — no unnecessary rejection
- **False Positives (5,307)**: Safe loans incorrectly flagged — business cost: lost revenue
- **False Negatives (3,007)**: Defaulters missed — financial loss to institution
- **True Positives (1,958)**: Correctly caught defaulters — primary model objective

---

## Hyperparameter Configuration

```python
lgb.LGBMClassifier(
    n_estimators      = 1000,      # trees (early stopping applied)
    learning_rate     = 0.05,      # conservative learning rate
    num_leaves        = 63,        # 2^6 - 1 leaves (slightly underfit)
    max_depth         = 7,         # bounds tree growth
    min_child_samples = 100,       # regularization: min leaf size
    subsample         = 0.8,       # 80% row sampling per tree
    subsample_freq    = 1,         # apply subsampling at every iteration
    colsample_bytree  = 0.8,       # 80% feature sampling per tree
    reg_alpha         = 0.1,       # L1 regularization (feature sparsity)
    reg_lambda        = 1.0,       # L2 regularization (smoothing)
    metric            = 'auc',     # track single metric for early stopping
    scale_pos_weight  = ~11.3,     # computed as neg_count / pos_count
    random_state      = 42,
    n_jobs            = -1,        # use all available CPU cores
    verbose           = -1,        # silent during training
)
```

### Early Stopping
```python
callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=True),
    lgb.log_evaluation(period=50),
]
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="auc", callbacks=callbacks)
```

Setting `metric='auc'` in the constructor and `eval_metric='auc'` in `fit()` is **critical** — using
both `binary_logloss` and `auc` simultaneously causes early stopping to fire immediately because
`scale_pos_weight` degrades `binary_logloss` while improving `auc`.

---

## Global Feature Importance (Top 20)

| Rank | Feature | Description |
|------|---------|-------------|
| 1 | `EXT_SOURCE_2` | External credit bureau score (strongest signal) |
| 2 | `EXT_SOURCE_3` | Second external credit score |
| 3 | `EXT_SOURCE_MEAN` | Mean of all three external scores |
| 4 | `AGE_YEARS` | Client age in years |
| 5 | `CREDIT_INCOME_RATIO` | Credit amount / annual income |
| 6 | `EMPLOYMENT_YEARS` | Duration of employment |
| 7 | `ANNUITY_INCOME_RATIO` | Monthly annuity / annual income |
| 8 | `CREDIT_TERM_MONTHS` | Credit amount / monthly annuity |
| 9 | `AMT_CREDIT` | Requested credit amount |
| 10 | `INCOME_PER_PERSON` | Income / (family members + 1) |
| 11 | `EXT_SOURCE_MIN` | Minimum external score |
| 12 | `AMT_INCOME_TOTAL` | Annual income |
| 13 | `AMT_ANNUITY` | Monthly annuity payment |
| 14 | `REGION_RATING_CLIENT` | Region creditworthiness rating |
| 15 | `REGION_POPULATION_RELATIVE` | Regional population density |
| 16 | `NAME_INCOME_TYPE` (freq) | Frequency-encoded income type |
| 17 | `CNT_CHILDREN` | Number of children |
| 18 | `CODE_GENDER` | Gender (binary) |
| 19 | `CNT_FAM_MEMBERS` | Family member count |
| 20 | `DOCUMENT_COUNT` | Count of submitted documents |

---

## Preprocessing Summary

| Step | Details |
|------|---------|
| Column dropping | Columns with >60% missing values removed |
| Feature engineering | 7 derived features added (see README §5) |
| Imputation | Numerical: median; Categorical: "Unknown" |
| Encoding | Binary→int, Low-cardinality→OHE, High-cardinality→freq |
| Train/Val split | 80/20 stratified on TARGET |
| Final features | ~80 features after OHE expansion |

---

## Decision Rules (Surrogate Model)

The surrogate `DecisionTreeClassifier` (max_depth=4, min_samples_leaf=200) achieves
**~85% fidelity** to the LightGBM model on training data.

### Sample High-Confidence Rules

**Low Risk Rule (High Support)**
```
IF External Credit Score 2 > 0.54
AND Avg External Credit Score > 0.49
→ PREDICTED: Repaid
   Confidence: ~91% | Support: ~34% | Avg Default Prob: ~12%
```

**High Risk Rule**
```
IF External Credit Score 2 ≤ 0.31
AND Credit-to-Income Ratio > 4.2
→ PREDICTED: Default
   Confidence: ~78% | Support: ~9% | Avg Default Prob: ~69%
```

**Medium Risk Rule (Borderline)**
```
IF External Credit Score 2 > 0.31
AND External Credit Score 2 ≤ 0.54
AND Age (years) ≤ 35
→ PREDICTED: Default (requires manual review)
   Confidence: ~65% | Support: ~12% | Avg Default Prob: ~48%
```

---

## Risk Band Thresholds

The `RiskPredictor` uses **calibrated thresholds** derived from the model's optimal threshold:

| Band | Threshold | Recommendation |
|------|-----------|---------------|
| **Low** | probability < optimal/2 (~0.34) | Auto-approve |
| **Medium** | optimal/2 ≤ prob < optimal (~0.68) | Manual underwriting review |
| **High** | probability ≥ optimal (~0.68) | Decline application |

For batch predictions, **percentile-based thresholds** are used:
- Low: bottom 70th percentile
- Medium: 70th–90th percentile
- High: top 10th percentile (above 90th)

---

## Known Model Limitations

1. **Training subset**: Only 100k of 307k rows used — more data would improve recall
2. **Single table**: Bureau, installment, and balance tables not incorporated
3. **No calibration**: Probabilities are not Platt/isotonic calibrated — may not be perfectly reliable as true probabilities
4. **Static threshold**: Optimal threshold derived on validation set may drift on deployment data
5. **Feature drift**: EXT_SOURCE scores computed by credit bureaus — changes in bureau methodology would affect predictions

---

## Ethical Considerations

- `CODE_GENDER` is included as a feature — in regulated markets, gender-based lending decisions
  may be legally prohibited (Equal Credit Opportunity Act, EU Gender Directive)
- Model should be regularly audited for demographic parity and disparate impact
- Decisions in the "Medium" (manual review) band should involve human underwriters
- All model outputs in the Explainability tab should be provided to applicants upon request
  (EU AI Act Art. 86 — right to explanation)
