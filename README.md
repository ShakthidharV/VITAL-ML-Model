# VITAL_ML_Model — Project documentation

This repository contains a small suite of Jupyter notebooks that implement a machine learning pipeline for risk prediction using an NHANES-derived dataset. The notebooks differ only in the number of top features selected for model training (8, 11 and 15 features respectively). This README documents the dataset assumptions, preprocessing, feature-selection protocols, model training and evaluation, explainability and reproducibility details to help reproduce and extend the work.

Repository contents (relevant files)
- `merged_nhanes_readable(1).csv` — input dataset (not included in repo). Place it in the project root.
- `test_for_8_features.ipynb` — notebook using TOP_N_FEATURES = 8.
- `test_for_11_features.ipynb` — notebook using TOP_N_FEATURES = 11.
- `test_for_15_features.ipynb` — notebook using TOP_N_FEATURES = 15.

Overview and goals
------------------
The notebooks implement a reproducible pipeline to:

- Select a compact set of predictive features (8 / 11 / 15) using ranking strategies.
- Train and evaluate a small collection of classifiers on multiple disease outcomes.
- Produce per-sample risk scores (predicted probabilities) and aggregate performance summaries.

The main motivations are model parsimony (fewer features = easier clinical adoption), robust evaluation (cross-validation) and explainability.

Dataset assumptions and recommended schema
----------------------------------------
The code expects a CSV table where each row is an individual/sample and each column is either a predictor (continuous or categorical), demographic variable, or a binary disease outcome. Recommended column organization (example):

- `id` (optional): unique sample identifier.
- `age`, `sex`, `race`, ... : demographic covariates.
- `feature_1`, `feature_2`, ... : clinical / laboratory predictors (continuous or categorical).
- `disease_X` (binary 0/1): target outcome columns (one or more diseases).

If your CSV deviates, inspect the first data-loading cell in the notebooks and adjust the column names or filtering there.

Preprocessing steps (what the notebooks do / what to check)
---------------------------------------------------------
The notebooks follow these general preprocessing steps (check the code cells for exact implementation details):

1. Read CSV into pandas DataFrame `df` and optional filtered version `df_filtered`.
2. Drop or flag columns with very high missingness (threshold depends on implementation). Biologically informed curation was applied to remove redundant or disease-irrelevant columns, ensuring that only clinically meaningful variables were retained for efficient data handling and biologically grounded model training.
3. Handle missing values: the notebooks may impute or drop rows/columns — inspect the code. Typical strategies:
	- Numeric: median imputation or simple mean.
	- Categorical: mode imputation or adding an explicit "missing" category.
4. Encode categorical variables: one-hot encoding or ordinal encoding depending on the variable and model used.
5. Optional scaling/normalization: tree-based models (RandomForest/XGBoost) do not require feature scaling; linear models and some distance-based approaches do.
6. Filter low-variance features and highly-correlated duplicates (optional). When removing correlated features, a representative is kept.

Feature ranking and selection
---------------------------
Each notebook selects a small subset of the most significant features for every target disease. For each disease, a predefined set of clinically relevant features is provided to the model beforehand. The model then performs automated feature selection to identify additional variables, resulting in a final set of 8, 11, or 15 features — those most important and biologically relevant for training and optimizing the predictive model. Typical strategies implemented or commonly used in similar pipelines are:

- Univariate scoring (e.g., correlation with the binary target, mutual information): quick and interpretable.
- Tree-based feature importance (RandomForest / XGBoost): captures non-linear relationships and feature interactions.

In this model, a Hybrid selection strategy — combining correlation filtering and Random Forest importance was implemented to efficiently capture both linear and non-linear relationships within the data. The correlation step effectively removed redundant or weakly associated variables, while the Random Forest step identified features with the strongest predictive influence on disease outcomes. Together, they yielded a more stable, interpretable, and biologically relevant feature set for model training.

In these notebooks the selected method is implemented in code (check the selection cell). The only deliberate difference among the three notebooks is the value of `TOP_N_FEATURES` (8, 11, or 15).

Modeling details
----------------
Models used and rationale:

- RandomForestClassifier: robust baseline for tabular data, handles mixed feature types, provides feature importance.
- GradientBoostingClassifier / XGBoost: often yields higher accuracy on tabular data via boosting; includes regularization and advanced hyperparameters.
- LogisticRegression: interpretable linear baseline; useful for comparing calibration and coefficient-based interpretability.

Typical hyperparameters (inspect the notebooks for exact values):

- RandomForest: n_estimators (e.g., 100 or 200), max_depth (optional), random_state set to `RANDOM_STATE`.
- XGBoost: use default boosters with tuned learning_rate, n_estimators, max_depth; `random_state` seeded.
- LogisticRegression: use `solver='lbfgs'` or `saga` with `max_iter` increased if needed; consider class_weight='balanced' for imbalanced data.

Upon comparing the performance of the three classifiers, XGBoost consistently produced the most reliable results. While the Random Forest model showed comparatively weaker performance across metrics, Logistic Regression achieved a similar ROC–AUC value to XGBoost. However, XGBoost was ultimately selected for its ability to capture non-linear relationships, model complex feature interactions, handle class imbalance, and apply strong regularization to prevent overfitting. These properties make the XGBoost-based model more robust, interpretable, and biologically meaningful for disease classification and risk score computation.

Training protocol
-----------------
- Dataset was randomly split into 80% training and 20% testing sets. (Only rows with valid binary target values (0/1) were used for training and testing; records with non-informative codes (e.g., 3 – “Rather not say”, 7 – “Don’t know”, 9 – “NaN”) were excluded during preprocessing.)
- Stratified sampling ensured balanced class distribution across disease categories.
- For multi-target experiments (several diseases), models are trained independently per target and results aggregated.

Evaluation metrics and how to interpret them
-------------------------------------------
Key metrics used in the notebooks:

- AUC-ROC: primary metric for ranking quality of predicted risk scores. Values near 1.0 indicate excellent ranking; 0.5 indicates random.
- Accuracy: proportion of correct predictions. Use with caution under class imbalance.
- Precision, Recall, F1-score: use when classes are imbalanced or when false positives/false negatives have different costs.
- Confusion matrix: raw counts of TP, FP, TN, FN 

Choice of threshold for binary predictions
-----------------------------------------
Models output probabilities. For applications requiring a binary decision, a threshold must be chosen. Common approaches:

- Default threshold 0.5 — simple but may not be appropriate for imbalanced data.
- Optimize threshold for a chosen metric (maximize F1, or set recall at a fixed sensitivity).
- Cost-sensitive thresholding (minimize expected cost given FP and FN costs).

Risk scores and exporting results
--------------------------------
- Risk scores are the predicted probability for the positive class saved in `risk_scores_df` (per-sample, per-target).
- Export these as CSV for downstream analysis, population stratification, or visualization.

Explainability and feature importance
------------------------------------
Recommended explainability steps used or suggested by the notebooks:

- Feature importances (from tree-based models) — quick global explanation.
- Partial dependence plots: visualize marginal effect of a feature on predicted outcome.

Reproducibility notes
---------------------
- Use the `RANDOM_STATE` variable present in the notebooks to seed model training and CV splitting.
- For full reproducibility, record library versions (use `pip freeze > requirements.txt`).
- Randomness sources to control: model random_state, NumPy random seed, and any parallel behavior (set `n_jobs=1` if deterministic runs required across machines).

Performance and runtime
-----------------------
- Models like RandomForest and XGBoost scale roughly linearly with number of trees and data size.
- For fast iteration, reduce `n_folds`, `n_estimators`, or sample down the data during development.

How to run locally (recommended)
--------------------------------
1. Create Python virtual environment and activate:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

2. Install dependencies (example):

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyterlab
```

3. Place `merged_nhanes_readable(1).csv` in this repository folder.

4. Start Jupyter Lab and open the notebook you want to run:

```bash
jupyter lab
```

5. Run the notebook cells top-to-bottom. If a cell expects `TOP_N_FEATURES` to be set, that is already set in each notebook (8 / 11 / 15).

Troubleshooting
---------------
- Missing package errors: install the missing package via pip (or add to `requirements.txt`).
- Out of memory: reduce dataset size, reduce number of trees (`n_estimators`), or run on a machine with more RAM.
- Unexpected results after edits: restart kernel and run all cells from top to bottom to ensure a clean state.

