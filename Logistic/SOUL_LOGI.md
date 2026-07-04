# ADEP - Logistic Regression Pipeline Soul

## 🏗️ Core Architecture & Design
The Logistic Regression Pipeline is a specialized data preparation and execution module within the ADEP supervised learning workflow. It ensures that incoming data undergoes structural validation, cleaning, and preprocessing steps required to meet the mathematical assumptions of logistic regression models.

* **Pipeline Workflow stages:**
  1. **Data Ingestion & Structural Formatting:** Enforces lowercase and snake_case column names, enforces appropriate numerical/categorical data types, and drops non-predictive identifiers (IDs, keys, names).
  2. **Handling Missing Values:** Drops columns or rows exceeding a missingness threshold (e.g., >60%). Imputes skewed numeric features with the median, normal numeric features with the mean, and categorical features with the mode or a dedicated "Unknown" class.
  3. **Outlier Treatment:** Identifies outliers via Interquartile Range (IQR) or Z-score. Treats them by capping/Winsorizing (at 1st and 99th percentiles) or applying log transformations to avoid distortion of the decision boundary.
  4. **Categorical Encoding:** Maps binary variables to 0/1, applies One-Hot Encoding to nominal variables (dropping the first column to avoid the dummy variable trap), and applies Ordinal Encoding to structured/hierarchical text features.
  5. **Multicollinearity Mitigation:** Uses a Pearson correlation matrix and Variance Inflation Factor (VIF) checks to identify and remove highly collinear features, preventing standard error inflation and unstable coefficient estimates.
  6. **Feature Scaling:** Standardizes features (Z-score scaling) to center data around a mean of 0 and standard deviation of 1. This is critical for regularized (L1/L2) logistic regression and gradient-based optimization solvers.

## 📈 Current Progress
* `Logistic/SOUL_LOGI.md` [NEW]: Created the core architectural specifications and tracking document for the Logistic prep pipeline.

## 🎯 Next Steps
1. **Pipeline Constructor (`Logistic/pipeline.py`):** Define a modular class or function using `scikit-learn` `Pipeline` and `ColumnTransformer` to sequence the data ingestion, imputation, encoding, and scaling steps.
2. **Preventing Data Leakage:** Ensure all fit-transform steps (imputers, outlier thresholds, scalers) are fitted only on the training split and then transformed on the validation/test splits.
3. **Multicollinearity Filter class:** Build a custom `scikit-learn` compatible transformer that calculates VIF and drops highly collinear columns dynamically during the pipeline fit step.
