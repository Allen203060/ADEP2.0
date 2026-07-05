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

* **ADK Agent Integration:**
  - `Logistic/logistic_agent.py`: Implements `Logistic_Agent` using `google.adk.agents.Agent`. It registers the preprocessing pipeline and the model training wrapper as `FunctionTool` objects.
  - `Logistic/runner.py`: Bootstraps the agent, configures `SHARED_GLOBALS` with raw datasets, and runs the preprocessing and model fitting tasks.

## 📈 Current Progress
* `Logistic/SOUL_LOGI.md` & `Logistic/SOUL_LOGI`: Tracks architecture, progress, and next steps.
* `Logistic/pipeline.py`: Custom sklearn-compatible preprocessing pipeline fully implemented.
* `Logistic/logistic_agent.py`: Created the Google ADK Agent containing preprocessing and training tools.
* `Logistic/runner.py`: Integrated `InMemoryRunner` and `asyncio.run` supporting dynamic dataset configurations.
* **Environment Setup:** Documented the requirement of `GEMINI_API_KEY` in `.env` for GenAI Client authentication.
* **Model Training:** Implemented 80/20 train-test splits, fitted `LogisticRegression`, evaluated metrics (classification report & ROC-AUC), and serialized the model to `logistic_model.pkl`.
* **Target Binarization:** Updated target preprocessing to automatically binarize multi-class numeric targets (collapsing values > 0 to 1) to address class imbalance and multi-class classification difficulty on datasets like heart disease.
* **High Cardinality Handling:** Configured `ColumnStandardizer` to drop columns with more than 100 unique categories or matching high-cardinality metadata (e.g. `name`, `ticket`, `cabin`) to avoid high dimensionality.
* **De-fragmentation Optimization:** Refactored `CategoricalEncoder` to perform batch concatenation of one-hot columns via `pd.concat` with index alignment, eliminating Pandas performance warnings and slow memory reallocation.

## 🎯 Next Steps
1. **Pipeline Integration:** Integrate this logistic sub-pipeline within the parent ADEP Orchestrator routing workflow.


