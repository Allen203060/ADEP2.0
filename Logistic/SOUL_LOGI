# ADEP - Logistic Regression Pipeline Soul

## 🏗️ Core Architecture & Design
The Logistic Regression Pipeline uses an agentic delegation workflow within the ADEP supervised learning system:
* **Delegation to Coding Agent:** Rather than using a static, hardcoded pipeline script, the `Logistic_Agent` orchestrates preprocessing and model training by formulating detailed descriptive prompts and delegating code generation and execution tasks to the `Coding_Agent` in `agents/coding_agent.py`.
* **Coding Agent Execution:** The `Coding_Agent` generates Python code dynamically based on the dataset shape and variables, executes it in-memory via `execute_python_code` (with direct access to `SHARED_GLOBALS`), and saves the results.
* **Three Operational Rules enforced:**
  1. *The Retention Rule:* Never drop more than 5% of total rows. Prioritize heavy imputation over row deletion.
  2. *The Simplicity Rule:* Do not apply advanced statistical filtering (like Winsorizing or collinearity drops) unless explicitly requested. Stick to standard scaling, imputation, and one-hot encoding.
  3. *The State Hook Rule:* The final generated code block MUST conclude by appending the exact literal lines:
     `SHARED_GLOBALS['logistic_processed_path'] = 'data/processed/logistic_ready_train.csv'`
     `SHARED_GLOBALS['logistic_processed_shape'] = df.shape`
  4. *The Feature Keep Rule:* Do NOT drop a column just because it contains text. You may only drop text/categorical columns if they have more than 15 unique values (e.g. Names, Tickets). Numeric columns must never be dropped based on unique values.
  5. *The No-Truncation Rule:* Load and process 100% of the dataset. Truncation methods like `nrows`, `.head()`, or `.sample()` are strictly forbidden.
  6. *The Reality Rule:* Strictly process the dataframe loaded from the CSV. Writing mock data, dummy arrays, or generating artificial rows to bypass errors is strictly forbidden.
* **Simplified Imputation Strategy:** Enforces imputing all numeric missing values with the median and all categorical missing values with the mode, reducing code generation complexity and avoiding runtime failures.
* **Deterministic State Hook Retrieval:** The runner (`runner.py`) itself retrieves the preprocessed file and updates `SHARED_GLOBALS` directly as a fallback, ensuring robust state management even if the generated script omits it.


* **Model Evaluation Metrics:** Enforces printing of Accuracy, AUC-ROC, Confusion Matrix, and Classification Report.



* **ADK Agent Integration:**
  - `Logistic/logistic_agent.py`: Implements `Logistic_Agent` using `google.adk.agents.Agent` with `AgentTool(agent=coding_agent)`.

  - `Logistic/runner.py`: Bootstraps the agent, configures `SHARED_GLOBALS` with raw datasets, and runs the orchestrator task.
  - `Logistic/pipeline_backup.py`: A backup of the original hardcoded custom preprocessing pipeline for reference.

## 📈 Current Progress
* `Logistic/SOUL_LOGI.md` & `Logistic/SOUL_LOGI`: Tracks architecture, progress, and next steps.
* `Logistic/pipeline_backup.py`: Original custom preprocessing pipeline backed up for reference.
* `Logistic/logistic_agent.py`: Refactored to delegate data engineering and modeling tasks to the `Coding_Agent` dynamically.
* `Logistic/runner.py`: Bootstraps the new agent-to-agent delegation pipeline.
* **Environment Setup:** Documented the requirement of `GEMINI_API_KEY` in `.env` for GenAI Client authentication.

* **Model Training:** Implemented 80/20 train-test splits, fitted `LogisticRegression`, evaluated metrics (classification report & ROC-AUC), and serialized the model to `logistic_model.pkl`.
* **Target Binarization:** Updated target preprocessing to automatically binarize multi-class numeric targets (collapsing values > 0 to 1) to address class imbalance and multi-class classification difficulty on datasets like heart disease.
* **High Cardinality Handling:** Configured `ColumnStandardizer` to drop columns with more than 100 unique categories or matching high-cardinality metadata (e.g. `name`, `ticket`, `cabin`) to avoid high dimensionality.
* **De-fragmentation Optimization:** Refactored `CategoricalEncoder` to perform batch concatenation of one-hot columns via `pd.concat` with index alignment, eliminating Pandas performance warnings and slow memory reallocation.

## 🎯 Next Steps
1. **Pipeline Integration:** Integrate this logistic sub-pipeline within the parent ADEP Orchestrator routing workflow.


