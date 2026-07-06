# ADEP - Project Soul

## 🏗️ Core Architecture & Design
ADEP (Agentic Data Engineering Pipeline) is a multi-agent system built using the Google ADK and Gemini models.
* **Orchestrator Routing:** Evaluates raw CSV structures and guides the workflow into either **Supervised** (Linear/Logistic Regression) or **Unsupervised** (Clustering) prep pipelines based on user target selection.

### Logistic Regression Pipeline Architecture
The Logistic Regression Pipeline uses an agentic delegation workflow within the ADEP supervised learning system:
* **Delegation to Coding Agent:** Rather than using a static, hardcoded pipeline script, the `Logistic_Agent` orchestrates preprocessing and model training by formulating detailed descriptive prompts and delegating code generation and execution tasks to the `Coding_Agent` in `agents/coding_agent.py`.
* **Coding Agent Execution:** The `Coding_Agent` generates Python code dynamically based on the dataset shape and variables, executes it in-memory via `execute_python_code` (with direct access to `SHARED_GLOBALS`), and saves the results.
* **Operational Rules Enforced:**
  1. *The Retention Rule:* Never drop more than 5% of total rows. Prioritize heavy imputation over row deletion.
  2. *The Simplicity Rule:* Do not apply advanced statistical filtering (like Winsorizing or collinearity drops) unless explicitly requested. Stick to standard scaling, imputation, and one-hot encoding.
  3. *The State Hook Rule:* The final generated code block MUST conclude by appending the exact literal lines to update `SHARED_GLOBALS`.
  4. *The Feature Keep Rule:* Do NOT drop a column just because it contains text. You may only drop text/categorical columns if they have more than 15 unique values. Numeric columns must never be dropped based on unique values.
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
* `utils/shared_environment.py`: Implemented thread-safe `SHARED_GLOBALS` state registry.
* `config/agent_config.py`: Centralized configuration supporting native Google Gemini and LiteLLM model backends.
* `tools/coding_tools.py`: Created in-memory and subprocess executors with self-healing `pip` package installation.
* `agents/coding_agent.py`: Built universal `Coding_Agent` to act as a code generation & execution engine for downstream orchestrators.
* `agents/problem_identifier_agent.py`: Created `Problem_Identifier_Agent` delegating scanning and task categorization to `Coding_Agent` with console HITL target selection.
* `requirements.txt`: Specified core project dependencies (ADK, LiteLLM, pandas, numpy, scikit-learn).
* `utils/observability.py` / `tools/observability.py`: Fixed `TraceLogger` to initialize/touch log files, ensuring they exist immediately upon setup.
* `tests/`: Verified the full test suite runs and passes cleanly inside the corrected virtual environment, including the integration test (`test_problem_identification.py`) running against `data/raw/raw_dataset.csv`.
* `venv/`: Fixed virtualenv shebangs and activate script paths after the workspace rename.
* `utils/observability.py`: Created structured `TraceLogger` categorizing system, thinking, and tool execution logs into dedicated files.
* `tests/`: Created test suite covering state management, environment configurations, self-healing executors, and logger outputs.
* `Logistic/`: Completed preprocessing pipeline (`pipeline.py`), Google ADK agent (`logistic_agent.py`) with preprocessing and model training tools, and dynamic parameterized runner (`runner.py`) to preprocess data, train models, and serialize outputs.
* **Environment Setup:** Documented the requirement of `GEMINI_API_KEY` in `.env` for GenAI Client authentication.
* **Model Training:** Implemented 80/20 train-test splits, fitted `LogisticRegression`, evaluated metrics (classification report & ROC-AUC), and serialized the model to `logistic_model.pkl`.
* **Target Binarization:** Updated target preprocessing to automatically binarize multi-class numeric targets (collapsing values > 0 to 1) to address class imbalance and multi-class classification difficulty on datasets like heart disease.
* **High Cardinality Handling:** Configured `ColumnStandardizer` to drop columns with more than 100 unique categories or matching high-cardinality metadata (e.g. `name`, `ticket`, `cabin`) to avoid high dimensionality.
* **De-fragmentation Optimization:** Refactored `CategoricalEncoder` to perform batch concatenation of one-hot columns via `pd.concat` with index alignment, eliminating Pandas performance warnings and slow memory reallocation.

## 🎯 Next Steps
1. **Runner Entry Point (`main.py`):** Create the primary command-line execution entry point to run the problem identification and test agent invocation.
2. **Supervised & Unsupervised Abstractions:** Set up directory scaffolds for linear, logistic, and clustering orchestrators.
3. **Pipeline Integration:** Integrate the logistic sub-pipeline within the parent ADEP Orchestrator routing workflow.
