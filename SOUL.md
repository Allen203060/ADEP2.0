# ADEP - Project Soul

## 🏗️ Core Architecture & Design
ADEP (Agentic Data Engineering Pipeline) is a multi-agent system built using the Google ADK and Gemini models.
* **Orchestrator Routing:** Evaluates raw CSV structures and guides the workflow into either **Supervised** (Linear/Logistic Regression) or **Unsupervised** (Clustering) prep pipelines based on user target selection.

## 📈 Current Progress
* `utils/shared_environment.py`: Implemented thread-safe `SHARED_GLOBALS` state registry.
* `config/agent_config.py`: Centralized configuration supporting native Google Gemini and LiteLLM model backends.
* `tools/coding_tools.py`: Created in-memory and subprocess executors with self-healing `pip` package installation.
* `agents/coding_agent.py`: Built universal `Coding_Agent` to act as a code generation & execution engine for downstream orchestrators.
* `agents/problem_identifier_agent.py`: Created `Problem_Identifier_Agent` delegating scanning and task categorization to `Coding_Agent` with console HITL target selection.
* `requirements.txt`: Specified core project dependencies (ADK, LiteLLM, pandas, numpy, scikit-learn).
* `utils/observability.py`: Created structured `TraceLogger` categorizing system, thinking, and tool execution logs into dedicated files.
* `tests/`: Created test suite covering state management, environment configurations, self-healing executors, and logger outputs.

## 🎯 Next Steps
1. **Runner Entry Point (`main.py`):** Create the primary command-line execution entry point to run the problem identification and test agent invocation.
2. **Supervised & Unsupervised Abstractions:** Set up directory scaffolds for linear, logistic, and clustering orchestrators.
