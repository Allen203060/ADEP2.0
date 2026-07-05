import os
from google.adk.agents import Agent
from google.adk.tools import AgentTool
from config.agent_config import config
from agents.coding_agent import create_coding_agent

def create_logistic_agent(model=config.MODEL) -> Agent:
    """
    Constructs and returns the Google ADK Agent for logistic regression,
    which delegates preprocessing and training code generation to the Coding Agent
    while enforcing Retention, Simplicity, Feature Keep, No-Truncation, and literal State Hook rules.
    """
    coding_agent = create_coding_agent(model=model)
    
    return Agent(
        name="Logistic_Agent",
        model=model,
        tools=[
            AgentTool(agent=coding_agent)
        ],
        description="An agent that orchestrates logistic regression preprocessing and training by prompting the Coding Agent.",
        instruction="""
        You are the Logistic Regression Pipeline Orchestrator.
        Your goal is to coordinate data preprocessing and training of a Logistic Regression model.
        
        You DO NOT execute code or write scripts directly. You must delegate all coding and execution tasks to the 'Coding_Agent' tool.
        
        CRITICAL RULES FOR GENERATED CODE:
        1. The Retention Rule: You must never drop more than 5% of the total rows in the dataset. Prioritize heavy imputation over row deletion.
        2. The Simplicity Rule: Do not apply advanced statistical filtering (like Winsorizing or collinearity drops) unless explicitly requested. Stick to standard scaling, imputation, and one-hot encoding.
        3. The State Hook Rule: At the very end of your generated Python script, you MUST append these exact lines of code to update the global memory. Do not change the dictionary keys:
           SHARED_GLOBALS['logistic_processed_path'] = 'data/processed/logistic_ready_train.csv'
           SHARED_GLOBALS['logistic_processed_shape'] = df.shape
        4. The Feature Keep Rule: Do NOT drop a column just because it contains text. You may only drop text/categorical columns if they have more than 15 unique values (e.g., Names, Tickets). You MUST retain low-cardinality columns (like gender, class, or location) and apply one-hot encoding to them.
        5. The No-Truncation Rule: You MUST load and process 100% of the dataset. NEVER use `nrows`, `.head()`, `.sample()`, or any truncation methods when loading or processing the data.
        
        STEPS:
        1. Query the 'Coding_Agent' tool to preprocess the raw CSV dataset located at SHARED_GLOBALS['file_path'] for the target column specified in SHARED_GLOBALS['target_column'].
           Your prompt to the 'Coding_Agent' must instruct it to generate and run Python code that does the following:
           a. Load the ENTIRE dataset from SHARED_GLOBALS['file_path'] using pd.read_csv(). Do NOT use the nrows parameter or any truncation method.
           b. Handle target binarization: if the target column has more than 2 unique values and is numeric, collapse it into binary classes (values > 0 become 1, else 0). If it has exactly 2 unique values, map them to 0 and 1.
           c. Adhere to the Feature Keep Rule: Drop ID columns, passenger names, tickets, cabins, or any text/object column with high cardinality (more than 15 unique values) to prevent high dimensionality. Retain low-cardinality columns (like gender, class, or location) for model training.
           d. Adhere to the Retention Rule: Prioritize heavy imputation over row deletion, and do not drop more than 5% of the total rows.
           e. Impute ALL missing numeric values with the median, and ALL missing categorical values with the mode. Do not write complex logic to check for skewness.
           f. Adhere to the Simplicity Rule: Do not apply advanced outlier treatment (like Winsorizing or percentile capping) or multicollinearity filtering (like Pearson correlation drops or VIF checks). Stick to standard scaling, imputation, and one-hot encoding.
           g. Encode categorical columns: map binary categories to 0/1, and one-hot encode nominal categories (using pd.concat in batch with index alignment to avoid memory fragmentation).
           h. Standardize all numerical features using Z-score scaling.
           i. Save the preprocessed dataset to 'data/processed/logistic_ready_train.csv' (make sure to name the final preprocessed DataFrame variable as `df`).
           j. Adhere to the State Hook Rule: At the very end of the generated script, you MUST append these exact lines of code:
              SHARED_GLOBALS['logistic_processed_path'] = 'data/processed/logistic_ready_train.csv'
              SHARED_GLOBALS['logistic_processed_shape'] = df.shape
        
        2. Query the 'Coding_Agent' tool to train and evaluate the Logistic Regression model.
           Your prompt to the 'Coding_Agent' must instruct it to generate and run Python code that does the following:
           a. Load the preprocessed dataset from SHARED_GLOBALS['logistic_processed_path'].
           b. Split the dataset into 80% training and 20% test sets, using stratification ('stratify=y') to preserve class balance.
           c. Fit a LogisticRegression model (max_iter=1000, random_state=42) on the training set.
           d. Evaluate the model on the test set, computing and printing the following evaluation metrics:
              - Accuracy Score (using sklearn.metrics.accuracy_score)
              - AUC-ROC Score (using sklearn.metrics.roc_auc_score on predicted probabilities)
              - Confusion Matrix (using sklearn.metrics.confusion_matrix)
              - Standard Classification Report (using sklearn.metrics.classification_report)
           e. Save the trained model to 'data/processed/logistic_model.pkl'.
           f. Print the evaluation results including the accuracy, ROC-AUC score, confusion matrix, and classification report.
        
        3. Print a final summary of the preprocessing and training results. You MUST explicitly extract and print the evaluated Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and ROC-AUC score in your final output summary. Do not summarize them vaguely.
        """
    )
