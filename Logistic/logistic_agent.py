import os
from google.adk.agents import Agent
from google.adk.tools import AgentTool
from config.agent_config import config
from agents.coding_agent import create_coding_agent

def create_logistic_agent(model=config.MODEL) -> Agent:
    """
    Constructs and returns the Google ADK Agent for logistic regression,
    which delegates preprocessing and training code generation to the Coding Agent
    while enforcing exact instructions for preprocessing pipelines and training validation.
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
        Your goal is to coordinate advanced data preprocessing and training of a Logistic Regression model.
        
        You DO NOT execute code or write scripts directly. You must delegate all coding and execution tasks to the 'Coding_Agent' tool.
        
        CRITICAL RULES FOR GENERATED CODE:
        1. The No-Truncation Rule: You MUST load and process 100% of the dataset. NEVER use `nrows`, `.head()`, `.sample()`, or any truncation methods.
        2. The Reality Rule: You MUST strictly process the dataframe loaded from the CSV. You are strictly FORBIDDEN from creating mock data, dummy arrays, or generating artificial rows to bypass errors. If your code fails, fix the math.
        3. The State Hook Rule: At the very end of your generated Python script, you MUST append these exact lines of code to update the global memory. Do not change the dictionary keys:
           SHARED_GLOBALS['logistic_processed_path'] = 'data/processed/logistic_ready_train.csv'
           SHARED_GLOBALS['logistic_processed_shape'] = df.shape
        
        STEPS:
        1. Query the 'Coding_Agent' tool to preprocess the raw CSV dataset located at SHARED_GLOBALS['file_path'] for the target column specified in SHARED_GLOBALS['target_column'].
           Your prompt to the 'Coding_Agent' must instruct it to generate and run Python code that executes this EXACT pipeline in order:
           
           a. Load the ENTIRE dataset using pd.read_csv(). Separate the target column from the features.
           b. Target Binarization: If the target has exactly 2 unique values, map them to 0 and 1. If it is numeric with >2 unique values, collapse it to binary (values > 0 become 1, else 0).
           c. Column Standardization & Pruning: 
              - Convert all column names to lowercase and replace spaces with underscores.
              - Drop identifier columns: any column named 'id', 'uuid', 'pk', 'key', 'index', 'name', 'ticket', 'cabin', or ending in '_id'.
              - Drop extreme cardinality text: Drop any non-numeric column that has more than 100 unique values.
              - Force types: Convert columns to float if numeric, else string.
           d. Missingness Drop: Drop any feature column where more than 60% (0.60) of the values are missing.
           e. Imputation: 
              - For missing categorical values: Impute with the mode.
              - For missing numeric values: Calculate the skewness. If abs(skew) > 0.5, impute with the median. Otherwise, impute with the mean.
           f. Outlier Treatment: 
              - For numeric columns where skewness > 1.0 AND all values are >= 0: apply a log1p transformation (np.log1p).
              - For all other numeric columns: Clip (Winsorize) the values at the 1st (0.01) and 99th (0.99) percentiles.
           g. Categorical Encoding: 
              - Map binary text categories (exactly 2 unique values) to 0 and 1.
              - One-hot encode remaining nominal categories. You MUST drop the first reference category (e.g., using `drop_first=True` in pd.get_dummies) to prevent the dummy variable trap.
           h. Multicollinearity Filter: 
              - Drop constant columns (standard deviation approx 0).
              - Calculate a Pearson correlation matrix. If a pair of features has abs(correlation) > 0.8, drop one of them.
              - (Optional but recommended) Run a Variance Inflation Factor (VIF) check. Iteratively drop features with a VIF > 10.0 until all remaining features are below 10.0.
           i. Scaling: Standardize all remaining numerical features using Z-score scaling (StandardScaler).
           j. Merge the processed features and binary target back together, save to 'data/processed/logistic_ready_train.csv' (ensure the variable is named `df`), and append the State Hook Rule lines.

        2. Query the 'Coding_Agent' tool to train and evaluate the Logistic Regression model.
           Your prompt to the 'Coding_Agent' must instruct it to generate and run Python code that does the following:
           a. Load the preprocessed dataset from SHARED_GLOBALS['logistic_processed_path'].
           b. Split the dataset into 80% training and 20% test sets, using stratification ('stratify=y') to preserve class balance.
           c. Fit a LogisticRegression model (max_iter=1000, random_state=42) on the training set.
           d. Evaluate the model on the test set, computing and printing: Accuracy Score, AUC-ROC Score, Confusion Matrix, and Standard Classification Report.
           e. Save the trained model to 'data/processed/logistic_model.pkl'.
           
        3. Print a final summary of the preprocessing and training results. You MUST explicitly extract and print the evaluated Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and ROC-AUC score in your final output summary. Do not summarize them vaguely.
        """
    )
