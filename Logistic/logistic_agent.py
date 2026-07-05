import os
import pandas as pd
import pickle
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from config.agent_config import config
from utils.shared_environment import SHARED_GLOBALS
from Logistic.pipeline import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

def run_logistic_pipeline() -> str:
    """
    Executes the logistic regression preprocessing pipeline on the raw dataset.
    
    Reads from:
      - SHARED_GLOBALS['file_path'] (path to raw CSV)
      - SHARED_GLOBALS['target_column'] (name of prediction column)
      
    Outputs:
      - 'data/processed/logistic_ready_train.csv' (processed dataset)
      
    Registers to SHARED_GLOBALS:
      - 'logistic_pipeline' (fitted sklearn Pipeline)
      - 'logistic_processed_path' (path to outputs)
      - 'logistic_processed_shape' (tuple shape of outputs)
    """
    file_path = SHARED_GLOBALS.get('file_path')
    target_col = SHARED_GLOBALS.get('target_column')
    
    if not file_path:
        return "Error: SHARED_GLOBALS['file_path'] is not set."
    if not os.path.exists(file_path):
        return f"Error: Raw dataset file not found at {file_path}."
    if not target_col:
        return "Error: SHARED_GLOBALS['target_column'] is not set."
        
    try:
        # 1. Load dataset
        df = pd.read_csv(file_path)
        
        # 2. Run prep pipeline
        # is_train=True fits scaling, imputation and encoding on training data
        X_proc, y_proc, pipeline = preprocess_data(df, target_col=target_col, is_train=True)
        
        # 3. Recombine preprocessed features and target
        processed_df = X_proc.copy()
        if y_proc is not None:
            processed_df[target_col] = y_proc
            
        # 4. Save results to disk
        os.makedirs("data/processed", exist_ok=True)
        output_path = "data/processed/logistic_ready_train.csv"
        processed_df.to_csv(output_path, index=False)
        
        # 5. Populate thread-safe global registry
        SHARED_GLOBALS['logistic_pipeline'] = pipeline
        SHARED_GLOBALS['logistic_processed_path'] = output_path
        SHARED_GLOBALS['logistic_processed_shape'] = processed_df.shape
        
        return (
            f"Success: Preprocessed dataset shape: {processed_df.shape}. "
            f"Saved to {output_path}."
        )
    except Exception as e:
        import traceback
        return f"Error running preprocessing pipeline: {str(e)}\n{traceback.format_exc()}"

def train_logistic_model() -> str:
    """
    Trains a Logistic Regression model on the preprocessed dataset.
    
    Reads from:
      - SHARED_GLOBALS['logistic_processed_path'] (path to preprocessed CSV)
      - SHARED_GLOBALS['target_column'] (name of target column)
      
    Outputs:
      - 'data/processed/logistic_model.pkl' (serialized trained model)
      
    Registers to SHARED_GLOBALS:
      - 'logistic_model' (fitted LogisticRegression object)
    """
    processed_path = SHARED_GLOBALS.get('logistic_processed_path')
    target_col = SHARED_GLOBALS.get('target_column')
    
    if not processed_path:
        return "Error: SHARED_GLOBALS['logistic_processed_path'] is not set. Have you run the preprocessing pipeline?"
    if not os.path.exists(processed_path):
        return f"Error: Preprocessed dataset file not found at {processed_path}."
    if not target_col:
        return "Error: SHARED_GLOBALS['target_column'] is not set."
        
    try:
        # 1. Load preprocessed data
        df = pd.read_csv(processed_path)
        
        # Standardizer converts headers to lowercase and snake_case
        norm_target_col = target_col.strip().lower().replace(" ", "_")
        if norm_target_col not in df.columns:
            if target_col in df.columns:
                norm_target_col = target_col
            else:
                return f"Error: Target column '{norm_target_col}' not found in preprocessed columns: {list(df.columns)}."
                
        # 2. Split features and target
        y = df[norm_target_col]
        X = df.drop(columns=[norm_target_col])
        
        # 3. Train-Test Split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 4. Fit Logistic Regression
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        # 5. Evaluate
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        
        # ROC AUC
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            auc_str = f"ROC-AUC Score: {auc:.4f}\n"
        except Exception:
            auc_str = ""
            
        # 6. Save model object
        os.makedirs("data/processed", exist_ok=True)
        model_path = "data/processed/logistic_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
            
        SHARED_GLOBALS['logistic_model'] = model
        SHARED_GLOBALS['logistic_model_path'] = model_path
        
        return (
            f"Success: Logistic Regression model trained successfully.\n"
            f"Saved model to {model_path}.\n"
            f"Test Set Evaluation Results:\n"
            f"{auc_str}"
            f"Classification Report:\n{report}"
        )
    except Exception as e:
        import traceback
        return f"Error during model training: {str(e)}\n{traceback.format_exc()}"

def create_logistic_agent(model=config.MODEL) -> Agent:
    """
    Constructs and returns the Google ADK Agent for logistic regression.
    """
    return Agent(
        name="Logistic_Agent",
        model=model,
        tools=[
            FunctionTool(run_logistic_pipeline),
            FunctionTool(train_logistic_model)
        ],
        description="An agent that prepares data and trains logistic regression models using custom pipelines and sklearn.",
        instruction="""
        You are the Logistic Preprocessing and Training Agent.
        
        GOAL:
        1. Run the preprocessing pipeline on the input dataset specified in SHARED_GLOBALS['file_path']
           under the target column specified in SHARED_GLOBALS['target_column'].
        2. Train a Logistic Regression model on the preprocessed output features.
        
        STEPS:
        1. Call the tool 'run_logistic_pipeline' to run the preprocessing pipeline and save output files.
        2. Print the exact response returned by 'run_logistic_pipeline'.
        3. Call the tool 'train_logistic_model' to split the preprocessed dataset, train the model, evaluate it, and serialize the trained model object.
        4. Print the exact response returned by 'train_logistic_model'.
        """
    )
