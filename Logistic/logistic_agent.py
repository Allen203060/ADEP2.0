import os
import pandas as pd
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from config.agent_config import config
from utils.shared_environment import SHARED_GLOBALS
from Logistic.pipeline import preprocess_data

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

def create_logistic_agent(model=config.MODEL) -> Agent:
    """
    Constructs and returns the Google ADK Agent for logistic regression preprocessing.
    """
    return Agent(
        name="Logistic_Agent",
        model=model,
        tools=[
            FunctionTool(run_logistic_pipeline)
        ],
        description="An agent that prepares data for logistic regression models using a custom scikit-learn preprocessing pipeline.",
        instruction="""
        You are the Logistic Preprocessing Agent.
        
        GOAL:
        Run the preprocessing pipeline on the input dataset specified in SHARED_GLOBALS['file_path']
        under the target column specified in SHARED_GLOBALS['target_column'].
        
        STEPS:
        1. Call the tool 'run_logistic_pipeline' to run the preprocessing pipeline and save output files.
        2. Print the exact response returned by the tool.
        """
    )
