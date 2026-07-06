import os
import asyncio
import sys
# Inject parent directory into path so it can resolve folders like config, agents, and tools
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from google.adk.agents import Agent
from google.adk.tools import AgentTool
from google.adk.runners import Runner
from google.genai import types 
from google.adk.sessions import InMemorySessionService
from config.agent_config import config
from agents.coding_agent import create_coding_agent
from tools.observability import TraceLogger

def create_linear_orchestrator(model=config.MODEL) -> Agent:
    """Builds the Orchestrator Agent responsible for Linear Regression data preparation."""
    
    # Initialize the coding agent which will act as the "worker" tool
    coding_agent = create_coding_agent(model=model)
    
    return Agent(
        name="Linear_Orchestrator",
        model=model,
        tools=[AgentTool(agent=coding_agent)],
        description="Orchestrates the sequential data cleaning pipeline for Linear Regression.",
        instruction="""
        You are the Linear Regression Data Preparation Orchestrator.
        Your goal is to transform the raw dataset stored in SHARED_GLOBALS.raw_data into a fully cleaned ML-ready dataset.
        
        You do not write or execute python code directly. You must delegate ALL code generation and execution to the 'Coding_Agent'.
        
        EXECUTION WORKFLOW:
        You must process the dataset through the following stages strictly in order. Wait for Coding_Agent to confirm success before moving to the next stage.
        
        STAGE 0: Structural Data Cleaning
        - Prompt Coding_Agent: "Analyze SHARED_GLOBALS.raw_data to identify and remove structural anomalies. First, drop any columns that function purely as row identifiers (e.g., columns named 'ID', 'Id', 'index', or 'Unnamed: 0'). Second, calculate the variance of all remaining features and drop any feature that has exactly zero variance (i.e., all values in the column are identical). Update SHARED_GLOBALS.raw_data with the cleaned dataset."

        STAGE 1: Missing Value Imputation
        - Prompt Coding_Agent: "Identify missing values in SHARED_GLOBALS.raw_data. Impute numerical columns with median and categorical columns with mode. Update SHARED_GLOBALS.raw_data with the imputed dataset."
        
        STAGE 2: Outlier Treatment
        - Prompt Coding_Agent: "Check for outliers in numerical columns of SHARED_GLOBALS.raw_data using the IQR method. Cap the outliers (Winsorize) and update SHARED_GLOBALS.raw_data with the capped dataset."
        
        STAGE 3: Skewness Correction
        - Prompt Coding_Agent: "Check numerical columns of SHARED_GLOBALS.raw_data for high skewness (skew > 1 or < -1). Apply np.log1p to highly skewed columns. Update SHARED_GLOBALS.raw_data."
        
        STAGE 4: Categorical Encoding
        - Prompt Coding_Agent: "Apply One-Hot Encoding to categorical columns in SHARED_GLOBALS.raw_data. MUST use drop_first=True to avoid dummy variable trap. Update SHARED_GLOBALS.raw_data."
        
        STAGE 5: Multicollinearity Audit (VIF)
        - Prompt Coding_Agent: "Identify the top 3 features with the highest absolute correlation to the target column (name specified in SHARED_GLOBALS.target_column). These 3 features, along with the target column, are PROTECTED. Calculate VIF for all features (excluding the target). Iteratively find the UNPROTECTED feature with the highest VIF; if its VIF > 30, drop it and recalculate VIFs. Repeat until no UNPROTECTED feature has VIF > 10. Update SHARED_GLOBALS.raw_data."
        
        STAGE 6: Scaling & Splitting
        - Prompt Coding_Agent: "FIRST, split the data into 80/20 training and testing sets. SECOND, fit a StandardScaler on the training features ONLY, and use it to transform both the training features and testing features. THIRD, reconstruct the final DataFrames. CRITICAL: When converting the scaled numpy arrays back to DataFrames, you MUST explicitly assign the index of the split target Series (e.g., `pd.DataFrame(X_train_scaled, index=y_train.index, columns=X.columns)`) so that `pd.concat([scaled_features, target], axis=1)` perfectly aligns the rows without scrambling the data or generating NaNs. Save to SHARED_GLOBALS.train_data and SHARED_GLOBALS.test_data."
        
        STAGE 7: Model Training & Evaluation
        - Prompt Coding_Agent: "Load the preprocessed training and testing DataFrames from SHARED_GLOBALS.train_data and SHARED_GLOBALS.test_data. Separate the features and the target (the target column name is in SHARED_GLOBALS.target_column). Fit a sklearn LinearRegression model on the training set. Evaluate the model on the testing set and print the R² Score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE). Finally, serialize and save the trained model to 'data/processed/linear_model.pkl' using joblib or pickle."
        
        When all stages are successfully completed, output a final summary of the dataset shapes and explicitly print the evaluated R² Score, MAE, and RMSE.
        """
    )

async def run_linear_pipeline():
    """Isolated runner to execute and log the Linear Orchestrator locally."""
    # 1. Initialize Observability pointing strictly to the Linear folder
    logger = TraceLogger(log_dir="Linear/logs")
    logger.log_system("INIT", "Initializing Linear Orchestrator Pipeline...")
    
    # 2. Spin up the orchestrator, session service, and runner
    orchestrator = create_linear_orchestrator()
    session_service = InMemorySessionService()
    
    app_name = "ADEP"
    user_id = "local_tester"
    session_id = "test_run_01"
    
    runner = Runner(
        agent=orchestrator,
        app_name=app_name,
        session_service=session_service
    )
    
    # 3. Ensure the session exists in the session service before starting the runner
    session = await session_service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id
    )
    if not session:
        await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id
        )
        logger.log_system("SESSION", f"Created new session: {session_id}")
    
    logger.log_system("START", "Kicking off pipeline execution...")
    
    # 4. Build the structured trigger prompt message
    trigger_message = types.Content(
        role='user',
        parts=[types.Part(text="Begin the Linear Regression EDA workflow strictly sequentially, starting from Stage 0 and progressing through all stages.")]
    )
    
    # 5. Stream events asynchronously to our TraceLogger
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=trigger_message
    ):
        logger.log_adk_event(default_agent_name="LINEAR_ORCHESTRATOR", event=event)
        # 6. Save the final datasets to disk
    logger.log_system("PERSIST", "Saving cleaned datasets to data/processed/...")
    
    processed_dir = os.path.abspath(os.path.join(script_dir, "..", "data", "processed"))
    os.makedirs(processed_dir, exist_ok=True)
    
    # Check if the keys exist in SHARED_GLOBALS before writing
    from utils.shared_environment import SHARED_GLOBALS
    if 'train_data' in SHARED_GLOBALS and 'test_data' in SHARED_GLOBALS:
        # Convert back to DataFrame if they are in array formats, or save directly if they are DataFrames
        train_df = pd.DataFrame(SHARED_GLOBALS['train_data']) if not isinstance(SHARED_GLOBALS['train_data'], pd.DataFrame) else SHARED_GLOBALS['train_data']
        test_df = pd.DataFrame(SHARED_GLOBALS['test_data']) if not isinstance(SHARED_GLOBALS['test_data'], pd.DataFrame) else SHARED_GLOBALS['test_data']
        
        train_df.to_csv(os.path.join(processed_dir, "linear_train.csv"), index=False)
        test_df.to_csv(os.path.join(processed_dir, "linear_test.csv"), index=False)
        
        logger.log_system("PERSIST", f"Successfully saved cleaned datasets to: {processed_dir}")
    else:
        logger.log_system("PERSIST-FAILED", "Could not find train_data/test_data in SHARED_GLOBALS to persist.")

    logger.log_system("COMPLETE", "Linear Orchestrator Pipeline finished.")

