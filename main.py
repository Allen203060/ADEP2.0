import os
import sys
import asyncio
import argparse
import pandas as pd
from google.genai import types
from utils.shared_environment import SHARED_GLOBALS
from google.adk.runners import Runner, InMemoryRunner
from google.adk.sessions import InMemorySessionService
from tools.observability import TraceLogger

# Import Agents
from agents.problem_identifier_agent import create_problem_identifier_agent
from Linear.linear_orchestrator import create_linear_orchestrator
from Logistic.logistic_agent import create_logistic_agent

async def main():
    parser = argparse.ArgumentParser(description="ADEP Master Orchestrator")
    parser.add_argument("--file", type=str, default="data/raw/raw_dataset.csv", help="Path to raw dataset")
    args = parser.parse_args()

    # 1. Setup Global State
    csv_path = os.path.abspath(args.file)
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        sys.exit(1)
        
    SHARED_GLOBALS['file_path'] = csv_path
    SHARED_GLOBALS['raw_data'] = pd.read_csv(csv_path)

    # Initialize Universal Logger
    logger = TraceLogger(log_dir="logs")
    logger.log_system("INIT", f"Initializing ADEP 2.0 Master Pipeline for: {csv_path}")

    # 2. Run the Problem Identifier Agent
    logger.log_system("STAGE 1", "Problem Identification & ML Routing")
    
    identifier_agent = create_problem_identifier_agent()
    runner_id = InMemoryRunner(agent=identifier_agent)
    
    # We still use run_debug here because it requires interactive console input for target selection
    await runner_id.run_debug(
        "Analyze the dataset, prompt the user for the target column, determine the task type, and update SHARED_GLOBALS."
    )
    
    task_type = SHARED_GLOBALS.get('chosen_task')
    target = SHARED_GLOBALS.get('target_column')
    
    logger.log_system("STAGE 1 COMPLETE", f"Target Column: {target} | Chosen Task: {task_type}")

    # 3. Route to the correct pipeline
    if task_type == 'linear':
        agent = create_linear_orchestrator()
        task_prompt = "Begin the Linear Regression EDA workflow strictly sequentially, starting from Stage 0 and progressing through all stages."
    elif task_type == 'logistic':
        agent = create_logistic_agent()
        task_prompt = f"Preprocess the dataset at {csv_path} for target column {target}, and then train a Logistic Regression model on the preprocessed data."
    elif task_type == 'unsupervised_clustering':
        logger.log_system("ERROR", "Clustering pipeline not yet implemented.")
        return
    else:
        logger.log_system("ERROR", f"Unknown or missing task type: {task_type}")
        return

    logger.log_system("STAGE 2", f"Delegating to {str(task_type).upper()} Orchestrator")

    # 4. Execute the chosen sub-pipeline using Event Streaming (Universal Logger)
    session_service = InMemorySessionService()
    user_id = "local_user"
    session_id = "master_run_01"
    
    await session_service.create_session(app_name="ADEP", user_id=user_id, session_id=session_id)
    
    runner = Runner(agent=agent, app_name="ADEP", session_service=session_service)
    
    trigger_message = types.Content(role='user', parts=[types.Part(text=task_prompt)])
    
    # Stream the ADK events quietly to our TraceLogger
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=trigger_message):
        logger.log_adk_event(default_agent_name=f"{str(task_type).upper()}_ORCHESTRATOR", event=event)

    logger.log_system("COMPLETE", "Master Pipeline Execution Finished Successfully!")

if __name__ == "__main__":
    asyncio.run(main())
