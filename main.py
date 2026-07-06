import os
import sys
import asyncio
import argparse
from utils.shared_environment import SHARED_GLOBALS
from google.adk.runners import InMemoryRunner

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
    print(f"🚀 Initializing ADEP 2.0 Master Pipeline for: {csv_path}")

    # 2. Run the Problem Identifier Agent
    print("\n" + "="*50)
    print("🔍 STAGE 1: Problem Identification & ML Routing")
    print("="*50)
    
    identifier_agent = create_problem_identifier_agent()
    # Using InMemoryRunner for quick console interactions
    runner_id = InMemoryRunner(agent=identifier_agent)
    
    await runner_id.run_debug(
        "Analyze the dataset, prompt the user for the target column, determine the task type, and update SHARED_GLOBALS."
    )
    
    task_type = SHARED_GLOBALS.get('chosen_task')
    target = SHARED_GLOBALS.get('target_column')
    
    print("\n✅ Problem Identification Complete!")
    print(f"Target Column: {target}")
    print(f"Chosen Task: {task_type}")

    # 3. Route to the correct pipeline
    print("\n" + "="*50)
    print(f"⚙️ STAGE 2: Delegating to {str(task_type).upper()} Orchestrator")
    print("="*50)

    if task_type == 'linear':
        agent = create_linear_orchestrator()
        task_prompt = "Begin the Linear Regression EDA workflow strictly sequentially, starting from Stage 0 and progressing through all stages."
    elif task_type == 'logistic':
        agent = create_logistic_agent()
        task_prompt = f"Preprocess the dataset at {csv_path} for target column {target}, and then train a Logistic Regression model on the preprocessed data."
    elif task_type == 'unsupervised_clustering':
        print("Clustering pipeline not yet implemented.")
        return
    else:
        print(f"⚠️ Error: Unknown or missing task type: {task_type}")
        return

    # 4. Execute the chosen sub-pipeline
    pipeline_runner = InMemoryRunner(agent=agent)
    await pipeline_runner.run_debug(task_prompt)

    print("\n🎉 Master Pipeline Execution Finished Successfully!")

if __name__ == "__main__":
    asyncio.run(main())
