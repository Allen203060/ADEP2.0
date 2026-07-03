import os
import pandas as pd
from google.adk.agents import Agent
from config.agent_config import config
from google.adk.tools import FunctionTool, AgentTool

from utils.shared_environment import SHARED_GLOBALS
from agents.coding_agent import create_coding_agent

def prompt_user_target(all_columns: list) -> dict:
    """
    Prompt the user to select the prediction (target) column and ML type.
    If no column is specified, defaults to Unsupervised Learning.
    """
    print("\n" + "="*60)
    print("🤖 [HITL] PROBLEM DEFINITION ASSISTANT")
    print(f"📄 Available Columns: {all_columns}")
    print("="*60)
    
    target_col = input("👉 Enter target column (or press ENTER for Unsupervised): ").strip()
    
    if not target_col:
        return {
            "ml_category": "unsupervised",
            "target_col": None,
            "regression_type": None
        }
        
    if target_col not in all_columns:
        # Fallback to last column if input invalid
        target_col = all_columns[-1]
        print(f"⚠️ Column not found. Defaulting to last column: '{target_col}'")
        
    print("\nChoose ML task type:")
    print("1. Auto-Detect")
    print("2. Linear Regression (Regression)")
    print("3. Logistic Regression (Classification)")
    choice = input("👉 Enter choice (1-3) [default: 1]: ").strip()
    
    regression_type = "auto"
    if choice == "2":
        regression_type = "linear"
    elif choice == "3":
        regression_type = "logistic"
        
    return {
        "ml_category": "supervised",
        "target_col": target_col,
        "regression_type": regression_type
    }

def create_problem_identifier_agent(model=config.MODEL) -> Agent:
    """Builds the Head ML Architect agent, which delegates tasks to the Coding Agent."""
    coding_agent = create_coding_agent(model=model)
    
    return Agent(
        name="Problem_Identifier_Agent",
        model=model,
        tools=[
            AgentTool(agent=coding_agent),
            FunctionTool(prompt_user_target)
        ],
        description="Coordinates dataset scanning and categorizes the ML task using the Coding Agent.",
        instruction="""
        You are the Head ML Architect.
        Your goal is to analyze the dataset structure and decide whether the ML problem is Supervised (Linear or Logistic Regression) or Unsupervised.
        
        You do not run code directly. You must delegate all coding tasks to the 'Coding_Agent'.
        
        STEPS:
        1. Query 'Coding_Agent' to load the file path from SHARED_GLOBALS['file_path'], inspect the column list, and save it to the registry.
           - Example prompt: "Read the CSV file from SHARED_GLOBALS['file_path']. Extract the list of columns and print them. Save this list to SHARED_GLOBALS['columns_list']."
        2. From the column list returned, call the tool `prompt_user_target` to get the target setup from the user.
        3. Save the chosen target to `SHARED_GLOBALS['target_column']`.
        4. If the user's choice category is 'unsupervised':
           - Query 'Coding_Agent': "Save 'unsupervised_clustering' to SHARED_GLOBALS['chosen_task'] and print confirmation."
        5. If the user's choice is 'supervised':
           - If they selected a specific task type (e.g. 'linear' or 'logistic'), query 'Coding_Agent': "Save 'linear' (or 'logistic') to SHARED_GLOBALS['chosen_task'] and print confirmation."
           - If they selected 'auto': Query 'Coding_Agent' to run auto-detection:
             "Analyze the target column in the dataset at SHARED_GLOBALS['file_path']. If the column is non-numeric, or numeric but has 20 or fewer unique values, categorize it as 'logistic'. Otherwise, categorize it as 'linear'. Save this result ('linear' or 'logistic') to SHARED_GLOBALS['chosen_task'] and print the categorization choice."
        6. Summarize the final configured target column and regression task.
        """
    )
