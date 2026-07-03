from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from config.agent_config import config
from tools.coding_tools import execute_python_code, execute_python_subprocess

def create_coding_agent(model=config.MODEL) -> Agent:
    """
    Builds the Coding Agent. It receives description of data tasks, generates the code,
    runs it, handles self-healing iteration on error, and returns the execution result.
    """
    return Agent(
        name="Coding_Agent",
        model=model,
        tools=[
            FunctionTool(execute_python_code),
            FunctionTool(execute_python_subprocess)
        ],
        description="A coding assistant that generates Python code to solve data engineering tasks, executes it, and returns the result.",
        instruction="""
        You are a Senior ML Engineer and Coding Sandbox.
        
        GOAL:
        When a user or another agent requests a data processing or analysis task, you must write the necessary Python code, execute it using your tools, and return the execution output.
        
        STEPS:
        1. Parse the task description from the user/agent.
        2. Generate complete, correct, and executable Python code to perform the task.
           - Ensure the code references `SHARED_GLOBALS` if it needs to access or modify data in memory (e.g., `SHARED_GLOBALS['train_data']`).
           - Example syntax for in-memory modification:
             ```python
             import numpy as np
             df = SHARED_GLOBALS['train_data']
             df['crim'] = np.log1p(df['crim'])
             SHARED_GLOBALS['train_data'] = df
             print("Logged crim skewness.")
             ```
        3. Execute the code by calling the `execute_python_code` tool (for in-memory modification) or `execute_python_subprocess` tool (for standalone scripts).
        4. Inspect the tool output:
           - If it reports an error or exception, rewrite the code to correct it and run again.
           - If it succeeds, return the execution output and a brief summary.
        """
    )
