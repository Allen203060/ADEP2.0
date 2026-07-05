import os
import sys
import asyncio
import argparse

# Ensure parent directory is in python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.shared_environment import SHARED_GLOBALS
from google.adk.runners import InMemoryRunner
from Logistic.logistic_agent import create_logistic_agent

def main():
    # Configure command line arguments
    parser = argparse.ArgumentParser(description="ADEP Logistic Preprocessing Runner")
    parser.add_argument(
        "--file", 
        type=str, 
        default="data/raw/raw_dataset1.csv",
        help="Path to the raw CSV dataset file (default: data/raw/raw_dataset1.csv)"
    )
    parser.add_argument(
        "--target", 
        type=str, 
        default="num",
        help="Name of the target column (default: num)"
    )
    
    args = parser.parse_args()
    
    print("=== ADEP Logistic Agent Runner ===")
    
    # 1. Setup global state configurations
    if not os.path.exists(args.file):
        print(f"Error: Raw dataset file not found at '{args.file}'.")
        return
        
    SHARED_GLOBALS['file_path'] = args.file
    SHARED_GLOBALS['target_column'] = args.target
    
    print(f"Loaded raw file: {args.file}")
    print(f"Target column set to: '{args.target}'")
    
    # 2. Build Agent
    print("\n[Runner] Instantiating Logistic Agent...")
    logistic_agent = create_logistic_agent()
    
    # 3. Run Agent using InMemoryRunner and asyncio
    print("\n[Runner] Running Agent task...")
    runner = InMemoryRunner(agent=logistic_agent)
    
    # Run the async debug helper synchronously via asyncio
    asyncio.run(runner.run_debug(
        f"Preprocess the dataset at {args.file} for target column {args.target}, "
        f"and then train a Logistic Regression model on the preprocessed data."
    ))


    # 4. Verify updates
    print("\n=== Verifying SHARED_GLOBALS ===")
    print(f"logistic_processed_path: {SHARED_GLOBALS.get('logistic_processed_path')}")
    print(f"logistic_processed_shape: {SHARED_GLOBALS.get('logistic_processed_shape')}")

if __name__ == "__main__":
    main()
