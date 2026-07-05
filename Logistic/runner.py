import os
import sys

# Ensure parent directory is in python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.shared_environment import SHARED_GLOBALS
from Logistic.logistic_agent import create_logistic_agent

def main():
    print("=== ADEP Logistic Agent Runner ===")
    
    # 1. Setup global state configurations
    raw_file = "data/raw/raw_dataset1.csv"
    if not os.path.exists(raw_file):
        print(f"Error: Raw dataset not found at {raw_file}.")
        return
        
    SHARED_GLOBALS['file_path'] = raw_file
    SHARED_GLOBALS['target_column'] = 'num'
    
    print(f"Loaded raw file: {raw_file}")
    print("Target column set to: 'num'")
    
    # 2. Build Agent
    print("\n[Runner] Instantiating Logistic Agent...")
    logistic_agent = create_logistic_agent()
    
    # 3. Run Agent
    print("\n[Runner] Running Agent task...")
    response = logistic_agent.run("Preprocess the heart disease dataset at SHARED_GLOBALS['file_path'].")
    
    print("\n=== Agent Response ===")
    print(response)
    
    # 4. Verify updates
    print("\n=== Verifying SHARED_GLOBALS ===")
    print(f"logistic_processed_path: {SHARED_GLOBALS.get('logistic_processed_path')}")
    print(f"logistic_processed_shape: {SHARED_GLOBALS.get('logistic_processed_shape')}")

if __name__ == "__main__":
    main()
