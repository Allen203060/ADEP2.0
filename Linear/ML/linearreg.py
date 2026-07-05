import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.abspath(os.path.join(script_dir, "../../data/processed/linear_train.csv"))
    test_path = os.path.abspath(os.path.join(script_dir, "../../data/processed/linear_test.csv"))
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("❌ Error: Cleaned datasets not found in data/processed/")
        return

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Handle any index misalignment NaNs from the splitting stage
    if train_df.isnull().any().any() or test_df.isnull().any().any():
        print("⚠️ Warning: NaNs detected. Dropping incomplete rows.")
        train_df = train_df.dropna()
        test_df = test_df.dropna()
        
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    
    target_col = 'Sales'
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("\n================ Metrics ================")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print("=========================================")

if __name__ == '__main__':
    main()
