import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class ColumnStandardizer(BaseEstimator, TransformerMixin):
    """
    Standardizes column names to lowercase and snake_case.
    Enforces clean data types (float for numeric, string for categorical).
    Optionally drops non-predictive identifiers (e.g. ID, uuid, key, names).
    """
    def __init__(self, drop_ids=True):
        self.drop_ids = drop_ids
        self.cols_to_drop = []

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        # Generate target standardized names
        standardized_cols = [str(c).strip().lower().replace(" ", "_") for c in X_df.columns]
        
        self.cols_to_drop = []
        if self.drop_ids:
            for orig_col, std_col in zip(X_df.columns, standardized_cols):
                # Match common unique key formats
                if std_col in ['id', 'uuid', 'pk', 'key', 'index'] or std_col.endswith('_id'):
                    self.cols_to_drop.append(orig_col)
                    continue
                # Match high cardinality text columns (e.g. string/categorical columns with high cardinality or specific names)
                if not pd.api.types.is_numeric_dtype(X_df[orig_col]):
                    unique_count = X_df[orig_col].nunique()
                    is_high_cardinality = unique_count > 100
                    is_common_id_name = std_col in ['name', 'ticket', 'cabin', 'home.dest', 'home_dest']
                    if (unique_count == len(X_df) and len(X_df) > 5) or is_high_cardinality or is_common_id_name:
                        self.cols_to_drop.append(orig_col)


                        
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        if self.cols_to_drop:
            present_drops = [c for c in self.cols_to_drop if c in X_df.columns]
            X_df = X_df.drop(columns=present_drops)
            
        # Apply snake_case naming
        X_df.columns = [str(c).strip().lower().replace(" ", "_") for c in X_df.columns]
        
        # Enforce types
        for col in X_df.columns:
            if not pd.api.types.is_numeric_dtype(X_df[col]):
                try:
                    non_nulls = X_df[col].dropna()
                    if len(non_nulls) > 0:
                        pd.to_numeric(non_nulls)
                        X_df[col] = pd.to_numeric(X_df[col])
                except (ValueError, TypeError):
                    X_df[col] = X_df[col].astype(str)

            
            if pd.api.types.is_numeric_dtype(X_df[col]):
                X_df[col] = X_df[col].astype(float)
            else:
                X_df[col] = X_df[col].astype(str)
                
        return X_df

class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values. Drops columns exceeding the missingness threshold.
    Imputes skewed numeric features with median, symmetric with mean, and categorical with mode.
    """
    def __init__(self, threshold=0.6, skew_threshold=0.5):
        self.threshold = threshold
        self.skew_threshold = skew_threshold
        self.cols_to_drop_ = []
        self.impute_values_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        n_rows = len(X_df)
        self.cols_to_drop_ = []
        self.impute_values_ = {}
        
        for col in X_df.columns:
            null_fraction = X_df[col].isnull().sum() / n_rows if n_rows > 0 else 0
            if null_fraction > self.threshold:
                self.cols_to_drop_.append(col)
                continue
                
            if pd.api.types.is_numeric_dtype(X_df[col]):
                non_null_vals = X_df[col].dropna()
                if len(non_null_vals) > 0:
                    skew = non_null_vals.skew()
                    if abs(skew) > self.skew_threshold:
                        self.impute_values_[col] = non_null_vals.median()
                    else:
                        self.impute_values_[col] = non_null_vals.mean()
                else:
                    self.impute_values_[col] = 0.0
            else:
                non_null_vals = X_df[col].dropna()
                if len(non_null_vals) > 0:
                    modes = non_null_vals.mode()
                    self.impute_values_[col] = modes.iloc[0] if not modes.empty else "Unknown"
                else:
                    self.impute_values_[col] = "Unknown"
                    
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        if self.cols_to_drop_:
            present_drops = [c for c in self.cols_to_drop_ if c in X_df.columns]
            X_df = X_df.drop(columns=present_drops)
            
        for col, val in self.impute_values_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].fillna(val)
        return X_df

class OutlierTreater(BaseEstimator, TransformerMixin):
    """
    Identifies outliers and treats them using log transforms (skewed features)
    or percentile capping/Winsorizing (1st and 99th percentiles).
    """
    def __init__(self, skew_log_threshold=1.0):
        self.skew_log_threshold = skew_log_threshold
        self.cap_limits_ = {}
        self.log_transform_cols_ = []

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.cap_limits_ = {}
        self.log_transform_cols_ = []
        
        for col in X_df.columns:
            if pd.api.types.is_numeric_dtype(X_df[col]):
                skew = X_df[col].skew()
                if skew > self.skew_log_threshold and (X_df[col] >= 0).all():
                    self.log_transform_cols_.append(col)
                    continue
                
                q1 = X_df[col].quantile(0.01)
                q3 = X_df[col].quantile(0.99)
                self.cap_limits_[col] = (q1, q3)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in self.log_transform_cols_:
            if col in X_df.columns:
                X_df[col] = np.log1p(X_df[col].clip(lower=0))
                
        for col, (q1, q3) in self.cap_limits_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].clip(lower=q1, upper=q3)
        return X_df

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categorical features. Binary values mapped to 0/1.
    Nominal classes mapped using One-Hot encoding, dropping the first column
    to prevent the dummy variable trap (perfect multicollinearity).
    """
    def __init__(self, ordinal_mappings=None):
        self.ordinal_mappings = ordinal_mappings or {}
        self.binary_mappings_ = {}
        self.categories_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.binary_mappings_ = {}
        self.categories_ = {}
        
        for col in X_df.columns:
            if not pd.api.types.is_numeric_dtype(X_df[col]):
                unique_vals = X_df[col].unique()
                if col in self.ordinal_mappings:
                    continue
                if len(unique_vals) == 2:
                    sorted_vals = sorted(unique_vals)
                    self.binary_mappings_[col] = {sorted_vals[0]: 0, sorted_vals[1]: 1}
                else:
                    sorted_cats = sorted(list(unique_vals))
                    if len(sorted_cats) > 0:
                        # Drop reference category (first element) to prevent collinearity
                        self.categories_[col] = sorted_cats[1:]
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col, mapping in self.ordinal_mappings.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].map(mapping).fillna(-1).astype(float)
                
        for col, mapping in self.binary_mappings_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].map(mapping).fillna(0).astype(float)
                
        for col, cats in self.categories_.items():
            if col in X_df.columns:
                new_cols = {}
                for cat in cats:
                    new_cols[f"{col}_{cat}"] = (X_df[col] == cat).astype(float)
                if new_cols:
                    # Concat in one batch, aligning on the existing DataFrame index
                    X_df = pd.concat([X_df, pd.DataFrame(new_cols, index=X_df.index)], axis=1)
                X_df = X_df.drop(columns=[col])
        return X_df 


class MulticollinearityFilter(BaseEstimator, TransformerMixin):
    """
    Identifies and removes highly collinear features using Pearson pairwise
    correlation matrices and iterative Variance Inflation Factor (VIF) checks.
    """
    def __init__(self, corr_threshold=0.8, vif_threshold=10.0):
        self.corr_threshold = corr_threshold
        self.vif_threshold = vif_threshold
        self.cols_to_drop_ = []

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.cols_to_drop_ = []
        
        numeric_cols = []
        for col in X_df.columns:
            if pd.api.types.is_numeric_dtype(X_df[col]):
                # Exclude constant columns (standard deviation near 0)
                if X_df[col].std(ddof=0) > 1e-9:
                    numeric_cols.append(col)
                else:
                    self.cols_to_drop_.append(col)
                    
        if len(numeric_cols) < 2:

            return self
            
        current_cols = list(numeric_cols)
        dropped_by_corr = set()
        
        # 1. Pearson Correlation Filter
        for i in range(len(current_cols)):
            col1 = current_cols[i]
            if col1 in dropped_by_corr:
                continue
            for j in range(i + 1, len(current_cols)):
                col2 = current_cols[j]
                if col2 in dropped_by_corr:
                    continue
                if abs(X_df[col1].corr(X_df[col2])) > self.corr_threshold:
                    dropped_by_corr.add(col2)
                    
        self.cols_to_drop_.extend(list(dropped_by_corr))
        vif_check_cols = [c for c in current_cols if c not in dropped_by_corr]
        
        # 2. Iterative VIF Filter
        while len(vif_check_cols) >= 2:
            df_vif = X_df[vif_check_cols]
            try:
                corr_matrix = df_vif.corr().values
                ridge = np.eye(corr_matrix.shape[0]) * 1e-6
                inv_corr = np.linalg.pinv(corr_matrix + ridge)
                vifs = np.diag(inv_corr)
            except Exception:
                # Least Squares Fallback
                vifs = []
                for col in vif_check_cols:
                    other_cols = [c for c in vif_check_cols if c != col]
                    X_other = df_vif[other_cols].values
                    X_other = np.hstack([np.ones((X_other.shape[0], 1)), X_other])
                    y_val = df_vif[col].values
                    try:
                        _, residuals, _, _ = np.linalg.lstsq(X_other, y_val, rcond=None)
                        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
                        ss_res = np.sum(residuals)
                        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0
                        vifs.append(1.0 / (1.0 - r2) if r2 < 1.0 else float('inf'))
                    except Exception:
                        vifs.append(1.0)
                        
            max_vif_idx = np.argmax(vifs)
            max_vif = vifs[max_vif_idx]
            
            if max_vif > self.vif_threshold:
                col_to_drop = vif_check_cols[max_vif_idx]
                self.cols_to_drop_.append(col_to_drop)
                vif_check_cols.remove(col_to_drop)
            else:
                break
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        if self.cols_to_drop_:
            present_drops = [c for c in self.cols_to_drop_ if c in X_df.columns]
            X_df = X_df.drop(columns=present_drops)
        return X_df

class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Standardizes all numerical features using Z-score scaling.
    """
    def __init__(self):
        self.means_ = {}
        self.stds_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.means_ = {}
        self.stds_ = {}
        
        for col in X_df.columns:
            if pd.api.types.is_numeric_dtype(X_df[col]):
                self.means_[col] = X_df[col].mean()
                std = X_df[col].std(ddof=0)
                self.stds_[col] = std if std > 0 else 1.0
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            if col in self.means_ and col in self.stds_:
                X_df[col] = (X_df[col] - self.means_[col]) / self.stds_[col]
        return X_df

def create_logistic_pipeline(drop_ids=True, corr_threshold=0.8, vif_threshold=10.0):
    """
    Helper constructor function to return a ready-to-run Pipeline structure.
    """
    steps = [
        ("standardizer", ColumnStandardizer(drop_ids=drop_ids)),
        ("imputer", MissingValueImputer(threshold=0.6, skew_threshold=0.5)),
        ("outliers", OutlierTreater(skew_log_threshold=1.0)),
        ("encoder", CategoricalEncoder()),
        ("multicollinearity", MulticollinearityFilter(corr_threshold=corr_threshold, vif_threshold=vif_threshold)),
        ("scaler", FeatureScaler())
    ]
    return Pipeline(steps)

def preprocess_data(df, target_col=None, is_train=True, pipeline=None):
    """
    Splits features and target, processes target to binary 0/1 labels,
    and runs features through the fitting or transforming stages.
    """
    df_copy = df.copy()
    if target_col and target_col in df_copy.columns:
        y = df_copy[target_col]
        X = df_copy.drop(columns=[target_col])
    else:
        y = None
        X = df_copy
        
    y_processed = None
    if y is not None:
        unique_y = y.dropna().unique()
        if len(unique_y) == 2:
            sorted_y = sorted(list(unique_y))
            mapping = {sorted_y[0]: 0, sorted_y[1]: 1}
            y_processed = y.map(mapping)
        else:
            if not pd.api.types.is_numeric_dtype(y):
                # Nominal/Text multiclass targets -> map to category codes
                y_processed = y.astype('category').cat.codes
            else:
                # Collapse multiclass numeric targets into binary classes (0 = absent, 1 = present)
                # Any value greater than 0 represents presence of the condition/class.
                print(f"[Preprocessing] Collapsing {len(unique_y)} numeric classes into binary (0 vs 1)")
                y_processed = (y > 0).astype(int)

                
    if is_train:
        pipeline = create_logistic_pipeline()
        X_processed = pipeline.fit_transform(X)
        return X_processed, y_processed, pipeline
    else:
        if pipeline is None:
            raise ValueError("A fitted pipeline must be provided when is_train=False.")
        X_processed = pipeline.transform(X)
        return X_processed, y_processed
