import pandas as pd
from sklearn.model_selection import train_test_split

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that carry no predictive signal."""
    return df.drop(columns=['customer_id'])

def add_zero_balance_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag customers with exactly zero balance — a distinct, lower-churn segment found in EDA."""
    df = df.copy()
    df['has_zero_balance'] = df['balance'] == 0
    return df

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode country and gender."""
    return pd.get_dummies(df, columns=['country', 'gender'], drop_first=True)

def split_data(df: pd.DataFrame, target_col: str = 'churn', test_size: float = 0.2, random_state: int = 42):
    """Split into train/test sets, stratified on the target to preserve class balance."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def run_pipeline(df: pd.DataFrame):
    """Run the full preprocessing pipeline: clean, engineer features, encode, and split."""
    df = drop_unused_columns(df)
    df = add_zero_balance_flag(df)
    df = encode_categoricals(df)
    return split_data(df)

def save_processed_data(X_train, X_test, y_train, y_test, output_dir='../data'):
    """Save the processed train/test splits to disk as Parquet files."""
    X_train.to_parquet(f'{output_dir}/X_train.parquet')
    X_test.to_parquet(f'{output_dir}/X_test.parquet')
    y_train.to_frame().to_parquet(f'{output_dir}/y_train.parquet')
    y_test.to_frame().to_parquet(f'{output_dir}/y_test.parquet')
    print(f"Saved processed data to {output_dir}")

def drop_weak_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop features consistently shown as weak/noise across EDA, permutation importance, and XGBoost importance."""
    return df.drop(columns=['estimated_salary', 'credit_score', 'tenure', 'credit_card'])