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