"""Train/test split helpers."""

from sklearn.model_selection import train_test_split
import pandas as pd


def split_data(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42):
    """Split DataFrame into train and test sets.

    Returns: X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    print("data_split module - import functions in pipelines")
