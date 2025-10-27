"""Model training script placeholder."""

from sklearn.ensemble import RandomForestRegressor
import joblib
import pandas as pd


def train_model(X_train, y_train, params=None):
    params = params or {}
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model


def save_model(model, path: str):
    joblib.dump(model, path)


if __name__ == "__main__":
    print("train module placeholder - wire into pipeline or CLI")
