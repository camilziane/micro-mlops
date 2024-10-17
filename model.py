def build_model():
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    import joblib
    import os

    df = pd.read_csv("data/houses.csv")
    X = df[["size", "nb_rooms", "garden"]]
    y = df["price"]
    model = LinearRegression()
    model.fit(X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/regression.joblib")


def load_model():
    import joblib

    model = joblib.load("models/regression.joblib")
    return model


if __name__ == "__main__":
    build_model()
