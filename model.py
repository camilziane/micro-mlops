data_path = "data/houses.csv"
model_path = "models/regression.joblib"


def build_model():
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    import joblib
    import os

    df = pd.read_csv(data_path)
    X = df[["size", "nb_rooms", "garden"]]
    y = df["price"]
    model = LinearRegression()
    model.fit(X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path)


def load_model():
    import joblib
    import os

    if not os.path.exists(model_path):
        build_model()

    model = joblib.load(model_path)
    return model


if __name__ == "__main__":
    build_model()
