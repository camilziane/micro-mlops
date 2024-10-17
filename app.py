from fastapi import FastAPI
from pydantic import BaseModel
from model import load_model

app = FastAPI()

model = load_model()


class PredictionInput(BaseModel):
    size: float
    nb_rooms: int
    garden: bool


@app.post("/predict")
def predict(input_data: PredictionInput):
    prediction = model.predict(
        [[input_data.size, input_data.nb_rooms, input_data.garden]]
    )[0]
    return {"prediction": prediction}


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
