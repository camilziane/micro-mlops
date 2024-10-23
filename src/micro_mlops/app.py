from fastapi import FastAPI
from pydantic import BaseModel
from micro_mlops.reg_model import load_model
from micro_mlops.sentiment_analysis import load_model as load_sentiment_analysis_model

app = FastAPI()

model = load_model()

sentiment_analyis_model = load_sentiment_analysis_model()


class PredictionInput(BaseModel):
    size: float
    nb_rooms: int
    garden: bool


class SentimentAnalysisInput(BaseModel):
    sentence: str


@app.post("/predict")
def predict(input_data: PredictionInput):
    prediction = model.predict(
        [[input_data.size, input_data.nb_rooms, input_data.garden]]
    )[0]
    return {"prediction": prediction}


@app.post("/sentiment_analysis")
def sentiment_analyis_predict(input_data: SentimentAnalysisInput):
    prediction = sentiment_analyis_model.predict(input_data.sentence)
    return {"prediction": f"Sentiment {'positive' if prediction else 'negative'}"}


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
