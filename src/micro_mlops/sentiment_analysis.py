from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import torch

model_name = os.getenv(
    "SENTIMENT_ANALYSIS_MODEL_NAME", "philschmid/tiny-bert-sst2-distilled"
)


class SentimentAnalysis:
    def __init__(
        self,
        model_name: str,
    ) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def predict(self, sentence: str) -> bool:
        inputs = self.tokenizer(
            sentence, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=-1).item()
        return predicted_class == 1


def load_model():
    return SentimentAnalysis(model_name)
