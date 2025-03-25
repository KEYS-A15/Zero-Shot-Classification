from transformers import pipeline
from typing import List
from loguru import logger

class ZeroShotClassifier:
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        logger.info(f"loading zero-shot model: {model_name}")
        self.model_name = model_name
        self.classifier = pipeline("zero-shot-classification", model=self.model_name)
    
    def predict(self, text: str, candidate_labels: List[str]) -> dict:
        if not text or not candidate_labels:
            raise ValueError("Text and candidate labels must be provided")
        logger.debug(f"User Query : {text}")
        logger.debug(f"Labels : {candidate_labels}")
        result = self.classifier(text, candidate_labels)
        logger.debug(f"Response : {result}")
        return {
            "labels" : result["labels"][0],
            "scores" : result["scores"][0]
        }