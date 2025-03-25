from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.classifier import ZeroShotClassifier

app = FastAPI(title="Zero-Shot Classification API")

classifier = ZeroShotClassifier()

class ClassificationRequest(BaseModel):
    text : str
    labels : list[str]

@app.get("/health")
def health():
    return {"status" : "OK"}

@app.post("/classify")
def classify(request: ClassificationRequest):
    try:
        result = classifier.predict(request.text, request.labels)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
