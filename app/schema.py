from pydantic import BaseModel,Field

class PredictionRequest(BaseModel):
    feature1: float = Field(..., ge=0, le=100)
    feature2: float = Field(..., ge=-50, le=50)
    class Config:
        extra = "forbid"

class PredictionResponse(BaseModel):
    prediction: int