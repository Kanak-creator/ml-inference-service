from pydantic import BaseModel,Field


class PredictionRequest(BaseModel):
    Pclass: int = Field(..., ge=1, le=3)
    Sex: int = Field(..., ge=0, le=1)  # 0=male, 1=female
    Age: float = Field(..., ge=0, le=120)
    Fare: float = Field(..., ge=0)
    class Config:
        extra = "forbid"

class PredictionResponse(BaseModel):
    probability: float