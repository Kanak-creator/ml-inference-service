# from fastapi import APIRouter,status,HTTPException
# from app.schema import PredictionRequest, PredictionResponse

# router = APIRouter()

# @router.post("/predict", response_model=PredictionResponse)
# def predict(req: PredictionRequest):
#     if req.feature1 == 0 and req.feature2 == 0:
#         raise HTTPException(
#             status_code=400,
#             detail="Invalid feature combination"
#         )
    
#     return PredictionResponse(prediction=1)



from fastapi import APIRouter,status,HTTPException
from app.schema import PredictionRequest, PredictionResponse
from app.service import ModelService
import logging
import time

logger = logging.getLogger()

router = APIRouter()

model_service = ModelService()

@router.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    features = {
        "Pclass": req.Pclass,
        "Sex": req.Sex,
        "Age": req.Age,
        "Fare": req.Fare,
    }

    prob = model_service.predict(features)

    return PredictionResponse(probability=prob)

@router.get("/health")
def health_check():
    if not model_service.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}
