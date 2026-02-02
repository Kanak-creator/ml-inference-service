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
    start = time.time()

    try:
        print(">>> PREDICT CALLED <<<")
        logger.info(">>> LOGGER PREDICT CALLED <<<")
        pred = model_service.predict([req.feature1, req.feature2])
    except RuntimeError as e:
        logger.error("Prediction failed", exc_info=e)
        raise HTTPException(status_code=503, detail="Model not available")

    total_latency = (time.time() - start) * 1000

    logger.info(
        "Request handled",
        extra={
            "feature1": req.feature1,
            "feature2": req.feature2,
            "total_latency_ms": round(total_latency, 2)
        }
    )

    return PredictionResponse(prediction=pred)

@router.get("/health")
def health_check():
    if not model_service.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}
