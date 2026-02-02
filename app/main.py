print(">>> MAIN.PY LOADED <<<")

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    force=True
)
from fastapi import FastAPI
from app.api import router, model_service

app = FastAPI()
app.include_router(router)

@app.on_event("startup")
def startup_check():
    if not model_service.is_ready():
        raise RuntimeError("Model failed to load at startup")