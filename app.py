from fastapi import FastAPI, Request
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from uvicorn import run as app_run
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.pipeline.train_pipeline import TrainPipeline
from src.constant.application import *

import warnings
import logging
import time

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MailGuard AI",
    description="Intelligent Spam Detection — SVM + TF-IDF",
    version="1.0.0"
)

templates = Jinja2Templates(directory='templates')
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.text: Optional[str] = None

    async def get_text_data(self):
        form = await self.request.form()
        self.text = form.get('input_text')


@app.get("/health")
async def health_check():
    return {"status": "ok", "app": "MailGuard AI", "version": "1.0.0"}


@app.get("/")
async def home(request: Request):
    try:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "context": "Rendering"},
        )
    except Exception as e:
        logger.error(f"Home route error: {e}")
        return Response(f"Error Occurred! {e}", status_code=500)


@app.get("/predict")
async def predict_form(request: Request):
    try:
        return templates.TemplateResponse(
            "prediction.html",
            {"request": request, "context": False},
        )
    except Exception as e:
        logger.error(f"Predict GET error: {e}")
        return Response(f"Error Occurred! {e}", status_code=500)


@app.post("/predict")
async def predict_result(request: Request):
    start = time.time()
    try:
        form = DataForm(request)
        await form.get_text_data()

        if not form.text or not form.text.strip():
            return templates.TemplateResponse(
                "prediction.html",
                {
                    "request": request,
                    "context": False,
                    "error": "Please enter some text before predicting."
                }
            )

        input_text = form.text.strip()
        logger.info(f"Predicting: '{input_text[:60]}'")

        prediction_pipeline = PredictionPipeline()
        prediction = prediction_pipeline.run_pipeline(input_data=[input_text])
        
        # Fix: handle both bool and int
        result = 1 if prediction[0] else 0
        
        elapsed = round((time.time() - start) * 1000)
        logger.info(f"Result: {'SPAM' if result == 1 else 'HAM'} | {elapsed}ms")
        print(f"RESULT VALUE: {result}", flush=True)

        return templates.TemplateResponse(
            "prediction.html",
            {
                "request":    request,
                "context":    True,
                "prediction": result,
                "input_text": input_text,
                "elapsed_ms": elapsed,
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Predict POST error: {e}")
        return templates.TemplateResponse(
            "prediction.html",
            {
                "request": request,
                "context": False,
                "error": f"Prediction failed: {str(e)}"
            }
        )


@app.get("/train")
async def train_model(request: Request):
    try:
        logger.info("Training pipeline triggered...")
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        logger.info("Training pipeline completed.")
        return Response("Training successful!", status_code=200)
    except Exception as e:
        logger.error(f"Training error: {e}")
        return Response(f"Training failed: {e}", status_code=500)


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)