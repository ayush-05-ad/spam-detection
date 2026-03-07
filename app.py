from fastapi import FastAPI, Request
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, RedirectResponse
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

# ── Logger setup ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── App init ──────────────────────────────────────────────────
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


# ── Form helper ───────────────────────────────────────────────
class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.text: Optional[str] = None

    async def get_text_data(self):
        form = await self.request.form()
        self.text = form.get('input_text')


# ── Health check ──────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Quick ping to verify server is running."""
    return {"status": "ok", "app": "MailGuard AI", "version": "1.0.0"}


# ── Home page ─────────────────────────────────────────────────
@app.get("/")
async def home(request: Request):                        # ✅ FIX: unique function name
    try:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "context": "Rendering"},
        )
    except Exception as e:
        logger.error(f"Home route error: {e}")
        return Response(f"Error Occurred! {e}", status_code=500)


# ── Predict GET (show form) ───────────────────────────────────
@app.get("/predict")
async def predict_form(request: Request):                # ✅ FIX: unique function name
    try:
        return templates.TemplateResponse(
            "prediction.html",
            {"request": request, "context": False},
        )
    except Exception as e:
        logger.error(f"Predict GET error: {e}")
        return Response(f"Error Occurred! {e}", status_code=500)


# ── Predict POST (run model) ──────────────────────────────────
@app.post("/predict")
async def predict_result(request: Request):              # ✅ FIX: unique function name
    start = time.time()
    try:
        form = DataForm(request)
        await form.get_text_data()

        # ── Validate input ──
        if not form.text or not form.text.strip():
            return templates.TemplateResponse(
                "prediction.html",
                {
                    "request": request,
                    "context": False,
                    "error": "Please enter some text before predicting."   # ✅ NEW: error msg
                }
            )

        input_text = form.text.strip()
        logger.info(f"Predicting for text: '{input_text[:60]}...' " if len(input_text) > 60 else f"Predicting: '{input_text}'")

        # ── Run prediction ──
        prediction_pipeline = PredictionPipeline()
        prediction: int = prediction_pipeline.run_pipeline(input_data=[input_text])
        result = int(prediction[0])

        # ── Confidence score ──────────────────────────────────
        # If your model has decision_function, use this for real confidence:
        # raw_score  = prediction_pipeline.model.decision_function([vectorized_input])[0]
        # confidence = round(min(100, max(0, 50 + abs(raw_score) * 15)), 1)
        #
        # For now, we pass -1 so the frontend shows a simulated bar.
        # Replace with real value once decision_function is wired up.
        confidence = -1                                  # ✅ NEW: placeholder, replace with real

        elapsed = round((time.time() - start) * 1000)   # ms
        logger.info(f"Prediction result: {'SPAM' if result == 1 else 'HAM'} | {elapsed}ms")

        return templates.TemplateResponse(
            "prediction.html",
            {
                "request":    request,
                "context":    True,
                "prediction": result,
                "confidence": confidence,                # ✅ NEW: pass to template
                "input_text": input_text,               # ✅ NEW: echo text back
                "elapsed_ms": elapsed,                  # ✅ NEW: response time
            }
        )

    except Exception as e:
        logger.error(f"Predict POST error: {e}")
        return templates.TemplateResponse(
            "prediction.html",
            {
                "request": request,
                "context": False,
                "error": f"Prediction failed: {str(e)}"  # ✅ NEW: show error in UI
            }
        )


# ── Train pipeline ────────────────────────────────────────────
@app.get("/train")
async def train_model(request: Request):
    """
    Trigger training pipeline.
    NOTE: In production, protect this route with a secret token or
    move it to a POST endpoint so it can't be triggered accidentally.

    Example protection (optional):
        from fastapi import Header, HTTPException
        async def train_model(x_train_token: str = Header(None)):
            if x_train_token != "your-secret-token":
                raise HTTPException(status_code=403, detail="Forbidden")
    """
    try:
        logger.info("Training pipeline triggered...")
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        logger.info("Training pipeline completed.")
        return Response("✅ Training successful!", status_code=200)
    except Exception as e:
        logger.error(f"Training error: {e}")
        return Response(f"❌ Training failed: {e}", status_code=500)


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)