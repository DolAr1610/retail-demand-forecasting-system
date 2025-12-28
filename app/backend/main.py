from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.backend.settings import get_settings
from app.backend.utils.logging import setup_logging
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.backend.api.routes_health import router as health_router
from app.backend.api.routes_data import router as data_router
from app.backend.api.routes_metrics import router as metrics_router
from app.backend.api.routes_predict import router as predict_router


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level)

    app = FastAPI(
        title="Demand Forecast MVP API",
        version="0.1.0",
    )

    # CORS for Streamlit frontend
    origins = settings.cors_origins_list()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(data_router)
    app.include_router(metrics_router)
    app.include_router(predict_router)

    return app


app = create_app()

FRONT_DIR = Path(__file__).resolve().parents[1] / "frontend"
ASSETS_DIR = FRONT_DIR / "assets"

# Статичні файли
app.mount("/ui/assets", StaticFiles(directory=str(ASSETS_DIR)), name="ui-assets")

# Сторінки UI
@app.get("/ui", include_in_schema=False)
def ui_index():
    return FileResponse(str(FRONT_DIR / "index.html"))

@app.get("/ui/", include_in_schema=False)
def ui_index_slash():
    return FileResponse(str(FRONT_DIR / "index.html"))

@app.get("/ui/predict", include_in_schema=False)
def ui_predict():
    return FileResponse(str(FRONT_DIR / "predict.html"))

@app.get("/ui/analytics", include_in_schema=False)
def ui_analytics():
    return FileResponse(str(FRONT_DIR / "analytics.html"))

@app.get("/ui/about", include_in_schema=False)
def ui_about():
    return FileResponse(str(FRONT_DIR / "about.html"))
