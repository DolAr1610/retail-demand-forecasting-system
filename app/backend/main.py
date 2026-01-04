from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse, PlainTextResponse, RedirectResponse

from app.backend.settings import settings

# якщо ці роутери в тебе є — ок
try:
    from app.backend.api.routes_predict import router as predict_router
except Exception:
    predict_router = None

try:
    from app.backend.api.routes_data import router as data_router
except Exception:
    data_router = None

def _cors_list(v: str) -> list[str]:
    if not v:
        return []
    return [x.strip() for x in v.split(",") if x.strip()]


def create_app() -> FastAPI:
    app = FastAPI(title="Forecast API", version="0.1.0")

    origins = _cors_list(getattr(settings, "cors_origins", ""))

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- API routers ---
    if data_router:
        app.include_router(data_router, prefix=settings.api_prefix, tags=["data"])
    if predict_router:
        app.include_router(predict_router, prefix=settings.api_prefix, tags=["predict"])

    # --- optional: прибрати 404 в логах (не критично, але акуратно) ---
    @app.get("/metrics", include_in_schema=False)
    def metrics_stub():
        return PlainTextResponse("ok\n")

    @app.get(f"{settings.api_prefix}/metrics", include_in_schema=False)
    @app.get(f"{settings.api_prefix}/analytics/metrics", include_in_schema=False)
    def api_metrics_stub():
        return {"ok": True}

    # --- UI ---
    frontend_dir = Path(settings.frontend_dir).resolve()
    assets_dir = frontend_dir / "assets"

    if assets_dir.exists():
        app.mount(f"{settings.ui_prefix}/assets", StaticFiles(directory=str(assets_dir)), name="ui-assets")

    # ✅ важливо: цей роут має бути ДО app.mount("/ui", ...)
    predict_html = frontend_dir / "predict.html"

    @app.get(f"{settings.ui_prefix}/predict", include_in_schema=False)
    def ui_predict():
        if predict_html.exists():
            return FileResponse(str(predict_html))
        # fallback: якщо нема predict.html — покажемо index.html
        index_html = frontend_dir / "index.html"
        if index_html.exists():
            return FileResponse(str(index_html))
        return PlainTextResponse("predict.html not found", status_code=404)

    # root → /ui/
    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url=f"{settings.ui_prefix}/")

    # Static UI (index.html і т.д.)
    if frontend_dir.exists():
        app.mount(settings.ui_prefix, StaticFiles(directory=str(frontend_dir), html=True), name="ui")

    return app


app = create_app()
