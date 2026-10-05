from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import APP_DIR, HISTORY_FILE, ensure_runtime_dirs, AppConfig
from app.services.diagnosis_service import DiagnosisService
from app.services.graph_service import KnowledgeGraphService
from app.services.history_service import HistoryService
from app.services.llm_service import LLMService
from app.services.model_generation_service import ModelGenerationService
from app.services.ocr_service import OCRService
from app.services.pipeline_service import PipelineService
from app.services.tencent_ocr_client import TencentOCRClient


def create_app() -> FastAPI:
    ensure_runtime_dirs()
    config = AppConfig()
    graph_service = KnowledgeGraphService(config.knowledge_graph_path)
    llm_service = LLMService()
    tencent_ocr_client = TencentOCRClient(config)
    history_service = HistoryService(HISTORY_FILE)
    pipeline_service = PipelineService(
        ocr_service=OCRService(llm_service, tencent_ocr_client),
        diagnosis_service=DiagnosisService(graph_service, llm_service),
        model_generation_service=ModelGenerationService(config),
        history_service=history_service,
    )
    app = FastAPI(title=config.app_name)
    app.state.config = config
    app.state.graph_service = graph_service
    app.state.llm_service = llm_service
    app.state.history_service = history_service
    app.state.pipeline_service = pipeline_service
    app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
    app.include_router(router)
    return app


app = create_app()
