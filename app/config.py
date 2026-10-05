from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_DIR = APP_DIR / "data"
UPLOAD_DIR = APP_DIR / "uploads"
GENERATED_DIR = APP_DIR / "generated"
MODEL_DIR = GENERATED_DIR / "models"
HISTORY_FILE = DATA_DIR / "analysis_history.json"
GRAPH_FILE = DATA_DIR / "knowledge_graph.json"


def resolve_project_path(raw_value: str, fallback: Path) -> Path:
    value = (raw_value or "").strip()
    if not value:
        return fallback
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists() and fallback.exists():
        return fallback
    return path


def load_env_files() -> None:
    merged_values: dict[str, str] = {}
    for env_path in (PROJECT_ROOT / ".env", PROJECT_ROOT / ".envs"):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                merged_values[key] = value
    for key, value in merged_values.items():
        os.environ[key] = value


load_env_files()


@dataclass
class AppConfig:
    app_name: str = os.getenv("APP_NAME", "智诊AI")
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8000"))
    default_ocr_provider: str = os.getenv("DEFAULT_OCR_PROVIDER", "demo")
    default_llm_provider: str = os.getenv("DEFAULT_LLM_PROVIDER", "mock")
    default_model_provider: str = os.getenv("DEFAULT_MODEL_PROVIDER", "demo")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    tencent_secret_id: str = os.getenv("TENCENT_SECRET_ID", "")
    tencent_secret_key: str = os.getenv("TENCENT_SECRET_KEY", "")
    tencent_ocr_region: str = os.getenv("TENCENT_OCR_REGION", "ap-beijing")
    tencent_ocr_version: str = os.getenv("TENCENT_OCR_VERSION", "2024-07-18")
    tripo_api_key: str = os.getenv("TRIPO_API_KEY", "")
    tripo_base_url: str = os.getenv("TRIPO_BASE_URL", "https://openapi.tripo3d.com/v3")
    tripo_model_version: str = os.getenv("TRIPO_MODEL_VERSION", "v3.1-20260211")
    tripo_poll_interval_seconds: float = float(os.getenv("TRIPO_POLL_INTERVAL_SECONDS", "2"))
    tripo_timeout_seconds: float = float(os.getenv("TRIPO_TIMEOUT_SECONDS", "120"))
    knowledge_graph_path: Path = field(
        default_factory=lambda: resolve_project_path(os.getenv("KNOWLEDGE_GRAPH_PATH", ""), GRAPH_FILE)
    )


def ensure_runtime_dirs() -> None:
    for path in (DATA_DIR, UPLOAD_DIR, GENERATED_DIR, MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)
