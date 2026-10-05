from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from app.config import UPLOAD_DIR
from app.models import (
    GraphAdditionRequest,
    GraphAdditionResponse,
    GraphCandidate,
    GraphSnapshot,
    GraphSuggestionRequest,
    GraphSuggestionResponse,
)


router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[1] / "templates"))


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    defaults = request.app.state.config
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_name": defaults.app_name,
            "default_ocr_provider": defaults.default_ocr_provider,
            "default_llm_provider": defaults.default_llm_provider,
            "default_model_provider": defaults.default_model_provider,
            "default_ollama_model": defaults.ollama_model,
            "default_ollama_base_url": defaults.ollama_base_url,
            "default_deepseek_model": defaults.deepseek_model,
            "default_deepseek_base_url": defaults.deepseek_base_url,
            "has_deepseek_api_key": bool(defaults.deepseek_api_key),
            "has_tencent_ocr_key": bool(defaults.tencent_secret_id and defaults.tencent_secret_key),
            "has_tripo_api_key": bool(defaults.tripo_api_key),
        },
    )


@router.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/api/history/summary")
async def history_summary(request: Request, student_name: Optional[str] = None) -> dict:
    summary = request.app.state.history_service.summary(student_name)
    return summary.model_dump()


@router.post("/api/pipeline/run")
async def run_pipeline(
    request: Request,
    image: UploadFile = File(...),
    student_name: str = Form(""),
    subject: str = Form(""),
    grade: str = Form(""),
    question_type: str = Form(""),
    ocr_provider: str = Form("demo"),
    llm_provider: str = Form("mock"),
    llm_model: str = Form(""),
    ollama_base_url: str = Form(""),
    deepseek_base_url: str = Form(""),
    model_provider: str = Form("demo"),
) -> dict:
    suffix = Path(image.filename or "upload.png").suffix or ".png"
    target_path = UPLOAD_DIR / "{0}{1}".format(uuid4().hex, suffix)
    target_path.write_bytes(await image.read())
    config = request.app.state.config
    result = await request.app.state.pipeline_service.run(
        image_path=target_path,
        student_name=student_name,
        subject=subject,
        grade=grade,
        question_type=question_type,
        ocr_provider=ocr_provider or config.default_ocr_provider,
        llm_provider=llm_provider or config.default_llm_provider,
        llm_model=llm_model or (
            config.deepseek_model if llm_provider == "deepseek" else config.ollama_model
        ),
        ollama_base_url=ollama_base_url or config.ollama_base_url,
        deepseek_base_url=deepseek_base_url or config.deepseek_base_url,
        deepseek_api_key=config.deepseek_api_key,
        model_provider=model_provider or config.default_model_provider,
    )
    return result.model_dump()


@router.post("/api/pipeline/stream")
async def stream_pipeline(
    request: Request,
    image: UploadFile = File(...),
    student_name: str = Form(""),
    subject: str = Form(""),
    grade: str = Form(""),
    question_type: str = Form(""),
    ocr_provider: str = Form("demo"),
    llm_provider: str = Form("mock"),
    llm_model: str = Form(""),
    ollama_base_url: str = Form(""),
    deepseek_base_url: str = Form(""),
    model_provider: str = Form("demo"),
) -> StreamingResponse:
    suffix = Path(image.filename or "upload.png").suffix or ".png"
    target_path = UPLOAD_DIR / "{0}{1}".format(uuid4().hex, suffix)
    target_path.write_bytes(await image.read())
    config = request.app.state.config

    async def event_generator():
        queue = asyncio.Queue()

        def emit_event(event: dict) -> None:
            queue.put_nowait(event)

        async def run_task():
            try:
                await request.app.state.pipeline_service.run_with_events(
                    image_path=target_path,
                    student_name=student_name,
                    subject=subject,
                    grade=grade,
                    question_type=question_type,
                    ocr_provider=ocr_provider or config.default_ocr_provider,
                    llm_provider=llm_provider or config.default_llm_provider,
                    llm_model=llm_model or (
                        config.deepseek_model if llm_provider == "deepseek" else config.ollama_model
                    ),
                    ollama_base_url=ollama_base_url or config.ollama_base_url,
                    deepseek_base_url=deepseek_base_url or config.deepseek_base_url,
                    deepseek_api_key=config.deepseek_api_key,
                    model_provider=model_provider or config.default_model_provider,
                    emit_event=emit_event,
                )
            except Exception as exc:
                emit_event({"type": "error", "message": str(exc)})

        task = asyncio.create_task(run_task())
        try:
            while True:
                if task.done() and queue.empty():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue
                yield json.dumps(event, ensure_ascii=False) + "\n"
        finally:
            await task

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@router.post("/api/graph/suggest")
async def suggest_graph_addition(request: Request, payload: GraphSuggestionRequest) -> dict[str, Any]:
    graph_service = request.app.state.graph_service
    llm_service = request.app.state.llm_service
    config = request.app.state.config
    missing_points = graph_service.find_missing_knowledge_points(payload.knowledge_points)
    if not missing_points:
        return GraphSuggestionResponse(
            missing_knowledge_points=[],
            suggestions=[],
            message="当前知识点已经存在于图谱中，无需新增节点。",
        ).model_dump()

    provider = payload.llm_provider or config.default_llm_provider
    model = payload.llm_model or (config.deepseek_model if provider == "deepseek" else config.ollama_model)
    suggestions: list[GraphCandidate] = []
    for missing_name in missing_points:
        similar_nodes = graph_service.search_similar_nodes(missing_name, limit=6)
        candidate = await _build_graph_candidate(
            graph_service=graph_service,
            llm_service=llm_service,
            missing_name=missing_name,
            question_text=payload.question_text,
            subject=payload.subject,
            grade=payload.grade,
            knowledge_points=payload.knowledge_points,
            similar_nodes=similar_nodes,
            provider=provider,
            model=model,
            ollama_base_url=payload.ollama_base_url or config.ollama_base_url,
            deepseek_base_url=payload.deepseek_base_url or config.deepseek_base_url,
            deepseek_api_key=config.deepseek_api_key,
        )
        suggestions.append(candidate)

    message = "已生成可加入图谱的候选节点，请确认后写入 K12-KGraph 合并图谱。"
    return GraphSuggestionResponse(
        missing_knowledge_points=missing_points,
        suggestions=suggestions,
        message=message,
    ).model_dump()


@router.post("/api/graph/add")
async def add_graph_nodes(request: Request, payload: GraphAdditionRequest) -> dict[str, Any]:
    graph_service = request.app.state.graph_service
    added_nodes = graph_service.add_graph_candidates([item.model_dump() for item in payload.suggestions])
    focus_node = _select_focus_node(graph_service, payload.focus_knowledge_points or added_nodes)
    graph_paths = graph_service.get_all_paths(focus_node)
    prerequisite_gap = payload.prerequisite_gap if payload.prerequisite_gap in {node for path in graph_paths for node in path} else ""
    if not prerequisite_gap and focus_node:
        prerequisite_gap = graph_service.get_lowest_prerequisite(focus_node)
    snapshot_dict = graph_service.build_learning_snapshot(
        focus_name=focus_node,
        prerequisite_gap=prerequisite_gap,
    )
    return GraphAdditionResponse(
        added_nodes=added_nodes,
        focus_node=focus_node,
        graph_paths=graph_paths,
        graph_snapshot=GraphSnapshot.model_validate(snapshot_dict),
        message="已将新知识点写入当前 K12-KGraph 合并图谱。",
    ).model_dump()


def _select_focus_node(graph_service, knowledge_points: list[str]) -> str:
    best_name = ""
    best_depth = -1
    best_length = -1
    for item in knowledge_points:
        name = graph_service.normalize_node_name(item)
        if name not in graph_service.nodes:
            continue
        paths = graph_service.get_all_paths(name)
        depth = max((len(path) for path in paths), default=0)
        if depth > best_depth or (depth == best_depth and len(name) > best_length):
            best_name = name
            best_depth = depth
            best_length = len(name)
    return best_name or (knowledge_points[0] if knowledge_points else "")


async def _build_graph_candidate(
    *,
    graph_service,
    llm_service,
    missing_name: str,
    question_text: str,
    subject: str,
    grade: str,
    knowledge_points: list[str],
    similar_nodes: list[dict[str, Any]],
    provider: str,
    model: str,
    ollama_base_url: str,
    deepseek_base_url: str,
    deepseek_api_key: str,
) -> GraphCandidate:
    similar_names = [item.get("name", "") for item in similar_nodes if item.get("name")]
    if provider in {"ollama", "deepseek"}:
        prompt = (
            "你是一位 K12 知识图谱构建助手，需要帮助把新知识点加入现有图谱。\n"
            "目标知识点: {0}\n"
            "题目文本: {1}\n"
            "学科: {2}\n"
            "年级: {3}\n"
            "OCR/诊断抽取到的知识点: {4}\n"
            "图谱中最相近的已有节点: {5}\n\n"
            "请只返回 JSON，字段必须包含：name, description, aliases, prerequisites, related_to, is_a。\n"
            "要求：\n"
            "1. name 必须使用目标知识点，除非你确信它应归一化到某个更标准表达；\n"
            "2. prerequisites/related_to/is_a 优先从“图谱中最相近的已有节点”中选择；\n"
            "3. 如果没有合适连接，可以返回空数组；\n"
            "4. 不要生成与题目明显无关的学科节点。"
        ).format(
            missing_name,
            question_text or "未提供",
            subject or "未提供",
            grade or "未提供",
            knowledge_points,
            similar_nodes,
        )
        try:
            payload = await llm_service.generate_json(
                provider=provider,
                prompt=prompt,
                model=model,
                ollama_base_url=ollama_base_url,
                deepseek_base_url=deepseek_base_url,
                deepseek_api_key=deepseek_api_key,
            )
            return _normalize_graph_candidate(payload, missing_name, subject, similar_names, source="llm", graph_service=graph_service)
        except Exception:
            pass

    fallback_prerequisites = similar_names[:2]
    return GraphCandidate(
        name=missing_name,
        subject=subject,
        node_type="Knowledge",
        description="根据题目内容自动补充到现有图谱的新知识点。",
        aliases=[],
        prerequisites=fallback_prerequisites[:1],
        related_to=fallback_prerequisites[1:2],
        is_a=[],
        similar_existing_nodes=similar_names,
        source="heuristic",
    )


def _normalize_graph_candidate(
    payload: dict[str, Any],
    fallback_name: str,
    subject: str,
    similar_names: list[str],
    *,
    source: str,
    graph_service,
) -> GraphCandidate:
    def text_list(value: Any) -> list[str]:
        if isinstance(value, list):
            items = []
            for item in value:
                text = str(item).strip()
                if text and text not in items:
                    items.append(text)
            return items
        text = str(value).strip() if value is not None else ""
        return [text] if text else []

    name = str(payload.get("name") or fallback_name).strip() or fallback_name
    prerequisites = []
    for item in text_list(payload.get("prerequisites")):
        normalized = graph_service.normalize_node_name(item)
        final_name = normalized if normalized in graph_service.nodes else item
        if final_name not in prerequisites and final_name != name:
            prerequisites.append(final_name)
    related_to = []
    for item in text_list(payload.get("related_to")):
        normalized = graph_service.normalize_node_name(item)
        final_name = normalized if normalized in graph_service.nodes else item
        if final_name not in related_to and final_name != name:
            related_to.append(final_name)
    is_a = []
    for item in text_list(payload.get("is_a")):
        normalized = graph_service.normalize_node_name(item)
        final_name = normalized if normalized in graph_service.nodes else item
        if final_name not in is_a and final_name != name:
            is_a.append(final_name)
    return GraphCandidate(
        name=name,
        subject=str(payload.get("subject") or subject or "").strip(),
        node_type=str(payload.get("node_type") or "Knowledge").strip() or "Knowledge",
        description=str(payload.get("description") or "").strip(),
        aliases=[item for item in text_list(payload.get("aliases")) if item != name],
        prerequisites=prerequisites,
        related_to=related_to,
        is_a=is_a,
        similar_existing_nodes=similar_names,
        source=source,
    )
