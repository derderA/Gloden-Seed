from __future__ import annotations

from typing import Callable
from pathlib import Path
from uuid import uuid4

from app.models import PipelineResult, ProcessStep
from app.services.diagnosis_service import DiagnosisService
from app.services.history_service import HistoryService
from app.services.model_generation_service import ModelGenerationService
from app.services.ocr_service import OCRService


class PipelineService:
    def __init__(
        self,
        ocr_service: OCRService,
        diagnosis_service: DiagnosisService,
        model_generation_service: ModelGenerationService,
        history_service: HistoryService,
    ) -> None:
        self.ocr_service = ocr_service
        self.diagnosis_service = diagnosis_service
        self.model_generation_service = model_generation_service
        self.history_service = history_service

    def _build_initial_process_steps(self) -> list[ProcessStep]:
        return [
            ProcessStep(key="upload", label="图片上传", status="completed", detail="作业图片已接收并缓存。"),
            ProcessStep(key="ocr", label="OCR 结构化", status="running", detail="正在提取题目、答案和知识点。"),
            ProcessStep(key="graph", label="知识图谱回溯", status="pending", detail="等待定位前置依赖路径。"),
            ProcessStep(key="diagnosis", label="根因诊断", status="pending", detail="等待判断错误层级与前置漏洞。"),
            ProcessStep(key="scene", label="3D 干预生成", status="pending", detail="等待生成 3D 干预内容。"),
        ]

    @staticmethod
    def _dump_process_steps(process_steps: list[ProcessStep]) -> list[dict]:
        return [step.model_dump() for step in process_steps]

    async def run(
        self,
        *,
        image_path: Path,
        student_name: str,
        subject: str,
        grade: str,
        question_type: str,
        ocr_provider: str,
        llm_provider: str,
        llm_model: str,
        ollama_base_url: str,
        deepseek_base_url: str,
        deepseek_api_key: str,
        model_provider: str,
    ) -> PipelineResult:
        return await self._run_internal(
            image_path=image_path,
            student_name=student_name,
            subject=subject,
            grade=grade,
            question_type=question_type,
            ocr_provider=ocr_provider,
            llm_provider=llm_provider,
            llm_model=llm_model,
            ollama_base_url=ollama_base_url,
            deepseek_base_url=deepseek_base_url,
            deepseek_api_key=deepseek_api_key,
            model_provider=model_provider,
            emit_event=lambda event: None,
        )

    async def run_with_events(
        self,
        *,
        image_path: Path,
        student_name: str,
        subject: str,
        grade: str,
        question_type: str,
        ocr_provider: str,
        llm_provider: str,
        llm_model: str,
        ollama_base_url: str,
        deepseek_base_url: str,
        deepseek_api_key: str,
        model_provider: str,
        emit_event: Callable[[dict], None],
    ) -> PipelineResult:
        return await self._run_internal(
            image_path=image_path,
            student_name=student_name,
            subject=subject,
            grade=grade,
            question_type=question_type,
            ocr_provider=ocr_provider,
            llm_provider=llm_provider,
            llm_model=llm_model,
            ollama_base_url=ollama_base_url,
            deepseek_base_url=deepseek_base_url,
            deepseek_api_key=deepseek_api_key,
            model_provider=model_provider,
            emit_event=emit_event,
        )

    async def _run_internal(
        self,
        *,
        image_path: Path,
        student_name: str,
        subject: str,
        grade: str,
        question_type: str,
        ocr_provider: str,
        llm_provider: str,
        llm_model: str,
        ollama_base_url: str,
        deepseek_base_url: str,
        deepseek_api_key: str,
        model_provider: str,
        emit_event: Callable[[dict], None],
    ) -> PipelineResult:
        warnings: list[str] = []
        process_steps = self._build_initial_process_steps()
        emit_event(
            {
                "type": "process_update",
                "process_steps": self._dump_process_steps(process_steps),
            }
        )
        ocr_result = await self.ocr_service.extract(
            image_path=image_path,
            provider=ocr_provider,
            subject=subject,
            grade=grade,
            question_type=question_type,
            llm_provider=llm_provider,
            llm_model=llm_model,
            ollama_base_url=ollama_base_url,
            deepseek_base_url=deepseek_base_url,
            deepseek_api_key=deepseek_api_key,
        )
        process_steps[1].status = "completed"
        process_steps[1].detail = "已提取 {0} 道错题结构化结果。".format(len(ocr_result.questions))
        process_steps[2].status = "running"
        process_steps[2].detail = "正在根据知识点路径回溯前置依赖。"
        emit_event(
            {
                "type": "ocr_result",
                "ocr_result": ocr_result.model_dump(),
                "process_steps": self._dump_process_steps(process_steps),
            }
        )
        diagnosis_context = self.history_service.build_diagnosis_context(
            student_name=student_name or None,
            focus_knowledge_points=ocr_result.questions[0].knowledge_points if ocr_result.questions else [],
        )
        diagnostic_result = await self.diagnosis_service.diagnose(
            ocr_result=ocr_result,
            provider=llm_provider,
            model=llm_model,
            ollama_base_url=ollama_base_url,
            deepseek_base_url=deepseek_base_url,
            deepseek_api_key=deepseek_api_key,
            diagnosis_context=diagnosis_context,
        )
        process_steps[2].status = "completed"
        process_steps[2].detail = "已定位 {0} 条知识图谱路径。".format(len(diagnostic_result.graph_paths))
        process_steps[3].status = "completed"
        process_steps[3].detail = "已判断为 {0}，前置漏洞为 {1}。".format(
            diagnostic_result.error_level, diagnostic_result.prerequisite_gap
        )
        if diagnostic_result.used_fallback and llm_provider in {"ollama", "deepseek"}:
            warning_message = "大模型调用未成功，已自动回退到启发式诊断结果。"
            if diagnostic_result.llm_error:
                warning_message += " 原因：{0}".format(diagnostic_result.llm_error)
            warnings.append(warning_message)
            process_steps[3].detail += " 本次使用了回退诊断。"
        process_steps[4].status = "running"
        process_steps[4].detail = "正在根据诊断结果生成 3D 干预内容。"
        emit_event(
            {
                "type": "diagnosis_result",
                "diagnostic_result": diagnostic_result.model_dump(),
                "warnings": warnings,
                "process_steps": self._dump_process_steps(process_steps),
            }
        )
        intervention_result = await self.model_generation_service.generate(
            provider=model_provider,
            ocr_result=ocr_result,
            diagnostic_result=diagnostic_result,
        )
        process_steps[4].status = "completed"
        process_steps[4].detail = intervention_result.generation_status
        emit_event(
            {
                "type": "scene_result",
                "intervention_result": intervention_result.model_dump(),
                "process_steps": self._dump_process_steps(process_steps),
            }
        )
        analysis_id = uuid4().hex
        self.history_service.append(
            {
                "analysis_id": analysis_id,
                "student_name": student_name,
                "subject": ocr_result.subject,
                "grade": ocr_result.grade,
                "knowledge_points": ocr_result.questions[0].knowledge_points if ocr_result.questions else [],
                "prerequisite_gap": diagnostic_result.prerequisite_gap,
                "error_level": diagnostic_result.error_level,
            }
        )
        history_summary = self.history_service.summary(student_name or None)
        result = PipelineResult(
            analysis_id=analysis_id,
            ocr_result=ocr_result,
            diagnostic_result=diagnostic_result,
            intervention_result=intervention_result,
            history_summary=history_summary,
            process_steps=process_steps,
            warnings=warnings,
        )
        emit_event(
            {
                "type": "complete",
                "payload": result.model_dump(),
            }
        )
        return result
