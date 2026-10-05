from __future__ import annotations

import json
from pathlib import Path
import logging
import tempfile
from typing import Optional, Tuple

from PIL import Image

from app.models import OCRQuestion, OCRResult
from app.services.llm_service import LLMService
from app.services.tencent_ocr_client import TencentOCRClient

logger = logging.getLogger(__name__)


class OCRService:
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        tencent_client: Optional[TencentOCRClient] = None,
    ) -> None:
        self.llm_service = llm_service
        self.tencent_client = tencent_client

    async def extract(
        self,
        image_path: Path,
        provider: str,
        subject: str,
        grade: str,
        question_type: str,
        llm_provider: str,
        llm_model: str,
        ollama_base_url: str,
        deepseek_base_url: str,
        deepseek_api_key: str,
    ) -> OCRResult:
        if provider == "ai":
            return await self._ai_result(
                image_path=image_path,
                subject=subject,
                grade=grade,
                question_type=question_type,
                llm_provider=llm_provider,
                llm_model=llm_model,
                ollama_base_url=ollama_base_url,
                deepseek_base_url=deepseek_base_url,
                deepseek_api_key=deepseek_api_key,
            )
        if provider == "demo":
            return self._demo_result(image_path, subject, grade, question_type)
        if provider == "tencent":
            return await self._tencent_result(
                image_path=image_path,
                subject=subject,
                grade=grade,
                question_type=question_type,
                llm_provider=llm_provider,
                llm_model=llm_model,
                ollama_base_url=ollama_base_url,
                deepseek_base_url=deepseek_base_url,
                deepseek_api_key=deepseek_api_key,
            )
        raise RuntimeError("不支持的 OCR Provider: {0}".format(provider))

    async def _tencent_result(
        self,
        *,
        image_path: Path,
        subject: str,
        grade: str,
        question_type: str,
        llm_provider: str,
        llm_model: str,
        ollama_base_url: str,
        deepseek_base_url: str,
        deepseek_api_key: str,
    ) -> OCRResult:
        if not self.tencent_client:
            raise RuntimeError("腾讯云 OCR 客户端未初始化。")
        response = await self.tencent_client.submit_question_mark_agent_job(
            image_path=image_path,
            question_type=question_type,
        )
        result = self._tencent_payload_to_result(
            payload=response,
            image_path=image_path,
            subject=subject,
            grade=grade,
            question_type=question_type,
        )
        if self._needs_tencent_enrichment(result):
            enriched, failure_reason = await self._enrich_tencent_result_with_llm(
                image_path=image_path,
                subject=subject,
                grade=grade,
                question_type=question_type,
                tencent_payload=response,
                llm_provider=llm_provider,
                llm_model=llm_model,
                ollama_base_url=ollama_base_url,
                deepseek_base_url=deepseek_base_url,
                deepseek_api_key=deepseek_api_key,
            )
            if enriched is not None:
                return enriched
        return result

    async def _ai_result(
        self,
        *,
        image_path: Path,
        subject: str,
        grade: str,
        question_type: str,
        llm_provider: str,
        llm_model: str,
        ollama_base_url: str,
        deepseek_base_url: str,
        deepseek_api_key: str,
    ) -> OCRResult:
        if not self.llm_service or llm_provider not in {"ollama", "deepseek"}:
            demo_result = self._demo_result(image_path, subject, grade, question_type)
            demo_result.provider = "ai-fallback"
            demo_result.raw_summary = "当前未配置可用多模态模型，已回退到题型推断演示结果。"
            return demo_result
        prompt = self._build_image_prompt(subject, grade, question_type)
        try:
            payload = await self.llm_service.generate_json_with_image(
                provider=llm_provider,
                prompt=prompt,
                image_path=image_path,
                model=llm_model,
                ollama_base_url=ollama_base_url,
                deepseek_base_url=deepseek_base_url,
                deepseek_api_key=deepseek_api_key,
            )
            return self._payload_to_result(payload, image_path, subject, grade)
        except Exception:
            demo_result = self._demo_result(image_path, subject, grade, question_type)
            demo_result.provider = "ai-fallback"
            demo_result.raw_summary = "多模态识图失败，已回退到题型推断演示结果。"
            return demo_result

    def _demo_result(
        self,
        image_path: Path,
        subject: str,
        grade: str,
        question_type: str,
    ) -> OCRResult:
        inferred_case = self._infer_case(question_type, subject)
        question = self._build_demo_question(inferred_case)
        return OCRResult(
            provider="demo",
            image_name=image_path.name,
            subject=subject or ("物理" if inferred_case == "magnet" else "数学"),
            grade=grade or "小学/初中演示",
            questions=[question],
            raw_summary="已使用演示 OCR 模式返回结构化错题结果。题目类型提示：{0}".format(
                question_type or "未填写"
            ),
        )

    def _build_image_prompt(self, subject: str, grade: str, question_type: str) -> str:
        return (
            "请分析这张 K12 作业图片，并输出结构化 JSON。\n"
            "已知信息：\n"
            "- 学科: {0}\n"
            "- 年级: {1}\n"
            "- 题目类型提示: {2}\n\n"
            "任务要求：\n"
            "1. 识别题目文本；\n"
            "2. 结合学生作答推断学生答案；\n"
            "3. 给出正确答案；\n"
            "4. 提取知识点标签；\n"
            "5. 定位错误步骤；\n"
            "6. 若图片信息不足，请尽量给出最合理的结构化分析，并在 raw_summary 中说明不确定性。\n\n"
            "输出 JSON 字段必须包含：subject, grade, raw_summary, questions。\n"
            "questions 是数组，每项包含：question_text, knowledge_points, student_answer, correct_answer, error_steps, handwriting_boxes。".format(
                subject or "未填写",
                grade or "未填写",
                question_type or "未填写",
            )
        )

    def _payload_to_result(
        self,
        payload: dict,
        image_path: Path,
        subject: str,
        grade: str,
    ) -> OCRResult:
        questions = []
        for item in payload.get("questions", []):
            questions.append(
                OCRQuestion(
                    question_text=self._normalize_text_value(item.get("question_text")),
                    knowledge_points=self._normalize_text_list(item.get("knowledge_points")),
                    student_answer=self._normalize_text_value(item.get("student_answer")),
                    correct_answer=self._normalize_text_value(item.get("correct_answer")),
                    error_steps=self._normalize_text_list(item.get("error_steps")),
                    handwriting_boxes=self._normalize_boxes(item.get("handwriting_boxes")),
                )
            )
        if not questions:
            raise RuntimeError("多模态模型未返回有效 questions。")
        return OCRResult(
            provider="ai",
            image_name=image_path.name,
            subject=payload.get("subject", subject),
            grade=payload.get("grade", grade),
            questions=questions,
            raw_summary=payload.get("raw_summary", "已完成 AI 图片分析。"),
        )

    def _tencent_payload_to_result(
        self,
        *,
        payload: dict,
        image_path: Path,
        subject: str,
        grade: str,
        question_type: str,
    ) -> OCRResult:
        question_blocks = []
        for page in payload.get("QuestionInfo", []):
            question_blocks.extend(page.get("ResultList") or [])

        scored_questions: list[tuple[int, OCRQuestion]] = []
        for index, item in enumerate(question_blocks, start=1):
            question_text = self._first_non_empty_text(
                item,
                "Question",
                "SubQuestion",
                "QuestionText",
                "Title",
                "Content",
                "Problem",
                "Stem",
            )
            student_answer = self._join_text_block(item.get("Answer"))
            correct_answer = self._first_non_empty_text(
                item,
                "TrueAnswer",
                "ReferenceAnswer",
                "CorrectAnswer",
            )
            knowledge_points = self._extract_text_list(
                item,
                "KnowledgePoints",
                "KnowledgePoint",
                "Tags",
            )
            error_steps = self._extract_text_list(
                item,
                "StepCorrection",
                "ErrorReason",
                "Correction",
                "Parse",
                "AnswerAnalysis",
            )
            boxes = self._extract_coord_boxes(item.get("Coord") or [])
            if not question_text:
                question_text = "腾讯 OCR 未返回题干文本（题块 {0}）".format(index)
            if not knowledge_points:
                knowledge_points = []
            if not error_steps:
                error_steps = ["腾讯 OCR 未返回步骤级批改信息，待后续诊断补充"]
            question = OCRQuestion(
                question_text=question_text,
                knowledge_points=knowledge_points,
                student_answer=student_answer,
                correct_answer=correct_answer,
                error_steps=error_steps,
                handwriting_boxes=boxes,
            )
            scored_questions.append((self._score_ocr_question(question), question))

        meaningful_questions = [question for score, question in scored_questions if score > 0]
        questions = meaningful_questions or [question for _, question in scored_questions]

        if not questions:
            questions.append(
                OCRQuestion(
                    question_text="腾讯 OCR 未返回可解析题目结构",
                    knowledge_points=[],
                    student_answer="",
                    correct_answer="",
                    error_steps=["腾讯 OCR 未返回可解析的题目结构"],
                    handwriting_boxes=[],
                )
            )

        summary_parts = [
            "腾讯云 OCR 已完成试题批改 Agent 调用。",
            "JobId: {0}".format(payload.get("JobId", "未知")),
        ]
        if question_type.strip():
            summary_parts.append("题目类型提示: {0}".format(question_type.strip()))
        if payload.get("QuestionCount"):
            summary_parts.append("题块数量: {0}".format(payload.get("QuestionCount")))
        if payload.get("OriginalImageUrl"):
            summary_parts.append("已回传原图地址。")

        return OCRResult(
            provider="tencent",
            image_name=image_path.name,
            subject=subject,
            grade=grade,
            questions=questions,
            raw_summary=" ".join(summary_parts),
        )

    async def _enrich_tencent_result_with_llm(
        self,
        *,
        image_path: Path,
        subject: str,
        grade: str,
        question_type: str,
        tencent_payload: dict,
        llm_provider: str,
        llm_model: str,
        ollama_base_url: str,
        deepseek_base_url: str,
        deepseek_api_key: str,
    ) -> Tuple[Optional[OCRResult], str]:
        if not self.llm_service:
            return None, "LLM 服务未初始化，无法执行二次补全。"

        provider = llm_provider
        model = llm_model
        if provider not in {"ollama", "deepseek"}:
            if deepseek_api_key:
                provider = "deepseek"
                model = "deepseek-chat"
            else:
                return None, "当前 LLM Provider 不支持图片分析，且未配置 DeepSeek API Key 作为兜底。"

        try:
            crop_result = await self._enrich_tencent_result_with_crops(
                image_path=image_path,
                subject=subject,
                grade=grade,
                question_type=question_type,
                tencent_payload=tencent_payload,
                provider=provider,
                model=model,
                ollama_base_url=ollama_base_url,
                deepseek_base_url=deepseek_base_url,
                deepseek_api_key=deepseek_api_key,
            )
            if crop_result is not None:
                return crop_result, ""
        except Exception as exc:
            logger.exception("Tencent OCR crop enrichment failed.")
            crop_error = "题块裁剪补全失败: {0}".format(exc)
        else:
            crop_error = ""

        prompt = self._build_tencent_enrichment_prompt(
            subject=subject,
            grade=grade,
            question_type=question_type,
            tencent_payload=tencent_payload,
        )
        try:
            payload = await self.llm_service.generate_json_with_image(
                provider=provider,
                prompt=prompt,
                image_path=image_path,
                model=model,
                ollama_base_url=ollama_base_url,
                deepseek_base_url=deepseek_base_url,
                deepseek_api_key=deepseek_api_key,
            )
            result = self._payload_to_result(payload, image_path, subject, grade)
            result.provider = "tencent"
            result.raw_summary = (
                "腾讯 OCR 结构化结果缺失，已结合原图和腾讯返回数据进行二次补全。"
            )
            return result, ""
        except Exception as exc:
            logger.exception("Tencent OCR enrichment failed.")
            if crop_error:
                return None, "{0}；整图补全失败: {1}".format(crop_error, exc)
            return None, str(exc)

    async def _enrich_tencent_result_with_crops(
        self,
        *,
        image_path: Path,
        subject: str,
        grade: str,
        question_type: str,
        tencent_payload: dict,
        provider: str,
        model: str,
        ollama_base_url: str,
        deepseek_base_url: str,
        deepseek_api_key: str,
    ) -> Optional[OCRResult]:
        crop_specs = self._build_tencent_crop_specs(image_path=image_path, payload=tencent_payload)
        if not crop_specs:
            return None

        enriched_questions: list[OCRQuestion] = []
        last_summary = ""
        for index, crop_paths, boxes in crop_specs:
            prompt = self._build_tencent_crop_prompt(
                subject=subject,
                grade=grade,
                question_type=question_type,
                question_index=index,
            )
            question = None
            summary = ""
            for crop_path in crop_paths:
                payload = await self.llm_service.generate_json_with_image(
                    provider=provider,
                    prompt=prompt,
                    image_path=crop_path,
                    model=model,
                    ollama_base_url=ollama_base_url,
                    deepseek_base_url=deepseek_base_url,
                    deepseek_api_key=deepseek_api_key,
                )
                question, summary = self._single_question_payload_to_question(payload, boxes)
                if self._score_ocr_question(question) > 0:
                    break
            if question.question_text:
                enriched_questions.append(question)
            if summary:
                last_summary = summary

        if not enriched_questions:
            return None

        result = OCRResult(
            provider="tencent",
            image_name=image_path.name,
            subject=subject,
            grade=grade,
            questions=enriched_questions,
            raw_summary=last_summary or "腾讯 OCR 未返回题干文本，已基于题块裁图完成二次补全。",
        )
        return result

    def _build_tencent_enrichment_prompt(
        self,
        *,
        subject: str,
        grade: str,
        question_type: str,
        tencent_payload: dict,
    ) -> str:
        payload_excerpt = json.dumps(tencent_payload, ensure_ascii=False)[:4000]
        return (
            "你是一位 K12 题目结构化助手。腾讯 OCR 已返回题块坐标，但文本字段不完整。"
            "请结合上传图片和腾讯返回结果，补全结构化题目信息，并只输出 JSON。\n"
            "已知信息：\n"
            "- 学科: {0}\n"
            "- 年级: {1}\n"
            "- 题目类型提示: {2}\n"
            "- 腾讯 OCR 返回摘录: {3}\n\n"
            "输出 JSON 字段必须包含：subject, grade, raw_summary, questions。\n"
            "questions 是数组，每项包含：question_text, knowledge_points, student_answer, correct_answer, error_steps, handwriting_boxes。\n"
            "如果图片中无法确定正确答案或学生答案，可保留为空字符串，但 question_text 尽量补全。".format(
                subject or "未填写",
                grade or "未填写",
                question_type or "未填写",
                payload_excerpt,
            )
        )

    def _build_tencent_crop_prompt(
        self,
        *,
        subject: str,
        grade: str,
        question_type: str,
        question_index: int,
    ) -> str:
        return (
            "请只分析这张单题裁剪图，并输出单题结构化 JSON。\n"
            "已知信息：\n"
            "- 学科: {0}\n"
            "- 年级: {1}\n"
            "- 题目类型提示: {2}\n"
            "- 题块序号: {3}\n\n"
            "要求：\n"
            "1. 尽量还原题目文本；\n"
            "2. 识别学生作答；\n"
            "3. 推断正确答案；\n"
            "4. 给出知识点标签；\n"
            "5. 给出错误步骤定位；\n"
            "6. 只返回 JSON。\n\n"
            "输出 JSON 支持两种格式之一：\n"
            "A. 直接返回 question_text, knowledge_points, student_answer, correct_answer, error_steps, raw_summary；\n"
            "B. 返回 questions 数组，但数组中只保留一题。\n"
            "若某字段无法确定，可返回空字符串或空数组。".format(
                subject or "未填写",
                grade or "未填写",
                question_type or "未填写",
                question_index,
            )
        )

    def _infer_case(self, question_type: str, subject: str) -> str:
        hint = "{0} {1}".format(subject or "", question_type or "").lower()
        if any(keyword in hint for keyword in ("磁", "电磁", "磁场", "磁感线", "magnet")):
            return "magnet"
        if any(keyword in hint for keyword in ("圆柱", "体积", "切片", "几何", "cylinder", "volume")):
            return "cylinder"
        return "fraction"

    def _build_demo_question(self, inferred_case: str) -> OCRQuestion:
        if inferred_case == "cylinder":
            return OCRQuestion(
                question_text="把一个底面半径为 3cm、高为 5cm 的圆柱体切成很多薄片，求它的体积。",
                knowledge_points=["圆柱体积", "圆面积公式"],
                student_answer="体积=3.14×3×5=47.1",
                correct_answer="体积=3.14×3×3×5=141.3",
                error_steps=["第1步未理解底面积应为 pi*r^2", "无法把切片累加与体积公式对应起来"],
                handwriting_boxes=[[70, 80, 310, 160]],
            )
        if inferred_case == "magnet":
            return OCRQuestion(
                question_text="请判断条形磁铁周围磁感线方向，并标出小磁针 N 极偏转方向。",
                knowledge_points=["电磁场方向判断", "磁场方向"],
                student_answer="磁感线从 S 极出发回到 N 极",
                correct_answer="磁感线在磁体外部从 N 极出发回到 S 极",
                error_steps=["第2步方向判断反了", "空间旋转后无法保持方向一致"],
                handwriting_boxes=[[88, 92, 336, 188]],
            )
        return OCRQuestion(
            question_text="计算 1/3 + 1/6，并写出通分过程。",
            knowledge_points=["分数加法", "分数通分"],
            student_answer="1/3 + 1/6 = 2/9",
            correct_answer="1/3 + 1/6 = 2/6 + 1/6 = 3/6 = 1/2",
            error_steps=["第2步通分错误", "把分母直接相加了"],
            handwriting_boxes=[[64, 96, 280, 164]],
        )

    def _extract_text_list(self, item: dict, *keys: str) -> list[str]:
        values: list[str] = []
        for key in keys:
            values.extend(self._flatten_text_values(item.get(key)))
        unique_values: list[str] = []
        for value in values:
            if value and value not in unique_values:
                unique_values.append(value)
        return unique_values

    def _first_non_empty_text(self, item: dict, *keys: str) -> str:
        for key in keys:
            values = self._flatten_text_values(item.get(key))
            if values:
                return " ".join(values)
        return ""

    def _join_text_block(self, value: object) -> str:
        values = self._flatten_text_values(value)
        return " ".join(values)

    def _flatten_text_values(self, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            cleaned = value.strip()
            return [cleaned] if cleaned else []
        if isinstance(value, (int, float)):
            return [str(value)]
        if isinstance(value, list):
            result: list[str] = []
            for item in value:
                result.extend(self._flatten_text_values(item))
            return result
        if isinstance(value, dict):
            result: list[str] = []
            preferred_keys = [
                "Text",
                "Content",
                "Value",
                "Name",
                "Label",
                "Word",
                "Words",
                "DetectedText",
                "Item",
            ]
            seen_key = False
            for key in preferred_keys:
                if key in value:
                    seen_key = True
                    result.extend(self._flatten_text_values(value[key]))
            if seen_key:
                return result
            for child in value.values():
                result.extend(self._flatten_text_values(child))
            return result
        return []

    def _extract_coord_boxes(self, coords: list) -> list[list[int]]:
        boxes: list[list[int]] = []
        for coord in coords:
            points = []
            for key in ("LeftTop", "RightTop", "RightBottom", "LeftBottom"):
                point = coord.get(key) if isinstance(coord, dict) else None
                if isinstance(point, dict) and "X" in point and "Y" in point:
                    points.append((int(point["X"]), int(point["Y"])))
            if not points:
                continue
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            boxes.append([min(xs), min(ys), max(xs), max(ys)])
        return boxes

    def _score_ocr_question(self, question: OCRQuestion) -> int:
        score = 0
        if question.question_text and not question.question_text.startswith("腾讯 OCR 未返回"):
            score += 4
        if question.student_answer:
            score += 2
        if question.correct_answer:
            score += 2
        if question.knowledge_points:
            score += 2
        if question.error_steps and not (
            len(question.error_steps) == 1 and question.error_steps[0].startswith("腾讯 OCR 未返回")
        ):
            score += 1
        return score

    def _needs_tencent_enrichment(self, result: OCRResult) -> bool:
        if not result.questions:
            return True
        best_score = max(self._score_ocr_question(question) for question in result.questions)
        return best_score < 4

    def _normalize_text_value(self, value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, list):
            return " ".join(self._normalize_text_list(value)).strip()
        if isinstance(value, dict):
            return " ".join(self._flatten_text_values(value)).strip()
        return str(value).strip()

    def _normalize_text_list(self, value: object) -> list[str]:
        values = self._flatten_text_values(value)
        unique_values: list[str] = []
        for item in values:
            cleaned = item.strip()
            if cleaned and cleaned not in unique_values:
                unique_values.append(cleaned)
        return unique_values

    def _normalize_boxes(self, value: object) -> list[list[int]]:
        if not isinstance(value, list):
            return []
        boxes: list[list[int]] = []
        for item in value:
            if isinstance(item, list) and len(item) == 4:
                try:
                    boxes.append([int(item[0]), int(item[1]), int(item[2]), int(item[3])])
                except (TypeError, ValueError):
                    continue
        return boxes

    def _single_question_payload_to_question(
        self,
        payload: dict,
        default_boxes: list[list[int]],
    ) -> tuple[OCRQuestion, str]:
        candidate = payload
        if isinstance(payload.get("questions"), list) and payload["questions"]:
            candidate = payload["questions"][0]
        question = OCRQuestion(
            question_text=self._normalize_text_value(candidate.get("question_text")),
            knowledge_points=self._normalize_text_list(candidate.get("knowledge_points")),
            student_answer=self._normalize_text_value(candidate.get("student_answer")),
            correct_answer=self._normalize_text_value(candidate.get("correct_answer")),
            error_steps=self._normalize_text_list(candidate.get("error_steps")),
            handwriting_boxes=self._normalize_boxes(candidate.get("handwriting_boxes")) or default_boxes,
        )
        return question, self._normalize_text_value(payload.get("raw_summary"))

    def _build_tencent_crop_specs(
        self,
        *,
        image_path: Path,
        payload: dict,
    ) -> list[tuple[int, list[Path], list[list[int]]]]:
        try:
            image = Image.open(image_path)
            image.load()
        except Exception:
            logger.exception("Failed to open uploaded image for Tencent crop enrichment.")
            return []

        crop_specs: list[tuple[int, list[Path], list[list[int]]]] = []
        crop_root = Path(tempfile.mkdtemp(prefix="tencent_ocr_", dir=str(image_path.parent)))
        question_index = 0
        for page in payload.get("QuestionInfo", []):
            angle = int(page.get("Angle") or 0)
            page_width = int(page.get("Width") or image.width or 1)
            page_height = int(page.get("Height") or image.height or 1)
            for item in page.get("ResultList") or []:
                boxes = self._extract_coord_boxes(item.get("Coord") or [])
                if not boxes:
                    continue
                question_index += 1
                crop_box = self._transform_box_to_image(
                    box=boxes[0],
                    angle=angle,
                    page_width=page_width,
                    page_height=page_height,
                    image_width=image.width,
                    image_height=image.height,
                )
                if not crop_box:
                    continue
                crop_paths = self._save_crop_variants(
                    image=image,
                    crop_box=crop_box,
                    angle=angle,
                    crop_root=crop_root,
                    stem="{0}_q{1}".format(image_path.stem, question_index),
                )
                crop_specs.append((question_index, crop_paths, [list(crop_box)]))
        return crop_specs

    def _save_crop_variants(
        self,
        *,
        image: Image.Image,
        crop_box: tuple[int, int, int, int],
        angle: int,
        crop_root: Path,
        stem: str,
    ) -> list[Path]:
        crop = image.crop(crop_box)
        paths: list[Path] = []

        primary_path = crop_root / "{0}.png".format(stem)
        crop.save(primary_path)
        paths.append(primary_path)

        corrected_angle = (360 - angle) % 360
        if corrected_angle:
            corrected = crop.rotate(corrected_angle, expand=True)
            corrected_path = crop_root / "{0}_upright.png".format(stem)
            corrected.save(corrected_path)
            paths.insert(0, corrected_path)
        return paths

    def _transform_box_to_image(
        self,
        *,
        box: list[int],
        angle: int,
        page_width: int,
        page_height: int,
        image_width: int,
        image_height: int,
    ) -> Optional[tuple[int, int, int, int]]:
        x1, y1, x2, y2 = box
        if angle == 90:
            x1, y1, x2, y2 = y1, max(page_width - x2, 0), y2, max(page_width - x1, 0)
        elif angle == 180:
            x1, y1, x2, y2 = max(page_width - x2, 0), max(page_height - y2, 0), max(page_width - x1, 0), max(page_height - y1, 0)
        elif angle == 270:
            x1, y1, x2, y2 = max(page_height - y2, 0), x1, max(page_height - y1, 0), x2

        scale_x = float(image_width) / float(page_width or 1)
        scale_y = float(image_height) / float(page_height or 1)
        left = max(0, min(int(x1 * scale_x), image_width - 1))
        top = max(0, min(int(y1 * scale_y), image_height - 1))
        right = max(left + 1, min(int(x2 * scale_x), image_width))
        bottom = max(top + 1, min(int(y2 * scale_y), image_height))
        if right <= left or bottom <= top:
            return None
        pad_x = max(8, int((right - left) * 0.04))
        pad_y = max(8, int((bottom - top) * 0.04))
        return (
            max(0, left - pad_x),
            max(0, top - pad_y),
            min(image_width, right + pad_x),
            min(image_height, bottom + pad_y),
        )
