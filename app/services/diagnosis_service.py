from __future__ import annotations

import json

from app.models import ConfidenceItem, DiagnosticResult, OCRQuestion, OCRResult
from app.services.graph_service import KnowledgeGraphService
from app.services.llm_service import LLMService


class DiagnosisService:
    def __init__(self, graph_service: KnowledgeGraphService, llm_service: LLMService) -> None:
        self.graph_service = graph_service
        self.llm_service = llm_service

    async def diagnose(
        self,
        ocr_result: OCRResult,
        provider: str,
        model: str,
        ollama_base_url: str,
        deepseek_base_url: str,
        deepseek_api_key: str,
        diagnosis_context: dict | None = None,
    ) -> DiagnosticResult:
        diagnosis_context = diagnosis_context or {}
        question = self._primary_question(ocr_result)
        traceback_context = self._build_traceback_context(question)
        prompt = self._build_prompt(ocr_result, diagnosis_context, traceback_context)
        if provider in {"ollama", "deepseek"}:
            try:
                payload = await self.llm_service.generate_json(
                    provider=provider,
                    prompt=prompt,
                    model=model,
                    ollama_base_url=ollama_base_url,
                    deepseek_base_url=deepseek_base_url,
                    deepseek_api_key=deepseek_api_key,
                )
                return self._payload_to_result(
                    payload,
                    ocr_result,
                    traceback_context,
                    provider,
                    prompt,
                    used_fallback=False,
                    diagnosis_context=diagnosis_context,
                )
            except Exception as exc:
                return self._heuristic_result(
                    ocr_result,
                    prompt,
                    provider,
                    llm_error=str(exc),
                    diagnosis_context=diagnosis_context,
                    traceback_context=traceback_context,
                )
        return self._heuristic_result(
            ocr_result,
            prompt,
            provider,
            diagnosis_context=diagnosis_context,
            traceback_context=traceback_context,
        )

    def _build_prompt(self, ocr_result: OCRResult, diagnosis_context: dict, traceback_context: dict) -> str:
        question = self._primary_question(ocr_result)
        repeated_gaps = diagnosis_context.get("top_repeated_gaps", [])
        error_levels = diagnosis_context.get("error_level_distribution", {})
        recent_examples = diagnosis_context.get("recent_examples", [])
        return (
            "【角色】你是一位 K12 学情诊断专家。\n"
            "【目标】根据 OCR 结构化结果与知识图谱前置依赖路径，输出可行动的错因归因结论。\n"
            "【原则】错因归因是推断问题，不要强行单一分类；如果证据不足，请通过置信度分布体现不确定性。\n"
            "【硬约束】你只能在给定的图谱候选节点和候选路径内做回溯判断，不能生成图谱外的新知识点，也不能改写路径节点顺序。\n\n"
            "【输入一：OCR结构化数据】\n"
            "题目: {0}\n"
            "知识点: {1}\n"
            "学生答案: {2}\n"
            "正确答案: {3}\n"
            "错误步骤定位: {4}\n\n"
            "【输入二A：题目主知识点锚点】\n"
            "{5}\n\n"
            "【输入二B：知识图谱候选回溯路径】\n"
            "{6}\n\n"
            "【输入二C：允许选择的图谱节点】\n"
            "{7}\n\n"
            "【输入二D：候选前置漏洞优先级】\n"
            "{8}\n\n"
            "【输入三：跨题聚合信号】\n"
            "该生历史诊断总数: {9}\n"
            "与本题知识点相关的历史记录数: {10}\n"
            "重复出现的前置漏洞: {11}\n"
            "相关历史错误层级分布: {12}\n"
            "最近相关样本: {13}\n\n"
            "【任务】\n"
            "1. 判断错误发生在“理解层”还是“执行层”；\n"
            "2. 仅沿给定知识图谱候选路径回溯，找到最底层、最可行动的前置漏洞；\n"
            "3. 输出带置信度的错因分布，而不是强行只给一个标签；\n"
            "4. 如果同类知识点在多次历史记录中反复出现，优先判断为稳定盲区；若历史证据不足，保留不确定性；\n"
            "5. 给出具体建议：该生需先补【XX前置知识点】；\n"
            "6. 额外给出一个适合 3D 可视化干预的教学提示词。\n\n"
            "【输出限制】\n"
            "1. prerequisite_gap 必须从“允许选择的图谱节点”中选，且优先选择“候选前置漏洞优先级”靠前的节点；\n"
            "2. confidence_distribution 中的 knowledge_point 只能来自“允许选择的图谱节点”；\n"
            "3. graph_paths 必须直接复用“知识图谱候选回溯路径”中的原始路径，不能自行造路径；\n"
            "4. 如果证据不足，请保守地选择当前主知识点或其直接前置节点，不要扩展到无关概念。\n\n"
            "【输出格式】\n"
            "仅输出 JSON，对象字段必须包含：error_level, prerequisite_gap, confidence_distribution, advice, evidence, graph_paths, three_d_prompt。\n"
            "其中 confidence_distribution 是数组，每项包含 knowledge_point, confidence, reason；evidence 必须是证据列表。".format(
                question.question_text,
                question.knowledge_points,
                question.student_answer,
                question.correct_answer,
                question.error_steps,
                traceback_context.get("focus_node", ""),
                traceback_context.get("graph_paths", []),
                traceback_context.get("allowed_nodes", []),
                traceback_context.get("candidate_gaps", []),
                diagnosis_context.get("student_record_count", 0),
                diagnosis_context.get("related_record_count", 0),
                repeated_gaps,
                error_levels,
                recent_examples,
            )
        )

    def _payload_to_result(
        self,
        payload: dict,
        ocr_result: OCRResult,
        traceback_context: dict,
        provider: str,
        prompt: str,
        used_fallback: bool,
        diagnosis_context: dict | None = None,
    ) -> DiagnosticResult:
        diagnosis_context = diagnosis_context or {}
        question = self._primary_question(ocr_result)
        primary_point = traceback_context.get("focus_node") or self._primary_knowledge_point(question)
        authoritative_paths = list(traceback_context.get("graph_paths") or self.graph_service.get_all_paths(primary_point))
        normalized_payload = self._normalize_payload(payload)
        distribution = self._normalize_confidence_distribution(
            normalized_payload.get("confidence_distribution", [])
        )
        prerequisite_gap = self._resolve_prerequisite_gap(
            primary_point=primary_point,
            graph_paths=authoritative_paths,
            suggested_gap=self.graph_service.normalize_node_name(
                self._normalize_scalar_text(normalized_payload.get("prerequisite_gap", ""))
            ),
            distribution=distribution,
        )
        distribution = self._align_distribution_to_traceback(
            distribution=distribution,
            primary_point=primary_point,
            prerequisite_gap=prerequisite_gap,
            graph_paths=authoritative_paths,
        )
        distribution = self._apply_history_signal_to_distribution(distribution, diagnosis_context)
        return DiagnosticResult(
            provider=provider,
            error_level=self._normalize_scalar_text(normalized_payload.get("error_level", "理解层")) or "理解层",
            prerequisite_gap=prerequisite_gap,
            confidence_distribution=distribution,
            advice=self._normalize_scalar_text(normalized_payload.get("advice", "")),
            evidence=self._normalize_text_list(normalized_payload.get("evidence", [])),
            graph_paths=authoritative_paths,
            graph_snapshot=self.graph_service.build_learning_snapshot(
                focus_name=primary_point,
                prerequisite_gap=prerequisite_gap,
            ),
            prompt_used=prompt,
            used_fallback=used_fallback,
            llm_error="",
        )

    def _heuristic_result(
        self,
        ocr_result: OCRResult,
        prompt: str,
        provider: str,
        llm_error: str = "",
        diagnosis_context: dict | None = None,
        traceback_context: dict | None = None,
    ) -> DiagnosticResult:
        diagnosis_context = diagnosis_context or {}
        question = self._primary_question(ocr_result)
        traceback_context = traceback_context or self._build_traceback_context(question)
        primary_point = traceback_context.get("focus_node") or self._primary_knowledge_point(question)
        lowest_gap = traceback_context.get("default_gap") or self.graph_service.get_lowest_prerequisite(primary_point)
        error_text = " ".join(question.error_steps)
        error_level = "理解层" if any(flag in error_text for flag in ("无法", "不懂", "理解")) else "执行层"
        distribution = [
            ConfidenceItem(
                knowledge_point=lowest_gap,
                confidence=0.78,
                reason="根据前置路径回溯到最基础依赖点，且错误步骤反复指向该环节。",
            ),
            ConfidenceItem(
                knowledge_point=primary_point,
                confidence=0.62,
                reason="题目直接考查该知识点，当前答题结果暴露出掌握不稳定。",
            ),
        ]
        distribution = self._apply_history_signal_to_distribution(distribution, diagnosis_context)
        evidence = [
            "错误步骤中出现：{0}".format("；".join(question.error_steps)),
            "学生答案与正确答案在关键推导环节不一致。",
            "知识图谱回溯路径：{0}".format(self.graph_service.get_all_paths(primary_point)),
        ]
        if diagnosis_context.get("related_record_count", 0) >= 2:
            evidence.append(
                "同类知识点历史记录已累计 {0} 次，重复出现前置漏洞信号。".format(
                    diagnosis_context.get("related_record_count", 0)
                )
            )
        return DiagnosticResult(
            provider=provider,
            error_level=error_level,
            prerequisite_gap=lowest_gap,
            confidence_distribution=distribution,
            advice="建议优先补强【{0}】后，再回到【{1}】做 3 道同型题强化。".format(lowest_gap, primary_point),
            evidence=evidence,
            graph_paths=list(traceback_context.get("graph_paths") or self.graph_service.get_all_paths(primary_point)),
            graph_snapshot=self.graph_service.build_learning_snapshot(
                focus_name=primary_point,
                prerequisite_gap=lowest_gap,
            ),
            prompt_used=prompt,
            used_fallback=True,
            llm_error=llm_error,
        )

    def _primary_question(self, ocr_result: OCRResult) -> OCRQuestion:
        if ocr_result.questions:
            return ocr_result.questions[0]
        return OCRQuestion(
            question_text="未提取到有效题目文本",
            knowledge_points=["待补充知识点"],
            student_answer="",
            correct_answer="",
            error_steps=["OCR 未返回可用题目结构"],
            handwriting_boxes=[],
        )

    def _primary_knowledge_point(self, question: OCRQuestion) -> str:
        if question.knowledge_points:
            candidates: list[tuple[int, int, int, str]] = []
            for index, item in enumerate(question.knowledge_points):
                normalized = self.graph_service.normalize_node_name(item)
                if normalized not in self.graph_service.nodes:
                    continue
                depth = self._graph_depth(normalized)
                candidates.append((depth, len(normalized), -index, normalized))
            if candidates:
                candidates.sort(reverse=True)
                return candidates[0][-1]
            return self.graph_service.normalize_node_name(question.knowledge_points[0])
        return "待补充知识点"

    def _normalize_payload(self, payload: object) -> dict:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, str):
            text = payload.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return {}
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, str):
                return self._normalize_payload(parsed)
        return {}

    def _normalize_confidence_distribution(self, value: object) -> list[ConfidenceItem]:
        items: list[object]
        if isinstance(value, list):
            items = value
        elif value:
            items = [value]
        else:
            items = []

        normalized: list[ConfidenceItem] = []
        for item in items:
            if isinstance(item, dict):
                knowledge_point = self._normalize_scalar_text(item.get("knowledge_point"))
                reason = self._normalize_scalar_text(item.get("reason"))
                confidence = self._normalize_confidence(item.get("confidence", 0.0))
            else:
                knowledge_point = self._normalize_scalar_text(item)
                reason = ""
                confidence = 0.0
            if not knowledge_point:
                continue
            normalized.append(
                ConfidenceItem(
                    knowledge_point=self.graph_service.normalize_node_name(knowledge_point),
                    confidence=confidence,
                    reason=reason,
                )
            )
        return normalized

    def _normalize_graph_paths(self, value: object) -> list[list[str]]:
        if not value:
            return []
        candidate = value
        if isinstance(candidate, str):
            text = candidate.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return [[text]]
            candidate = parsed

        if isinstance(candidate, list):
            if candidate and all(isinstance(item, str) for item in candidate):
                path = []
                for item in candidate:
                    text = self._normalize_scalar_text(item)
                    if text:
                        path.append(self.graph_service.normalize_node_name(text))
                return [path] if path else []
            paths: list[list[str]] = []
            for item in candidate:
                if isinstance(item, list):
                    path = []
                    for node in item:
                        text = self._normalize_scalar_text(node)
                        if text:
                            path.append(self.graph_service.normalize_node_name(text))
                    if path:
                        paths.append(path)
                else:
                    node = self._normalize_scalar_text(item)
                    if node:
                        paths.append([self.graph_service.normalize_node_name(node)])
            return paths
        return []

    @staticmethod
    def _focus_node_from_paths(paths: list[list[str]]) -> str:
        if not paths:
            return ""
        first_path = paths[0]
        return first_path[-1] if first_path else ""

    def _apply_history_signal_to_distribution(
        self,
        distribution: list[ConfidenceItem],
        diagnosis_context: dict,
    ) -> list[ConfidenceItem]:
        if not distribution:
            return distribution
        repeated = {
            item.get("knowledge_point"): item.get("count", 0)
            for item in diagnosis_context.get("top_repeated_gaps", [])
            if item.get("knowledge_point")
        }
        boosted: list[ConfidenceItem] = []
        for item in distribution:
            count = repeated.get(item.knowledge_point, 0)
            confidence = item.confidence
            reason = item.reason
            if count >= 2:
                confidence = min(0.98, confidence + min(0.18, 0.05 * count))
                extra = "该前置漏洞在历史相关题中重复出现 {0} 次。".format(count)
                reason = "{0} {1}".format(reason, extra).strip()
            boosted.append(
                ConfidenceItem(
                    knowledge_point=item.knowledge_point,
                    confidence=confidence,
                    reason=reason,
                )
            )
        boosted.sort(key=lambda item: item.confidence, reverse=True)
        return boosted

    def _resolve_prerequisite_gap(
        self,
        *,
        primary_point: str,
        graph_paths: list[list[str]],
        suggested_gap: str,
        distribution: list[ConfidenceItem],
    ) -> str:
        if not primary_point:
            return suggested_gap
        if not graph_paths:
            return primary_point
        path_nodes = self._path_node_set(graph_paths)
        prerequisite_nodes = {name for name in path_nodes if name != primary_point}
        if suggested_gap in prerequisite_nodes:
            return suggested_gap
        for item in distribution:
            if item.knowledge_point in prerequisite_nodes:
                return item.knowledge_point
        if prerequisite_nodes:
            return self.graph_service.get_lowest_prerequisite(primary_point)
        return suggested_gap or primary_point

    def _align_distribution_to_traceback(
        self,
        *,
        distribution: list[ConfidenceItem],
        primary_point: str,
        prerequisite_gap: str,
        graph_paths: list[list[str]],
    ) -> list[ConfidenceItem]:
        if not primary_point:
            return distribution
        allowed_nodes = self._path_node_set(graph_paths)
        if not allowed_nodes:
            return [
                ConfidenceItem(
                    knowledge_point=primary_point,
                    confidence=0.72,
                    reason="当前题目知识点尚未命中现有图谱，暂不采纳图谱外回溯结论。",
                )
            ]

        aligned: list[ConfidenceItem] = []
        seen: set[str] = set()
        for item in distribution:
            if item.knowledge_point not in allowed_nodes or item.knowledge_point in seen:
                continue
            aligned.append(item)
            seen.add(item.knowledge_point)

        default_distribution: list[ConfidenceItem] = []
        if prerequisite_gap and prerequisite_gap != primary_point:
            default_distribution.append(
                ConfidenceItem(
                    knowledge_point=prerequisite_gap,
                    confidence=0.78,
                    reason="该知识点位于当前题目知识点的前置回溯路径上，适合作为优先补强目标。",
                )
            )
        default_distribution.append(
            ConfidenceItem(
                knowledge_point=primary_point,
                confidence=0.62,
                reason="该知识点与当前题目直接对应，是本次回溯的主锚点。",
            )
        )

        for item in default_distribution:
            if item.knowledge_point and item.knowledge_point not in seen:
                aligned.append(item)
                seen.add(item.knowledge_point)

        if not aligned:
            return default_distribution
        return aligned

    @staticmethod
    def _path_node_set(paths: list[list[str]]) -> set[str]:
        names: set[str] = set()
        for path in paths:
            names.update(path)
        return names

    def _graph_depth(self, node_name: str) -> int:
        paths = self.graph_service.get_all_paths(node_name)
        if not paths:
            return 0
        return max(len(path) for path in paths)

    def _build_traceback_context(self, question: OCRQuestion) -> dict:
        focus_node = self._primary_knowledge_point(question)
        graph_paths = self.graph_service.get_all_paths(focus_node)
        allowed_nodes: list[str] = []
        for path in graph_paths:
            for node in path:
                if node not in allowed_nodes:
                    allowed_nodes.append(node)
        direct_prerequisites = self.graph_service.get_prerequisites(focus_node)
        default_gap = self.graph_service.get_lowest_prerequisite(focus_node)
        candidate_gaps: list[dict[str, object]] = []
        if direct_prerequisites:
            for index, node in enumerate(direct_prerequisites, start=1):
                candidate_gaps.append(
                    {
                        "rank": index,
                        "knowledge_point": node,
                        "reason": "当前主知识点的直接前置节点。",
                    }
                )
        elif default_gap and default_gap != focus_node:
            candidate_gaps.append(
                {
                    "rank": 1,
                    "knowledge_point": default_gap,
                    "reason": "当前主知识点回溯到的最底层前置节点。",
                }
            )
        return {
            "focus_node": focus_node,
            "graph_paths": graph_paths,
            "allowed_nodes": allowed_nodes,
            "direct_prerequisites": direct_prerequisites,
            "candidate_gaps": candidate_gaps,
            "default_gap": default_gap,
        }

    def _normalize_text_list(self, value: object) -> list[str]:
        if not value:
            return []
        if isinstance(value, list):
            result: list[str] = []
            for item in value:
                text = self._normalize_scalar_text(item)
                if text:
                    result.append(text)
            return result
        text = self._normalize_scalar_text(value)
        return [text] if text else []

    def _normalize_scalar_text(self, value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return ""
            if text.startswith("[") or text.startswith("{"):
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    return text
                if isinstance(parsed, list):
                    parsed_list = self._normalize_text_list(parsed)
                    return parsed_list[0] if parsed_list else ""
                if isinstance(parsed, dict):
                    return self._normalize_scalar_text(
                        parsed.get("knowledge_point")
                        or parsed.get("name")
                        or parsed.get("text")
                        or parsed.get("reason")
                    )
                return str(parsed)
            return text
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, list):
            items = self._normalize_text_list(value)
            return items[0] if items else ""
        if isinstance(value, dict):
            return self._normalize_scalar_text(
                value.get("knowledge_point") or value.get("name") or value.get("text") or value.get("reason")
            )
        return str(value)

    @staticmethod
    def _normalize_confidence(value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
