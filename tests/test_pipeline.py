from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import create_app  # noqa: E402
from app.models import OCRQuestion, OCRResult  # noqa: E402
from app.services.graph_service import KnowledgeGraphService  # noqa: E402
from app.services.llm_service import LLMService  # noqa: E402
from app.services.tencent_ocr_client import TencentOCRClient  # noqa: E402


def build_client(tmp_path: Path) -> TestClient:
    app = create_app()
    history_file = tmp_path / "history.json"
    history_file.write_text("[]", encoding="utf-8")
    app.state.history_service.history_file = history_file
    app.state.pipeline_service.history_service.history_file = history_file
    return TestClient(app)


def test_demo_pipeline_returns_complete_payload(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    response = client.post(
        "/api/pipeline/run",
        data={
            "student_name": "测试学生",
            "subject": "数学",
            "grade": "五年级",
            "question_type": "分数通分应用题",
            "ocr_provider": "demo",
            "llm_provider": "mock",
            "model_provider": "demo",
        },
        files={"image": ("demo.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ocr_result"]["provider"] == "demo"
    assert payload["diagnostic_result"]["prerequisite_gap"]
    assert payload["intervention_result"]["provider"] == "demo"
    assert payload["intervention_result"]["scene_recipe"]["scene_type"] == "fraction_blocks"
    assert payload["history_summary"]["total_records"] == 1
    assert len(payload["process_steps"]) == 5
    assert payload["process_steps"][-1]["status"] == "completed"
    assert payload["diagnostic_result"]["graph_snapshot"]["node_count"] >= 3
    assert payload["diagnostic_result"]["graph_snapshot"]["focus_node"]


def test_graph_addition_api_can_suggest_and_persist_missing_node(tmp_path: Path) -> None:
    app = create_app()
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"name": "数列基础", "subject": "数学"},
                    {"name": "代数式化简", "subject": "数学"},
                ],
                "edges": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    class DummyLLMService:
        async def generate_json(self, **kwargs):
            return {
                "name": "数列的递推规律",
                "description": "从递推关系理解数列各项之间的构造规律。",
                "aliases": ["兔子数列"],
                "prerequisites": ["数列基础"],
                "related_to": ["代数式化简"],
                "is_a": [],
            }

    app.state.graph_service = KnowledgeGraphService(graph_path)
    app.state.llm_service = DummyLLMService()
    client = TestClient(app)

    suggest_response = client.post(
        "/api/graph/suggest",
        json={
            "question_text": "兔子数列递推证明题",
            "subject": "数学",
            "grade": "高中",
            "knowledge_points": ["数列的递推规律"],
            "llm_provider": "deepseek",
            "llm_model": "deepseek-chat",
        },
    )

    assert suggest_response.status_code == 200
    suggest_payload = suggest_response.json()
    assert suggest_payload["missing_knowledge_points"] == ["数列的递推规律"]
    assert suggest_payload["suggestions"][0]["prerequisites"] == ["数列基础"]

    add_response = client.post(
        "/api/graph/add",
        json={
            "suggestions": suggest_payload["suggestions"],
            "focus_knowledge_points": ["数列的递推规律"],
            "prerequisite_gap": "数列基础",
        },
    )

    assert add_response.status_code == 200
    add_payload = add_response.json()
    assert add_payload["added_nodes"] == ["数列的递推规律"]
    assert add_payload["graph_paths"] == [["数列基础", "数列的递推规律"]]
    assert add_payload["graph_snapshot"]["focus_node"] == "数列的递推规律"


def test_llm_graph_paths_are_not_used_when_question_knowledge_point_is_missing_from_graph() -> None:
    class DummyLLMService:
        async def generate_json(self, **kwargs):
            return {
                "error_level": "理解层",
                "prerequisite_gap": "立体几何初步",
                "confidence_distribution": [
                    {"knowledge_point": "空间想象力", "confidence": 0.91, "reason": "模型误判。"}
                ],
                "advice": "先补立体几何",
                "evidence": ["模型给出了错误学科路径"],
                "graph_paths": [["空间想象力", "立体几何初步", "三视图"]],
            }
    service = create_app().state.pipeline_service.diagnosis_service
    service.llm_service = DummyLLMService()
    ocr_result = OCRResult(
        provider="ai",
        image_name="sequence.png",
        subject="数学",
        grade="高中",
        questions=[
            OCRQuestion(
                question_text="兔子数列递推证明题",
                knowledge_points=["数列的递推规律", "代数式化简求值"],
                student_answer="m^2-1",
                correct_answer="m^2-1",
                error_steps=["需要根据递推关系做恒等变形"],
            )
        ],
        raw_summary="测试数列题",
    )

    import asyncio

    result = asyncio.run(
        service.diagnose(
            ocr_result=ocr_result,
            provider="deepseek",
            model="deepseek-chat",
            ollama_base_url="http://127.0.0.1:11434",
            deepseek_base_url="https://api.deepseek.com",
            deepseek_api_key="test-key",
            diagnosis_context={},
        )
    )

    assert result.graph_paths == []
    assert result.graph_snapshot.node_count == 0
    assert result.prerequisite_gap == "数列的递推规律"
    assert result.confidence_distribution[0].knowledge_point == "数列的递推规律"


def test_history_summary_accumulates_blind_spots(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    for _ in range(2):
        client.post(
            "/api/pipeline/run",
            data={
                "student_name": "张三",
                "question_type": "圆柱体积计算题",
                "ocr_provider": "demo",
                "llm_provider": "mock",
                "model_provider": "demo",
            },
            files={"image": ("demo.png", b"fake-image", "image/png")},
        )

    response = client.get("/api/history/summary", params={"student_name": "张三"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["total_records"] == 2
    assert payload["top_blind_spots"][0]["count"] == 2


def test_history_context_participates_in_fallback_diagnosis(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    history = client.app.state.pipeline_service.history_service
    history.append(
        {
            "analysis_id": "a1",
            "student_name": "历史学生",
            "subject": "数学",
            "grade": "五年级",
            "knowledge_points": ["分数加法"],
            "prerequisite_gap": "分数意义",
            "error_level": "理解层",
        }
    )
    history.append(
        {
            "analysis_id": "a2",
            "student_name": "历史学生",
            "subject": "数学",
            "grade": "五年级",
            "knowledge_points": ["分数加法"],
            "prerequisite_gap": "分数意义",
            "error_level": "理解层",
        }
    )

    response = client.post(
        "/api/pipeline/run",
        data={
            "student_name": "历史学生",
            "subject": "数学",
            "grade": "五年级",
            "question_type": "分数通分应用题",
            "ocr_provider": "demo",
            "llm_provider": "mock",
            "llm_model": "mock",
            "model_provider": "demo",
        },
        files={"image": ("demo.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["diagnostic_result"]["used_fallback"] is True
    assert payload["diagnostic_result"]["confidence_distribution"][0]["knowledge_point"] == "分数意义"
    assert payload["diagnostic_result"]["confidence_distribution"][0]["confidence"] > 0.78
    assert "重复出现 2 次" in payload["diagnostic_result"]["confidence_distribution"][0]["reason"]


def test_diagnosis_prompt_contains_cross_question_context(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    history = client.app.state.pipeline_service.history_service
    history.append(
        {
            "analysis_id": "b1",
            "student_name": "聚合学生",
            "subject": "数学",
            "grade": "五年级",
            "knowledge_points": ["分数加法"],
            "prerequisite_gap": "分数意义",
            "error_level": "理解层",
        }
    )

    captured = {}

    class CapturingLLMService:
        async def generate_json(
            self,
            *,
            provider: str,
            prompt: str,
            model: str,
            ollama_base_url: str,
            deepseek_base_url: str,
            deepseek_api_key: str,
        ) -> dict:
            captured["prompt"] = prompt
            return {
                "error_level": "理解层",
                "prerequisite_gap": "分数意义",
                "confidence_distribution": [{"knowledge_point": "分数意义", "confidence": 0.88, "reason": "历史与单题证据一致"}],
                "advice": "先补分数意义",
                "evidence": ["多次出现同类问题"],
                "graph_paths": [["分数意义", "分数通分", "分数加法"]],
            }

    client.app.state.pipeline_service.diagnosis_service.llm_service = CapturingLLMService()
    response = client.post(
        "/api/pipeline/run",
        data={
            "student_name": "聚合学生",
            "subject": "数学",
            "grade": "五年级",
            "question_type": "分数通分应用题",
            "ocr_provider": "demo",
            "llm_provider": "deepseek",
            "llm_model": "deepseek-chat",
            "model_provider": "demo",
        },
        files={"image": ("demo.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 200
    assert "【输入三：跨题聚合信号】" in captured["prompt"]
    assert "【输入二A：题目主知识点锚点】" in captured["prompt"]
    assert "【输入二B：知识图谱候选回溯路径】" in captured["prompt"]
    assert "【输入二C：允许选择的图谱节点】" in captured["prompt"]
    assert "只能在给定的图谱候选节点和候选路径内做回溯判断" in captured["prompt"]
    assert "与本题知识点相关的历史记录数: 1" in captured["prompt"]
    assert "重复出现的前置漏洞" in captured["prompt"]


def test_diagnosis_normalizes_graph_node_names_from_llm_payload(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    class AliasLLMService:
        async def generate_json(
            self,
            *,
            provider: str,
            prompt: str,
            model: str,
            ollama_base_url: str,
            deepseek_base_url: str,
            deepseek_api_key: str,
        ) -> dict:
            return {
                "error_level": "理解层",
                "prerequisite_gap": "分数意义问题",
                "confidence_distribution": [
                    {"knowledge_point": "分数意义相关题", "confidence": 0.91, "reason": "概念理解不稳"},
                    {"knowledge_point": "分数加法应用题", "confidence": 0.72, "reason": "当前题面直接考查"},
                ],
                "advice": "建议先补分数意义",
                "evidence": ["学生在通分前置概念上反复失误"],
                "graph_paths": [["分数意义相关题", "分数通分应用题", "分数加法应用题"]],
            }

    client.app.state.pipeline_service.diagnosis_service.llm_service = AliasLLMService()
    response = client.post(
        "/api/pipeline/run",
        data={
            "student_name": "归一化学生",
            "subject": "数学",
            "grade": "五年级",
            "question_type": "分数通分应用题",
            "ocr_provider": "demo",
            "llm_provider": "deepseek",
            "llm_model": "deepseek-chat",
            "model_provider": "demo",
        },
        files={"image": ("demo.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()["diagnostic_result"]
    assert payload["prerequisite_gap"] == "分数意义"
    assert payload["confidence_distribution"][0]["knowledge_point"] == "分数意义"
    assert payload["confidence_distribution"][1]["knowledge_point"] == "分数加法"
    assert payload["graph_paths"] == [
        ["分数意义", "分数通分", "分数加法"],
        ["最小公倍数", "分数通分", "分数加法"],
    ]
    assert payload["graph_snapshot"]["focus_node"] == "分数加法"


def test_traceback_recomputes_paths_from_question_knowledge_point(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    class DriftedLLMService:
        async def generate_json(
            self,
            *,
            provider: str,
            prompt: str,
            model: str,
            ollama_base_url: str,
            deepseek_base_url: str,
            deepseek_api_key: str,
        ) -> dict:
            return {
                "error_level": "理解层",
                "prerequisite_gap": "磁感线",
                "confidence_distribution": [
                    {"knowledge_point": "磁场方向", "confidence": 0.93, "reason": "错误漂移示例"},
                ],
                "advice": "先补磁感线",
                "evidence": ["这是故意构造的错误回溯结果"],
                "graph_paths": [["磁感线", "磁场方向", "电磁场方向判断"]],
            }

    client.app.state.pipeline_service.diagnosis_service.llm_service = DriftedLLMService()
    response = client.post(
        "/api/pipeline/run",
        data={
            "student_name": "回溯学生",
            "subject": "数学",
            "grade": "五年级",
            "question_type": "分数通分应用题",
            "ocr_provider": "demo",
            "llm_provider": "deepseek",
            "llm_model": "deepseek-chat",
            "model_provider": "demo",
        },
        files={"image": ("demo.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()["diagnostic_result"]
    assert payload["graph_paths"] == [
        ["分数意义", "分数通分", "分数加法"],
        ["最小公倍数", "分数通分", "分数加法"],
    ]
    assert payload["graph_snapshot"]["focus_node"] == "分数加法"
    assert payload["prerequisite_gap"] == "分数意义"
    assert payload["confidence_distribution"][0]["knowledge_point"] == "分数意义"
    assert all(item["knowledge_point"] in {"分数意义", "分数加法"} for item in payload["confidence_distribution"])


def test_traceback_prefers_more_specific_question_focus_node(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    async def fake_extract(*args, **kwargs):
        from app.models import OCRQuestion, OCRResult

        return OCRResult(
            provider="demo",
            image_name="demo.png",
            subject="物理",
            grade="初中",
            questions=[
                OCRQuestion(
                    question_text="判断磁感线方向并标出小磁针偏转方向。",
                    knowledge_points=["磁场方向", "电磁场方向判断"],
                    student_answer="略",
                    correct_answer="略",
                    error_steps=["方向判断错误"],
                    handwriting_boxes=[],
                )
            ],
            raw_summary="测试更具体的知识点焦点选择。",
        )

    client.app.state.pipeline_service.ocr_service.extract = fake_extract
    response = client.post(
        "/api/pipeline/run",
        data={
            "student_name": "焦点学生",
            "subject": "物理",
            "grade": "初中",
            "question_type": "磁感线方向判断",
            "ocr_provider": "demo",
            "llm_provider": "mock",
            "model_provider": "demo",
        },
        files={"image": ("demo.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()["diagnostic_result"]
    assert payload["graph_snapshot"]["focus_node"] == "电磁场方向判断"
    assert payload["graph_paths"] == [["磁感线", "磁场方向", "电磁场方向判断"], ["空间想象力", "电磁场方向判断"]]


def test_demo_ocr_infers_case_by_question_type(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    response = client.post(
        "/api/pipeline/run",
        data={
            "student_name": "李四",
            "question_type": "磁场方向判断题",
            "ocr_provider": "demo",
            "llm_provider": "mock",
            "model_provider": "demo",
        },
        files={"image": ("demo.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ocr_result"]["provider"] == "demo"
    assert payload["ocr_result"]["subject"] == "物理"
    assert "磁感线" in payload["ocr_result"]["questions"][0]["question_text"]


def test_tripo_model_provider_returns_online_result(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    service = client.app.state.pipeline_service.model_generation_service
    service.config.tripo_api_key = "test-key"

    async def fake_submit(prompt: str) -> str:
        assert "K12 学生" in prompt
        return "task-123"

    async def fake_poll(task_id: str) -> dict:
        assert task_id == "task-123"
        return {
            "status": "success",
            "output": {
                "model_url": "https://example.com/model.glb",
                "rendered_image_url": "https://example.com/preview.png",
            },
        }

    service._submit_tripo_text_to_model = fake_submit
    service._poll_tripo_task = fake_poll

    response = client.post(
        "/api/pipeline/run",
        data={
            "student_name": "Tripo 学生",
            "subject": "数学",
            "grade": "五年级",
            "question_type": "分数通分应用题",
            "ocr_provider": "demo",
            "llm_provider": "mock",
            "model_provider": "tripo",
        },
        files={"image": ("demo.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["intervention_result"]["provider"] == "tripo"
    assert payload["intervention_result"]["model_download_url"] == "https://example.com/model.glb"
    assert payload["intervention_result"]["preview_image_url"] == "https://example.com/preview.png"
    assert "Tripo 在线 3D 生成完成" in payload["intervention_result"]["generation_status"]


def test_tripo_model_provider_reports_missing_api_key(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    service = client.app.state.pipeline_service.model_generation_service
    service.config.tripo_api_key = ""

    response = client.post(
        "/api/pipeline/run",
        data={
            "student_name": "Tripo 学生",
            "subject": "数学",
            "grade": "五年级",
            "question_type": "分数通分应用题",
            "ocr_provider": "demo",
            "llm_provider": "mock",
            "model_provider": "tripo",
        },
        files={"image": ("demo.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["intervention_result"]["provider"] == "tripo"
    assert payload["intervention_result"]["model_download_url"] is None
    assert "Tripo API Key 未配置" in payload["intervention_result"]["generation_status"]


def test_stream_pipeline_returns_incremental_events(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    with client.stream(
        "POST",
        "/api/pipeline/stream",
        data={
            "student_name": "流式学生",
            "question_type": "分数加法题",
            "ocr_provider": "demo",
            "llm_provider": "mock",
            "model_provider": "demo",
        },
        files={"image": ("demo.png", b"fake-image", "image/png")},
    ) as response:
        assert response.status_code == 200
        lines = [line for line in response.iter_lines() if line]

    assert len(lines) >= 4
    assert '"type": "process_update"' in lines[0]
    assert any('"type": "ocr_result"' in line for line in lines)
    assert any('"type": "diagnosis_result"' in line for line in lines)
    assert any('"type": "scene_result"' in line for line in lines)
    assert any('"type": "complete"' in line for line in lines)


def test_tencent_ocr_pipeline_maps_agent_response(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    class FakeTencentOCRClient:
        async def submit_question_mark_agent_job(self, *, image_path: Path, question_type: str) -> dict:
            assert image_path.exists()
            assert question_type == "分数通分应用题"
            return {
                "JobId": "job-123",
                "QuestionCount": "1",
                "QuestionInfo": [
                    {
                        "ResultList": [
                            {
                                "Question": ["计算 1/3 + 1/6，并写出通分过程。"],
                                "Answer": ["1/3 + 1/6 = 2/9"],
                                "TrueAnswer": ["1/2"],
                                "KnowledgePoints": ["分数通分", "分数加法"],
                                "StepCorrection": ["第2步通分错误", "把分母直接相加了"],
                                "Coord": [
                                    {
                                        "LeftTop": {"X": 64, "Y": 96},
                                        "RightTop": {"X": 280, "Y": 96},
                                        "RightBottom": {"X": 280, "Y": 164},
                                        "LeftBottom": {"X": 64, "Y": 164},
                                    }
                                ],
                            }
                        ]
                    }
                ],
            }

    client.app.state.pipeline_service.ocr_service.tencent_client = FakeTencentOCRClient()
    response = client.post(
        "/api/pipeline/run",
        data={
            "student_name": "腾讯学生",
            "subject": "数学",
            "grade": "五年级",
            "question_type": "分数通分应用题",
            "ocr_provider": "tencent",
            "llm_provider": "mock",
            "model_provider": "demo",
        },
        files={"image": ("demo.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ocr_result"]["provider"] == "tencent"
    assert payload["ocr_result"]["questions"][0]["question_text"] == "计算 1/3 + 1/6，并写出通分过程。"
    assert payload["ocr_result"]["questions"][0]["correct_answer"] == "1/2"
    assert payload["ocr_result"]["questions"][0]["handwriting_boxes"] == [[64, 96, 280, 164]]


def test_tencent_ocr_pipeline_tolerates_empty_question_lists(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    class FakeTencentOCRClient:
        async def submit_question_mark_agent_job(self, *, image_path: Path, question_type: str) -> dict:
            return {
                "JobId": "job-empty",
                "QuestionCount": "0",
                "QuestionInfo": [{"ResultList": []}],
            }

    class FailingLLMService:
        async def generate_json_with_image(
            self,
            *,
            provider: str,
            prompt: str,
            image_path: Path,
            model: str,
            ollama_base_url: str,
            deepseek_base_url: str,
            deepseek_api_key: str,
        ) -> dict:
            raise RuntimeError("模拟二次补全失败")

    client.app.state.pipeline_service.ocr_service.tencent_client = FakeTencentOCRClient()
    client.app.state.pipeline_service.ocr_service.llm_service = FailingLLMService()
    response = client.post(
        "/api/pipeline/run",
        data={
            "student_name": "空结果学生",
            "subject": "数学",
            "grade": "五年级",
            "question_type": "分数通分应用题",
            "ocr_provider": "tencent",
            "llm_provider": "mock",
            "model_provider": "demo",
        },
        files={"image": ("demo.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ocr_result"]["provider"] == "tencent"
    assert payload["ocr_result"]["questions"][0]["question_text"] == "腾讯 OCR 未返回可解析题目结构"
    assert payload["diagnostic_result"]["prerequisite_gap"]


def test_tencent_ocr_pipeline_uses_llm_enrichment_when_text_missing(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    class FakeTencentOCRClient:
        async def submit_question_mark_agent_job(self, *, image_path: Path, question_type: str) -> dict:
            return {
                "JobId": "job-enrich",
                "QuestionCount": "1",
                "QuestionInfo": [
                    {
                        "ResultList": [
                            {
                                "Question": [],
                                "Answer": [],
                                "TrueAnswer": [],
                                "KnowledgePoints": [],
                                "StepCorrection": [],
                                "Coord": [
                                    {
                                        "LeftTop": {"X": 10, "Y": 20},
                                        "RightTop": {"X": 200, "Y": 20},
                                        "RightBottom": {"X": 200, "Y": 80},
                                        "LeftBottom": {"X": 10, "Y": 80},
                                    }
                                ],
                            }
                        ]
                    }
                ],
            }

    class FakeLLMService:
        async def generate_json_with_image(
            self,
            *,
            provider: str,
            prompt: str,
            image_path: Path,
            model: str,
            ollama_base_url: str,
            deepseek_base_url: str,
            deepseek_api_key: str,
        ) -> dict:
            assert provider == "deepseek"
            assert image_path.exists()
            return {
                "subject": "数学",
                "grade": "五年级",
                "raw_summary": "已根据图片补全结构化结果。",
                "questions": [
                    {
                        "question_text": "12+8=",
                        "knowledge_points": ["20以内口算"],
                        "student_answer": "19",
                        "correct_answer": "20",
                        "error_steps": ["加法结果计算错误"],
                        "handwriting_boxes": [[10, 20, 200, 80]],
                    }
                ],
            }

    client.app.state.pipeline_service.ocr_service.tencent_client = FakeTencentOCRClient()
    client.app.state.pipeline_service.ocr_service.llm_service = FakeLLMService()
    response = client.post(
        "/api/pipeline/run",
        data={
            "student_name": "补全学生",
            "subject": "数学",
            "grade": "五年级",
            "question_type": "口算",
            "ocr_provider": "tencent",
            "llm_provider": "deepseek",
            "llm_model": "deepseek-chat",
            "model_provider": "demo",
        },
        files={"image": ("demo.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ocr_result"]["provider"] == "tencent"
    assert payload["ocr_result"]["questions"][0]["question_text"] == "12+8="
    assert payload["ocr_result"]["questions"][0]["knowledge_points"] == ["20以内口算"]
    assert payload["ocr_result"]["raw_summary"] == "腾讯 OCR 结构化结果缺失，已结合原图和腾讯返回数据进行二次补全。"


def test_diagnosis_fallback_exposes_llm_error_reason(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    class FailingLLMService:
        async def generate_json(
            self,
            *,
            provider: str,
            prompt: str,
            model: str,
            ollama_base_url: str,
            deepseek_base_url: str,
            deepseek_api_key: str,
        ) -> dict:
            raise RuntimeError("DeepSeek API Key 为空，无法调用。")

    client.app.state.pipeline_service.diagnosis_service.llm_service = FailingLLMService()
    response = client.post(
        "/api/pipeline/run",
        data={
            "student_name": "诊断学生",
            "subject": "数学",
            "grade": "五年级",
            "question_type": "分数通分应用题",
            "ocr_provider": "demo",
            "llm_provider": "deepseek",
            "llm_model": "deepseek-chat",
            "model_provider": "demo",
        },
        files={"image": ("demo.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["diagnostic_result"]["used_fallback"] is True
    assert payload["diagnostic_result"]["llm_error"] == "DeepSeek API Key 为空，无法调用。"
    assert "原因：DeepSeek API Key 为空，无法调用。" in payload["warnings"][0]


def test_diagnosis_accepts_stringified_llm_payload_shapes(tmp_path: Path) -> None:
    client = build_client(tmp_path)

    class StringyLLMService:
        async def generate_json(
            self,
            *,
            provider: str,
            prompt: str,
            model: str,
            ollama_base_url: str,
            deepseek_base_url: str,
            deepseek_api_key: str,
        ) -> dict:
            return {
                "error_level": "理解层",
                "prerequisite_gap": ["分数意义"],
                "confidence_distribution": [
                    "分数意义",
                    {"knowledge_point": "分数通分", "confidence": "0.66", "reason": "直接考查该知识点"},
                ],
                "advice": ["先复习分数意义，再做通分训练"],
                "evidence": "学生在通分前没有统一分母",
                "graph_paths": "[[\"分数意义\", \"分数通分\", \"分数加法\"]]",
            }

    client.app.state.pipeline_service.diagnosis_service.llm_service = StringyLLMService()
    response = client.post(
        "/api/pipeline/run",
        data={
            "student_name": "字符串诊断学生",
            "subject": "数学",
            "grade": "五年级",
            "question_type": "分数通分应用题",
            "ocr_provider": "demo",
            "llm_provider": "deepseek",
            "llm_model": "deepseek-chat",
            "model_provider": "demo",
        },
        files={"image": ("demo.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["diagnostic_result"]["used_fallback"] is False
    assert payload["diagnostic_result"]["prerequisite_gap"] == "分数意义"
    assert payload["diagnostic_result"]["advice"] == "先复习分数意义，再做通分训练"
    assert payload["diagnostic_result"]["evidence"] == ["学生在通分前没有统一分母"]
    assert payload["diagnostic_result"]["graph_paths"] == [["分数意义", "分数通分", "分数加法"]]
    assert payload["diagnostic_result"]["confidence_distribution"][0]["knowledge_point"] == "分数意义"


def test_tencent_ocr_client_uses_version_from_config() -> None:
    config = SimpleNamespace(
        tencent_secret_id="id",
        tencent_secret_key="secret",
        tencent_ocr_region="ap-shanghai",
        tencent_ocr_version="2024-07-18",
    )
    client = TencentOCRClient(config)

    headers = client._build_headers(
        action="SubmitQuestionMarkAgentJob",
        payload="{}",
        timestamp=1735689600,
    )

    assert headers["X-TC-Version"] == "2024-07-18"


def test_llm_service_parses_code_fenced_json() -> None:
    service = LLMService()

    payload = service._parse_json_content(
        """```json
{"error_level":"理解层","prerequisite_gap":"分数意义"}
```"""
    )

    assert payload["error_level"] == "理解层"
    assert payload["prerequisite_gap"] == "分数意义"


def test_tencent_ocr_client_builds_recommended_question_config(tmp_path: Path) -> None:
    config = SimpleNamespace(
        tencent_secret_id="id",
        tencent_secret_key="secret",
        tencent_ocr_region="ap-shanghai",
        tencent_ocr_version="2018-11-19",
    )
    client = TencentOCRClient(config)
    image_path = tmp_path / "paper.png"
    image_path.write_bytes(b"fake-image")
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    image_body = client._build_request_body(image_path=image_path, question_type="口算")
    pdf_body = client._build_request_body(image_path=pdf_path, question_type="口算")
    question_config = json.loads(image_body["QuestionConfigMap"])

    assert question_config == {
        "KnowledgePoints": True,
        "TrueAnswer": True,
        "StepCorrection": True,
        "DisableAnswerAnalysis": False,
        "OutputSubQuestionsAndCoords": True,
        "UseCoordAssist": True,
    }
    assert "PdfPageNumber" not in image_body
    assert pdf_body["PdfPageNumber"] == 1


def test_llm_payload_to_result_normalizes_string_fields(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    service = client.app.state.pipeline_service.ocr_service

    result = service._payload_to_result(
        payload={
            "subject": "数学",
            "grade": "五年级",
            "questions": [
                {
                    "question_text": ["12+8="],
                    "knowledge_points": "20以内口算",
                    "student_answer": 19,
                    "correct_answer": "20",
                    "error_steps": "加法结果计算错误",
                    "handwriting_boxes": [["10", "20", "200", "80"]],
                }
            ],
        },
        image_path=tmp_path / "demo.png",
        subject="数学",
        grade="五年级",
    )

    question = result.questions[0]
    assert question.question_text == "12+8="
    assert question.knowledge_points == ["20以内口算"]
    assert question.student_answer == "19"
    assert question.error_steps == ["加法结果计算错误"]
    assert question.handwriting_boxes == [[10, 20, 200, 80]]


def test_tencent_crop_enrichment_recovers_question_text(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    image_path = tmp_path / "source.png"
    Image.new("RGB", (120, 80), color=(255, 255, 255)).save(image_path)

    class FakeTencentOCRClient:
        async def submit_question_mark_agent_job(self, *, image_path: Path, question_type: str) -> dict:
            return {
                "JobId": "job-crop",
                "QuestionCount": "1",
                "QuestionInfo": [
                    {
                        "Angle": 0,
                        "Width": 120,
                        "Height": 80,
                        "ResultList": [
                            {
                                "Question": [],
                                "Answer": [],
                                "TrueAnswer": [],
                                "KnowledgePoints": [],
                                "StepCorrection": [],
                                "Coord": [
                                    {
                                        "LeftTop": {"X": 10, "Y": 10},
                                        "RightTop": {"X": 100, "Y": 10},
                                        "RightBottom": {"X": 100, "Y": 60},
                                        "LeftBottom": {"X": 10, "Y": 60},
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }

    class FakeLLMService:
        async def generate_json_with_image(
            self,
            *,
            provider: str,
            prompt: str,
            image_path: Path,
            model: str,
            ollama_base_url: str,
            deepseek_base_url: str,
            deepseek_api_key: str,
        ) -> dict:
            assert image_path.exists()
            assert "单题裁剪图" in prompt
            with Image.open(image_path) as cropped:
                assert cropped.size[0] > 0
                assert cropped.size[1] > 0
            return {
                "question_text": "12+8=",
                "knowledge_points": "20以内口算",
                "student_answer": "19",
                "correct_answer": "20",
                "error_steps": "加法结果计算错误",
                "raw_summary": "已通过题块裁图补全。",
            }

    client.app.state.pipeline_service.ocr_service.tencent_client = FakeTencentOCRClient()
    client.app.state.pipeline_service.ocr_service.llm_service = FakeLLMService()
    response = client.post(
        "/api/pipeline/run",
        data={
            "student_name": "裁图学生",
            "subject": "数学",
            "grade": "五年级",
            "question_type": "口算",
            "ocr_provider": "tencent",
            "llm_provider": "deepseek",
            "llm_model": "deepseek-chat",
            "model_provider": "demo",
        },
        files={"image": ("source.png", image_path.read_bytes(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ocr_result"]["questions"][0]["question_text"] == "12+8="
    assert payload["ocr_result"]["questions"][0]["error_steps"] == ["加法结果计算错误"]
