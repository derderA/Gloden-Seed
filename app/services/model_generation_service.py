from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from app.config import AppConfig
from app.models import DiagnosticResult, InterventionResult, OCRResult, SceneRecipe


class ModelGenerationService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    async def generate(
        self,
        *,
        provider: str,
        ocr_result: OCRResult,
        diagnostic_result: DiagnosticResult,
    ) -> InterventionResult:
        prompt = self._build_prompt(ocr_result, diagnostic_result)
        scene_recipe = self._build_demo_recipe(diagnostic_result.prerequisite_gap)

        if provider == "tripo":
            return await self._generate_with_tripo(
                prompt=prompt,
                scene_recipe=scene_recipe,
            )
        return InterventionResult(
            provider="demo",
            prompt=prompt,
            generation_status="已生成本地 Three.js 干预场景。",
            scene_recipe=scene_recipe,
        )

    async def _generate_with_tripo(self, *, prompt: str, scene_recipe: SceneRecipe) -> InterventionResult:
        if not self.config.tripo_api_key:
            return InterventionResult(
                provider="tripo",
                prompt=prompt,
                generation_status="Tripo API Key 未配置，已回退到本地 Three.js 演示场景。",
                scene_recipe=scene_recipe,
            )

        try:
            task_id = await self._submit_tripo_text_to_model(prompt)
            task = await self._poll_tripo_task(task_id)
            output = task.get("output", {}) if isinstance(task.get("output"), dict) else {}
            model_url = self._first_text(
                output.get("model_url"),
                output.get("model"),
                output.get("base_model"),
                output.get("pbr_model"),
            )
            preview_url = self._first_text(
                output.get("rendered_image_url"),
                output.get("rendered_image"),
                output.get("generated_image_url"),
                output.get("generated_image"),
            )
            status = "Tripo 在线 3D 生成完成。模型下载链接通常会在数分钟后过期，请尽快查看或下载。"
            if not model_url:
                status = "Tripo 任务已完成，但接口未返回可用模型链接，当前先保留本地演示场景。"
            return InterventionResult(
                provider="tripo",
                prompt=prompt,
                generation_status=status,
                model_download_url=model_url or None,
                preview_image_url=preview_url or None,
                scene_recipe=scene_recipe,
            )
        except Exception as exc:
            return InterventionResult(
                provider="tripo",
                prompt=prompt,
                generation_status="Tripo 在线 3D 生成失败，已回退到本地 Three.js 演示场景。原因：{0}".format(exc),
                scene_recipe=scene_recipe,
            )

    async def _submit_tripo_text_to_model(self, prompt: str) -> str:
        url = self.config.tripo_base_url.rstrip("/") + "/generation/text-to-model"
        payload = {
            "prompt": prompt,
            "model": self.config.tripo_model_version,
        }
        response = await self._request("POST", url, json=payload)
        data = self._extract_tripo_data(response)
        task_id = self._first_text(data.get("task_id"))
        if not task_id:
            raise RuntimeError("Tripo 创建任务成功，但未返回 task_id。")
        return task_id

    async def _poll_tripo_task(self, task_id: str) -> dict[str, Any]:
        url = self.config.tripo_base_url.rstrip("/") + "/tasks/{0}".format(task_id)
        deadline = time.monotonic() + max(self.config.tripo_timeout_seconds, 1.0)
        while True:
            response = await self._request("GET", url)
            data = self._extract_tripo_data(response)
            status = self._first_text(data.get("status")).lower()
            if status == "success":
                return data
            if status in {"failed", "cancelled", "banned", "expired", "unknown"}:
                message = self._first_text(
                    data.get("message"),
                    data.get("error"),
                    data.get("error_message"),
                ) or status
                raise RuntimeError("Tripo 任务失败：{0}".format(message))
            if time.monotonic() >= deadline:
                raise RuntimeError("Tripo 任务超时，状态仍为 {0}。".format(status or "unknown"))
            await asyncio.sleep(max(self.config.tripo_poll_interval_seconds, 0.5))

    async def _request(self, method: str, url: str, json: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {
            "Authorization": "Bearer {0}".format(self.config.tripo_api_key),
        }
        if json is not None:
            headers["Content-Type"] = "application/json"
        async with httpx.AsyncClient(timeout=max(self.config.tripo_timeout_seconds, 10.0)) as client:
            response = await client.request(method, url, headers=headers, json=json)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text.strip()[:800]
            raise RuntimeError(
                "Tripo API 调用失败：HTTP {0} - {1}".format(response.status_code, detail or response.reason_phrase)
            ) from exc
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Tripo API 返回格式异常。")
        return payload

    @staticmethod
    def _extract_tripo_data(payload: dict[str, Any]) -> dict[str, Any]:
        code = payload.get("code", 0)
        if code not in (0, "0", None):
            message = payload.get("message") or payload.get("msg") or payload.get("error") or "unknown error"
            raise RuntimeError("Tripo API 返回错误：{0}".format(message))
        data = payload.get("data", payload)
        if not isinstance(data, dict):
            raise RuntimeError("Tripo API 返回 data 字段异常。")
        return data

    @staticmethod
    def _first_text(*values: Any) -> str:
        for value in values:
            if value is None:
                continue
            if isinstance(value, str):
                text = value.strip()
                if text:
                    return text
            elif isinstance(value, (int, float)):
                return str(value)
        return ""

    def _build_prompt(self, ocr_result: OCRResult, diagnostic_result: DiagnosticResult) -> str:
        weak_point = diagnostic_result.prerequisite_gap
        question = ocr_result.questions[0].question_text if ocr_result.questions else "当前上传的作业题目"
        return (
            "为 K12 学生生成一个可交互 3D 教学模型，重点补救“{0}”。"
            "模型需围绕题目“{1}”，支持旋转、拆解、逐步播放和关键步骤讲解。".format(
                weak_point, question
            )
        )

    def _build_demo_recipe(self, prerequisite_gap: str) -> SceneRecipe:
        if "磁" in prerequisite_gap or "空间" in prerequisite_gap:
            return SceneRecipe(
                scene_type="magnet_field",
                title="磁感线方向干预模型",
                description="通过条形磁铁与方向箭头演示，帮助学生建立磁场方向与空间旋转的一致性。",
                steps=[
                    "观察条形磁铁的 N 极和 S 极。",
                    "查看磁体外部磁感线从 N 极指向 S 极。",
                    "旋转模型后再次判断小磁针方向。",
                ],
                controls=["rotate", "zoom", "explode", "top", "side", "section"],
                payload={"arrowCount": 10},
            )
        if "圆" in prerequisite_gap or "体积" in prerequisite_gap:
            return SceneRecipe(
                scene_type="cylinder_slices",
                title="圆柱切片体积模型",
                description="把圆柱拆成很多薄片，让底面积乘高的体积公式变成可视化的动态过程。",
                steps=[
                    "观察完整圆柱结构。",
                    "点击拆解查看切片逐层堆叠。",
                    "切换剖面视角理解底面积和高度的关系。",
                ],
                controls=["rotate", "zoom", "explode", "top", "side", "section"],
                payload={"sliceCount": 12, "radius": 2.4, "height": 4.8},
            )
        return SceneRecipe(
            scene_type="fraction_blocks",
            title="分数通分干预模型",
            description="用不同分母的分数条块重组，帮助理解通分和分数加法。",
            steps=[
                "观察 1/3 和 1/6 的原始分数条。",
                "点击拆解把 1/3 转换为 2/6。",
                "将 2/6 和 1/6 拼成 3/6，理解结果为何为 1/2。",
            ],
            controls=["rotate", "zoom", "explode", "top", "side"],
            payload={"segmentsA": 3, "segmentsB": 6},
        )
