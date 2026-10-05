from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

import httpx


class LLMService:
    async def generate_json(
        self,
        *,
        provider: str,
        prompt: str,
        model: str,
        ollama_base_url: str,
        deepseek_base_url: str,
        deepseek_api_key: str,
    ) -> dict[str, Any]:
        if provider == "ollama":
            return await self._call_ollama(prompt, model, ollama_base_url)
        if provider == "deepseek":
            return await self._call_deepseek(prompt, model, deepseek_base_url, deepseek_api_key)
        raise RuntimeError("不支持的 LLM Provider: {0}".format(provider))

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
    ) -> dict[str, Any]:
        if provider == "ollama":
            return await self._call_ollama_with_image(prompt, image_path, model, ollama_base_url)
        if provider == "deepseek":
            return await self._call_deepseek_with_image(
                prompt, image_path, model, deepseek_base_url, deepseek_api_key
            )
        raise RuntimeError("当前 LLM Provider 不支持图片分析: {0}".format(provider))

    async def _call_ollama(self, prompt: str, model: str, base_url: str) -> dict[str, Any]:
        url = base_url.rstrip("/") + "/api/generate"
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                },
            )
            self._raise_for_status(response)
            payload = response.json()
        content = payload.get("response", "")
        return self._parse_json_content(content)

    async def _call_deepseek(
        self,
        prompt: str,
        model: str,
        base_url: str,
        api_key: str,
    ) -> dict[str, Any]:
        if not api_key:
            raise RuntimeError("DeepSeek API Key 为空，无法调用。")
        url = base_url.rstrip("/") + "/chat/completions"
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                url,
                headers={
                    "Authorization": "Bearer {0}".format(api_key),
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": "你是一位 K12 学情诊断专家，只能返回 JSON。"},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            self._raise_for_status(response)
            payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        return self._parse_json_content(content)

    async def _call_ollama_with_image(
        self,
        prompt: str,
        image_path: Path,
        model: str,
        base_url: str,
    ) -> dict[str, Any]:
        url = base_url.rstrip("/") + "/api/generate"
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "format": "json",
                    "stream": False,
                },
            )
            self._raise_for_status(response)
            payload = response.json()
        content = payload.get("response", "")
        return self._parse_json_content(content)

    async def _call_deepseek_with_image(
        self,
        prompt: str,
        image_path: Path,
        model: str,
        base_url: str,
        api_key: str,
    ) -> dict[str, Any]:
        if not api_key:
            raise RuntimeError("DeepSeek API Key 为空，无法调用。")
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        url = base_url.rstrip("/") + "/chat/completions"
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                url,
                headers={
                    "Authorization": "Bearer {0}".format(api_key),
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是一位 K12 教育图像分析助手，只能返回 JSON。",
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "data:{0};base64,{1}".format(mime_type, image_b64)
                                    },
                                },
                            ],
                        },
                    ],
                },
            )
            self._raise_for_status(response)
            payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        return self._parse_json_content(content)

    def _raise_for_status(self, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text.strip()
            if detail:
                detail = detail[:800]
                raise RuntimeError(
                    "LLM 接口调用失败：HTTP {0} - {1}".format(response.status_code, detail)
                ) from exc
            raise

    def _parse_json_content(self, content: str) -> dict[str, Any]:
        text = (content or "").strip()
        if not text:
            raise RuntimeError("LLM 返回内容为空，无法解析 JSON。")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        cleaned = self._strip_code_fence(text)
        if cleaned != text:
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                text = cleaned

        json_block = self._extract_first_json_object(text)
        if json_block:
            try:
                return json.loads(json_block)
            except json.JSONDecodeError:
                pass

        raise RuntimeError("LLM 返回的内容不是有效 JSON：{0}".format(text[:300]))

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3:
                return "\n".join(lines[1:-1]).strip()
        return stripped

    @staticmethod
    def _extract_first_json_object(text: str) -> str:
        start = text.find("{")
        if start < 0:
            return ""
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == "\"":
                    in_string = False
                continue
            if char == "\"":
                in_string = True
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        return ""
