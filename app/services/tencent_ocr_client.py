from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from datetime import datetime
from pathlib import Path

import httpx

from app.config import AppConfig


class TencentOCRClient:
    HOST = "ocr.tencentcloudapi.com"
    SERVICE = "ocr"
    ACTION = "SubmitQuestionMarkAgentJob"
    ALGORITHM = "TC3-HMAC-SHA256"

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    async def submit_question_mark_agent_job(
        self,
        *,
        image_path: Path,
        question_type: str,
    ) -> dict:
        if not self.config.tencent_secret_id or not self.config.tencent_secret_key:
            raise RuntimeError("未检测到腾讯云 OCR 密钥，请在 .envs 中配置 TENCENT_SECRET_ID 和 TENCENT_SECRET_KEY。")

        body = self._build_request_body(image_path=image_path, question_type=question_type)
        payload = await self._post(self.ACTION, body)
        response = payload.get("Response", {})
        if "Error" in response:
            error = response["Error"]
            code = error.get("Code", "UnknownError")
            message = error.get("Message", "腾讯云 OCR 调用失败。")
            raise RuntimeError("腾讯云 OCR 调用失败：{0} - {1}".format(code, message))
        return response

    def _build_request_body(self, *, image_path: Path, question_type: str) -> dict:
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        body = {
            "ImageBase64": encoded,
            "QuestionConfigMap": json.dumps(
                {
                    "KnowledgePoints": True,
                    "TrueAnswer": True,
                    "StepCorrection": True,
                    "DisableAnswerAnalysis": False,
                    "OutputSubQuestionsAndCoords": True,
                    "UseCoordAssist": True,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        }
        if image_path.suffix.lower() == ".pdf":
            body["PdfPageNumber"] = 1
        return body

    async def _post(self, action: str, body: dict) -> dict:
        payload = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
        timestamp = int(time.time())
        headers = self._build_headers(action=action, payload=payload, timestamp=timestamp)
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                "https://{0}/".format(self.HOST),
                content=payload.encode("utf-8"),
                headers=headers,
            )
            response.raise_for_status()
            return response.json()

    def _build_headers(self, *, action: str, payload: str, timestamp: int) -> dict[str, str]:
        signed_headers = "content-type;host;x-tc-action"
        content_type = "application/json; charset=utf-8"
        date = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")
        credential_scope = "{0}/{1}/tc3_request".format(date, self.SERVICE)
        hashed_payload = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        canonical_headers = (
            "content-type:{0}\n"
            "host:{1}\n"
            "x-tc-action:{2}\n"
        ).format(content_type, self.HOST, action.lower())
        canonical_request = "\n".join(
            [
                "POST",
                "/",
                "",
                canonical_headers,
                signed_headers,
                hashed_payload,
            ]
        )
        string_to_sign = "\n".join(
            [
                self.ALGORITHM,
                str(timestamp),
                credential_scope,
                hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
            ]
        )
        signature = self._sign(string_to_sign=string_to_sign, date=date)
        authorization = (
            "{0} Credential={1}/{2}, SignedHeaders={3}, Signature={4}".format(
                self.ALGORITHM,
                self.config.tencent_secret_id,
                credential_scope,
                signed_headers,
                signature,
            )
        )
        return {
            "Authorization": authorization,
            "Content-Type": content_type,
            "Host": self.HOST,
            "X-TC-Action": action,
            "X-TC-Version": self.config.tencent_ocr_version,
            "X-TC-Region": self.config.tencent_ocr_region,
            "X-TC-Timestamp": str(timestamp),
        }

    def _sign(self, *, string_to_sign: str, date: str) -> str:
        secret_date = self._hmac_sha256(date.encode("utf-8"), ("TC3" + self.config.tencent_secret_key).encode("utf-8"))
        secret_service = self._hmac_sha256(self.SERVICE.encode("utf-8"), secret_date)
        secret_signing = self._hmac_sha256(b"tc3_request", secret_service)
        return hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    @staticmethod
    def _hmac_sha256(message: bytes, key: bytes) -> bytes:
        return hmac.new(key, message, hashlib.sha256).digest()
