#!/usr/bin/env python3
"""Minimal secret-safe OpenAI HTTP client for the Phase 2 Responses/Batch APIs."""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Mapping


API_BASE = "https://api.openai.com/v1"
NON_RETRYABLE_BILLING_CODES = {
    "billing_hard_limit_reached",
    "insufficient_quota",
    "billing_not_active",
}


class OpenAIRequestError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, code: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code

    @property
    def retryable(self) -> bool:
        return self.status in {408, 409, 429, 500, 502, 503, 504} and self.code not in NON_RETRYABLE_BILLING_CODES


def load_api_key(env_path: Path) -> str:
    if not env_path.exists():
        raise OpenAIRequestError(f"Missing environment file: {env_path}")
    found: str | None = None
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == "OPENAI_API_KEY":
            found = value.strip().strip('"').strip("'")
            break
    if not found:
        raise OpenAIRequestError("OPENAI_API_KEY is missing or blank")
    return found


def sanitize(text: str, api_key: str) -> str:
    value = text.replace(api_key, "[REDACTED]") if api_key else text
    return value.replace("Bearer [REDACTED]", "[REDACTED_AUTH]")


class OpenAIHTTPClient:
    def __init__(self, api_key: str, *, timeout_seconds: int = 180) -> None:
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds

    def _request_bytes(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        content_type: str | None = "application/json",
        retries: int = 4,
    ) -> tuple[bytes, Mapping[str, str]]:
        url = API_BASE + path
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if content_type is not None:
            headers["Content-Type"] = content_type
        for attempt in range(retries + 1):
            request = urllib.request.Request(url, data=body, headers=headers, method=method)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return response.read(), dict(response.headers.items())
            except urllib.error.HTTPError as exc:
                raw = exc.read().decode("utf-8", errors="replace")
                code: str | None = None
                message = raw
                try:
                    payload = json.loads(raw)
                    error = payload.get("error", {}) if isinstance(payload, dict) else {}
                    if isinstance(error, dict):
                        code = str(error.get("code") or error.get("type") or "") or None
                        message = str(error.get("message") or raw)
                except json.JSONDecodeError:
                    pass
                failure = OpenAIRequestError(
                    sanitize(message, self._api_key), status=exc.code, code=code
                )
                if not failure.retryable or attempt >= retries:
                    raise failure from None
            except (urllib.error.URLError, TimeoutError) as exc:
                failure = OpenAIRequestError(
                    sanitize(str(exc), self._api_key), status=None, code="network_error"
                )
                if attempt >= retries:
                    raise failure from None
            delay = min(30.0, 2.0**attempt) + random.Random(attempt).random()
            time.sleep(delay)
        raise AssertionError("unreachable")

    def request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, object] | None = None,
        *,
        retries: int = 4,
    ) -> dict[str, object]:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        raw, _headers = self._request_bytes(method, path, body=body, retries=retries)
        parsed = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise OpenAIRequestError("OpenAI API returned a non-object JSON response")
        return parsed

    def create_response(self, body: Mapping[str, object]) -> dict[str, object]:
        return self.request_json("POST", "/responses", body)

    def upload_batch_file(self, path: Path) -> dict[str, object]:
        boundary = "----phase2-" + uuid.uuid4().hex
        file_bytes = path.read_bytes()
        chunks = [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n".encode(),
            (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                f"filename=\"{path.name}\"\r\nContent-Type: application/jsonl\r\n\r\n"
            ).encode(),
            file_bytes,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
        raw, _headers = self._request_bytes(
            "POST",
            "/files",
            body=b"".join(chunks),
            content_type=f"multipart/form-data; boundary={boundary}",
        )
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise OpenAIRequestError("File upload returned non-object JSON")
        return payload

    def create_batch(self, input_file_id: str, metadata: Mapping[str, str]) -> dict[str, object]:
        return self.request_json(
            "POST",
            "/batches",
            {
                "input_file_id": input_file_id,
                "endpoint": "/v1/responses",
                "completion_window": "24h",
                "metadata": dict(metadata),
            },
        )

    def retrieve_batch(self, batch_id: str) -> dict[str, object]:
        return self.request_json("GET", f"/batches/{batch_id}", retries=2)

    def download_file(self, file_id: str) -> bytes:
        raw, _headers = self._request_bytes("GET", f"/files/{file_id}/content", content_type=None)
        return raw
