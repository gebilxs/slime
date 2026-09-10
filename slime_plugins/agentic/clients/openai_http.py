"""OpenAI-compatible HTTP client for Harbor / API eval and the ECHO judge.

Not a slime engine. Training SGLang stays inside slime. urllib only so login
nodes without aiohttp can unit-test with an injected ``http`` callable.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

HttpFn = Callable[[str, dict[str, Any], dict[str, str]], dict[str, Any]]


def _join_chat_url(base_url: str) -> str:
    url = (base_url or "").rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url + "/v1/chat/completions"


def _join_models_url(base_url: str) -> str:
    url = (base_url or "").rstrip("/")
    if url.endswith("/models"):
        return url
    if url.endswith("/v1"):
        return url + "/models"
    if url.endswith("/chat/completions"):
        return url[: -len("/chat/completions")] + "/models"
    return url + "/v1/models"


def _proxy_handler(proxy: str | None) -> list[urllib.request.BaseHandler]:
    """proxy="" forces a direct connection (bypasses http_proxy env vars)."""
    if proxy is None:
        return []
    if proxy == "":
        return [urllib.request.ProxyHandler({})]
    return [urllib.request.ProxyHandler({"http": proxy, "https": proxy})]


def urllib_http(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    timeout: float = 120.0,
    proxy: str | None = None,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    opener = urllib.request.build_opener(*_proxy_handler(proxy))
    try:
        with opener.open(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"HTTP {exc.code} {url}: {detail}") from exc
    return json.loads(body)


def urllib_get(
    url: str,
    headers: dict[str, str],
    *,
    timeout: float = 8.0,
    proxy: str | None = None,
) -> dict[str, Any]:
    req = urllib.request.Request(url, headers=headers, method="GET")
    opener = urllib.request.build_opener(*_proxy_handler(proxy))
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


@dataclass
class OpenAIHttpClient:
    base_url: str
    api_key: str = ""
    model: str = ""
    timeout: float = 120.0
    proxy: str | None = None
    http: HttpFn | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls, *, prefix: str = "JUDGE") -> OpenAIHttpClient | None:
        """Build from JUDGE_* (legacy ECHO_JUDGE_*) or OPENAI_* env."""

        def _get(suffix: str, default: str = "") -> str:
            return (
                os.environ.get(f"{prefix}_{suffix}")
                or os.environ.get(f"ECHO_{prefix}_{suffix}")
                or os.environ.get(f"ECHO_JUDGE_{suffix}" if prefix == "JUDGE" else "")
                or default
            )

        url = _get("URL") or os.environ.get("OPENAI_BASE_URL") or ""
        model = _get("MODEL") or os.environ.get("OPENAI_MODEL") or ""
        if not url.strip() or not model.strip():
            return None
        return cls(
            base_url=url.strip(),
            api_key=_get("API_KEY") or os.environ.get("OPENAI_API_KEY") or "",
            model=model.strip(),
            timeout=float(_get("TIMEOUT", "120") or "120"),
            proxy=_get("PROXY") or None,
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.http is not None:
            return self.http(url, payload, self._headers())
        return urllib_http(
            url, payload, self._headers(), timeout=self.timeout, proxy=self.proxy
        )

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "stream": False,
        }
        data = self._post(_join_chat_url(self.base_url), payload)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected chat payload: {data!r}") from exc

    def health(self) -> bool:
        """True if GET /v1/models succeeds. Used before Harbor run."""
        url = _join_models_url(self.base_url)
        try:
            if self.http is not None:
                self.http(url, {}, self._headers())
                return True
            urllib_get(url, self._headers(), timeout=min(8.0, self.timeout), proxy=self.proxy)
            return True
        except Exception:  # noqa: BLE001
            return False
