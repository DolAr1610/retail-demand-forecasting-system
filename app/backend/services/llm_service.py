from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from app.backend.settings import Settings


class LLMService:
    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def is_enabled(self) -> bool:
        return bool(self.settings.openrouter_api_key)

    async def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str | Dict[str, Any] = "auto",
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> Dict[str, Any]:
        if not self.is_enabled:
            raise RuntimeError("OpenRouter API key is not configured.")

        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }

        if self.settings.openrouter_site_url:
            headers["HTTP-Referer"] = self.settings.openrouter_site_url
        if self.settings.openrouter_app_name:
            headers["X-Title"] = self.settings.openrouter_app_name

        payload: Dict[str, Any] = {
            "model": self.settings.openrouter_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
            payload["parallel_tool_calls"] = False

        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                f"{self.settings.openrouter_base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            return response.json()