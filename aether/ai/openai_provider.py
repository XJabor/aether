"""OpenAI implementation of :class:`LLMProvider`.

Kept in its own module so the OpenAI and Anthropic SDKs are never mixed in
one file. Points at a configurable base URL, so it also drives any
OpenAI-compatible endpoint (LM Studio, Ollama, OpenRouter).
"""
from __future__ import annotations

from . import keys
from .provider import DeltaCallback, LLMProvider, Message, ProviderError


class OpenAIProvider(LLMProvider):
    key_name = "openai"
    display_name = "OpenAI"
    models = ["gpt-4o", "gpt-4o-mini", "o3-mini"]
    default_model = "gpt-4o"

    def __init__(self, model: str = "", base_url: str = "") -> None:
        super().__init__(model)
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        api_key = keys.get_key(self.key_name)
        if not api_key:
            raise ProviderError(
                "No OpenAI API key saved. Add one under Settings > AI."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderError(
                "The 'openai' package is not installed. "
                "Install it with: pip install openai"
            ) from exc
        kwargs = {"api_key": api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def stream_chat(self, messages: list[Message], system: str,
                    on_delta: DeltaCallback, max_tokens: int = 8000) -> str:
        import openai

        client = self._get_client()
        # OpenAI carries the system prompt as the first message rather than
        # a separate parameter.
        payload = [{"role": "system", "content": system}]
        payload.extend(m.as_dict() for m in messages)

        parts: list[str] = []
        try:
            stream = client.chat.completions.create(
                model=self.model,
                messages=payload,
                max_tokens=max_tokens,
                stream=True,
            )
            for event in stream:
                if not event.choices:
                    continue
                delta = event.choices[0].delta
                text = getattr(delta, "content", None)
                if text:
                    parts.append(text)
                    on_delta(text)
        except openai.AuthenticationError as exc:
            raise ProviderError(
                "OpenAI rejected the API key. Check it under Settings > AI."
            ) from exc
        except openai.RateLimitError as exc:
            raise ProviderError(
                "Rate limited by OpenAI. Wait a moment and try again."
            ) from exc
        except openai.APIConnectionError as exc:
            raise ProviderError("Could not reach the OpenAI API: %s" % exc) from exc
        except openai.APIStatusError as exc:
            raise ProviderError("OpenAI API error %s: %s"
                                % (exc.status_code, exc.message)) from exc

        return "".join(parts)
