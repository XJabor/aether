"""LLM provider interface and the Anthropic (Claude) implementation.

The OpenAI implementation lives in :mod:`aether.ai.openai_provider` so
the two SDKs never share a module.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from . import keys

# Streaming callback: receives each text delta as it arrives.
DeltaCallback = Callable[[str], None]


class ProviderError(RuntimeError):
    """A provider failed in a way worth showing the user verbatim."""


@dataclass
class Message:
    role: str          # "user" | "assistant"
    content: str

    def as_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class LLMProvider(ABC):
    """Everything the chat panel needs from a model backend."""

    key_name: str = ""
    display_name: str = ""
    models: list[str] = []
    default_model: str = ""

    def __init__(self, model: str = "") -> None:
        self.model = model or self.default_model

    @property
    def has_key(self) -> bool:
        return keys.has_key(self.key_name)

    @abstractmethod
    def stream_chat(self, messages: list[Message], system: str,
                    on_delta: DeltaCallback,
                    max_tokens: int = 8000) -> str:
        """Stream a reply, calling on_delta with each chunk.

        Returns the complete text. Raises ProviderError on failure.
        """


class AnthropicProvider(LLMProvider):
    key_name = "anthropic"
    display_name = "Anthropic (Claude)"
    models = ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"]
    default_model = "claude-opus-5"

    def __init__(self, model: str = "") -> None:
        super().__init__(model)
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        api_key = keys.get_key(self.key_name)
        if not api_key:
            raise ProviderError(
                "No Anthropic API key saved. Add one under Settings > AI."
            )
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderError(
                "The 'anthropic' package is not installed. "
                "Install it with: pip install anthropic"
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def stream_chat(self, messages: list[Message], system: str,
                    on_delta: DeltaCallback, max_tokens: int = 8000) -> str:
        import anthropic

        client = self._get_client()
        payload = [m.as_dict() for m in messages]

        try:
            with client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                # Adaptive thinking: interpreting a spectrum is genuinely
                # multi-step reasoning, and the model decides how much to
                # spend rather than us guessing a budget.
                thinking={"type": "adaptive"},
                messages=payload,
            ) as stream:
                for chunk in stream.text_stream:
                    on_delta(chunk)
                final = stream.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise ProviderError(
                "Anthropic rejected the API key. Check it under Settings > AI."
            ) from exc
        except anthropic.RateLimitError as exc:
            raise ProviderError(
                "Rate limited by Anthropic. Wait a moment and try again."
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError("Could not reach the Anthropic API: %s" % exc) from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError("Anthropic API error %s: %s"
                                % (exc.status_code, exc.message)) from exc

        if getattr(final, "stop_reason", None) == "refusal":
            detail = getattr(final, "stop_details", None)
            raise ProviderError(
                "The model declined to answer%s."
                % (" (%s)" % detail.category if detail and detail.category else "")
            )

        return "".join(
            b.text for b in final.content if getattr(b, "type", "") == "text"
        )


def available_providers() -> dict[str, type[LLMProvider]]:
    from .openai_provider import OpenAIProvider

    return {
        AnthropicProvider.key_name: AnthropicProvider,
        OpenAIProvider.key_name: OpenAIProvider,
    }


def make_provider(name: str, model: str = "") -> LLMProvider:
    registry = available_providers()
    cls = registry.get(name)
    if cls is None:
        raise ProviderError(
            "Unknown AI provider %r. Available: %s"
            % (name, ", ".join(sorted(registry)))
        )
    return cls(model)
