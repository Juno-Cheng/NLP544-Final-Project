"""Injectable text clients for live GPT generation and deterministic offline tests."""

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from openai import APIError, OpenAI

from src.schemas import GenerationConfig


class GenerationError(RuntimeError):
    """A model call failed or did not produce a usable completed reply."""


@dataclass(frozen=True)
class Completion:
    """Model text and safe metadata, kept separate from dialogue."""

    text: str
    metadata: dict[str, Any]


class LLMClient(Protocol):
    """Small provider boundary; implementations need only complete()."""

    def complete(
        self, instructions: str, payload: str, config: GenerationConfig
    ) -> Completion: ...


class OpenAILLMClient:
    """Use an injected SDK client so credentials, timeouts and retries stay explicit."""

    def __init__(self, sdk: OpenAI, *, send_temperature: bool = True) -> None:
        self._sdk = sdk
        self._send_temperature = send_temperature

    def complete(
        self, instructions: str, payload: str, config: GenerationConfig
    ) -> Completion:
        """Return a completed text response; never fabricate a fallback reply."""
        if config.max_output_tokens < 16:
            raise ValueError("Responses API max_output_tokens must be at least 16")
        request: dict[str, Any] = {
            "model": config.model_id,
            "instructions": instructions,
            "input": payload,
            "max_output_tokens": config.max_output_tokens,
            "store": False,
        }
        if self._send_temperature:
            request["temperature"] = config.temperature
        started = perf_counter()
        try:
            response = self._sdk.responses.create(**request)
        except APIError as exc:
            # Provider exception bodies may contain prompts or credentials.
            status = getattr(exc, "status_code", None)
            raise GenerationError(
                f"OpenAI request failed (HTTP status: {status})"
            ) from None
        if response.status != "completed":
            raise GenerationError(
                "OpenAI did not complete the reply; partial output discarded"
            )
        if any(
            content.type == "refusal"
            for item in response.output
            if item.type == "message"
            for content in item.content
        ):
            raise GenerationError("OpenAI refused the reply")
        text = response.output_text.strip()
        if not text:
            raise GenerationError("OpenAI returned an empty reply")
        return Completion(
            text,
            {
                "provider": "openai",
                "requested_model_id": config.model_id,
                "model_id": response.model,
                "response_id": response.id,
                "prompt_version": config.prompt_version,
                "temperature_requested": config.temperature,
                "temperature_applied": config.temperature
                if self._send_temperature
                else None,
                "max_output_tokens": config.max_output_tokens,
                "random_seed_requested": config.random_seed,
                "random_seed_applied": None,
                "token_usage": response.usage.model_dump() if response.usage else None,
                "latency_ms": (perf_counter() - started) * 1000,
                "cost": None,
            },
        )


class MockLLMClient:
    """Consume explicit canned replies and capture requests without network access."""

    def __init__(self, replies: Iterable[str]) -> None:
        self._replies = iter(replies)
        self.requests: list[dict[str, Any]] = []

    def complete(
        self, instructions: str, payload: str, config: GenerationConfig
    ) -> Completion:
        """Return the next canned reply, failing clearly when the fixture is exhausted."""
        self.requests.append(
            deepcopy(
                {
                    "instructions": instructions,
                    "payload": payload,
                    "config": config,
                }
            )
        )
        try:
            text = next(self._replies)
        except StopIteration:
            raise GenerationError("Mock reply sequence exhausted") from None
        return Completion(
            text,
            {
                "provider": "mock",
                "model_id": config.model_id,
                "prompt_version": config.prompt_version,
            },
        )
