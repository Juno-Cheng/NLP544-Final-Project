"""Model requests, an OpenAI-compatible client, and asking again when an answer is unusable."""

import json
import logging
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

import openai

from trustsim.config import RoleSettings

LOGGER = logging.getLogger(__name__)
PREVIEW_CHARS = 200
_FENCED = re.compile(r"\A```(?:json)?\s*(.*?)\s*```\Z", re.DOTALL)


class LLMError(RuntimeError):
    """A model call failed."""


class LLMSetupError(LLMError):
    """A problem every call would hit, such as a bad key or an unknown model."""


class UnusableAnswerError(LLMError):
    """The model answered, but the answer cannot be used (refused, empty, cut off, or malformed)."""


class AnswerFormatError(UnusableAnswerError):
    """An answer that does not match the JSON format its role asks for."""


@dataclass(frozen=True)
class LLMRequest:
    """One prompt to one role's model, with the JSON schema its answer must follow."""

    role: str
    system: str
    user: str
    schema_name: str
    schema: Mapping[str, Any]


class LLMClient(Protocol):
    """Anything that answers a request with text."""

    def complete(self, request: LLMRequest) -> str: ...


class UsageMeter:
    """Thread-safe count of model calls and tokens for each role."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._totals: dict[str, dict[str, int]] = {}

    def record(self, role: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Count one call."""
        with self._lock:
            totals = self._totals.setdefault(
                role, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
            )
            totals["calls"] += 1
            totals["prompt_tokens"] += prompt_tokens
            totals["completion_tokens"] += completion_tokens

    def totals(self) -> dict[str, dict[str, int]]:
        """Calls and tokens so far, by role."""
        with self._lock:
            return {role: dict(values) for role, values in self._totals.items()}


class OpenAICompatibleClient:
    """Chat Completions client for OpenAI, or for Gemini through its OpenAI-compatible endpoint."""

    def __init__(
        self,
        sdk: Any,
        settings: RoleSettings,
        token_limit_param: str,
        meter: UsageMeter | None = None,
    ) -> None:
        self._sdk = sdk
        self._settings = settings
        self._token_limit_param = token_limit_param
        self._meter = meter

    def complete(self, request: LLMRequest) -> str:
        """The answer text for one request."""
        started = perf_counter()
        try:
            response = self._sdk.chat.completions.create(**self.request_params(request))
        except openai.AuthenticationError:
            raise LLMSetupError(
                "the API key was rejected; check the key in your environment"
            ) from None
        except (openai.PermissionDeniedError, openai.NotFoundError) as exc:
            raise LLMSetupError(
                f"model {self._settings.model!r}: {_provider_message(exc)}"
            ) from None
        except openai.BadRequestError as exc:
            raise LLMError(f"request rejected: {_provider_message(exc)}") from None
        except openai.APIError as exc:
            raise LLMError(f"{type(exc).__name__} after the SDK's retries") from None
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise UnusableAnswerError(
                f"answer cut off at {self._settings.max_output_tokens} tokens; raise "
                f"max_output_tokens for the {request.role} role in configs/experiment.yaml"
            )
        if getattr(choice.message, "refusal", None):
            raise UnusableAnswerError(
                f"the model refused: {choice.message.refusal[:PREVIEW_CHARS]}"
            )
        content = choice.message.content
        if not content or not content.strip():
            raise UnusableAnswerError("the model returned an empty answer")
        if self._meter is not None:
            usage = response.usage
            self._meter.record(
                request.role,
                getattr(usage, "prompt_tokens", 0) or 0,
                getattr(usage, "completion_tokens", 0) or 0,
            )
        LOGGER.debug("%s answered in %.1f s", request.role, perf_counter() - started)
        return content

    def request_params(self, request: LLMRequest) -> dict[str, Any]:
        """Keyword arguments for chat.completions.create. Unset optional settings are left out,
        because some models reject them (gpt-5.5 rejects any temperature)."""
        params: dict[str, Any] = {
            "model": self._settings.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_name,
                    "schema": dict(request.schema),
                    "strict": True,
                },
            },
            self._token_limit_param: self._settings.max_output_tokens,
        }
        if self._settings.temperature is not None:
            params["temperature"] = self._settings.temperature
        if self._settings.reasoning_effort is not None:
            params["reasoning_effort"] = self._settings.reasoning_effort
        return params


def _provider_message(exc: openai.APIError) -> str:
    return str(getattr(exc, "message", "") or type(exc).__name__)[:PREVIEW_CHARS]


def preview(text: str) -> str:
    """The start of a model answer, for error messages."""
    text = " ".join(text.split())
    return text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + "..."


def decode_json_object(answer: str) -> dict[str, Any]:
    """The JSON object in a model answer. A surrounding Markdown code fence is allowed."""
    body = answer.strip()
    fenced = _FENCED.match(body)
    if fenced:
        body = fenced.group(1)
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        raise AnswerFormatError(f"not valid JSON: {preview(answer)}") from None
    if not isinstance(value, dict):
        raise AnswerFormatError(f"not a JSON object: {preview(answer)}")
    return value


def expect_keys(answer: dict[str, Any], keys: set[str]) -> None:
    """Check that an answer has exactly the keys its role's format defines."""
    if set(answer) != keys:
        raise AnswerFormatError(f"expected keys {sorted(keys)}, got {sorted(answer)}")


def ask[T](
    client: LLMClient,
    request: LLMRequest,
    parse: Callable[[dict[str, Any]], T],
    attempts: int,
) -> T:
    """The parsed answer to a request, asking again while answers are unusable."""
    problem: UnusableAnswerError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return parse(decode_json_object(client.complete(request)))
        except UnusableAnswerError as exc:
            problem = exc
            LOGGER.warning(
                "%s answer unusable (attempt %d of %d): %s", request.role, attempt, attempts, exc
            )
    raise UnusableAnswerError(f"no usable {request.role} answer in {attempts} attempts: {problem}")
