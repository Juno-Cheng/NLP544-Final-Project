import json
from pathlib import Path
from types import SimpleNamespace

import httpx2
import openai
import pytest

from trustsim.config import AGENT, CRITIC, JUDGE, ROLES, RoleSettings, with_overrides
from trustsim.judge.judge import VERDICT_SCHEMA, parse_verdict
from trustsim.llm import factory
from trustsim.llm.client import (
    AnswerFormatError,
    LLMError,
    LLMRequest,
    LLMSetupError,
    OpenAICompatibleClient,
    UnusableAnswerError,
    UsageMeter,
    ask,
    decode_json_object,
)
from trustsim.llm.factory import build_clients
from trustsim.llm.mock import MockLLMClient
from trustsim.simulation.agent import REPLY_SCHEMA, parse_reply
from trustsim.simulation.critic import RATING_SCHEMA, parse_rating

from helpers import ScriptedClient, rating_answer

ROLE = RoleSettings(
    model="test-model",
    prompt_path=Path("unused.txt"),
    max_output_tokens=256,
    temperature=None,
    reasoning_effort=None,
)
REQUEST = LLMRequest(CRITIC, "Rate it.", '{"reader": "a"}', "critic_rating", RATING_SCHEMA)
HTTP_REQUEST = httpx2.Request("POST", "https://example.invalid/v1/chat/completions")


class StubCompletions:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.params: list[dict] = []

    def create(self, **params):
        self.params.append(params)
        if self.error is not None:
            raise self.error
        return self.response


def completion(content='{"rating": 0.5}', finish_reason="stop", refusal=None):
    message = SimpleNamespace(content=content, refusal=refusal)
    usage = SimpleNamespace(prompt_tokens=12, completion_tokens=3)
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish_reason, message=message)], usage=usage
    )


def client_with(
    response=None, error=None, settings=ROLE, token_limit_param="max_completion_tokens", meter=None
):
    completions = StubCompletions(response, error)
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return OpenAICompatibleClient(sdk, settings, token_limit_param, meter), completions


def status_error(cls, code: int, message: str):
    return cls(message, response=httpx2.Response(code, request=HTTP_REQUEST), body=None)


def test_unset_temperature_and_reasoning_effort_are_not_sent():
    client, completions = client_with(completion())
    client.complete(REQUEST)
    params = completions.params[0]
    assert "temperature" not in params
    assert "reasoning_effort" not in params
    assert params["max_completion_tokens"] == 256


def test_configured_settings_and_token_limit_name_are_sent():
    settings = RoleSettings("m", Path("p"), 512, temperature=0.7, reasoning_effort="low")
    client, completions = client_with(
        completion(), settings=settings, token_limit_param="max_tokens"
    )
    client.complete(REQUEST)
    params = completions.params[0]
    assert params["temperature"] == 0.7
    assert params["reasoning_effort"] == "low"
    assert params["max_tokens"] == 512
    assert "max_completion_tokens" not in params


def test_request_carries_the_prompts_and_the_role_json_schema():
    client, completions = client_with(completion())
    client.complete(REQUEST)
    params = completions.params[0]
    assert params["messages"] == [
        {"role": "system", "content": "Rate it."},
        {"role": "user", "content": '{"reader": "a"}'},
    ]
    schema = params["response_format"]["json_schema"]
    assert params["response_format"]["type"] == "json_schema"
    assert schema["strict"] is True
    assert schema["schema"] == RATING_SCHEMA


def test_usage_is_counted_per_role():
    meter = UsageMeter()
    client, _ = client_with(completion(), meter=meter)
    client.complete(REQUEST)
    client.complete(REQUEST)
    assert meter.totals() == {CRITIC: {"calls": 2, "prompt_tokens": 24, "completion_tokens": 6}}


@pytest.mark.parametrize(
    "response",
    [
        completion(finish_reason="length"),
        completion(refusal="I can't help with that."),
        completion(content=""),
        completion(content=None),
    ],
)
def test_cut_off_refused_or_empty_answers_are_unusable(response):
    client, _ = client_with(response)
    with pytest.raises(UnusableAnswerError):
        client.complete(REQUEST)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (status_error(openai.AuthenticationError, 401, "bad key"), LLMSetupError),
        (status_error(openai.PermissionDeniedError, 403, "no access"), LLMSetupError),
        (status_error(openai.NotFoundError, 404, "no such model"), LLMSetupError),
        (status_error(openai.BadRequestError, 400, "bad request"), LLMError),
        (openai.APIConnectionError(request=HTTP_REQUEST), LLMError),
    ],
)
def test_provider_errors_become_llm_errors(error, expected):
    client, _ = client_with(error=error)
    with pytest.raises(LLMError) as caught:
        client.complete(REQUEST)
    assert type(caught.value) is expected


def test_rejected_key_is_never_repeated_in_the_error():
    error = status_error(openai.AuthenticationError, 401, "Incorrect API key provided: sk-secret")
    client, _ = client_with(error=error)
    with pytest.raises(LLMSetupError) as caught:
        client.complete(REQUEST)
    assert "sk-secret" not in str(caught.value)


def test_bad_request_message_is_kept_so_settings_can_be_fixed():
    error = status_error(openai.BadRequestError, 400, "Unsupported parameter: 'temperature'")
    client, _ = client_with(error=error)
    with pytest.raises(LLMError, match="Unsupported parameter"):
        client.complete(REQUEST)


def test_decode_accepts_a_json_code_fence():
    assert decode_json_object('```json\n{"rating": 0.4}\n```') == {"rating": 0.4}


@pytest.mark.parametrize("answer", ["0.4", "[0.4]", "rating: 0.4"])
def test_decode_rejects_anything_but_a_json_object(answer):
    with pytest.raises(AnswerFormatError):
        decode_json_object(answer)


def test_ask_asks_again_after_an_unusable_answer():
    client = ScriptedClient({CRITIC: ["not json", rating_answer(0.8)]})
    assert ask(client, REQUEST, parse_rating, attempts=3) == 0.8
    assert len(client.requests) == 2


def test_ask_gives_up_after_the_last_attempt():
    client = ScriptedClient({CRITIC: ["no", "still no"]})
    with pytest.raises(UnusableAnswerError, match="2 attempts"):
        ask(client, REQUEST, parse_rating, attempts=2)


def test_ask_does_not_retry_a_failed_call():
    class FailingClient:
        calls = 0

        def complete(self, request):
            FailingClient.calls += 1
            raise LLMError("server error")

    with pytest.raises(LLMError, match="server error"):
        ask(FailingClient(), REQUEST, parse_rating, attempts=3)
    assert FailingClient.calls == 1


def test_mock_answers_every_role_in_its_format():
    mock = MockLLMClient()
    for role, schema, parse in [
        (AGENT, REPLY_SCHEMA, parse_reply),
        (CRITIC, RATING_SCHEMA, parse_rating),
        (JUDGE, VERDICT_SCHEMA, parse_verdict),
    ]:
        for n in range(20):
            request = LLMRequest(role, "system", json.dumps({"n": n}), role, schema)
            parse(decode_json_object(mock.complete(request)))


def test_mock_is_repeatable():
    first = MockLLMClient().complete(REQUEST)
    assert MockLLMClient().complete(REQUEST) == first


def test_live_provider_needs_its_api_key(monkeypatch, experiment_settings):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = with_overrides(experiment_settings, provider="openai")
    with pytest.raises(LLMSetupError, match="OPENAI_API_KEY"):
        build_clients(settings.llm, UsageMeter())


def test_gemini_uses_its_endpoint_key_and_token_limit_name(monkeypatch, experiment_settings):
    created: dict = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(factory.openai, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    settings = with_overrides(experiment_settings, provider="gemini")
    clients = build_clients(settings.llm, UsageMeter())
    assert created["api_key"] == "test-key"
    assert created["base_url"].startswith("https://generativelanguage.googleapis.com/")
    assert set(clients) == set(ROLES)
    assert "max_tokens" in clients[JUDGE].request_params(REQUEST)
