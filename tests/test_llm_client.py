"""Exercise the actual OpenAI SDK through a fake HTTP transport; never use live APIs."""

import json
from dataclasses import replace

import httpx2 as httpx
import pytest
from openai import OpenAI

from src.llm_client import GenerationError, OpenAILLMClient
from src.schemas import GenerationConfig

CONFIG = GenerationConfig("gpt-test", 0.4, 256, "reply_v1", 544)


def response_body():
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-test-snapshot",
        "output": [
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": "A reply.", "annotations": []}
                ],
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 4,
            "total_tokens": 14,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def sdk_for(handler, retries=0):
    return OpenAI(
        api_key="fake-test-key",
        base_url="https://offline.invalid/v1",
        timeout=1.0,
        max_retries=retries,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


@pytest.mark.parametrize("send_temperature", [True, False])
def test_request_settings_and_safe_metadata(send_temperature):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response_body())

    with sdk_for(handler) as sdk:
        result = OpenAILLMClient(sdk, send_temperature=send_temperature).complete(
            "instructions", "dialogue", CONFIG
        )
    request = requests[0]
    assert request["model"] == CONFIG.model_id
    assert request["max_output_tokens"] == CONFIG.max_output_tokens
    assert request["store"] is False
    assert request["instructions"] == "instructions"
    assert request["input"] == "dialogue"
    assert ("temperature" in request) == send_temperature
    assert "seed" not in request
    assert "condition" not in request
    assert result.text == "A reply."
    assert result.metadata["random_seed_applied"] is None
    assert result.metadata["random_seed_requested"] == 544
    assert result.metadata["temperature_applied"] == (0.4 if send_temperature else None)
    assert result.metadata["model_id"] == "gpt-test-snapshot"
    assert result.metadata["token_usage"]["total_tokens"] == 14
    assert result.metadata["latency_ms"] >= 0
    assert "fake-test-key" not in str(result.metadata)


@pytest.mark.parametrize("kind", ["incomplete", "empty", "refusal"])
def test_unusable_responses_raise(kind):
    body = response_body()
    if kind == "incomplete":
        body["status"] = "incomplete"
    elif kind == "empty":
        body["output"] = []
    else:
        body["output"][0]["content"] = [{"type": "refusal", "refusal": "No"}]
    with (
        sdk_for(lambda _: httpx.Response(200, json=body)) as sdk,
        pytest.raises(GenerationError),
    ):
        OpenAILLMClient(sdk).complete("prompt", "input", CONFIG)


def test_sdk_retry_is_bounded_and_error_body_is_hidden():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            429,
            headers={"retry-after-ms": "1"},
            json={"error": {"message": "fake-test-key SECRET", "type": "rate_limit"}},
        )

    with (
        sdk_for(handler, retries=1) as sdk,
        pytest.raises(GenerationError) as caught,
    ):
        OpenAILLMClient(sdk).complete("prompt", "input", CONFIG)
    assert len(calls) == 2
    assert "429" in str(caught.value)
    assert "SECRET" not in str(caught.value)
    assert "fake-test-key" not in str(caught.value)
    assert caught.value.__suppress_context__


def test_authentication_error_is_not_retried():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    with sdk_for(handler, retries=1) as sdk, pytest.raises(GenerationError):
        OpenAILLMClient(sdk).complete("prompt", "input", CONFIG)
    assert len(calls) == 1


def test_timeout_is_reported_without_request_contents():
    def handler(request):
        raise httpx.ReadTimeout("sensitive body", request=request)

    with (
        sdk_for(handler) as sdk,
        pytest.raises(GenerationError, match="OpenAI request failed") as caught,
    ):
        OpenAILLMClient(sdk).complete("private prompt", "input", CONFIG)
    assert "sensitive" not in str(caught.value)


def test_token_limit_rejected_before_request():
    with (
        sdk_for(lambda _: pytest.fail("Unexpected network request")) as sdk,
        pytest.raises(ValueError, match="at least 16"),
    ):
        OpenAILLMClient(sdk).complete(
            "prompt", "input", replace(CONFIG, max_output_tokens=1)
        )
