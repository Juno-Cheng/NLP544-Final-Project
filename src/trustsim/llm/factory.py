"""Build one model client per role from the experiment settings."""

import os

import openai

from trustsim.config import MOCK_PROVIDER, LLMSettings
from trustsim.llm.client import LLMClient, LLMSetupError, OpenAICompatibleClient, UsageMeter
from trustsim.llm.mock import MockLLMClient


def build_clients(settings: LLMSettings, meter: UsageMeter) -> dict[str, LLMClient]:
    """A client for every role. The mock provider needs no key and makes no network calls."""
    if settings.provider == MOCK_PROVIDER:
        mock = MockLLMClient(meter)
        return {role: mock for role in settings.roles}
    provider = settings.providers[settings.provider]
    api_key = os.environ.get(provider.api_key_env, "").strip()
    if not api_key:
        raise LLMSetupError(
            f"{provider.api_key_env} is not set; export it before a live {settings.provider} run, "
            "or use --mock"
        )
    sdk = openai.OpenAI(
        api_key=api_key,
        base_url=provider.base_url,
        max_retries=settings.max_retries,
        timeout=settings.timeout_seconds,
    )
    return {
        role: OpenAICompatibleClient(sdk, role_settings, provider.token_limit_param, meter)
        for role, role_settings in settings.roles.items()
    }
