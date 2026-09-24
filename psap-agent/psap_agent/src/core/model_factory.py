"""Provider-aware chat model construction for the PSAP agent."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from psap_agent.src.core.exceptions.exceptions import AppException, AppExceptionCode
from psap_agent.src.settings import settings

_OPENAI_PREFIX = "openai:"
_OPENAI_BARE_MODEL_PREFIXES = ("gpt-", "o1", "o3", "o4")
_OPENAI_REASONING_EFFORTS = frozenset(
    {"none", "low", "medium", "high", "xhigh", "max"}
)


def get_model_provider(model_name: str) -> str:
    """Return the provider selected by a model name.

    ``openai:<model-id>`` is the explicit OpenAI form. Common bare OpenAI
    model IDs are also recognized to make environment-based configuration
    less error-prone. Existing Gemini and Claude names remain unchanged.
    """
    normalized = model_name.strip().lower()
    if normalized.startswith(_OPENAI_PREFIX) or normalized.startswith(
        _OPENAI_BARE_MODEL_PREFIXES
    ):
        return "openai"
    if normalized.startswith("claude-"):
        return "claude"
    return "gemini"


def is_gemini_model(model_name: str) -> bool:
    """Return whether ``model_name`` uses the existing Gemini provider path."""
    return get_model_provider(model_name) == "gemini"


def _get_openai_model_id(model_name: str) -> str:
    """Remove an optional provider prefix and validate the resulting model ID."""
    normalized = model_name.strip()
    if normalized.lower().startswith(_OPENAI_PREFIX):
        normalized = normalized.split(":", 1)[1].strip()
    if not normalized:
        raise AppException(
            "An OpenAI model ID is required (for example, openai:gpt-5.6)",
            AppExceptionCode.CONFIGURATION_VALIDATION_ERROR,
        )
    return normalized


def _create_openai_model(
    model_name: str, reasoning_effort: str | None = None
) -> BaseChatModel:
    """Create an OpenAI chat model without provider-specific unsupported knobs.

    Sampling and output-token parameters differ across current OpenAI models,
    especially reasoning models. Leaving them at the provider defaults keeps
    the configured model broadly compatible while preserving the existing
    provider-specific settings for Gemini and Claude. GPT-6 models use the
    Responses API so they can call tools with their configured reasoning level.
    """
    if not settings.OPENAI_API_KEY:
        raise AppException(
            "OPENAI_API_KEY must be set to use OpenAI models",
            AppExceptionCode.CONFIGURATION_VALIDATION_ERROR,
        )

    from langchain_openai import ChatOpenAI

    model_id = _get_openai_model_id(model_name)
    kwargs: dict = {"model": model_id, "api_key": settings.OPENAI_API_KEY}

    if reasoning_effort:
        normalized_effort = reasoning_effort.strip().lower()
        if normalized_effort not in _OPENAI_REASONING_EFFORTS:
            supported_efforts = ", ".join(sorted(_OPENAI_REASONING_EFFORTS))
            raise AppException(
                f"Unsupported OpenAI reasoning effort '{reasoning_effort}'. "
                f"Supported values: {supported_efforts}",
                AppExceptionCode.CONFIGURATION_VALIDATION_ERROR,
            )
        # ``reasoning`` selects the Responses API in langchain-openai and maps
        # directly to OpenAI's ``reasoning.effort`` request parameter.
        kwargs["reasoning"] = {"effort": normalized_effort}

    # GPT-6 tool calling requires the Responses API, even at its default effort.
    if reasoning_effort or model_id.lower().startswith("gpt-6-"):
        kwargs["use_responses_api"] = True

    return ChatOpenAI(**kwargs)


def _create_claude_model(model_name: str) -> BaseChatModel:
    """Create a ChatAnthropicVertex model for Claude via Vertex AI."""
    from langchain_google_vertexai.model_garden import ChatAnthropicVertex

    if not settings.ANTHROPIC_VERTEX_PROJECT_ID:
        raise AppException(
            "ANTHROPIC_VERTEX_PROJECT_ID must be set to use Claude models",
            AppExceptionCode.CONFIGURATION_VALIDATION_ERROR,
        )

    return ChatAnthropicVertex(
        model_name=model_name,
        project=settings.ANTHROPIC_VERTEX_PROJECT_ID,
        location=settings.CLOUD_ML_REGION,
        temperature=0.0,
        max_tokens=16384,
    )


def create_chat_model(
    model_name: str,
    *,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    cached_content: str | None = None,
    reasoning_effort: str | None = None,
) -> BaseChatModel:
    """Create a chat model while preserving each provider's existing behavior.

    ``reasoning_effort`` is used only by the OpenAI provider. The agent API
    validates the values sent by Streamlit before model construction.
    """
    provider = get_model_provider(model_name)
    if provider == "openai":
        return _create_openai_model(model_name, reasoning_effort=reasoning_effort)
    if provider == "claude":
        return _create_claude_model(model_name)

    from langchain_google_genai import ChatGoogleGenerativeAI

    kwargs: dict = {"model": model_name}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens
    if cached_content:
        kwargs["model_kwargs"] = {"cached_content": cached_content}
    return ChatGoogleGenerativeAI(**kwargs)
