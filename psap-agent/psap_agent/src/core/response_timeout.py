"""Agent response timeout policy for model-specific execution limits."""


def get_agent_response_timeout_seconds(
    model_name: str | None,
    default_timeout_seconds: int,
    argon_timeout_seconds: int,
) -> int:
    """Use a longer minimum for Gemini 4 Argon without changing other models."""
    if (model_name or "").strip().lower().startswith("gemini-4-argon"):
        return max(default_timeout_seconds, argon_timeout_seconds)
    return default_timeout_seconds
