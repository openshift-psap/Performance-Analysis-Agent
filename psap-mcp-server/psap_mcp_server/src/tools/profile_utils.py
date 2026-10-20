"""Utilities for parsing and formatting benchmark prompt/output profiles."""


def _parse_token_count(value: str) -> int:
    value = value.strip().lower()
    if value.endswith("k"):
        return int(float(value[:-1]) * 1000)
    return int(value)


def parse_profile(profile: str) -> tuple[int, int]:
    """Parse an ISL/OSL profile; ``k`` always means 1,000 tokens."""
    parts = profile.strip().lower().split("/")
    if len(parts) != 2:
        raise ValueError("Profile must contain prompt/output token counts")
    return _parse_token_count(parts[0]), _parse_token_count(parts[1])


def format_profile_shorthand(prompt_tokens: int, output_tokens: int) -> str:
    """Format token counts using decimal ``k`` shorthand only."""

    def format_token_count(value: int) -> str:
        if value >= 1000 and value % 1000 == 0:
            return f"{value // 1000}k"
        return str(value)

    return f"{format_token_count(prompt_tokens)}/{format_token_count(output_tokens)}"
