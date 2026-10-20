"""Tests for decimal benchmark-profile shorthand."""

import pytest

from psap_mcp_server.src.tools.profile_utils import (
    format_profile_shorthand,
    parse_profile,
)


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ("1k/1k", (1000, 1000)),
        ("1K/2k", (1000, 2000)),
        ("8k/800", (8000, 800)),
        ("1024/1024", (1024, 1024)),
    ],
)
def test_parse_profile_uses_decimal_k(profile, expected):
    assert parse_profile(profile) == expected


@pytest.mark.parametrize("profile", ["1k", "1k/", "/1k", "1kk/1k"])
def test_parse_profile_rejects_invalid_values(profile):
    with pytest.raises(ValueError):
        parse_profile(profile)


def test_format_profile_shorthand_only_uses_decimal_multiples():
    assert format_profile_shorthand(1000, 1000) == "1k/1k"
    assert format_profile_shorthand(8000, 800) == "8k/800"
    assert format_profile_shorthand(1024, 1024) == "1024/1024"
