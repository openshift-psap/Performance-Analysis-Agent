"""Tests for user-specific Mem0 exclusions."""

import sys
from types import ModuleType
from unittest.mock import AsyncMock, Mock

import pytest

from psap_agent.src.core.memory import Mem0Memory, MemoryManager
from psap_agent.src.settings import settings


@pytest.mark.asyncio
async def test_forge_user_skips_mem0_retrieval_without_initializing_client(monkeypatch):
    monkeypatch.setattr(settings, "MEM0_DISABLED_USER_IDS", "forge-rhaiis")
    memory = Mem0Memory()
    get_client = Mock(side_effect=AssertionError("Mem0 client should not be used"))
    memory._get_client = get_client

    result = await memory.retrieve_memories("follow-up", "forge-rhaiis")

    assert result == ""
    get_client.assert_not_called()


@pytest.mark.asyncio
async def test_forge_user_skips_mem0_storage_without_initializing_client(monkeypatch):
    monkeypatch.setattr(settings, "MEM0_DISABLED_USER_IDS", "forge-rhaiis")
    memory = Mem0Memory()
    get_client = Mock(side_effect=AssertionError("Mem0 client should not be used"))
    memory._get_client = get_client

    await memory.store_interaction("question", "answer", "forge-rhaiis")

    get_client.assert_not_called()


@pytest.mark.asyncio
async def test_disabled_mem0_keeps_skill_context(monkeypatch):
    monkeypatch.setattr(settings, "MEM0_DISABLED_USER_IDS", "forge-rhaiis")
    manager = MemoryManager.__new__(MemoryManager)
    manager._mem0 = Mem0Memory()
    manager._skills = Mock()
    manager._skills.retrieve_skills = AsyncMock(return_value="Relevant skill")

    context = await manager.get_memory_context("follow-up", "forge-rhaiis")

    assert context == "Relevant skill"
    manager._skills.retrieve_skills.assert_awaited_once_with("follow-up")


@pytest.mark.asyncio
async def test_other_users_still_use_mem0(monkeypatch):
    monkeypatch.setattr(settings, "MEM0_DISABLED_USER_IDS", "forge-rhaiis")
    mem0_module = ModuleType("mem0")
    mem0_module.Memory = type("LocalMemory", (), {})
    monkeypatch.setitem(sys.modules, "mem0", mem0_module)
    memory = Mem0Memory()
    client = Mock(search=Mock(return_value={"results": []}))
    memory._get_client = Mock(return_value=client)

    result = await memory.retrieve_memories("question", "streamlit-user")

    assert result == ""
    client.search.assert_called_once()


@pytest.mark.asyncio
async def test_non_forge_user_can_store_memories(monkeypatch):
    monkeypatch.setattr(settings, "MEM0_DISABLED_USER_IDS", "forge-rhaiis")
    memory = Mem0Memory()
    client = Mock(add=Mock())
    memory._get_client = Mock(return_value=client)

    await memory.store_interaction("question", "answer", "streamlit-user")

    client.add.assert_called_once_with(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        user_id="streamlit-user",
    )
