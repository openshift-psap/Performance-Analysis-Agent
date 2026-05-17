"""Long-term memory system for the PSAP agent.

Hybrid approach combining:
- Mem0 for semantic/episodic memory (automatic fact extraction + search)
- Hermes-style Skill Documents for procedural memory (tool-call recipes)
- LangGraph BaseStore as the coordination layer
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage

from psap_agent.src.settings import settings
from psap_agent.utils.pylogger import get_python_logger

logger = get_python_logger(log_level=settings.PYTHON_LOG_LEVEL)

# --- Mem0 Integration (Tier 3A) ---


class Mem0Memory:
    """Wrapper around Mem0 for semantic and episodic memory."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            if settings.MEM0_API_KEY:
                from mem0 import MemoryClient
                self._client = MemoryClient(api_key=settings.MEM0_API_KEY)
            else:
                from mem0 import Memory
                config = {
                    "llm": {
                        "provider": "gemini",
                        "config": {
                            "model": "gemini-2.5-flash",
                            "temperature": 0.1,
                        },
                    },
                    "embedder": {
                        "provider": "gemini",
                        "config": {
                            "model": "models/gemini-embedding-001",
                            "embedding_dims": 768,
                        },
                    },
                    "vector_store": {
                        "provider": "qdrant",
                        "config": {
                            "embedding_model_dims": 768,
                            "path": settings.MEM0_QDRANT_PATH,
                            "on_disk": True,
                        },
                    },
                    "version": "v1.1",
                }
                self._client = Memory.from_config(config)
        return self._client

    _RELEVANCE_THRESHOLD = 0.6
    _MAX_MEMORIES = 5

    async def retrieve_memories(self, query: str, user_id: str) -> str:
        """Retrieve relevant memories for the current query.

        Filters results by relevance score and caps the count to avoid
        injecting loosely-related context that confuses the agent.
        """
        try:
            client = self._get_client()
            from mem0 import Memory
            if isinstance(client, Memory):
                search_kwargs = {"top_k": self._MAX_MEMORIES * 2}
            else:
                search_kwargs = {"limit": self._MAX_MEMORIES * 2}

            memories = await asyncio.to_thread(
                client.search, query, filters={"user_id": user_id}, **search_kwargs
            )

            if isinstance(memories, dict):
                memory_list = memories.get("results", [])
            else:
                memory_list = memories if memories else []

            if not memory_list:
                logger.info(f"Mem0: no memories found for user {user_id}")
                return ""

            lines = ["Relevant context from past interactions:"]
            kept = 0
            for mem in memory_list:
                if isinstance(mem, dict):
                    score = mem.get("score", 0.0)
                    text = mem.get("memory", mem.get("text", str(mem)))
                    if score < self._RELEVANCE_THRESHOLD:
                        logger.debug(
                            f"Mem0: skipping memory (score={score:.2f}): {text[:80]}"
                        )
                        continue
                    logger.debug(f"Mem0: keeping memory (score={score:.2f}): {text[:80]}")
                else:
                    text = str(mem)
                lines.append(f"- {text}")
                kept += 1
                if kept >= self._MAX_MEMORIES:
                    break

            if kept == 0:
                logger.info(
                    f"Mem0: {len(memory_list)} memories found but none above "
                    f"threshold={self._RELEVANCE_THRESHOLD}"
                )
                return ""

            logger.info(
                f"Mem0: retrieved {kept}/{len(memory_list)} memories for user {user_id} "
                f"(threshold={self._RELEVANCE_THRESHOLD})"
            )
            return "\n".join(lines)

        except Exception as e:
            logger.warning(f"Mem0 retrieval failed: {e}")
            return ""

    async def store_interaction(
        self, user_query: str, agent_response: str, user_id: str
    ) -> None:
        """Store a conversation interaction in Mem0."""
        try:
            client = self._get_client()
            interaction = [
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": agent_response},
            ]
            await asyncio.to_thread(client.add, interaction, user_id=user_id)
            logger.info(f"Mem0: stored interaction for user {user_id}")
        except Exception as e:
            logger.warning(f"Mem0 storage failed: {e}")


# --- Skill Documents (Tier 3B) ---

SKILL_EXTRACTION_PROMPT = """Analyze this successful agent interaction and extract a reusable skill document.

User Query: {user_query}

Tool Calls (in order):
{tool_sequence}

Final Response Summary: {response_summary}

Create a skill document in this EXACT JSON format:
{{
  "title": "Short descriptive title (e.g., 'Deep Profiling Investigation')",
  "trigger": "When should this skill be used? (e.g., 'User asks to profile latency for a specific model')",
  "tool_sequence": ["ordered", "list", "of", "tool", "names"],
  "notes": "Important observations about order, prerequisites, or edge cases",
  "tags": ["searchable", "tags"]
}}

Only extract genuinely reusable patterns. If this interaction is too specific to generalize, respond with: {{"skip": true}}"""


class SkillDocumentStore:
    """Manages Hermes-style skill documents using LangGraph BaseStore."""

    def __init__(self, store=None):
        self._store = store
        self._local_skills: list[dict] = []

    async def retrieve_skills(self, query: str, limit: int = 3) -> str:
        """Retrieve relevant skill documents for the current query."""
        try:
            if self._store and hasattr(self._store, "asearch"):
                results = await self._store.asearch(
                    ("skills", "global"), query=query, limit=limit
                )
                if results:
                    lines = ["Relevant skill recipes from past successful investigations:"]
                    for item in results:
                        skill = item.value if hasattr(item, "value") else item
                        if isinstance(skill, dict):
                            lines.append(f"\n## {skill.get('title', 'Untitled Skill')}")
                            lines.append(f"Trigger: {skill.get('trigger', 'N/A')}")
                            seq = skill.get("tool_sequence", [])
                            if seq:
                                lines.append("Tool Sequence:")
                                for i, tool in enumerate(seq, 1):
                                    lines.append(f"  {i}. {tool}")
                            notes = skill.get("notes", "")
                            if notes:
                                lines.append(f"Notes: {notes}")
                    return "\n".join(lines)

            if self._local_skills:
                query_lower = query.lower()
                matched = []
                for skill in self._local_skills:
                    tags = " ".join(skill.get("tags", []))
                    trigger = skill.get("trigger", "")
                    if any(
                        word in tags.lower() or word in trigger.lower()
                        for word in query_lower.split()
                        if len(word) > 3
                    ):
                        matched.append(skill)

                if matched:
                    lines = ["Relevant skill recipes from past successful investigations:"]
                    for skill in matched[:limit]:
                        lines.append(f"\n## {skill.get('title', 'Untitled Skill')}")
                        lines.append(f"Trigger: {skill.get('trigger', 'N/A')}")
                        seq = skill.get("tool_sequence", [])
                        if seq:
                            lines.append("Tool Sequence:")
                            for i, tool in enumerate(seq, 1):
                                lines.append(f"  {i}. {tool}")
                        notes = skill.get("notes", "")
                        if notes:
                            lines.append(f"Notes: {notes}")
                    return "\n".join(lines)

            return ""

        except Exception as e:
            logger.warning(f"Skill retrieval failed: {e}")
            return ""

    async def maybe_generate_skill(self, messages: list[BaseMessage]) -> None:
        """Generate a skill document if the interaction was complex enough."""
        if not settings.ENABLE_SKILL_DOCUMENTS:
            return

        try:
            tool_calls = []
            user_query = ""
            agent_response = ""
            had_error_recovery = False

            for msg in messages:
                msg_type = getattr(msg, "type", "")
                if msg_type == "human":
                    user_query = str(getattr(msg, "content", ""))
                elif isinstance(msg, AIMessage):
                    if msg.content:
                        agent_response = str(msg.content)
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tool_calls.append(tc.get("name", "unknown"))
                elif msg_type == "tool":
                    content = str(getattr(msg, "content", ""))
                    if "error" in content.lower():
                        had_error_recovery = True

            is_complex = (
                len(tool_calls) >= settings.SKILL_GENERATION_THRESHOLD
                or had_error_recovery
            )

            if not is_complex or not user_query or not agent_response:
                return

            logger.info(
                f"Complex interaction detected ({len(tool_calls)} tool calls, "
                f"error_recovery={had_error_recovery}). Generating skill document."
            )

            from langchain_google_genai import ChatGoogleGenerativeAI

            model = ChatGoogleGenerativeAI(
                model=settings.LLM_JUDGE_MODEL, temperature=0.0
            )

            tool_sequence_text = "\n".join(
                f"  {i}. {name}" for i, name in enumerate(tool_calls, 1)
            )

            prompt = SKILL_EXTRACTION_PROMPT.format(
                user_query=user_query[:500],
                tool_sequence=tool_sequence_text,
                response_summary=agent_response[:500],
            )

            result = await model.ainvoke([{"role": "user", "content": prompt}])
            raw = result.content
            if isinstance(raw, list):
                raw = "\n".join(
                    item.get("text", str(item)) if isinstance(item, dict) else str(item)
                    for item in raw
                )
            response_text = str(raw).strip()
            if response_text.startswith("```"):
                response_text = (
                    response_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                )

            skill = json.loads(response_text)

            if skill.get("skip"):
                logger.debug("Skill generation skipped: interaction not generalizable")
                return

            skill["created_at"] = datetime.now(timezone.utc).isoformat()
            skill["source_query"] = user_query[:200]

            if self._store and hasattr(self._store, "aput"):
                skill_id = str(uuid4())
                await self._store.aput(
                    ("skills", "global"), skill_id, skill
                )
                logger.info(f"Stored skill document: {skill.get('title', 'untitled')}")
            else:
                self._local_skills.append(skill)
                logger.info(
                    f"Stored skill locally: {skill.get('title', 'untitled')} "
                    f"(total: {len(self._local_skills)})"
                )

        except Exception as e:
            logger.warning(f"Skill generation failed: {e}")


# --- Unified Memory Manager (Tier 3C coordination) ---


class MemoryManager:
    """Coordinates Mem0 and SkillDocuments behind a single interface."""

    def __init__(self, store=None):
        self._mem0: Optional[Mem0Memory] = None
        self._skills = SkillDocumentStore(store=store)

        if settings.ENABLE_MEM0:
            self._mem0 = Mem0Memory()

    async def get_memory_context(self, query: str, user_id: str) -> str:
        """Retrieve all relevant memory context for a query.

        Combines Mem0 episodic/semantic memories with skill document recipes.
        Returns a string to inject into the system prompt.
        """
        parts = []

        if self._mem0:
            mem0_context = await self._mem0.retrieve_memories(query, user_id)
            if mem0_context:
                parts.append(mem0_context)

        skill_context = await self._skills.retrieve_skills(query)
        if skill_context:
            parts.append(skill_context)

        return "\n\n".join(parts)

    async def post_interaction(
        self, messages: list[BaseMessage], user_id: str, user_query: str = ""
    ) -> None:
        """Process a completed interaction: store memories and maybe generate skills.

        Fire-and-forget -- errors are logged but never block the user.
        """
        try:
            extracted_query = ""
            agent_response = ""
            for msg in messages:
                if getattr(msg, "type", "") == "human":
                    extracted_query = str(getattr(msg, "content", ""))
                elif isinstance(msg, AIMessage) and msg.content:
                    agent_response = str(msg.content)

            effective_query = user_query or extracted_query
            if self._mem0 and effective_query and agent_response:
                await self._mem0.store_interaction(effective_query, agent_response, user_id)

            await self._skills.maybe_generate_skill(messages)

        except Exception as e:
            logger.warning(f"Post-interaction memory processing failed: {e}")
