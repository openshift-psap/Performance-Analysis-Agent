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

from psap_agent.src.core.model_factory import create_chat_model
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
  "prerequisites": ["discovery/listing tools needed to gather IDs or context before the core workflow"],
  "tool_sequence": ["ordered", "list", "of", "core", "workflow", "tool", "names"],
  "notes": "Important observations about order, prerequisites, or edge cases",
  "tags": ["searchable", "tags"]
}}

IMPORTANT: Separate discovery/setup tools from the core workflow.
- "prerequisites": Tools that gather IDs, list available items, or resolve names into identifiers \
(e.g., discover_configurations, list_models). A consumer starting from scratch will need these first.
- "tool_sequence": The core analysis/action tools that do the real work once IDs are known.
If the interaction did not use any discovery tools, set "prerequisites" to an empty list.

Only extract genuinely reusable patterns. If this interaction is too specific to generalize, respond with: {{"skip": true}}"""


class SkillDocumentStore:
    """Manages Hermes-style skill documents using LangGraph BaseStore."""

    def __init__(self, store=None):
        self._local_skills: list[dict] = []
        self._store = store

    @staticmethod
    def _render_skill(skill: dict) -> list[str]:
        """Render a single skill document into human-readable lines."""
        lines = [f"\n## {skill.get('title', 'Untitled Skill')}"]
        lines.append(f"Trigger: {skill.get('trigger', 'N/A')}")
        prereqs = skill.get("prerequisites", [])
        if prereqs:
            lines.append("Prerequisites (run first if starting from scratch):")
            for i, tool in enumerate(prereqs, 1):
                lines.append(f"  {i}. {tool}")
        seq = skill.get("tool_sequence", [])
        if seq:
            label = "Core Tool Sequence:" if prereqs else "Tool Sequence:"
            lines.append(label)
            for i, tool in enumerate(seq, 1):
                lines.append(f"  {i}. {tool}")
        notes = skill.get("notes", "")
        if notes:
            lines.append(f"Notes: {notes}")
        return lines

    _SKILL_RELEVANCE_THRESHOLD = 0.5

    @staticmethod
    def _skill_text(skill: dict) -> str:
        """Build a single string representing a skill for embedding."""
        parts = [
            skill.get("title", ""),
            skill.get("trigger", ""),
            " ".join(skill.get("tags", [])),
        ]
        return " ".join(p for p in parts if p)

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    async def _rank_skills(
        self, query: str, skills: list[dict], limit: int
    ) -> list[tuple[dict, float]]:
        """Rank skills by cosine similarity to the query using Gemini embeddings."""
        if not skills:
            return []
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            embedder = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
            )
            texts = [self._skill_text(s) for s in skills]
            all_texts = [query] + texts
            embeddings = await asyncio.to_thread(embedder.embed_documents, all_texts)
            query_emb = embeddings[0]
            scored = []
            for i, skill in enumerate(skills):
                score = self._cosine_similarity(query_emb, embeddings[i + 1])
                scored.append((skill, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:limit]
        except Exception as e:
            logger.warning(f"Skill embedding ranking failed: {e}")
            return [(s, 1.0) for s in skills[:limit]]

    async def retrieve_skills(self, query: str, limit: int = 1) -> str:
        """Retrieve relevant skill documents for the current query."""
        try:
            all_skills: list[dict] = []

            if self._store and hasattr(self._store, "asearch"):
                results = await self._store.asearch(
                    ("skills", "global"), query=query, limit=50
                )
                for item in results:
                    skill = item.value if hasattr(item, "value") else item
                    if isinstance(skill, dict):
                        all_skills.append(skill)

            all_skills.extend(self._local_skills)

            if not all_skills:
                return ""

            ranked = await self._rank_skills(query, all_skills, limit)
            lines = [
                "Suggested (non-authoritative) tool recipes from past successful "
                "investigations:"
            ]
            kept = 0
            for skill, score in ranked:
                if score < self._SKILL_RELEVANCE_THRESHOLD:
                    logger.debug(
                        f"Skill retrieval: skipping '{skill.get('title', '?')}' "
                        f"(score={score:.2f} < {self._SKILL_RELEVANCE_THRESHOLD})"
                    )
                    continue
                logger.info(
                    f"Skill retrieval: using '{skill.get('title', '?')}' "
                    f"(score={score:.2f})"
                )
                lines.extend(self._render_skill(skill))
                kept += 1

            return "\n".join(lines) if kept > 0 else ""

        except Exception as e:
            logger.warning(f"Skill retrieval failed: {e}")
            return ""

    async def maybe_generate_skill(
        self, messages: list[BaseMessage], *, user_query: str = ""
    ) -> None:
        """Generate a skill document if the interaction was complex enough."""
        if not settings.ENABLE_SKILL_DOCUMENTS:
            return

        try:
            tool_calls = []
            agent_response = ""
            had_error_recovery = False
            tool_error_count = 0
            tool_success_count = 0

            for msg in messages:
                msg_type = getattr(msg, "type", "")
                if msg_type == "human" and not user_query:
                    user_query = str(getattr(msg, "content", ""))
                elif isinstance(msg, AIMessage):
                    if msg.content:
                        agent_response = str(msg.content)
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tool_calls.append(tc.get("name", "unknown"))
                elif msg_type == "tool":
                    content = str(getattr(msg, "content", ""))[:500].lower()
                    if '"status":"error"' in content or '"status": "error"' in content:
                        tool_error_count += 1
                    else:
                        tool_success_count += 1

            # Error recovery: agent hit errors but also had successful calls
            # after them, indicating it adapted its approach.
            had_error_recovery = tool_error_count > 0 and tool_success_count > 0

            is_complex = (
                len(tool_calls) >= settings.SKILL_GENERATION_THRESHOLD
                or (had_error_recovery and len(tool_calls) >= 4)
            )

            if not is_complex:
                logger.debug(
                    f"Skill generation: not complex enough "
                    f"(tool_calls={len(tool_calls)}, threshold={settings.SKILL_GENERATION_THRESHOLD}, "
                    f"error_recovery={had_error_recovery})"
                )
                return
            if not user_query or not agent_response:
                logger.info(
                    f"Skill generation: skipped — missing user_query={bool(user_query)}, "
                    f"agent_response={bool(agent_response)}"
                )
                return

            logger.info(
                f"Complex interaction detected ({len(tool_calls)} tool calls, "
                f"error_recovery={had_error_recovery}). Generating skill document."
            )

            model = create_chat_model(
                settings.LLM_JUDGE_MODEL, temperature=0.0
            )

            deduped: list[tuple[str, int]] = []
            for name in tool_calls:
                if deduped and deduped[-1][0] == name:
                    deduped[-1] = (name, deduped[-1][1] + 1)
                else:
                    deduped.append((name, 1))
            tool_sequence_text = "\n".join(
                f"  {i}. {name}" if count == 1 else f"  {i}. {name} (x{count} parallel)"
                for i, (name, count) in enumerate(deduped, 1)
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

        except Exception as e:
            logger.warning(f"Post-interaction memory processing failed: {e}")
