"""Model backends. Agents never talk to a model directly; they call `backend.run(...)`.

Two interchangeable backends:

  OfflineBackend  Runs each agent's deterministic rule-based `offline` function. No API key, fully
                  reproducible. Use it to demo, test and CI the architecture.
  SKBackend       Runs each agent as a Semantic Kernel `ChatCompletionAgent` against a real model
                  (Azure OpenAI or OpenAI, chosen from environment variables). Debaters, the Research
                  Agent and the Rebuttal agents also get a `search_facts` retrieval plugin (RAG as an SK
                  tool); the Judge and Risk agents deliberately get none.

Either way the output goes through the same validators (citations.py), so guardrails never depend on
which backend is in use.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Annotated, Any, Callable, Protocol  # Annotated must be module-level: SK evaluates annotation strings

JSON_SUFFIX = "\n\nRespond with ONE JSON object only: no prose before or after it and no markdown fences."


class Backend(Protocol):
    name: str

    async def run(self, role: str, instructions: str, payload: dict[str, Any],
                  offline: Callable[[dict], dict], retrieval: bool = False) -> dict: ...


class OfflineBackend:
    name = "offline (deterministic rule-based reasoning, no model)"

    async def run(self, role, instructions, payload, offline, retrieval=False):
        await asyncio.sleep(0)
        return offline(payload)


def extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in model reply")
    return json.loads(text[start:end + 1])


def make_retrieval_plugin(index):
    """Wrap the RAG index as a Semantic Kernel plugin (imported lazily so offline mode needs no SK)."""
    from semantic_kernel.functions import kernel_function

    class RetrievalPlugin:
        @kernel_function(name="search_facts",
                         description="Search the asset's document set. Returns fact IDs with their text; cite facts by ID.")
        def search_facts(self,
                         query: Annotated[str, "What evidence to look for"],
                         top_k: Annotated[int, "Maximum number of facts to return"] = 5) -> Annotated[str, "JSON list of facts"]:
            hits = index.search(query, max(1, min(int(top_k), 10)))
            return json.dumps([{"id": f.id, "text": f.text, "source": f.source} for f, _ in hits])

    return RetrievalPlugin()


def build_kernel(service=None):
    """Create a Kernel with a chat service from the environment (or use the one supplied)."""
    from semantic_kernel import Kernel
    kernel = Kernel()
    if service is not None:
        kernel.add_service(service)
        return kernel, getattr(service, "ai_model_id", "custom-service")
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
        svc = AzureChatCompletion()          # reads AZURE_OPENAI_* env vars
        kernel.add_service(svc)
        return kernel, f"azure:{os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT_NAME', 'deployment')}"
    if os.getenv("OPENAI_API_KEY"):
        from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
        model = os.getenv("OPENAI_CHAT_MODEL_ID", "gpt-4o")
        kernel.add_service(OpenAIChatCompletion(ai_model_id=model))
        return kernel, f"openai:{model}"
    gkey = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gkey:   # Google Gemini through its OpenAI-compatible endpoint (supports tool calling)
        from openai import AsyncOpenAI
        from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
        model = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
        client = AsyncOpenAI(api_key=gkey, base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                             timeout=180.0, max_retries=4)
        kernel.add_service(OpenAIChatCompletion(ai_model_id=model, async_client=client))
        return kernel, f"gemini:{model}"
    raise RuntimeError("No model configured. Set OPENAI_API_KEY (and optionally OPENAI_CHAT_MODEL_ID) or "
                       "AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_CHAT_DEPLOYMENT_NAME, or GEMINI_API_KEY.")


def model_configured() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                or (os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT")))


class SKBackend:
    def __init__(self, index, service=None):
        self.index = index
        self.kernel, label = build_kernel(service)
        self.name = f"Semantic Kernel ChatCompletionAgent ({label})"
        # Free-tier model quotas are per-minute; limit simultaneous requests (override: DEBATE_MAX_CONCURRENCY=4).
        self._gate = asyncio.Semaphore(max(1, int(os.getenv("DEBATE_MAX_CONCURRENCY", "1"))))

    async def _call_with_retry(self, agent, message, role, attempts: int = 5):
        """Retry transient failures. Rate limits (429) wait 15/30/45/60s; network drops wait 2/4/8/16s."""
        for attempt in range(1, attempts + 1):
            try:
                async with self._gate:
                    t0 = time.monotonic()
                    print(f"  [{role}] calling model...", flush=True)
                    result = await agent.get_response(messages=message)
                    print(f"  [{role}] done in {time.monotonic() - t0:.1f}s", flush=True)
                    return result
            except Exception as e:                      # SK wraps provider errors in ServiceResponseException
                text = f"{type(e).__name__} {e}"
                limited = any(k in text for k in ("429", "RateLimit", "quota", "RESOURCE_EXHAUSTED"))
                transient = limited or any(k in text for k in ("Connection error", "ReadError", "Timeout", "timed out",
                                                               "500", "502", "503", "504", "overloaded"))
                if not transient or attempt == attempts:
                    raise
                wait = 15 * attempt if limited else 2 ** attempt
                why = "rate limit (429)" if limited else "network/server error"
                print(f"  [{role}] {why}, retrying in {wait}s (attempt {attempt}/{attempts - 1})", flush=True)
                await asyncio.sleep(wait)

    async def run(self, role, instructions, payload, offline, retrieval=False):
        from semantic_kernel.agents import ChatCompletionAgent
        kwargs: dict[str, Any] = {}
        if retrieval:
            from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
            kwargs["plugins"] = {"documents": make_retrieval_plugin(self.index)}
            kwargs["function_choice_behavior"] = FunctionChoiceBehavior.Auto()
        agent = ChatCompletionAgent(kernel=self.kernel, name=role, instructions=instructions + JSON_SUFFIX, **kwargs)
        message = json.dumps(payload)
        last_err: Exception | None = None
        for _ in range(2):
            response = await self._call_with_retry(agent, message, role)
            try:
                return extract_json(str(response.message.content))
            except (ValueError, json.JSONDecodeError) as e:
                last_err = e
                message = "Your previous reply was not a single valid JSON object. Reply with only the JSON.\n" + json.dumps(payload)
        raise RuntimeError(f"{role}: model did not return valid JSON twice ({last_err})")
