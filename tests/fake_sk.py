"""A scripted chat-completion service for Semantic Kernel, so the live code path can be tested without an API key.

It is a real SK `ChatCompletionClientBase`: the SKBackend drives a real `ChatCompletionAgent` against it.
Replies are produced by the agents' own offline functions, optionally tampered with to simulate a
misbehaving model.
"""
from __future__ import annotations

import json
from typing import Callable, ClassVar

from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import AuthorRole, ChatMessageContent

from debate_desk.agents.base import opening_offline
from debate_desk.agents.judge import judge_offline
from debate_desk.agents.rebuttal import rebuttal_offline
from debate_desk.agents.risk import risk_offline
from debate_desk.heuristics import analyze_fact
from debate_desk.models import Fact


def role_of(system_text: str) -> str:
    t = system_text
    if "RESEARCH Agent" in t:
        return "research"
    if "rebuttal round" in t:
        return "rebuttal_bull" if "BULL Agent" in t else "rebuttal_bear"
    if "BULL Agent" in t:
        return "bull"
    if "BEAR Agent" in t:
        return "bear"
    if "JUDGE" in t:
        return "judge"
    if "RISK Agent" in t:
        return "risk"
    raise ValueError("unknown role in system prompt")


def research_reply(payload: dict) -> dict:
    out = []
    for f in payload["facts"]:
        a = analyze_fact(Fact(f["id"], f["text"], f["source"], f["section"]))
        out.append({"id": f["id"], "theme": a.theme, "polarity": a.polarity, "strength": a.strength})
    return {"analysis": out}


class ScriptedChat(ChatCompletionClientBase):
    SUPPORTS_FUNCTION_CALLING: ClassVar[bool] = False
    calls: list = []                                   # (role, saw_tools) for assertions
    tamper: Callable[[str, dict], dict] | None = None  # optional hook to corrupt a reply

    def get_prompt_execution_settings_class(self):
        return PromptExecutionSettings

    async def _inner_get_chat_message_contents(self, chat_history, settings):
        system = next(m.content for m in chat_history.messages if m.role in (AuthorRole.SYSTEM, AuthorRole.DEVELOPER))
        user = [m.content for m in chat_history.messages if m.role == AuthorRole.USER][-1]
        role, payload = role_of(system), json.loads(user)
        reply = {
            "research": research_reply, "bull": opening_offline, "bear": opening_offline,
            "rebuttal_bull": rebuttal_offline, "rebuttal_bear": rebuttal_offline,
            "judge": judge_offline, "risk": risk_offline,
        }[role](payload)
        if self.tamper:
            reply = self.tamper(role, reply)
        self.calls.append(role)
        # models often wrap JSON in fences; the backend must cope
        text = "```json\n" + json.dumps(reply) + "\n```"
        return [ChatMessageContent(role=AuthorRole.ASSISTANT, content=text)]


def make_service(tamper=None) -> ScriptedChat:
    svc = ScriptedChat(ai_model_id="scripted-model", service_id="scripted")
    svc.calls = []
    svc.tamper = tamper
    return svc
