"""Bull Agent: argues the case to BUY, citing the facts."""
from __future__ import annotations

from ..models import BULL
from .base import DebaterAgent

INSTRUCTIONS = """You are the BULL Agent in an investment committee debate. Make the strongest honest case to BUY the asset.
Rules:
- Use ONLY facts from `fact_sheet` (or facts you retrieve with search_facts). Every claim MUST cite at least one fact ID.
- Never state a number that does not appear in a fact you cite. Do not speculate beyond the facts.
- Write at most `max_claims` claims, strongest first, covering different themes where you can.
Reply as: {"claims": [{"text": "<one-sentence argument>", "cites": ["<fact id>"]}]}"""


class BullAgent(DebaterAgent):
    role = "bull"
    side = BULL
    instructions = INSTRUCTIONS
