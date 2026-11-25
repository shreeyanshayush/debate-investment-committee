"""Citation enforcement. This is code, not a prompt: it holds no matter what a model says.

Rules applied to every debater claim and rebuttal:
  1. It must cite at least one fact ID.
  2. Every cited ID must exist in the document index (no invented sources).
  3. If a restricted allow-list is supplied (the Judge's case), IDs outside it are stripped.
  4. Every number in the claim text must appear in a cited fact (cheap hallucination guard).
Anything failing is dropped and recorded in the audit log.
"""
from __future__ import annotations

import re
from typing import Iterable

from .models import AuditEntry, Claim

_NUM = re.compile(r"\d+(?:\.\d+)?")
_TAG = re.compile(r"\[?\b[A-Z]{2,4}-\d{2,3}\b\]?")   # inline citation tags like [HEL-10]; their digits are IDs, not figures


def clean_cites(raw: Iterable[str] | None, valid_ids: set[str], allowed: set[str] | None = None) -> tuple[list[str], list[str]]:
    """Return (kept ids in order, rejected ids)."""
    kept: list[str] = []
    rejected: list[str] = []
    for c in raw or []:
        c = str(c).strip().strip("[]")
        if c in valid_ids and (allowed is None or c in allowed):
            if c not in kept:
                kept.append(c)
        else:
            rejected.append(c)
    return kept, rejected


def numbers_grounded(text: str, cited_texts: list[str]) -> list[str]:
    """Return numbers in `text` that do not occur in any cited fact."""
    pool_numbers = set(_NUM.findall(" ".join(cited_texts)))
    return [n for n in _NUM.findall(_TAG.sub(" ", text)) if n not in pool_numbers]


def validate_claims(
    agent: str,
    raw_claims: list[dict],
    fact_text: dict[str, str],
    audit: list[AuditEntry],
    allowed: set[str] | None = None,
) -> list[Claim]:
    valid_ids = set(fact_text)
    out: list[Claim] = []
    for raw in raw_claims or []:
        text = str(raw.get("text", "")).strip()
        cites, rejected = clean_cites(raw.get("cites"), valid_ids, allowed)
        if rejected:
            audit.append(AuditEntry(agent, "invalid_citation", f"Stripped unknown/not-allowed IDs {rejected} from: {text[:80]!r}"))
        if not text:
            audit.append(AuditEntry(agent, "empty_claim", "Dropped a claim with no text."))
            continue
        if not cites:
            audit.append(AuditEntry(agent, "uncited_claim", f"Dropped claim with no valid citation: {text[:80]!r}"))
            continue
        ungrounded = numbers_grounded(text, [fact_text[c] for c in cites])
        if ungrounded:
            audit.append(AuditEntry(agent, "ungrounded_number", f"Dropped claim; numbers {ungrounded} not in cited facts: {text[:80]!r}"))
            continue
        out.append(Claim(text=text, cites=cites))
    return out
