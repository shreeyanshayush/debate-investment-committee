"""RAG layer: load an asset's document set, split it into citable facts, retrieve by TF-IDF similarity.

Every paragraph becomes one Fact with a stable ID (e.g. HEL-03). Agents cite by ID, and every downstream
text shown to the user is looked up from this index by ID, never copied from model output.
"""
from __future__ import annotations

import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .models import Fact

_COMMENT = re.compile(r"<!--.*?-->", re.S)


def load_facts(asset_dir: Path) -> tuple[str, list[Fact]]:
    """Return (asset display name, facts). Files are read in sorted order so IDs are stable."""
    files = sorted(p for p in asset_dir.iterdir() if p.suffix.lower() in {".md", ".txt"})
    if not files:
        raise FileNotFoundError(f"No .md/.txt documents found in {asset_dir}")
    code = re.sub(r"[^A-Za-z]", "", asset_dir.name)[:3].upper() or "DOC"
    name = asset_dir.name
    facts: list[Fact] = []
    for f in files:
        section = ""
        text = _COMMENT.sub("", f.read_text(encoding="utf-8"))
        for block in re.split(r"\n\s*\n", text):
            block = " ".join(block.split())
            if not block:
                continue
            if block.startswith("#"):
                heading = block.lstrip("#").strip()
                if block.startswith("# ") and not facts and name == asset_dir.name:
                    name = heading.split(" - ")[0].strip()
                section = heading
                continue
            facts.append(Fact(id=f"{code}-{len(facts) + 1:02d}", text=block, source=f.name, section=section))
    return name, facts


class DocumentIndex:
    def __init__(self, asset_dir: str | Path):
        self.asset_dir = Path(asset_dir)
        self.asset_name, fact_list = load_facts(self.asset_dir)
        self.facts: dict[str, Fact] = {f.id: f for f in fact_list}
        self._ids = [f.id for f in fact_list]
        self._vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self._matrix = self._vec.fit_transform([f"{f.section}. {f.text}" for f in fact_list])

    def search(self, query: str, k: int = 6) -> list[tuple[Fact, float]]:
        q = self._vec.transform([query])
        scores = linear_kernel(q, self._matrix).ravel()
        order = scores.argsort()[::-1][:k]
        return [(self.facts[self._ids[i]], float(scores[i])) for i in order if scores[i] > 0]

    def __len__(self) -> int:
        return len(self.facts)
