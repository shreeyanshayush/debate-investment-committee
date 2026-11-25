#!/usr/bin/env python3
"""Run the investment committee on one asset's document set.

    python run_debate.py data/helios
    python run_debate.py data/verdant --backend sk          # live model via Semantic Kernel
    python run_debate.py data/verdant --limit leverage_x=6.0 --veto-score 80
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import sys
from pathlib import Path

from debate_desk.config import Config
from debate_desk.orchestrator import DebateOrchestrator
from debate_desk.report import console_summary, render_markdown, to_json


def build_config(limits: list[str], veto_score: int | None) -> Config:
    cfg = Config()
    if veto_score is not None:
        cfg.risk_veto_score = veto_score
    if limits:
        overrides = {}
        for item in limits:
            key, _, val = item.partition("=")
            overrides[key.strip()] = float(val)
        known = {l.key for l in cfg.limits}
        bad = set(overrides) - known
        if bad:
            sys.exit(f"Unknown limit(s) {sorted(bad)}. Known: {sorted(known)}")
        cfg.limits = tuple(dataclasses.replace(l, limit=overrides.get(l.key, l.limit)) for l in cfg.limits)
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("asset", type=Path, help="folder containing the asset's .md/.txt documents")
    p.add_argument("--backend", choices=["auto", "offline", "sk"], default="auto",
                   help="auto = Semantic Kernel if an API key is set, else offline")
    p.add_argument("--out", type=Path, default=Path("output"), help="where to write the report (default: output/)")
    p.add_argument("--limit", action="append", default=[], metavar="KEY=VALUE", help="override a risk limit, e.g. leverage_x=6")
    p.add_argument("--veto-score", type=int, help="override the composite risk score that triggers a veto")
    args = p.parse_args()

    orch = DebateOrchestrator(args.asset, backend=args.backend, cfg=build_config(args.limit, args.veto_score))
    state = asyncio.run(orch.run())

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.asset.name
    (args.out / f"{stem}_report.md").write_text(render_markdown(state), encoding="utf-8")
    (args.out / f"{stem}_report.json").write_text(to_json(state), encoding="utf-8")
    print(f"Backend: {state.backend}")
    print(console_summary(state))
    print(f"Full report: {args.out / (stem + '_report.md')}")


if __name__ == "__main__":
    main()
