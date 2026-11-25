# Debate-Driven Investment Committee

A multi-agent system that reaches a **BUY / HOLD / AVOID** recommendation on an asset through a structured
debate, built on **Semantic Kernel**. All company data in `data/` is synthetic.

```
Research ──► Bull ┐                     ┌─► Judge ┐
                  ├─► Rebuttal round ──┤         ├─► resolve  (Risk veto can force HOLD)
            Bear ─┘                     └─► Risk ─┘
```

| Agent | Module | Job | RAG search tool (live mode) |
|---|---|---|---|
| Research | `agents/research.py` | Retrieves facts from the document set, classifies theme / polarity / strength | yes |
| Bull | `agents/bull.py` | Argues BUY, every claim cites fact IDs | yes |
| Bear | `agents/bear.py` | Argues AVOID, every claim cites fact IDs | yes |
| Rebuttal round | `agents/rebuttal.py` | Each side answers the other's strongest point once, or concedes | yes |
| Judge | `agents/judge.py` | Weighs the cases, issues the verdict and deciding rationale | **no** |
| Risk | `agents/risk.py` | Holds a single veto; forces HOLD when a stated threshold is crossed | **no** |

Shared state (the blackboard) is `DebateState` in `orchestrator.py`.

## Project website

An interactive walkthrough of the system is in `docs/index.html`. Open it in a browser, or publish it with GitHub Pages
(Settings > Pages > Deploy from a branch > `main` / `/docs`).

## Run

```bash
pip install -r requirements.txt
python run_debate.py data/verdant                     # offline, deterministic, no API key
python run_debate.py data/verdant --backend sk        # live model through Semantic Kernel
python run_debate.py data/helios --limit leverage_x=1.0    # tighten a risk limit and watch the veto fire
pytest tests -q
```

Live mode needs one of: `OPENAI_API_KEY` (optional `OPENAI_CHAT_MODEL_ID`); `AZURE_OPENAI_API_KEY`,
`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME`; or `GEMINI_API_KEY` (optional `GEMINI_MODEL`,
for example `gemini-3.5-flash-lite`). On a free tier, `DEBATE_MAX_CONCURRENCY=1` (the default) avoids rate limits. With `--backend auto` (default) the system uses
Semantic Kernel if a key is set and otherwise runs offline. Reports land in `output/`.

## How the rules are enforced (in code, not in prompts)

- **Debaters must cite.** `citations.py` drops any claim with no citation, an unknown fact ID, or a number
  that does not appear in the cited fact. Drops are written to the audit log.
- **The Judge may only use cited facts.** It is only ever shown facts cited in the debate, gets no search
  tool, and any other fact ID in its output or rationale is stripped and logged.
- **Scoring is evidence-based.** Citing a bearish fact earns the Bull nothing; a rebuttal discounts the
  opponent's claim by how much counter-evidence outweighs it.
- **The Risk veto is rule-based.** It scans the whole fact sheet (including risks neither debater raised) against
  the limits in `config.py` (leverage, customer concentration, runway, drawdown, hard-flag disclosures, composite
  score). A model only writes the narrative and cannot remove a veto. By default the veto overrides a BUY; set
  `veto_applies_to` to include AVOID if you want it to override that too.
- **Risk is independent of the Judge.** Risk limits do not change how the Research Agent classifies facts.

## Demo results (offline backend)

| Asset | Judge | Risk Agent | Final |
|---|---|---|---|
| Helios Renewables | BUY | no veto | **BUY** |
| Verdant Mobility | BUY | VETO: 56% customer concentration vs 50% limit | **HOLD** |
| Ironbridge Retail | AVOID | no veto | **AVOID** |

## Limits worth knowing

- Offline mode uses rule-based reasoning (`heuristics.py`), not a language model. It makes the architecture
  reproducible and testable; it is not an investment model. The Semantic Kernel path is tested against a
  scripted SK service (`tests/fake_sk.py`) and was also run live against a Gemini model on all demo assets.
- Verdicts depend on tunable weights and margins in `config.py`; changing them can flip close calls.
- Not financial advice. The companies and figures are invented.
