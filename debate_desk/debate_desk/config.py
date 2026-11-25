"""Central configuration: every threshold the system uses is stated here, in one place."""
from __future__ import annotations

from dataclasses import dataclass, field

# How much each theme matters when the Judge weighs evidence (0..1).
THEME_WEIGHTS: dict[str, float] = {
    "growth": 1.0,
    "profitability": 1.0,
    "balance_sheet": 1.0,
    "regulation_legal": 0.9,
    "valuation": 0.8,
    "competition": 0.8,
    "customers": 0.8,
    "governance": 0.7,
    "market_risk": 0.7,
    "operations": 0.6,
}
THEMES = list(THEME_WEIGHTS)


@dataclass(frozen=True)
class RiskLimit:
    """One stated risk threshold. kind='max' breaches when observed > limit, 'min' when observed < limit."""
    key: str
    label: str
    limit: float
    kind: str  # "max" | "min"
    unit: str


@dataclass
class Config:
    # --- debate mechanics ---
    max_claims: int = 5              # claims per debater in the opening round (3 is a knife-edge for some assets)
    retrieval_top_k: int = 6         # chunks returned per RAG query
    # --- Judge ---
    buy_margin: float = 0.15         # normalised margin >= this  -> BUY
    avoid_margin: float = -0.15      # normalised margin <= this  -> AVOID, otherwise HOLD
    rebuttal_max_discount: float = 0.5   # a fully effective rebuttal removes up to 50% of a claim's weight
    rebuttal_fact_credit: float = 0.5    # new facts introduced only in a rebuttal count at half weight
    # --- Risk Agent veto ---
    risk_veto_score: int = 60        # composite score (0-100) at/above which the Risk Agent vetoes
    veto_applies_to: frozenset[str] = frozenset({"BUY"})  # verdicts the veto can override -> HOLD
    limits: tuple[RiskLimit, ...] = field(default_factory=lambda: (
        RiskLimit("leverage_x", "Net debt / EBITDA", 4.0, "max", "x"),
        RiskLimit("customer_concentration_pct", "Largest-customer revenue share", 50.0, "max", "%"),
        RiskLimit("runway_months", "Cash runway", 12.0, "min", "months"),
        RiskLimit("drawdown_pct", "Max share-price drawdown", 40.0, "max", "%"),
    ))
    near_limit_band: float = 0.15    # within 15% of a limit counts as "near"
    # Disclosures that are a veto on their own (matched with a negation guard).
    hard_flags: tuple[str, ...] = (
        "going concern", "covenant breach", "event of default", "material weakness",
        "wells notice", "sec enforcement", "delisting notice", "restatement", "bankruptcy",
    )
    # Composite score points
    points_breach: int = 30
    points_near: int = 10
    points_flag: int = 40
    points_strong_bearish: int = 5   # per strongly bearish fact (strength >= 2), capped
    cap_strong_bearish: int = 20
