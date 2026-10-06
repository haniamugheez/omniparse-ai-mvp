"""What a parsed document looks like once every agent has had its turn.

Deliberately plain dataclasses rather than a validation library: the point of
this file is that anyone reading the project can see, in thirty seconds, every
field the pipeline produces and every field the audit engine can reason about.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date


@dataclass
class LineItem:
    description: str = ""
    quantity: float = 0.0
    unit_price: float = 0.0
    amount: float = 0.0

    @property
    def expected_amount(self) -> float:
        return round(self.quantity * self.unit_price, 2)


@dataclass
class Invoice:
    """One document, extracted."""

    invoice_id: str = ""
    vendor: str = ""
    invoice_date: date | None = None
    currency: str = "PKR"
    subtotal: float = 0.0
    tax: float = 0.0
    total: float = 0.0
    line_items: list = field(default_factory=list)
    source_file: str = ""
    source_kind: str = ""          # pdf / image / table
    extraction_method: str = ""    # rules / llm / table
    confidence: float = 0.0
    raw_text: str = ""

    def to_json(self) -> dict:
        out = asdict(self)
        out["invoice_date"] = (self.invoice_date.isoformat()
                               if self.invoice_date else None)
        out.pop("raw_text", None)
        return out

    @property
    def line_total(self) -> float:
        return round(sum(li.amount for li in self.line_items), 2)


SEVERITIES = ("CRITICAL", "HIGH", "MEDIUM", "LOW")


@dataclass
class Anomaly:
    """One finding, in the form an auditor can act on.

    `evidence` is always the arithmetic — the numbers that produced the flag —
    because a finding a human cannot check is a finding they will not trust.
    """

    code: str
    severity: str
    title: str
    evidence: str
    recommendation: str

    @property
    def rank(self) -> int:
        return SEVERITIES.index(self.severity)


@dataclass
class AuditResult:
    invoice: Invoice
    anomalies: list = field(default_factory=list)
    checks_run: int = 0

    @property
    def risk_score(self) -> int:
        """0 clean, 100 unusable. Weighted so one CRITICAL outranks three LOWs."""
        weight = {"CRITICAL": 40, "HIGH": 22, "MEDIUM": 10, "LOW": 4}
        return min(100, sum(weight[a.severity] for a in self.anomalies))

    @property
    def verdict(self) -> str:
        score = self.risk_score
        if score == 0:
            return "CLEAR"
        if score < 20:
            return "REVIEW"
        return "HOLD"
