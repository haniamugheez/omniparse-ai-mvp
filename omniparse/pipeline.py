"""The agent graph — who runs, in what order, and what each one handed on.

The architecture diagram draws this as LangGraph. What LangGraph buys you is
state passed between nodes and a record of the path taken; at four nodes with
one branch, that is this file, with no framework to install and nothing to go
wrong on a free hosting tier.

The trace is not decoration. A document-processing system that cannot say
*which* agent produced a number is not auditable, and an audit tool that is
not itself auditable is a toy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from omniparse import extract, ingest
from omniparse.audit import audit
from omniparse.store import Baseline


@dataclass
class Step:
    agent: str
    detail: str
    ms: int
    ok: bool = True


@dataclass
class Run:
    filename: str
    results: list = field(default_factory=list)   # list[AuditResult]
    trace: list = field(default_factory=list)     # list[Step]
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def anomaly_count(self) -> int:
        return sum(len(r.anomalies) for r in self.results)


class _Clock:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = int((time.perf_counter() - self.start) * 1000)


def process(data: bytes, filename: str, baseline: Baseline | None = None) -> Run:
    """One file, all four agents."""
    run = Run(filename=filename)
    baseline = baseline or Baseline()

    # Agent 1 — ingestion & routing
    try:
        with _Clock() as clock:
            payload = ingest.ingest(data, filename)
        run.trace.append(Step(
            "Document Ingestion & Routing",
            f"routed as {payload['kind'].upper()} · read by {payload['method']}",
            clock.ms))
    except Exception as exc:   # noqa: BLE001 — a bad file is a message, not a crash
        run.error = str(exc)
        run.trace.append(Step("Document Ingestion & Routing", str(exc), 0, False))
        return run

    # Agent 2 — structured extraction
    #
    # Rules first, always. The model is only reached when they genuinely came
    # up short — a scan with no OCR engine behind it, or a sheet laid out so
    # unusually that nothing in it looks like a header. Which road was taken
    # is recorded, because "the agent read it" and "a model guessed at it"
    # are not the same claim.
    try:
        with _Clock() as clock:
            if payload["kind"] == "table":
                invoices = extract.from_frame(payload["frame"], filename,
                                              payload["method"])
                if _nothing_useful(invoices) and payload.get("frame") is not None:
                    rescued = _rescue_table(payload["frame"], filename)
                    if rescued is not None:
                        invoices = [rescued]
            elif (payload.get("text") or "").strip():
                invoices = [extract.from_text(payload["text"], filename,
                                              payload["kind"], payload["method"])]
                if _nothing_useful(invoices) and payload.get("image"):
                    looked = extract.from_image(payload["image"], filename,
                                                payload["kind"])
                    if looked.confidence > invoices[0].confidence:
                        invoices = [looked]
            else:
                invoices = [extract.from_image(payload.get("image"), filename,
                                               payload["kind"])]

        placed = ", ".join(f"{i.invoice_id or '(no id)'} "
                           f"{i.confidence:.0%}" for i in invoices[:4])
        run.trace.append(Step(
            "Structured Extraction",
            f"{len(invoices)} record(s) via {invoices[0].extraction_method} · "
            f"{placed}" + (" …" if len(invoices) > 4 else ""),
            clock.ms,
            ok=not _nothing_useful(invoices)))
    except Exception as exc:   # noqa: BLE001
        run.error = f"extraction failed: {exc}"
        run.trace.append(Step("Structured Extraction", str(exc), 0, False))
        return run

    # Agents 3 & 4 — validation, then the anomaly engine
    with _Clock() as clock:
        run.results = [audit(invoice, baseline) for invoice in invoices]
    checks = run.results[0].checks_run if run.results else 0
    run.trace.append(Step(
        "Validation & Audit",
        f"{checks} checks against {len(baseline.invoice_level())} historical "
        f"invoices", clock.ms))
    run.trace.append(Step(
        "Anomaly Discovery Engine",
        f"{run.anomaly_count} anomaly(ies) across {len(run.results)} record(s)",
        0))
    return run

def _nothing_useful(invoices: list) -> bool:
    """Did the reading actually produce a record worth auditing?"""
    if not invoices:
        return True
    best = max(invoices, key=lambda i: i.confidence)
    return not best.line_items and not best.total


def _rescue_table(frame, filename: str):
    """A sheet no header could be found in, read by a model instead."""
    from omniparse import extract as _extract

    if not _extract.vision_available():
        return None
    text = _extract.sheet_as_text(frame)
    if not text.strip():
        return None
    invoice = _extract.from_text(text, filename, "table", "llm-from-sheet")
    invoice.extraction_method = "llm (sheet had no recognisable header)"
    return invoice

