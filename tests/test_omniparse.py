"""Tests for the parts where being wrong costs money.

Every assertion below is arithmetic a person can do on paper, which is the
point: an audit engine nobody can check is not an audit engine.
"""

import pathlib
import sys
from datetime import date

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from omniparse import extract, ingest          # noqa: E402
from omniparse.audit import audit              # noqa: E402
from omniparse.pipeline import process         # noqa: E402
from omniparse.schema import Invoice, LineItem  # noqa: E402
from omniparse.store import Baseline           # noqa: E402

SAMPLES = pathlib.Path(__file__).resolve().parent.parent / "sample_data"


@pytest.fixture(scope="module")
def baseline():
    return Baseline()


def _invoice(**kwargs) -> Invoice:
    base = dict(invoice_id="INV-TEST-1", vendor="Test Vendor Ltd",
                invoice_date=date(2026, 9, 1), total=1170.0, tax=170.0,
                confidence=1.0, source_file="test.pdf",
                line_items=[LineItem("Widget", 10, 100.0, 1000.0)])
    base.update(kwargs)
    return Invoice(**base)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,expected", [
    ("a.pdf", "pdf"), ("scan.PNG", "image"), ("batch.csv", "table"),
    ("book.xlsx", "table"), ("notes.docx", ""),
])
def test_the_router_knows_which_engine_can_read_a_file(name, expected):
    assert ingest.route(name) == expected


def test_an_unreadable_file_is_a_message_not_a_crash():
    run = process(b"nonsense", "notes.docx")
    assert not run.ok
    assert "unsupported" in run.error.lower()


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def test_grand_total_is_not_confused_with_sub_total():
    """The bug that makes every clean invoice look like it fails its own
    arithmetic: 'Total' matches inside 'Sub-Total'."""
    text = ("Invoice No: INV-9001\nInvoice Date: 2026-09-28\n"
            "Vendor: Acme Ltd\n"
            "Sub-Total: 25,575.00\nSales Tax (17%): 4,347.75\n"
            "Grand Total: 29,922.75\n")
    invoice = extract.from_text(text, "x.pdf", "pdf", "test")
    assert invoice.subtotal == 25575.00
    assert invoice.tax == 4347.75
    assert invoice.total == 29922.75


def test_a_real_pdf_is_read_field_for_field():
    data = (SAMPLES / "invoice_clean.pdf").read_bytes()
    payload = ingest.ingest(data, "invoice_clean.pdf")
    invoice = extract.from_text(payload["text"], "invoice_clean.pdf", "pdf",
                                payload["method"])
    assert invoice.invoice_id == "INV-2026-5101"
    assert invoice.vendor.startswith("Indus Office Supplies")
    assert invoice.invoice_date == date(2026, 9, 28)
    assert invoice.total == 29922.75
    assert len(invoice.line_items) == 2
    assert invoice.line_items[0].amount == 17200.00


def test_a_spreadsheet_of_many_invoices_becomes_many_records():
    data = (SAMPLES / "new_invoices.csv").read_bytes()
    frame, method = ingest.read_table(data, "new_invoices.csv")
    invoices = extract.from_frame(frame, "new_invoices.csv", method)
    assert len(invoices) == 6
    assert {i.invoice_id for i in invoices} >= {"INV-2026-5001", "INV-2026-5003"}


def test_confidence_reflects_what_was_actually_found():
    poor = extract.from_text("some free text with no invoice in it at all",
                             "x.pdf", "pdf", "test")
    assert poor.confidence < 0.3
    good = extract.from_text(
        "Invoice No: INV-1\nInvoice Date: 2026-01-05\nVendor: Acme Ltd\n"
        "Grand Total: 100.00\nWidget   1   100.00   100.00\n",
        "x.pdf", "pdf", "test")
    assert good.confidence > 0.8


# ---------------------------------------------------------------------------
# The audit engine
# ---------------------------------------------------------------------------
def test_a_clean_invoice_raises_nothing(baseline):
    """The one that matters most. A tool that cries wolf gets switched off."""
    data = (SAMPLES / "invoice_clean.pdf").read_bytes()
    run = process(data, "invoice_clean.pdf", baseline)
    assert run.ok
    assert run.results[0].anomalies == []
    assert run.results[0].verdict == "CLEAR"


def test_a_line_that_does_not_multiply_is_caught(baseline):
    invoice = _invoice(line_items=[LineItem("Widget", 3, 12500.0, 39500.0)],
                       total=46215.0, tax=6715.0)
    codes = [a.code for a in audit(invoice, baseline).anomalies]
    assert "LINE_MATH" in codes


def test_a_total_above_its_own_lines_is_critical(baseline):
    invoice = _invoice(line_items=[LineItem("Widget", 2, 75600.0, 151200.0)],
                       tax=25704.0, total=198904.0)
    found = [a for a in audit(invoice, baseline).anomalies
             if a.code == "TOTAL_MATH"]
    assert found and found[0].severity == "CRITICAL"
    assert "22,000.00" in found[0].evidence


def test_an_invoice_number_already_in_the_ledger_is_critical(baseline):
    known = next(iter(baseline.invoice_ids))
    invoice = _invoice(invoice_id=known)
    found = [a for a in audit(invoice, baseline).anomalies
             if a.code == "DUPLICATE_ID"]
    assert found and found[0].severity == "CRITICAL"


def test_a_future_date_is_flagged(baseline):
    invoice = _invoice(invoice_date=date(2026, 12, 15))
    codes = [a.code for a in audit(invoice, baseline,
                                   today=date(2026, 10, 3)).anomalies]
    assert "FUTURE_DATE" in codes


def test_an_amount_just_under_an_approval_limit_is_flagged(baseline):
    invoice = _invoice(total=98_500.0, tax=0.0,
                       line_items=[LineItem("Retainer", 1, 98_500.0, 98_500.0)])
    codes = [a.code for a in audit(invoice, baseline).anomalies]
    assert "THRESHOLD_PROXIMITY" in codes


def test_an_impossible_tax_rate_is_flagged(baseline):
    invoice = _invoice(line_items=[LineItem("Lunch", 60, 470.0, 28_200.0)],
                       subtotal=28_200.0, tax=8_460.0, total=36_660.0)
    codes = [a.code for a in audit(invoice, baseline).anomalies]
    assert "TAX_RATE" in codes


def test_a_standard_tax_rate_is_not_flagged(baseline):
    invoice = _invoice(line_items=[LineItem("Lunch", 60, 470.0, 28_200.0)],
                       subtotal=28_200.0, tax=4_794.0, total=32_994.0)
    codes = [a.code for a in audit(invoice, baseline).anomalies]
    assert "TAX_RATE" not in codes


def test_a_vendor_never_seen_before_is_flagged(baseline):
    invoice = _invoice(vendor="Falcon Trading Company", total=5000.0, tax=0.0,
                       line_items=[LineItem("Thing", 1, 5000.0, 5000.0)])
    codes = [a.code for a in audit(invoice, baseline).anomalies]
    assert "NEW_VENDOR" in codes


def test_missing_fields_are_reported_as_an_audit_trail_gap(baseline):
    invoice = _invoice(invoice_id="", vendor="", confidence=0.2)
    codes = [a.code for a in audit(invoice, baseline).anomalies]
    assert "AUDIT_TRAIL_GAP" in codes
    assert "LOW_CONFIDENCE" in codes


def test_risk_score_ranks_a_critical_above_several_minor_findings():
    critical = _invoice(line_items=[LineItem("W", 1, 10.0, 10.0)],
                        tax=0.0, total=9999.0)
    result = audit(critical, Baseline(pd.DataFrame(
        columns=["invoice_id", "vendor", "invoice_date", "description",
                 "quantity", "unit_price", "amount", "total"])))
    assert result.verdict == "HOLD"
    assert result.risk_score >= 40


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
def test_the_sample_batch_produces_the_findings_it_was_built_to_produce(baseline):
    data = (SAMPLES / "new_invoices.csv").read_bytes()
    run = process(data, "new_invoices.csv", baseline)
    found = {a.code for r in run.results for a in r.anomalies}
    assert {"LINE_MATH", "TOTAL_MATH", "DUPLICATE_ID", "NEW_VENDOR",
            "THRESHOLD_PROXIMITY", "FUTURE_DATE", "TAX_RATE"} <= found
    # and the one clean invoice in the batch stays clean
    clean = [r for r in run.results if r.invoice.invoice_id == "INV-2026-5001"]
    assert clean and clean[0].verdict == "CLEAR"


def test_every_agent_reports_in_the_trace(baseline):
    run = process((SAMPLES / "invoice_clean.pdf").read_bytes(),
                  "invoice_clean.pdf", baseline)
    agents = [step.agent for step in run.trace]
    assert agents == ["Document Ingestion & Routing", "Structured Extraction",
                      "Validation & Audit", "Anomaly Discovery Engine"]
