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



@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """No test may call out to the internet.

    Two of these tests quietly started making real API calls the moment a
    GROQ_API_KEY appeared in .streamlit/secrets.toml — they passed on a
    machine with no key and failed on one with it, which is the worst kind of
    test there is. A suite has to say the same thing on every machine, so the
    key is taken away here and the tests that need a model stub it instead.
    """
    monkeypatch.setattr(extract, "_api_key", lambda: "")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OMNIPARSE_VISION_MODEL", raising=False)
    extract._vision_model_cache.clear()


@pytest.fixture
def baseline():
    return Baseline()


def _no_ocr(image):
    """Stand in for a machine with no Tesseract installed."""
    raise RuntimeError("OCR engine (Tesseract) is not installed in this "
                       "environment.")


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


# ===========================================================================
# A real invoice, in the shape real invoices come in
#
# A courier reconciliation run, exported to Excel. Column A is empty down the
# whole sheet. The vendor is on row 8, the invoice number on row 10 inside one
# cell as "Customer Invoice Number: INV-… # 010-…", the billing date on row 12
# with its label in one cell and its value in the next, and the column
# headings do not start until row 14. Each shipment is three rows — a freight
# charge, a fuel surcharge and a "Total" — with the shipment's details written
# only on the first, and amounts written as "$ -1.29".
#
# Read with the obvious assumptions, this file came out as a record with no
# invoice number, no vendor, no date, no total and no lines: 0% confidence.
# That is what these tests are here to stop happening again.
# ===========================================================================
FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"
COURIER = FIXTURES / "courier_reconciliation.xlsx"


@pytest.fixture
def courier(baseline):
    return process(COURIER.read_bytes(), "courier_reconciliation.xlsx", baseline)


def test_a_header_buried_on_row_fourteen_is_still_found(courier):
    assert courier.ok, courier.error
    assert len(courier.results) == 1


def test_the_invoice_number_is_dug_out_of_a_sentence(courier):
    """It is not in a column. It is inside one cell, after a label."""
    assert "INV-ADJ" in courier.results[0].invoice.invoice_id


def test_the_vendor_is_found_without_a_label(courier):
    """Nothing says "Vendor:". The name just sits there on row 8 — and the
    bill-to company, its street and its city sit above it as decoys."""
    vendor = courier.results[0].invoice.vendor
    assert vendor == "Global Express PK"
    assert "KINGSTON" not in vendor and "OAK BROOK" not in vendor


def test_a_label_in_one_cell_and_its_date_in_the_next(courier):
    assert courier.results[0].invoice.invoice_date == date(2026, 10, 5)


def test_every_charge_line_is_read_and_subtotals_are_not_counted_twice(courier):
    """Four charges, two group "Total" rows, one "GRAND TOTAL:". Only the four
    charges are line items — counting the sums as charges doubles the
    invoice."""
    invoice = courier.results[0].invoice
    assert len(invoice.line_items) == 4
    assert {li.description for li in invoice.line_items} == {
        "Freight Charges", "Fuel Surcharge"}


def test_the_grand_total_wins_over_the_group_sums(courier):
    """"GRAND TOTAL:" with a colon used to slip past as an ordinary charge."""
    assert courier.results[0].invoice.total == 0.86


def test_the_arithmetic_agrees_so_nothing_is_falsely_flagged(courier):
    """-1.29 - 0.35 + 2.00 + 0.50 = 0.86. The document is sound, and the audit
    must say so rather than inventing a failure."""
    invoice = courier.results[0].invoice
    assert invoice.line_total == 0.86
    codes = {a.code for a in courier.results[0].anomalies}
    assert "TOTAL_MATH" not in codes
    assert "AUDIT_TRAIL_GAP" not in codes
    assert "LOW_CONFIDENCE" not in codes


def test_a_document_this_complete_reads_at_full_confidence(courier):
    assert courier.results[0].invoice.confidence >= 0.95


@pytest.mark.parametrize("written,expected", [
    ("$ -1.29", -1.29), ("$ 1,234.56", 1234.56), ("(500.00)", -500.0),
    ("Rs 2,500/-", 2500.0), ("  ", 0.0), ("—", 0.0), (None, 0.0), (42, 42.0),
])
def test_money_is_read_in_the_shapes_money_is_written(written, expected):
    from omniparse.extract import _amount
    assert _amount(written) == expected


@pytest.mark.parametrize("label,is_grand,is_sub", [
    ("GRAND TOTAL:", True, False), ("Grand Total", True, False),
    ("Total", False, True), ("total ", False, True),
    ("Sub-Total", False, True), ("Freight Charges", False, False),
])
def test_a_trailing_colon_does_not_hide_a_total(label, is_grand, is_sub):
    from omniparse.extract import _is_grand_total, _is_subtotal
    assert _is_grand_total(label) is is_grand
    assert _is_subtotal(label) is is_sub


# ===========================================================================
# Any invoice, not this invoice
#
# The point of the extractor is that it reads whatever arrives, so a fixture
# built from one real file proves very little on its own. These six are
# deliberately unalike — different header rows, different column wording,
# different currencies and conventions, one invoice or several per sheet,
# brackets for negatives, a line of prose above the table. None of them looks
# like the courier file that prompted the rewrite.
#
# Each one is checked for the four things a record is useless without: who
# billed, which document, when, and what it adds up to.
# ===========================================================================
SHAPES = [
    # file, invoice_id, vendor, date, lines, total
    ("A_plain.xlsx",  "SI-900",     "Nova Supplies Ltd",        date(2026, 3, 11), 2, 5500.00),
    ("B_desi.xlsx",   "BT-2026-77", "KARACHI TRADERS",          date(2026, 3, 11), 2, 31000.00),
    ("C_us.xlsx",     "BR-88421",   "Blue Ridge Logistics LLC", date(2026, 3, 11), 2, 720.00),
    ("F_decoy.csv",   "SV-77",      "Crescent IT Services",     date(2026, 5, 20), 2, 37000.00),
]


@pytest.mark.parametrize("name,invoice_id,vendor,when,lines,total", SHAPES)
def test_an_invoice_of_any_shape_is_read(baseline, name, invoice_id, vendor,
                                         when, lines, total):
    run = process((FIXTURES / name).read_bytes(), name, baseline)
    assert run.ok, run.error
    invoice = run.results[0].invoice
    assert invoice.invoice_id == invoice_id
    assert invoice.vendor == vendor
    assert invoice.invoice_date == when
    assert len(invoice.line_items) == lines
    assert invoice.total == total
    assert invoice.confidence >= 0.9


def test_several_invoices_in_one_sheet_come_out_separately(baseline):
    """Grouped by an invoice-number column, with the vendor and date written
    once at the top of each group and left blank beneath."""
    run = process((FIXTURES / "D_multi.xlsx").read_bytes(), "D_multi.xlsx",
                  baseline)
    assert len(run.results) == 2
    first, second = (r.invoice for r in run.results)
    assert (first.invoice_id, first.vendor) == ("MX-1", "Alpha Foods")
    assert (second.invoice_id, second.vendor) == ("MX-2", "Beta Paper")
    assert len(first.line_items) == 2        # the second line inherits both


def test_a_tax_written_against_every_line_is_charged_once(baseline):
    """1,020 appears on both lines of MX-1. It is the invoice's tax, not each
    line's — summing it charges the customer twice."""
    run = process((FIXTURES / "D_multi.xlsx").read_bytes(), "D_multi.xlsx",
                  baseline)
    assert run.results[0].invoice.tax == 1020.00


def test_brackets_mean_a_credit_not_a_charge(baseline):
    """An accountant writes a negative as (1,250.00)."""
    run = process((FIXTURES / "E_credit.xlsx").read_bytes(), "E_credit.xlsx",
                  baseline)
    invoice = run.results[0].invoice
    assert invoice.invoice_id == "ADJ-5512"
    assert invoice.line_items[0].amount == -1250.00
    assert invoice.total == -950.00


def test_a_missing_vendor_is_reported_honestly_not_invented(baseline):
    """E_credit names no supplier anywhere. The right answer is to say so and
    drop the confidence — not to put the document's own title in the field."""
    run = process((FIXTURES / "E_credit.xlsx").read_bytes(), "E_credit.xlsx",
                  baseline)
    invoice = run.results[0].invoice
    assert invoice.vendor == ""
    assert invoice.confidence < 0.9
    assert "AUDIT_TRAIL_GAP" in {a.code for a in run.results[0].anomalies}


def test_a_blank_cell_never_becomes_the_word_nan(baseline):
    """str() on an empty spreadsheet cell gives "nan" — three letters, not an
    address, not a label, which is exactly the shape a vendor name has. One
    invoice came back billed by "nan"."""
    for name, *_ in SHAPES:
        run = process((FIXTURES / name).read_bytes(), name, baseline)
        for result in run.results:
            assert "nan" not in result.invoice.vendor.lower()
            assert "nan" not in result.invoice.invoice_id.lower()


def test_a_ragged_csv_does_not_defeat_the_reader(baseline):
    """One cell of prose, a blank line, then a seven-column table. pandas
    decides the file has one column and refuses the rest."""
    run = process((FIXTURES / "F_decoy.csv").read_bytes(), "F_decoy.csv",
                  baseline)
    assert run.ok, run.error
    assert len(run.results[0].invoice.line_items) == 2


# ===========================================================================
# The last resort, and only the last resort
#
# Rules run first, every time. A model is reached when they genuinely came up
# short — a scan with no OCR engine behind it, a sheet with no header anywhere
# in it. Two things have to stay true: the model must not take over documents
# the rules can already read, and with no key configured the record must say
# plainly that it could not be read rather than quietly inventing one.
# ===========================================================================
def test_a_readable_document_never_reaches_the_model(baseline, monkeypatch):
    """If the rules can read it, nothing is sent anywhere and nothing is paid
    for."""
    called = []
    monkeypatch.setattr(extract, "from_image",
                        lambda *a, **k: called.append(a) or Invoice())

    run = process((SAMPLES / "invoice_clean.pdf").read_bytes(),
                  "invoice_clean.pdf", baseline)
    assert not called
    assert run.results[0].invoice.extraction_method.startswith("rules")


def test_no_ocr_engine_is_a_message_not_a_crash(baseline, monkeypatch):
    """Her Windows has no Tesseract. The file must still go through the
    pipeline and come out saying why it is empty."""
    monkeypatch.setattr(ingest, "_ocr", _no_ocr)

    run = process((SAMPLES / "invoice_scan.png").read_bytes(), "scan.png",
                  baseline)
    assert run.ok, run.error
    invoice = run.results[0].invoice
    assert "no OCR engine" in invoice.extraction_method
    assert invoice.confidence == 0


def test_an_unreadable_scan_is_flagged_rather_than_guessed(baseline, monkeypatch):
    """The worst outcome would be a confident record nobody can check."""
    monkeypatch.setattr(ingest, "_ocr", _no_ocr)

    run = process((SAMPLES / "invoice_scan.png").read_bytes(), "scan.png",
                  baseline)
    codes = {a.code for a in run.results[0].anomalies}
    assert "AUDIT_TRAIL_GAP" in codes
    assert "LOW_CONFIDENCE" in codes
    assert run.results[0].verdict == "HOLD"


def test_the_trace_says_which_engine_read_the_document(baseline):
    """"The agent read it" and "a model guessed at it" are not the same
    claim, and an audit tool has to be able to tell them apart."""
    run = process((SAMPLES / "invoice_clean.pdf").read_bytes(),
                  "invoice_clean.pdf", baseline)
    extraction = next(s for s in run.trace if s.agent == "Structured Extraction")
    assert "rules" in extraction.detail


def test_the_vision_reader_stays_quiet_without_a_key():
    """No key, no call, no invented record."""
    invoice = extract.from_image(b"not really a png", "x.png")
    assert invoice.confidence == 0
    assert invoice.invoice_id == ""
    assert "no vision key" in invoice.extraction_method


def test_a_sheet_is_rendered_for_a_model_to_read():
    """The fallback for a sheet with no header: hand the model the lines."""
    frame, _ = ingest.read_table(
        (FIXTURES / "F_decoy.csv").read_bytes(), "F_decoy.csv")
    text = extract.sheet_as_text(frame)
    assert "Crescent IT Services" in text
    assert "nan" not in text.lower()


# ===========================================================================
# A model name is not a constant
#
# This code first shipped pointing at llama-4-scout, which Groq retired, and
# every scan came back "vision failed (404)". A dead string in a config file
# took down a working feature. The account's own model list decides now.
# ===========================================================================
def test_the_vision_model_is_chosen_not_hard_coded(monkeypatch):
    from omniparse import extract as ex

    ex._vision_model_cache.clear()
    monkeypatch.setattr(ex, "_models_on_this_account",
                        lambda: {"qwen/qwen3.8-27b", "llama-3.1-8b-instant"})
    monkeypatch.setattr(ex, "_api_key", lambda: "k")
    assert ex.vision_model() == "qwen/qwen3.8-27b"


def test_a_retired_first_choice_falls_through_to_the_next(monkeypatch):
    from omniparse import extract as ex

    ex._vision_model_cache.clear()
    monkeypatch.setattr(ex, "_api_key", lambda: "k")
    monkeypatch.setattr(ex, "_models_on_this_account",
                        lambda: {"meta-llama/llama-4-maverick-17b-128e-instruct"})
    assert ex.vision_model() == "meta-llama/llama-4-maverick-17b-128e-instruct"


def test_an_unreachable_model_list_does_not_stop_us_trying(monkeypatch):
    """Better a 404 we can report than a refusal to even attempt the page."""
    from omniparse import extract as ex

    ex._vision_model_cache.clear()
    monkeypatch.setattr(ex, "_api_key", lambda: "k")
    monkeypatch.setattr(ex, "_models_on_this_account", lambda: set())
    assert ex.vision_model() == ex.VISION_CANDIDATES[0]


def test_an_explicit_override_always_wins(monkeypatch):
    from omniparse import extract as ex

    ex._vision_model_cache.clear()
    monkeypatch.setenv("OMNIPARSE_VISION_MODEL", "some/other-model")
    assert ex.vision_model() == "some/other-model"
