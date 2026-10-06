"""Agents 3 and 4 — Validation & Audit, and the Anomaly Discovery Engine.

Thirteen checks. Every one of them is arithmetic or a comparison against the
historical baseline, and every finding carries the numbers that produced it,
because an auditor who cannot reproduce a flag will stop reading the report.

Nothing here is a language model's opinion. A model can read a document; it
cannot be the control that decides whether money leaves the company.
"""

from __future__ import annotations

from datetime import date, timedelta

from omniparse.schema import Anomaly, AuditResult, Invoice
from omniparse.store import Baseline

# Tax rates a Pakistani invoice plausibly carries. An amount that is clearly
# "tax" but matches none of these is worth a human glance, not a refusal.
PLAUSIBLE_TAX_RATES = (0.0, 0.05, 0.13, 0.15, 0.16, 0.17, 0.18)
TAX_TOLERANCE = 0.01

# Approval ceilings. An invoice landing just under one, again and again, is the
# oldest trick in procurement.
APPROVAL_THRESHOLDS = (50_000, 100_000, 250_000, 500_000, 1_000_000)
JUST_UNDER = 0.02          # within 2% below a ceiling

VARIANCE_SIGMA = 2.5       # how far above a vendor's own average is unusual
PRICE_TOLERANCE = 0.25     # 25% above the historical unit price
QTY_TOLERANCE = 3.0        # three times the usual quantity
LOW_CONFIDENCE = 0.6
STALE_DAYS = 365


def _pct(part: float, whole: float) -> str:
    return f"{(part / whole * 100):.1f}%" if whole else "n/a"


def audit(invoice: Invoice, baseline: Baseline | None = None,
          today: date | None = None) -> AuditResult:
    baseline = baseline or Baseline()
    today = today or date.today()
    found: list = []
    checks = 0

    # -- 1. every line's own arithmetic ------------------------------------
    checks += 1
    for item in invoice.line_items:
        if item.quantity and item.unit_price:
            gap = round(item.amount - item.expected_amount, 2)
            if abs(gap) > 1.0:
                found.append(Anomaly(
                    "LINE_MATH", "HIGH",
                    f"Line does not multiply: {item.description}",
                    f"{item.quantity} x {item.unit_price:,.2f} = "
                    f"{item.expected_amount:,.2f}, but the invoice says "
                    f"{item.amount:,.2f} (off by {gap:,.2f})",
                    "Ask the vendor for a corrected line before payment."))

    # -- 2. the invoice adds up --------------------------------------------
    checks += 1
    if invoice.line_items and invoice.total:
        expected = round(invoice.line_total + invoice.tax, 2)
        gap = round(invoice.total - expected, 2)
        if abs(gap) > 1.0:
            found.append(Anomaly(
                "TOTAL_MATH", "CRITICAL",
                "Invoice total does not match its own lines",
                f"lines {invoice.line_total:,.2f} + tax {invoice.tax:,.2f} = "
                f"{expected:,.2f}, invoice says {invoice.total:,.2f} "
                f"(off by {gap:,.2f})",
                "Hold payment. A total above its own lines is the single most "
                "common overbilling pattern."))

    # -- 3. an invoice number we have already paid --------------------------
    checks += 1
    if baseline.seen(invoice.invoice_id):
        found.append(Anomaly(
            "DUPLICATE_ID", "CRITICAL",
            f"Invoice number {invoice.invoice_id} already exists",
            f"{invoice.invoice_id} is already recorded in the ledger",
            "Do not pay. Confirm with the vendor whether this is a resend."))

    # -- 4. same vendor, same amount ---------------------------------------
    checks += 1
    twin = baseline.near_duplicate(invoice.vendor, invoice.total)
    if twin is not None and not baseline.seen(invoice.invoice_id):
        found.append(Anomaly(
            "DUPLICATE_AMOUNT", "HIGH",
            "An invoice for this exact amount from this vendor already exists",
            f"{invoice.vendor} — {invoice.currency} {invoice.total:,.2f} "
            f"also appears as {twin['invoice_id']}",
            "Check whether the same work has been billed twice."))

    # -- 5. far above what this vendor usually bills ------------------------
    checks += 1
    stats = baseline.vendor_stats(invoice.vendor)
    if stats.known and stats.count >= 3 and stats.std > 0 and invoice.total:
        z = (invoice.total - stats.mean) / stats.std
        if z >= VARIANCE_SIGMA:
            found.append(Anomaly(
                "HIGH_VALUE_VARIANCE", "HIGH",
                f"Unusually large for {invoice.vendor}",
                f"{invoice.currency} {invoice.total:,.2f} against an average of "
                f"{stats.mean:,.2f} over {stats.count} invoices "
                f"(previous highest {stats.maximum:,.2f}; {z:.1f} standard "
                "deviations above the mean)",
                "Confirm the scope of work before approving."))

    # -- 6. a vendor with no history ---------------------------------------
    checks += 1
    if invoice.vendor and not stats.known:
        found.append(Anomaly(
            "NEW_VENDOR", "MEDIUM",
            f"First invoice from {invoice.vendor}",
            "No previous invoices from this vendor in the ledger",
            "Verify bank details independently before the first payment."))

    # -- 7. paying more than last time for the same thing -------------------
    checks += 1
    for item in invoice.line_items:
        was, seen = baseline.item_price(item.description)
        if seen >= 2 and was > 0 and item.unit_price > was * (1 + PRICE_TOLERANCE):
            rise = (item.unit_price - was) / was
            found.append(Anomaly(
                "PRICE_MISMATCH", "MEDIUM",
                f"Unit price up {rise * 100:.0f}% — {item.description}",
                f"{item.unit_price:,.2f} now against {was:,.2f} average "
                f"over {seen} previous purchases",
                "Check the price against the contract or current quotation."))

    # -- 8. ordering far more than usual -----------------------------------
    checks += 1
    for item in invoice.line_items:
        usual, seen = baseline.item_quantity(item.description)
        if seen >= 2 and usual > 0 and item.quantity > usual * QTY_TOLERANCE:
            found.append(Anomaly(
                "QUANTITY_OUTLIER", "MEDIUM",
                f"Quantity far above the norm — {item.description}",
                f"{item.quantity:g} units against a usual {usual:g} "
                f"over {seen} purchases",
                "Confirm the order was actually this size."))

    # -- 9. the tax does not look like tax ---------------------------------
    checks += 1
    base = invoice.subtotal or invoice.line_total
    if invoice.tax and base:
        rate = invoice.tax / base
        if not any(abs(rate - r) <= TAX_TOLERANCE for r in PLAUSIBLE_TAX_RATES):
            found.append(Anomaly(
                "TAX_RATE", "MEDIUM",
                f"Tax is {_pct(invoice.tax, base)} of the net amount",
                f"tax {invoice.tax:,.2f} on {base:,.2f} = {rate * 100:.1f}%, "
                "which matches no standard rate",
                "Ask the vendor which rate was applied."))

    # -- 10. sitting just under an approval ceiling ------------------------
    checks += 1
    for ceiling in APPROVAL_THRESHOLDS:
        if ceiling * (1 - JUST_UNDER) <= invoice.total < ceiling:
            found.append(Anomaly(
                "THRESHOLD_PROXIMITY", "MEDIUM",
                f"Lands just under the {ceiling:,} approval limit",
                f"{invoice.currency} {invoice.total:,.2f} is "
                f"{ceiling - invoice.total:,.2f} below {ceiling:,}",
                "Check whether a larger order was split to stay under the "
                "limit."))
            break

    # -- 11. the date ------------------------------------------------------
    checks += 1
    if invoice.invoice_date is None:
        found.append(Anomaly(
            "MISSING_DATE", "MEDIUM", "No invoice date could be read",
            "the date field is empty after extraction",
            "A dated document is required for the audit trail."))
    elif invoice.invoice_date > today:
        found.append(Anomaly(
            "FUTURE_DATE", "HIGH",
            f"Dated in the future — {invoice.invoice_date}",
            f"invoice date {invoice.invoice_date} is after today ({today})",
            "Do not post to this period until the date is corrected."))
    elif invoice.invoice_date < today - timedelta(days=STALE_DAYS):
        found.append(Anomaly(
            "STALE_DATE", "LOW",
            f"Over a year old — {invoice.invoice_date}",
            f"{(today - invoice.invoice_date).days} days old",
            "Confirm it has not already been settled."))

    # -- 12. gaps in the audit trail ---------------------------------------
    checks += 1
    gaps = [name for name, value in (
        ("invoice number", invoice.invoice_id),
        ("vendor name", invoice.vendor),
        ("total", invoice.total),
        ("line items", invoice.line_items)) if not value]
    if gaps:
        found.append(Anomaly(
            "AUDIT_TRAIL_GAP", "HIGH" if len(gaps) > 1 else "MEDIUM",
            "Missing fields: " + ", ".join(gaps),
            f"{len(gaps)} of 4 required fields could not be read from "
            f"{invoice.source_file}",
            "A record with gaps cannot be relied on; re-scan or key manually."))

    # -- 13. how well we read it -------------------------------------------
    checks += 1
    if invoice.confidence < LOW_CONFIDENCE:
        found.append(Anomaly(
            "LOW_CONFIDENCE", "MEDIUM",
            f"Extraction confidence {invoice.confidence:.0%}",
            f"read by {invoice.extraction_method}; "
            f"{invoice.confidence:.0%} of the weighted fields were placed",
            "Have a person check this record against the original."))

    found.sort(key=lambda a: a.rank)
    return AuditResult(invoice=invoice, anomalies=found, checks_run=checks)
