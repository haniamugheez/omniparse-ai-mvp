"""Agent 2 — Structured Extraction.

Turns whatever came out of ingestion into an `Invoice`. Two routes, and the
order matters:

  1. Rules first. Invoice numbers, dates, totals and line tables follow a small
     number of shapes, and a regular expression that matches is exact, free and
     instant. Most documents are finished here.

  2. A language model second, and only for what the rules could not place. It
     is optional — set GROQ_API_KEY to switch it on. With no key the app still
     runs end to end, which is the whole point: a demo must not depend on
     somebody else's rate limit.

Every field carries a confidence, and the confidence is computed from what was
actually found, not decided in advance. A number nobody can check is worse than
no number.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import date, datetime

import pandas as pd

from omniparse.schema import Invoice, LineItem

logger = logging.getLogger("omniparse.extract")

MONEY = r"[-+]?[\d,]+(?:\.\d{1,2})?"

INVOICE_ID_PATTERNS = (
    r"invoice\s*(?:no\.?|number|#|id)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-/]{2,})",
    r"\bINV[-/ ]?([A-Z0-9\-/]{3,})",
    r"bill\s*(?:no\.?|number|#)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-/]{2,})",
)

VENDOR_PATTERNS = (
    r"(?:vendor|supplier|billed\s*by|from|sold\s*by)\s*[:\-]\s*(.+)",
    r"^([A-Z][\w&.,'\- ]{3,40}(?:Ltd|LLC|Inc|Pvt|Limited|Co\.|Company|Traders|Enterprises|Supplies|Solutions))\s*$",
)

# Totals are read line by line, not with one clever expression, because the
# clever expression finds "Total" inside "Sub-Total" and quietly reports the
# net amount as the grand total — which then makes every clean invoice look
# like it fails its own arithmetic. Labels are matched in order of how
# specific they are, and a line carrying a disqualifying word is skipped
# outright.
TOTAL_LABELS = (
    (("grand total",), ()),
    (("total payable", "amount due", "total due", "net payable"), ()),
    (("total",), ("sub", "subtotal", "sub-total", "line total", "tax")),
)
SUBTOTAL_LABELS = ((("sub total", "sub-total", "subtotal", "net amount",
                     "taxable amount"), ()),)
TAX_LABELS = ((("sales tax", "gst", "vat", "tax"), ("taxable", "tax id",
                                                    "ntn", "tax invoice")),)

AMOUNT_ON_LINE = re.compile(rf"({MONEY})\s*$")

DATE_PATTERNS = (
    r"(?:invoice\s*date|date\s*of\s*issue|dated|date)\s*[:\-]?\s*"
    r"(\d{1,2}[-/\s][A-Za-z]{3,9}[-/\s]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
)

CURRENCIES = {"PKR": ("pkr", "rs.", "rs ", "rupee"), "USD": ("usd", "$"),
              "EUR": ("eur", "€"), "GBP": ("gbp", "£"), "AED": ("aed", "dirham")}

DATE_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%d-%b-%Y",
                "%d %b %Y", "%d %B %Y", "%d-%B-%Y", "%d/%m/%y", "%d-%m-%y")


# ---------------------------------------------------------------------------
def _money(text: str) -> float:
    try:
        return round(float(str(text).replace(",", "").replace(" ", "")), 2)
    except (TypeError, ValueError):
        return 0.0


def _first(patterns, text: str, flags=re.I | re.M) -> str:
    for pattern in patterns:
        found = re.search(pattern, text, flags)
        if found:
            return found.group(1).strip()
    return ""


def _labelled_amount(text: str, label_sets) -> float:
    """The last number on the first line that carries one of these labels.

    Tried in the order given, so "Grand Total" always wins over a bare
    "Total", and any line containing an excluded word is passed over.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for include, exclude in label_sets:
        for line in lines:
            lowered = line.lower()
            if not any(word in lowered for word in include):
                continue
            if any(word in lowered for word in exclude):
                continue
            found = AMOUNT_ON_LINE.search(line.replace(":", " "))
            if found:
                value = _money(found.group(1))
                if value:
                    return value
    return 0.0


def _parse_date(raw: str) -> date | None:
    raw = (raw or "").strip().replace("  ", " ")
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def _currency(text: str) -> str:
    lowered = text.lower()
    for code, markers in CURRENCIES.items():
        if code.lower() in lowered or any(m in lowered for m in markers):
            return code
    return "PKR"


# ---------------------------------------------------------------------------
LINE_ROW = re.compile(
    rf"^(?P<desc>[A-Za-z][\w ()./&'\-]{{2,60}}?)\s{{1,}}"
    rf"(?P<qty>\d+(?:\.\d+)?)\s+"
    rf"(?P<price>{MONEY})\s+"
    rf"(?P<amount>{MONEY})\s*$")

SKIP_ROW = re.compile(r"sub\s*-?total|grand\s*total|^total|tax|vat|gst|balance",
                      re.I)


def _line_items_from_text(text: str) -> list:
    """Rows that look like 'description  qty  unit price  amount'."""
    items = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or SKIP_ROW.search(line):
            continue
        found = LINE_ROW.match(line.strip())
        if not found:
            continue
        items.append(LineItem(
            description=found.group("desc").strip(),
            quantity=_money(found.group("qty")),
            unit_price=_money(found.group("price")),
            amount=_money(found.group("amount")),
        ))
    return items


# ---------------------------------------------------------------------------
def from_text(text: str, filename: str, kind: str, method: str) -> Invoice:
    invoice = Invoice(source_file=filename, source_kind=kind, raw_text=text)
    invoice.invoice_id = _first(INVOICE_ID_PATTERNS, text)
    if invoice.invoice_id and not invoice.invoice_id.upper().startswith("INV"):
        pass    # keep whatever the document actually says

    vendor = _first(VENDOR_PATTERNS, text)
    invoice.vendor = re.sub(r"\s{2,}", " ", vendor)[:60]
    invoice.invoice_date = _parse_date(_first(DATE_PATTERNS, text))
    invoice.currency = _currency(text)
    invoice.subtotal = _labelled_amount(text, SUBTOTAL_LABELS)
    invoice.tax = _labelled_amount(text, TAX_LABELS)
    invoice.total = _labelled_amount(text, TOTAL_LABELS)
    invoice.line_items = _line_items_from_text(text)
    invoice.extraction_method = f"rules ({method})"

    missing = _missing_fields(invoice)
    if missing and llm_available():
        filled = _llm_fill(text, missing)
        if filled:
            _apply(invoice, filled)
            invoice.extraction_method = f"rules + llm ({method})"

    invoice.confidence = _confidence(invoice)
    return invoice


def _missing_fields(invoice: Invoice) -> list:
    missing = []
    if not invoice.invoice_id:
        missing.append("invoice_id")
    if not invoice.vendor:
        missing.append("vendor")
    if not invoice.invoice_date:
        missing.append("invoice_date")
    if not invoice.total:
        missing.append("total")
    if not invoice.line_items:
        missing.append("line_items")
    return missing


def _confidence(invoice: Invoice) -> float:
    """What share of the document we actually placed.

    Weighted by how much each field matters to an audit: a total nobody could
    read makes the whole record useless, a missing vendor name does not.
    """
    weights = {"invoice_id": 0.15, "vendor": 0.15, "invoice_date": 0.15,
               "total": 0.3, "line_items": 0.25}
    have = {
        "invoice_id": bool(invoice.invoice_id),
        "vendor": bool(invoice.vendor),
        "invoice_date": invoice.invoice_date is not None,
        "total": invoice.total > 0,
        "line_items": bool(invoice.line_items),
    }
    score = sum(w for k, w in weights.items() if have[k])
    # Arithmetic that agrees with itself is evidence the reading was right.
    if invoice.line_items and invoice.total:
        if abs(invoice.line_total + invoice.tax - invoice.total) <= 1.0:
            score = min(1.0, score + 0.05)
    return round(score, 2)


# ---------------------------------------------------------------------------
# Tables — no OCR, no model, no guessing
# ---------------------------------------------------------------------------
COLUMN_ALIASES = {
    "invoice_id": ("invoice_id", "invoice no", "invoice number", "invoice#",
                   "invoice", "bill no", "bill_no", "doc no", "id"),
    "vendor": ("vendor", "supplier", "vendor name", "supplier name", "party",
               "billed by", "company"),
    "invoice_date": ("invoice_date", "date", "invoice date", "bill date",
                     "issue date", "dated"),
    "description": ("description", "item", "particulars", "details",
                    "line item", "product", "service"),
    "quantity": ("quantity", "qty", "units", "no of units", "count"),
    "unit_price": ("unit_price", "unit price", "rate", "price", "unit cost",
                   "rate per unit"),
    "amount": ("amount", "line total", "total amount", "value", "line amount",
               "net amount"),
    "tax": ("tax", "vat", "gst", "sales tax", "tax amount"),
    "total": ("total", "grand total", "invoice total", "amount due",
              "total payable"),
    "currency": ("currency", "ccy"),
}


def _column_map(frame: pd.DataFrame) -> dict:
    found = {}
    lowered = {str(c).strip().lower(): c for c in frame.columns}
    for field, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                found[field] = lowered[alias]
                break
    return found


def from_frame(frame: pd.DataFrame, filename: str, method: str) -> list:
    """A spreadsheet may hold many invoices. Group by invoice id, or treat the
    whole sheet as one document when there is no id column."""
    cols = _column_map(frame)
    if "invoice_id" in cols:
        groups = frame.groupby(frame[cols["invoice_id"]].astype(str), sort=False)
    else:
        groups = [("", frame)]

    invoices = []
    for invoice_id, rows in groups:
        invoice = Invoice(source_file=filename, source_kind="table",
                          extraction_method=f"table ({method})")
        invoice.invoice_id = str(invoice_id).strip()
        if "vendor" in cols:
            invoice.vendor = str(rows[cols["vendor"]].iloc[0]).strip()
        if "invoice_date" in cols:
            invoice.invoice_date = _parse_date(
                str(rows[cols["invoice_date"]].iloc[0]).split(" ")[0])
        if "currency" in cols:
            invoice.currency = str(rows[cols["currency"]].iloc[0]).strip().upper()

        if "description" in cols:
            for _, row in rows.iterrows():
                invoice.line_items.append(LineItem(
                    description=str(row[cols["description"]]).strip(),
                    quantity=_money(row.get(cols.get("quantity"), 0)),
                    unit_price=_money(row.get(cols.get("unit_price"), 0)),
                    amount=_money(row.get(cols.get("amount"), 0)),
                ))

        invoice.tax = _money(rows[cols["tax"]].iloc[0]) if "tax" in cols else 0.0
        if "total" in cols:
            invoice.total = _money(rows[cols["total"]].iloc[0])
        elif invoice.line_items:
            invoice.total = round(invoice.line_total + invoice.tax, 2)
        invoice.subtotal = invoice.line_total
        invoice.confidence = _confidence(invoice)
        invoices.append(invoice)
    return invoices


# ---------------------------------------------------------------------------
# The optional language-model pass
# ---------------------------------------------------------------------------
def llm_available() -> bool:
    return bool(_api_key())


def _api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "")
    if key:
        return key
    try:                       # Streamlit secrets, when running in Streamlit
        import streamlit as st
        return st.secrets.get("GROQ_API_KEY", "")
    except Exception:          # noqa: BLE001 — no secrets file is normal
        return ""


LLM_MODEL = os.getenv("OMNIPARSE_LLM_MODEL", "llama-3.3-70b-versatile")


def _llm_fill(text: str, missing: list) -> dict:
    """Ask a model only for the fields the rules could not place."""
    import httpx

    prompt = (
        "You are reading one invoice. Return ONLY a JSON object with exactly "
        f"these keys: {', '.join(missing)}. Use null where the document does "
        "not say. Dates as YYYY-MM-DD. Numbers as plain numbers, no commas or "
        "currency symbols. line_items is a list of objects with description, "
        "quantity, unit_price, amount. Never invent a value that is not in the "
        "document.\n\n--- DOCUMENT ---\n" + text[:6000])
    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {_api_key()}"},
                json={"model": LLM_MODEL, "temperature": 0,
                      "response_format": {"type": "json_object"},
                      "messages": [{"role": "user", "content": prompt}]})
        if response.status_code >= 400:
            logger.warning("llm.error %s %s", response.status_code,
                           response.text[:200])
            return {}
        body = response.json()["choices"][0]["message"]["content"]
        return json.loads(body)
    except Exception:          # noqa: BLE001 — the app must survive this
        logger.exception("llm.failed")
        return {}


def _apply(invoice: Invoice, filled: dict) -> None:
    if not invoice.invoice_id and filled.get("invoice_id"):
        invoice.invoice_id = str(filled["invoice_id"]).strip()
    if not invoice.vendor and filled.get("vendor"):
        invoice.vendor = str(filled["vendor"]).strip()[:60]
    if invoice.invoice_date is None and filled.get("invoice_date"):
        invoice.invoice_date = _parse_date(str(filled["invoice_date"]))
    if not invoice.total and filled.get("total"):
        invoice.total = _money(filled["total"])
    if not invoice.tax and filled.get("tax"):
        invoice.tax = _money(filled["tax"])
    if not invoice.line_items and isinstance(filled.get("line_items"), list):
        for row in filled["line_items"]:
            if not isinstance(row, dict):
                continue
            invoice.line_items.append(LineItem(
                description=str(row.get("description", "")).strip(),
                quantity=_money(row.get("quantity", 0)),
                unit_price=_money(row.get("unit_price", 0)),
                amount=_money(row.get("amount", 0)),
            ))
