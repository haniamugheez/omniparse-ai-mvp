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
                "%d %b %Y", "%d %B %Y", "%d-%B-%Y", "%d/%m/%y", "%d-%m-%y",
                # "October 05, 2026" — how an American system writes it
                "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y",
                "%Y/%m/%d", "%d.%m.%Y")


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
# ---------------------------------------------------------------------------
# Tables — the part that has to survive a real invoice
#
# The file that forced this rewrite is a courier reconciliation run. Column A
# is empty down the whole sheet. The vendor sits on row 8, the invoice number
# on row 10 inside one cell as "Customer Invoice Number: INV-… # 010-…", the
# billing date on row 12 with its label in one cell and its value in the next,
# and the column headings do not appear until row 14. Below that, each
# shipment is three rows — a freight charge, a fuel surcharge, and a "Total"
# line — with the shipment's own details written only on the first of them,
# and amounts formatted as "$ -1.29".
#
# Read that with pandas' default header=0 and every column is named after a
# blank cell. That is why it came out as a record with no invoice number, no
# vendor, no date, no total and no lines: not a hard document, just one the
# reader was too polite to look at properly.
#
# So nothing is assumed. The header row is found, the preamble above it is
# mined for the facts an invoice carries outside its table, group columns are
# carried down, money is parsed in the shapes money actually comes in, and
# subtotal rows are recognised as subtotals rather than counted twice.
# ---------------------------------------------------------------------------
COLUMN_ALIASES = {
    "invoice_id": ("invoice_id", "invoice no", "invoice no.", "invoice number",
                   "invoice#", "invoice #", "invoice", "bill no", "bill_no",
                   "doc no", "document no", "reference", "ref no"),
    "vendor": ("vendor", "supplier", "vendor name", "supplier name", "party",
               "billed by", "company", "merchant", "payee"),
    "invoice_date": ("invoice_date", "date", "invoice date", "bill date",
                     "billing date", "issue date", "dated", "shipment date",
                     "transaction date", "posting date"),
    "description": ("description", "item", "particulars", "details",
                    "line item", "product", "service", "charge description",
                    "charge type", "narration", "expense", "charge"),
    "quantity": ("quantity", "qty", "units", "no of units", "count", "pieces"),
    "unit_price": ("unit_price", "unit price", "rate", "price", "unit cost",
                   "rate per unit", "unit rate"),
    "amount": ("amount", "line total", "total amount", "value", "line amount",
               "net amount", "charge amount", "amount (usd)", "amount usd",
               "debit", "credit"),
    "tax": ("tax", "vat", "gst", "sales tax", "tax amount"),
    "total": ("total", "grand total", "invoice total", "amount due",
              "total payable", "net payable"),
    "currency": ("currency", "ccy", "curr"),
}

# A header is matched on its words, not on an exact string. "Item Description",
# "Charge Description" and "Service Description" are all the description
# column, and no list of exact phrases will ever contain the next one a vendor
# invents. So: strip the decoration, then look for a known phrase inside.
def _norm(value) -> str:
    # An empty spreadsheet cell arrives as a float NaN, and str() turns that
    # into the word "nan" — three letters, no colon, not an address, which is
    # exactly the shape this code was looking for in a vendor name. One
    # invoice came back billed by "nan". Empty is empty, here and everywhere
    # that calls this.
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().lower()
    if text in ("nan", "none", "nat", "#n/a", "n/a", "-", "--"):
        return ""
    text = re.sub(r"[\u2013\u2014]", "-", text)
    text = re.sub(r"\s*\([^)]*\)", " ", text)      # "Amount (USD)" -> "Amount"
    text = re.sub(r"[^a-z0-9#/ .-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _alias_of(cell) -> tuple:
    """(field, how good a match) for a header cell, or (None, 0).

    The score matters because a sheet can offer two candidates for the same
    field — an accountant's export has "Debit", "Credit" AND "Amount", and the
    first two are usually empty. An exact match beats a word found inside a
    longer heading, so "Amount" wins over "Debit" and the real numbers are
    read.
    """
    text = _norm(cell)
    if not text or text.replace(".", "").isdigit():
        return None, 0
    best = (None, 0)
    for field, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if text == alias:
                return field, 3                      # exact
            if re.search(rf"\b{re.escape(alias)}\b", text) and 2 > best[1]:
                best = (field, 2)                    # contained
    return best


MAX_HEADER_SCAN = 40        # how far down to look for the column headings


def _find_header(frame: pd.DataFrame) -> int | None:
    """The row that names the columns, wherever the sheet put it.

    Scored, not guessed: a header row is the one where the most cells are
    recognisable field names. Two matches and three filled cells is the floor,
    so a stray line of prose cannot be mistaken for a header.
    """
    best_row, best_score = None, 0
    for i in range(min(len(frame), MAX_HEADER_SCAN)):
        row = frame.iloc[i]
        filled = sum(1 for c in row if _norm(c))
        hits = sum(1 for c in row if _alias_of(c)[0])
        if hits >= 2 and filled >= 3 and hits > best_score:
            best_row, best_score = i, hits
    return best_row


# The first number in the cell, with its own minus sign if it has one. Written
# this way rather than by stripping characters, because "Rs 2,500/-" ends in a
# dash that is punctuation, not arithmetic — strip-and-parse turns it into
# "2500.-" and then into nothing at all.
MONEY_PATTERN = re.compile(r"-?\d[\d,\u00a0 ]*(?:\.\d+)?")


def _amount(value) -> float:
    """Money as it is actually written: "$ -1.29", "(1,234.56)", "Rs 2,500/-"."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return round(float(value), 2)

    text = str(value).strip()
    if not text:
        return 0.0
    found = MONEY_PATTERN.search(text)
    if not found:
        return 0.0
    try:
        number = float(found.group().replace(",", "")
                       .replace("\u00a0", "").replace(" ", ""))
    except ValueError:
        return 0.0
    # Accountants wrap a negative in brackets instead of signing it.
    if text.startswith("(") and text.endswith(")"):
        number = -abs(number)
    return round(number, 2)


SUBTOTAL_WORDS = ("total", "subtotal", "sub-total", "sub total", "balance")
GRAND_TOTAL_WORDS = ("grand total", "invoice total", "amount due",
                     "total due", "net payable", "balance due")


def _bare(text: str) -> str:
    """Normalised, with the decoration a spreadsheet puts around a label.

    "GRAND TOTAL:" is the same word as "Grand Total". A trailing colon is why
    the grand total was once counted as an ordinary charge, which doubled the
    invoice and then produced an arithmetic failure the document never had.
    """
    return _norm(text).strip(" :.-*\u2013\u2014")


def _is_grand_total(text: str) -> bool:
    return _bare(text) in GRAND_TOTAL_WORDS


def _is_subtotal(text: str) -> bool:
    """A "Total" line inside the table is one group's own sum, not a charge."""
    return _bare(text) in SUBTOTAL_WORDS


LABELS = {
    "invoice_id": ("invoice number", "invoice no", "invoice #", "invoice:",
                   "customer invoice number", "bill number", "bill no",
                   "bill #", "document number", "document no", "doc no",
                   "reference number", "reference no", "ref no", "ref #"),
    "invoice_date": ("billing date", "invoice date", "bill date", "date of issue",
                     "issue date", "dated", "date:"),
    "vendor": ("vendor", "supplier", "billed by", "sold by", "from:",
               "remit to", "merchant"),
    "total": ("grand total", "invoice total", "total due", "amount due",
              "total amount due", "net payable", "balance due"),
}

COMPANY_WORDS = ("ltd", "limited", "llc", "inc", "pvt", "co.", "company",
                 "corp", "express", "services", "service", "logistics",
                 "traders", "enterprises", "supplies", "solutions", "group",
                 "international", "industries", "store", "mart")

# A document's own title is not the company that sent it.
TITLE_WORDS = ("invoice", "statement", "report", "summary", "adjustment",
               "reconciliation", "tax invoice", "credit note", "debit note",
               "bill of", "remittance", "account")

ADDRESS_HINT = re.compile(
    r"\b(road|street|st\.|drive|avenue|ave|block|sector|plot|house|suite|"
    r"floor|p\.?o\.?\s*box|zip|postal)\b|\d{4,}", re.I)


def _preamble_facts(frame: pd.DataFrame, header_row: int) -> dict:
    """Everything an invoice says ABOVE its table.

    Two shapes, both common and both here: the label and the value in one cell
    ("Customer Invoice Number: INV-…"), and the label in one cell with the
    value in the next ("Billing Date:" | "October 05, 2026").
    """
    found: dict = {}
    rows = frame.iloc[:header_row] if header_row else frame.iloc[:0]

    for _, row in rows.iterrows():
        cells = [str(c).strip() for c in row if _norm(c)]
        for index, cell in enumerate(cells):
            lowered = cell.lower()
            for field, labels in LABELS.items():
                if field in found:
                    continue
                for label in labels:
                    if label not in lowered:
                        continue
                    after = cell[lowered.index(label) + len(label):]
                    after = after.lstrip(" :-\u2013\u2014\t")
                    if not after and index + 1 < len(cells):
                        after = cells[index + 1]
                    if after.strip():
                        found[field] = after.strip()
                    break

    # The vendor rarely carries a label. Take the most company-looking line
    # above the table that is not an address and not a label.
    if "vendor" not in found:
        best = None
        for _, row in rows.iterrows():
            for cell in row:
                text = str(cell).strip() if _norm(cell) else ""
                if not (3 <= len(text) <= 60) or ":" in text:
                    continue
                if ADDRESS_HINT.search(text):
                    continue
                if any(w in text.lower() for w in TITLE_WORDS):
                    continue
                score = (sum(w in text.lower() for w in COMPANY_WORDS) * 10
                         + len(text.split()))
                if best is None or score > best[0]:
                    best = (score, text)
        if best and best[0] > 0:
            found["vendor"] = best[1]
    return found


def from_frame(frame: pd.DataFrame, filename: str, method: str) -> list:
    """One raw sheet in, one or more invoices out."""
    frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="all")
    frame = frame.reset_index(drop=True)
    if frame.empty:
        return [Invoice(source_file=filename, source_kind="table",
                        extraction_method=f"table ({method})")]

    header_row = _find_header(frame)
    if header_row is None:
        # No recognisable table. Still report what the preamble says rather
        # than returning an empty shell.
        invoice = Invoice(source_file=filename, source_kind="table",
                          extraction_method=f"table ({method}, no header found)")
        _apply_preamble(invoice, _preamble_facts(frame, len(frame)))
        invoice.confidence = _confidence(invoice)
        return [invoice]

    facts = _preamble_facts(frame, header_row)
    headers = [_alias_of(c) for c in frame.iloc[header_row]]
    body = frame.iloc[header_row + 1:].reset_index(drop=True)

    # Two columns can claim the same field. Keep the better-scoring one, and
    # break a tie with whichever column actually holds values — an empty
    # "Debit" column must not win over a full "Amount" one.
    cols: dict = {}
    scores: dict = {}
    for position, (field, score) in enumerate(headers):
        if not field:
            continue
        filled = sum(1 for v in frame.iloc[header_row + 1:, position]
                     if _norm(v) not in ("", "nan"))
        rank = (score, filled)
        if field not in cols or rank > scores[field]:
            cols[field], scores[field] = position, rank

    # Group columns are written once per group and left blank on the rows
    # beneath. Carry them down so every line knows which shipment it belongs to.
    for field in ("invoice_id", "vendor", "invoice_date", "currency"):
        if field in cols:
            body.iloc[:, cols[field]] = body.iloc[:, cols[field]].ffill()

    groups = ([(str(key), rows) for key, rows in
               body.groupby(body.iloc[:, cols["invoice_id"]].astype(str),
                            sort=False)]
              if "invoice_id" in cols else [("", body)])

    invoices = []
    for invoice_id, rows in groups:
        invoice = Invoice(source_file=filename, source_kind="table",
                          extraction_method=f"table ({method})")
        invoice.invoice_id = invoice_id.strip()
        _apply_preamble(invoice, facts)

        if "vendor" in cols and not invoice.vendor:
            invoice.vendor = _first_text(rows, cols["vendor"])
        if "currency" in cols:
            invoice.currency = (_first_text(rows, cols["currency"]).upper()
                                or invoice.currency)
        if invoice.invoice_date is None and "invoice_date" in cols:
            invoice.invoice_date = _parse_date(_first_text(rows,
                                                           cols["invoice_date"]))

        stated_total = 0.0
        grand_total = 0.0
        for _, row in rows.iterrows():
            description = (str(row.iloc[cols["description"]]).strip()
                           if "description" in cols else "")
            if _norm(description) in ("", "nan"):
                continue
            amount = _amount(row.iloc[cols["amount"]]) if "amount" in cols else 0.0
            if _is_grand_total(description):
                grand_total = amount      # the document's own answer; it wins
                continue
            if _is_subtotal(description):
                stated_total += amount
                continue
            invoice.line_items.append(LineItem(
                description=description,
                quantity=_amount(row.iloc[cols["quantity"]])
                if "quantity" in cols else 0.0,
                unit_price=_amount(row.iloc[cols["unit_price"]])
                if "unit_price" in cols else 0.0,
                amount=amount,
            ))

        if "tax" in cols:
            # The tax for the whole invoice is written against every line of
            # it. Adding those up charges the customer tax once per item.
            taxes = [_amount(row.iloc[cols["tax"]]) for _, row in rows.iterrows()]
            distinct = {t for t in taxes if t}
            invoice.tax = (sum(taxes) if len(distinct) == len(
                [t for t in taxes if t]) and len(distinct) > 1
                else (max(distinct, key=abs) if distinct else 0.0))
        if "total" in cols:
            invoice.total = max(
                (_amount(row.iloc[cols["total"]]) for _, row in rows.iterrows()),
                key=abs, default=0.0)
        if grand_total:
            invoice.total = grand_total
        if not invoice.total and stated_total:
            invoice.total = round(stated_total, 2)
        if not invoice.total and invoice.line_items:
            invoice.total = round(invoice.line_total + invoice.tax, 2)

        invoice.subtotal = invoice.line_total
        invoice.confidence = _confidence(invoice)
        invoices.append(invoice)
    return invoices


def _first_text(rows: pd.DataFrame, position: int) -> str:
    for value in rows.iloc[:, position]:
        text = str(value).strip() if _norm(value) else ""
        if text and text.lower() != "nan":
            return text
    return ""


def _apply_preamble(invoice: Invoice, facts: dict) -> None:
    if facts.get("invoice_id") and not invoice.invoice_id:
        invoice.invoice_id = facts["invoice_id"][:80]
    if facts.get("vendor") and not invoice.vendor:
        invoice.vendor = facts["vendor"][:60]
    if facts.get("invoice_date") and invoice.invoice_date is None:
        invoice.invoice_date = _parse_date(facts["invoice_date"])
    if facts.get("total") and not invoice.total:
        invoice.total = _amount(facts["total"])


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

# ---------------------------------------------------------------------------
# When nothing else could read the page
#
# Two documents arrive that rules cannot touch: a photographed invoice on a
# machine with no OCR engine, and a spreadsheet laid out so unusually that no
# row in it looks like a header. Both used to come back empty.
#
# A vision model reads either. It is slower and it costs, so it is the last
# resort and never the first — the rules run first, every time, and this is
# only reached when they have genuinely come up short. With no API key
# configured it does not run at all, and the record says honestly that it
# could not be read.
# ---------------------------------------------------------------------------
# Model names expire. This code first shipped pointing at
# `llama-4-scout-17b-16e-instruct`, which Groq retired, and every scan then
# came back "vision failed (404)" — a dead string in a config file taking down
# a working feature.
#
# So the model is not hard-coded. The account's own model list is read and the
# first one we know can see is used, newest preference first. Set
# OMNIPARSE_VISION_MODEL to override; if the list cannot be fetched, the first
# candidate is tried anyway, because a 404 is a better outcome than refusing
# to try.
VISION_CANDIDATES = (
    "qwen/qwen3.8-27b",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
)

_vision_model_cache: list = []


def vision_model() -> str:
    override = os.getenv("OMNIPARSE_VISION_MODEL")
    if override:
        return override
    if _vision_model_cache:
        return _vision_model_cache[0]

    available = _models_on_this_account()
    for candidate in VISION_CANDIDATES:
        if not available or candidate in available:
            _vision_model_cache.append(candidate)
            logger.info("vision.model_selected %s", candidate)
            return candidate
    _vision_model_cache.append(VISION_CANDIDATES[0])
    return VISION_CANDIDATES[0]


def _models_on_this_account() -> set:
    """What this key can actually reach. An empty set means "could not ask"."""
    import httpx

    if not _api_key():
        return set()
    try:
        with httpx.Client(timeout=15) as client:
            response = client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {_api_key()}"})
        if response.status_code >= 400:
            return set()
        return {m.get("id") for m in response.json().get("data", [])}
    except Exception:   # noqa: BLE001 — never let a lookup cost a document
        logger.exception("vision.model_list_failed")
        return set()

INVOICE_SCHEMA_PROMPT = (
    "Read this invoice and return ONLY a JSON object with these keys: "
    "invoice_id, vendor, invoice_date, currency, tax, total, line_items. "
    "line_items is a list of objects with description, quantity, unit_price, "
    "amount. Dates as YYYY-MM-DD. Numbers plain — no commas, no currency "
    "symbols; a credit or refund is negative. Use null for anything the "
    "document does not say. NEVER invent a value that is not visible in the "
    "document."
)


def vision_available() -> bool:
    return bool(_api_key())


def from_image(image: bytes, filename: str, kind: str = "image") -> Invoice:
    """Read a page by looking at it."""
    import base64

    import httpx

    invoice = Invoice(source_file=filename, source_kind=kind,
                      extraction_method="vision")
    if not image or not _api_key():
        invoice.extraction_method = "unreadable (no OCR engine, no vision key)"
        return invoice

    encoded = base64.b64encode(image).decode()
    model = vision_model()
    try:
        with httpx.Client(timeout=90) as client:
            response = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {_api_key()}"},
                json={
                    "model": model,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": INVOICE_SCHEMA_PROMPT},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{encoded}"}},
                    ]}],
                })
        if response.status_code == 404 and "model" in response.text.lower():
            # Retired between runs. Forget it and let the next call re-choose.
            logger.warning("vision.model_retired %s", model)
            _vision_model_cache.clear()
            if model in VISION_CANDIDATES:
                VISION_CANDIDATES_LEFT = [c for c in VISION_CANDIDATES
                                          if c != model]
                if VISION_CANDIDATES_LEFT:
                    os.environ.pop("OMNIPARSE_VISION_MODEL", None)
                    _vision_model_cache.append(VISION_CANDIDATES_LEFT[0])
                    return from_image(image, filename, kind)
            invoice.extraction_method = "vision failed (no usable model)"
            return invoice
        if response.status_code >= 400:
            logger.warning("vision.error %s %s", response.status_code,
                           response.text[:200])
            invoice.extraction_method = f"vision failed ({response.status_code})"
            return invoice
        filled = json.loads(response.json()["choices"][0]["message"]["content"])
    except Exception:   # noqa: BLE001 — a model outage is not a crash
        logger.exception("vision.failed")
        invoice.extraction_method = "vision failed"
        return invoice

    if filled.get("currency"):
        invoice.currency = str(filled["currency"]).strip().upper()[:4]
    _apply(invoice, filled)
    invoice.confidence = round(_confidence(invoice) * 0.9, 2)   # read, not parsed
    return invoice


def sheet_as_text(frame: pd.DataFrame, limit: int = 120) -> str:
    """A spreadsheet written out as lines, for a model to read."""
    lines = []
    for _, row in frame.head(limit).iterrows():
        cells = [str(c).strip() for c in row if _norm(c)]
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)

