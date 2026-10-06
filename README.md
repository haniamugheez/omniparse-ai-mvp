# OmniParse AI

**Autonomous Multimodal Document Processing & Anomaly Audit Engine**

Upload invoices, receipts, statements or exports — as PDFs, scans or
spreadsheets. Four agents read them, turn them into structured records, check
those records against the organisation's own historical ledger, and report
what a human needs to look at before money leaves the company.

---

## The problem

Finance, compliance and audit teams process thousands of unstructured
documents by hand. Hundreds of hours go into retyping, human error turns into
financial risk, and because every vendor's format differs, the work resists
automation.

| Beneficiary | What this does for them |
|---|---|
| Finance & accounting | Invoice parsing and reconciliation without retyping |
| Compliance & risk auditors | Variance detection, missing audit trails, non-compliant transactions |
| Operations managers | Multi-format operational files turned into one structured dataset |

---

## The four agents

```
            ┌──────────────────────────────────────────────┐
  file ───► │ 1 · Document Ingestion & Routing             │
            │     PDF → text layer, or render → OCR        │
            │     Image → OCR (Tesseract)                  │
            │     CSV / XLSX → pandas                      │
            └────────────────────┬─────────────────────────┘
                                 ▼
            ┌──────────────────────────────────────────────┐
            │ 2 · Structured Extraction                    │
            │     rules first (exact, free, instant)       │
            │     language model only for what is left     │
            │     every field carries a confidence         │
            └────────────────────┬─────────────────────────┘
                                 ▼
            ┌──────────────────────────────────────────────┐
            │ 3 · Validation & Audit                       │
            │     13 checks against the historical ledger  │
            └────────────────────┬─────────────────────────┘
                                 ▼
            ┌──────────────────────────────────────────────┐
            │ 4 · Anomaly Discovery Engine                 │
            │     severity, evidence, recommendation       │
            │     risk score → CLEAR / REVIEW / HOLD       │
            └──────────────────────────────────────────────┘
```

Every step is recorded in the **Agent trace** tab, with what it produced and
how long it took. A finding you cannot trace back to an agent is a finding you
cannot defend in an audit.

---

## The thirteen checks

| Code | What it catches |
|---|---|
| `LINE_MATH` | A line where quantity × unit price ≠ amount |
| `TOTAL_MATH` | An invoice total that disagrees with its own lines |
| `DUPLICATE_ID` | An invoice number already in the ledger |
| `DUPLICATE_AMOUNT` | Same vendor, same amount — a double payment |
| `HIGH_VALUE_VARIANCE` | Far above what this vendor normally bills (z-score) |
| `NEW_VENDOR` | First invoice from a payee with no history |
| `PRICE_MISMATCH` | A unit price well above what the item has cost before |
| `QUANTITY_OUTLIER` | A quantity far above the usual order size |
| `TAX_RATE` | Tax that matches no standard rate |
| `THRESHOLD_PROXIMITY` | Lands just under an approval ceiling |
| `FUTURE_DATE` / `STALE_DATE` / `MISSING_DATE` | Date problems |
| `AUDIT_TRAIL_GAP` | Required fields that could not be read |
| `LOW_CONFIDENCE` | The document was read poorly; a person should check it |

Every finding prints the arithmetic that produced it. Nothing here is a
language model's opinion — a model can read a document, but it should not be
the control that decides whether a payment goes out.

---

## Running it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 and sign in with the demo credentials shown on
the login screen.

Scanned images need Tesseract on the machine:

* Ubuntu/Debian — `sudo apt install tesseract-ocr`
* macOS — `brew install tesseract`
* Windows — https://github.com/UB-Mannheim/tesseract/wiki

Text PDFs, CSV and Excel work without it. On Streamlit Community Cloud it is
installed automatically from `packages.txt`.

---

## The optional language-model pass

The rules read every sample document on their own, and the app runs end to end
with **no API key at all** — a demo must not depend on somebody else's rate
limit.

For messy or unusual layouts, set a `GROQ_API_KEY` and the extraction agent
asks a model for the fields the rules could not place (never for the fields it
already found). Locally, export it; on Streamlit Cloud put it in
**Settings → Secrets**:

```toml
GROQ_API_KEY = "your-key"
```

---

## Tests

```bash
python -m pytest tests/ -q
```

23 tests. They cover routing, the sub-total/grand-total trap, extraction from
a real PDF and a real spreadsheet, each audit check, and the whole pipeline
end to end — including that a clean invoice raises nothing at all, which is
the test that matters most. A tool that cries wolf gets switched off.

---

## Project layout

```
app.py                  Streamlit dashboard
omniparse/
  schema.py             Invoice, LineItem, Anomaly, AuditResult
  ingest.py             Agent 1 — routing and reading
  extract.py            Agent 2 — rules, then optionally a model
  store.py              The historical baseline
  audit.py              Agents 3 & 4 — validation and anomaly discovery
  pipeline.py           The agent graph and its trace
sample_data/            Historical ledger + demo documents
tests/                  pytest
```

### About the stack

The technical design names LangGraph, ChromaDB, Ollama and PostgreSQL. This
MVP implements the same four-agent flow with the dependencies a free hosting
tier can actually carry: the graph is `pipeline.py`, the baseline is a pandas
ledger rather than a vector store, and extraction is rules-first with an
optional hosted model. The interfaces are kept narrow on purpose — swapping
ChromaDB in means changing `store.py` and nothing else.
