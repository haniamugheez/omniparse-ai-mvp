"""OmniParse AI — Autonomous Multimodal Document Processing & Anomaly Audit.

The dashboard. Everything it shows comes from the pipeline in `omniparse/`;
there is no demo mode, no hard-coded result, and no screen that pretends work
happened. If the engine cannot read a document it says so.
"""

from __future__ import annotations

import io
import json
import pathlib

import pandas as pd
import streamlit as st

from omniparse import extract, ingest
from omniparse.pipeline import process
from omniparse.store import Baseline

SAMPLES = pathlib.Path(__file__).resolve().parent / "sample_data"

DEMO_EMAIL = "admin@omniparse.ai"
DEMO_PASSWORD = "DemoAudit2026!"

SEVERITY_COLOUR = {"CRITICAL": "#b91c1c", "HIGH": "#c2410c",
                   "MEDIUM": "#a16207", "LOW": "#4d7c0f"}
VERDICT_COLOUR = {"CLEAR": "#15803d", "REVIEW": "#a16207", "HOLD": "#b91c1c"}

st.set_page_config(page_title="OmniParse AI", page_icon="🧾",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1200px;}
  .op-badge {display:inline-block; padding:2px 9px; border-radius:4px;
             font-size:.7rem; font-weight:700; letter-spacing:.04em;
             color:#fff;}
  .op-card {border:1px solid #e5e7eb; border-left-width:4px; border-radius:6px;
            padding:12px 14px; margin-bottom:10px; background:#fff;}
  .op-card h4 {margin:0 0 4px 0; font-size:.95rem;}
  .op-ev {font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
          font-size:.78rem; color:#374151; background:#f9fafb;
          padding:6px 8px; border-radius:4px; margin:6px 0;}
  .op-rec {font-size:.82rem; color:#4b5563;}
  .op-agent {border-left:3px solid #0f766e; padding:4px 0 4px 12px;
             margin-bottom:6px; font-size:.85rem;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 0 · Authentication gate
# ---------------------------------------------------------------------------
def login_gate() -> bool:
    if st.session_state.get("authenticated"):
        return True

    left, middle, right = st.columns([1, 1.1, 1])
    with middle:
        st.markdown("## 🧾 OmniParse AI")
        st.caption("Autonomous Multimodal Document Processing & Anomaly Audit "
                   "Engine")
        st.write("")
        with st.form("login"):
            email = st.text_input("Work email", value=DEMO_EMAIL)
            password = st.text_input("Password", type="password",
                                     value=DEMO_PASSWORD)
            submitted = st.form_submit_button("Sign in", use_container_width=True,
                                              type="primary")
        if submitted:
            if email.strip().lower() == DEMO_EMAIL and password == DEMO_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Those credentials were not recognised.")
        st.info("**Demo access** — this gate is a demonstration placeholder, "
                "not a security control. Credentials are shown above and are "
                "pre-filled.", icon="ℹ️")
    return False


if not login_gate():
    st.stop()


# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def baseline() -> Baseline:
    return Baseline()


def ocr_ready() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:   # noqa: BLE001
        return False


base = baseline()

with st.sidebar:
    st.markdown("### 🧾 OmniParse AI")
    st.caption("v1.0 · MVP")
    st.divider()

    st.markdown("**Engine status**")
    ledger = base.invoice_level()
    st.write(f"📚 Baseline ledger — **{len(ledger)}** invoices, "
             f"**{len(base.vendors)}** vendors")
    st.write("🔤 OCR (Tesseract) — " + ("**ready**" if ocr_ready()
                                        else "**not installed**"))
    st.write("🧠 LLM fallback — " + ("**connected**" if extract.llm_available()
                                     else "**off (rules only)**"))
    if not extract.llm_available():
        st.caption("Rules read every sample document on their own. Add a "
                   "`GROQ_API_KEY` secret to switch the language-model pass "
                   "on for messy or unusual layouts.")
    st.divider()
    st.markdown("**Accepted formats**")
    st.caption("PDF (text or scanned) · PNG, JPG, TIFF · CSV, TSV, XLSX")
    st.divider()
    if st.button("Sign out", use_container_width=True):
        st.session_state.clear()
        st.rerun()


# ---------------------------------------------------------------------------
st.title("Document Processing & Anomaly Audit")
st.caption("Upload financial documents. Four agents read them, check them "
           "against the historical ledger, and report what a human needs to "
           "look at.")

upload_tab, sample_tab = st.tabs(["📤 Upload documents", "⚡ Load sample data"])

files: list = []

with upload_tab:
    uploaded = st.file_uploader(
        "Invoices, receipts, statements or exports",
        type=[t.lstrip(".") for t in ingest.SUPPORTED],
        accept_multiple_files=True)
    if uploaded:
        files = [(f.name, f.getvalue()) for f in uploaded]

with sample_tab:
    st.write("Four documents that exercise every road through the pipeline — "
             "a clean PDF, a PDF with planted arithmetic errors, a scanned "
             "image that has to go through OCR, and a batch of six invoices "
             "as a spreadsheet.")
    choice = st.radio("Sample set", [
        "Everything", "Clean invoice (PDF)", "Suspect invoice (PDF)",
        "Scanned invoice (PNG → OCR)", "Invoice batch (CSV)"],
        horizontal=False, label_visibility="collapsed")
    if st.button("Load sample invoice data", type="primary"):
        wanted = {
            "Everything": ["invoice_clean.pdf", "invoice_suspect.pdf",
                           "invoice_scan.png", "new_invoices.csv"],
            "Clean invoice (PDF)": ["invoice_clean.pdf"],
            "Suspect invoice (PDF)": ["invoice_suspect.pdf"],
            "Scanned invoice (PNG → OCR)": ["invoice_scan.png"],
            "Invoice batch (CSV)": ["new_invoices.csv"],
        }[choice]
        st.session_state.sample_files = [
            (name, (SAMPLES / name).read_bytes()) for name in wanted]

if not files and st.session_state.get("sample_files"):
    files = st.session_state.sample_files

if not files:
    st.stop()


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------
st.divider()
st.write(f"**{len(files)} document(s) ready:** "
         + " · ".join(f"`{name}`" for name, _ in files))

# Collapsed by default: an A4 page rendered full width pushes the one button
# that matters off the screen, and in a demo nobody should have to scroll to
# find it.
with st.expander("👁 Preview the documents"):
    preview_tabs = st.tabs([name for name, _ in files])
    for tab, (name, data) in zip(preview_tabs, files):
        with tab:
            kind = ingest.route(name)
            if kind == "image":
                st.image(data, use_container_width=True)
            elif kind == "pdf":
                try:
                    import pymupdf
                    with pymupdf.open(stream=data, filetype="pdf") as doc:
                        st.image(doc[0].get_pixmap(dpi=110).tobytes("png"),
                                 width=560)
                        st.caption(f"{doc.page_count} page(s)")
                except Exception as exc:   # noqa: BLE001
                    st.warning(f"Could not render a preview: {exc}")
            else:
                try:
                    frame, _ = ingest.read_table(data, name)
                    st.dataframe(frame.head(12), use_container_width=True)
                    st.caption(f"{len(frame)} rows × {len(frame.columns)} "
                               "columns")
                except Exception as exc:   # noqa: BLE001
                    st.warning(f"Could not read the table: {exc}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
st.divider()
if not st.button("▶ Run agent pipeline", type="primary",
                 use_container_width=True):
    st.stop()

runs = []
progress = st.progress(0.0, text="Starting the agent graph…")
for index, (name, data) in enumerate(files, start=1):
    progress.progress((index - 0.5) / len(files), text=f"Processing {name}…")
    runs.append(process(data, name, base))
progress.progress(1.0, text="Done.")
progress.empty()

results = [r for run in runs for r in run.results]
failed = [run for run in runs if not run.ok]

if not results:
    for run in failed:
        st.error(f"**{run.filename}** — {run.error}")
    st.stop()


# ---------------------------------------------------------------------------
# Headline
# ---------------------------------------------------------------------------
st.subheader("Audit summary")

anomalies = [a for r in results for a in r.anomalies]
critical = sum(1 for a in anomalies if a.severity == "CRITICAL")
held = sum(1 for r in results if r.verdict == "HOLD")
exposure = sum(r.invoice.total for r in results if r.verdict == "HOLD")

one, two, three, four, five = st.columns(5)
one.metric("Documents", len(files))
two.metric("Records extracted", len(results))
three.metric("Anomalies", len(anomalies),
             delta=f"{critical} critical" if critical else "none critical",
             delta_color="inverse" if critical else "normal")
four.metric("Held for review", held)
five.metric("Value on hold", f"{exposure:,.0f}")

if failed:
    for run in failed:
        st.warning(f"**{run.filename}** could not be processed — {run.error}")

report_tab, records_tab, json_tab, trace_tab = st.tabs(
    ["🚨 Audit report", "📋 Extracted records", "{ } Structured JSON",
     "🔗 Agent trace"])


# ---------------------------------------------------------------------------
with report_tab:
    clean = [r for r in results if not r.anomalies]
    flagged = sorted([r for r in results if r.anomalies],
                     key=lambda r: -r.risk_score)

    if not flagged:
        st.success("No anomalies found. Every record passed all "
                   f"{results[0].checks_run} checks.", icon="✅")

    for result in flagged:
        invoice = result.invoice
        colour = VERDICT_COLOUR[result.verdict]
        st.markdown(
            f"#### {invoice.invoice_id or '(no invoice number)'} &nbsp;"
            f"<span class='op-badge' style='background:{colour}'>"
            f"{result.verdict}</span>", unsafe_allow_html=True)
        st.caption(
            f"{invoice.vendor or 'unknown vendor'} · "
            f"{invoice.invoice_date or 'no date'} · "
            f"{invoice.currency} {invoice.total:,.2f} · "
            f"risk {result.risk_score}/100 · read from {invoice.source_file}")

        for anomaly in result.anomalies:
            st.markdown(
                f"<div class='op-card' style='border-left-color:"
                f"{SEVERITY_COLOUR[anomaly.severity]}'>"
                f"<h4><span class='op-badge' style='background:"
                f"{SEVERITY_COLOUR[anomaly.severity]}'>{anomaly.severity}</span>"
                f"&nbsp; {anomaly.title}</h4>"
                f"<div class='op-ev'>{anomaly.evidence}</div>"
                f"<div class='op-rec'>→ {anomaly.recommendation}</div>"
                f"</div>", unsafe_allow_html=True)
        st.write("")

    if clean:
        with st.expander(f"✅ {len(clean)} record(s) passed every check"):
            st.dataframe(pd.DataFrame([{
                "Invoice": r.invoice.invoice_id,
                "Vendor": r.invoice.vendor,
                "Date": r.invoice.invoice_date,
                "Total": r.invoice.total,
            } for r in clean]), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
with records_tab:
    frame = pd.DataFrame([{
        "Invoice": r.invoice.invoice_id,
        "Vendor": r.invoice.vendor,
        "Date": r.invoice.invoice_date,
        "Currency": r.invoice.currency,
        "Lines": len(r.invoice.line_items),
        "Net": r.invoice.line_total,
        "Tax": r.invoice.tax,
        "Total": r.invoice.total,
        "Confidence": round(r.invoice.confidence * 100),
        "Verdict": r.verdict,
        "Risk": r.risk_score,
        "Source": r.invoice.source_file,
    } for r in results])
    st.dataframe(frame, use_container_width=True, hide_index=True,
                 column_config={
                     "Confidence": st.column_config.ProgressColumn(
                         "Confidence", min_value=0, max_value=100,
                         format="%d%%"),
                     "Risk": st.column_config.ProgressColumn(
                         "Risk", min_value=0, max_value=100, format="%d"),
                 })

    st.download_button("⬇ Download records (CSV)",
                       frame.to_csv(index=False).encode(),
                       file_name="omniparse_records.csv", mime="text/csv")

    st.markdown("##### Line items")
    lines = pd.DataFrame([{
        "Invoice": r.invoice.invoice_id,
        "Description": item.description,
        "Qty": item.quantity,
        "Unit price": item.unit_price,
        "Amount": item.amount,
        "Qty × price": item.expected_amount,
        "Difference": round(item.amount - item.expected_amount, 2),
    } for r in results for item in r.invoice.line_items])
    if lines.empty:
        st.caption("No line items were present in these documents.")
    else:
        st.dataframe(lines, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
with json_tab:
    payload = {
        "documents": len(files),
        "records": [{
            **r.invoice.to_json(),
            "audit": {
                "verdict": r.verdict,
                "risk_score": r.risk_score,
                "checks_run": r.checks_run,
                "anomalies": [{
                    "code": a.code, "severity": a.severity, "title": a.title,
                    "evidence": a.evidence, "recommendation": a.recommendation,
                } for a in r.anomalies],
            },
        } for r in results],
    }
    body = json.dumps(payload, indent=2, default=str)
    st.code(body, language="json")
    st.download_button("⬇ Download structured JSON", body.encode(),
                       file_name="omniparse_output.json",
                       mime="application/json")


# ---------------------------------------------------------------------------
with trace_tab:
    st.caption("Which agent produced what, and how long it took. A finding "
               "you cannot trace back to an agent is a finding you cannot "
               "defend in an audit.")
    for run in runs:
        st.markdown(f"**{run.filename}**")
        for step in run.trace:
            mark = "✅" if step.ok else "⛔"
            st.markdown(
                f"<div class='op-agent'>{mark} <b>{step.agent}</b> — "
                f"{step.detail} <span style='color:#9ca3af'>"
                f"({step.ms} ms)</span></div>", unsafe_allow_html=True)
        st.write("")

    st.divider()
    st.markdown("##### Baseline the audit compared against")
    st.dataframe(base.invoice_level().tail(10), use_container_width=True,
                 hide_index=True)
