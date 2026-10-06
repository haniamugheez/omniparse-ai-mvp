"""The historical baseline — what "normal" looks like for this business.

The architecture names ChromaDB here. ChromaDB's job in that design is to
answer one question: *have we seen something like this before?* For an MVP
that question is answered exactly, and for free, out of a table of past
invoices — exact ids for duplicates, per-vendor statistics for variance, and
per-item price history for price checks.

Keeping it in pandas means the demo starts in under a second, needs no server,
and every number the audit prints can be traced back to a row a human can
open. The interface below is the one a vector store would have to satisfy, so
swapping ChromaDB in later changes this file and nothing else.
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger("omniparse.store")

HISTORY_PATH = (pathlib.Path(__file__).resolve().parent.parent
                / "sample_data" / "history.csv")


@dataclass
class VendorStats:
    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    maximum: float = 0.0

    @property
    def known(self) -> bool:
        return self.count > 0


class Baseline:
    def __init__(self, frame: pd.DataFrame | None = None):
        self.frame = frame if frame is not None else _load()
        self._prepare()

    def _prepare(self) -> None:
        frame = self.frame
        for column in ("invoice_id", "vendor", "description"):
            if column in frame:
                frame[column] = frame[column].astype(str).str.strip()
        for column in ("quantity", "unit_price", "amount", "total"):
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if "invoice_date" in frame:
            frame["invoice_date"] = pd.to_datetime(frame["invoice_date"],
                                                   errors="coerce")

    # -- duplicates ---------------------------------------------------------
    @property
    def invoice_ids(self) -> set:
        return set(self.frame.get("invoice_id", pd.Series(dtype=str)))

    def seen(self, invoice_id: str) -> bool:
        return bool(invoice_id) and invoice_id in self.invoice_ids

    def near_duplicate(self, vendor: str, total: float, days: int = 14):
        """Same vendor, same amount, close in time — the classic double payment.

        Returns the matching row, or None. The amount is compared to the rupee
        so a genuine repeat order at a different price is not flagged.
        """
        if not vendor or not total:
            return None
        frame = self.invoice_level()
        hit = frame[(frame["vendor"].str.lower() == vendor.lower())
                    & ((frame["total"] - total).abs() <= 1.0)]
        return None if hit.empty else hit.iloc[0]

    # -- per vendor ---------------------------------------------------------
    def invoice_level(self) -> pd.DataFrame:
        """One row per past invoice, so a 6-line invoice does not count six
        times in a vendor's average."""
        frame = self.frame
        if "total" not in frame or "invoice_id" not in frame:
            return pd.DataFrame(columns=["invoice_id", "vendor", "total",
                                         "invoice_date"])
        return (frame.groupby("invoice_id", as_index=False)
                .agg(vendor=("vendor", "first"), total=("total", "first"),
                     invoice_date=("invoice_date", "first")))

    def vendor_stats(self, vendor: str) -> VendorStats:
        frame = self.invoice_level()
        rows = frame[frame["vendor"].str.lower() == (vendor or "").lower()]
        totals = rows["total"].dropna()
        if totals.empty:
            return VendorStats()
        return VendorStats(
            count=int(totals.count()),
            mean=round(float(totals.mean()), 2),
            # ddof=0 so a vendor with two invoices still yields a usable spread
            std=round(float(totals.std(ddof=0)), 2),
            maximum=round(float(totals.max()), 2),
        )

    # -- per item -----------------------------------------------------------
    def item_price(self, description: str) -> tuple:
        """(mean unit price, times seen) for an item bought before."""
        if "description" not in self.frame or not description:
            return 0.0, 0
        rows = self.frame[self.frame["description"].str.lower()
                          == description.strip().lower()]
        prices = rows["unit_price"].dropna()
        if prices.empty:
            return 0.0, 0
        return round(float(prices.mean()), 2), int(prices.count())

    def item_quantity(self, description: str) -> tuple:
        if "description" not in self.frame or not description:
            return 0.0, 0
        rows = self.frame[self.frame["description"].str.lower()
                          == description.strip().lower()]
        qty = rows["quantity"].dropna()
        if qty.empty:
            return 0.0, 0
        return round(float(qty.mean()), 2), int(qty.count())

    @property
    def vendors(self) -> set:
        return {v.lower() for v in self.frame.get("vendor", pd.Series(dtype=str))}


def _load() -> pd.DataFrame:
    if HISTORY_PATH.exists():
        return pd.read_csv(HISTORY_PATH)
    logger.warning("no history file at %s — running without a baseline",
                   HISTORY_PATH)
    return pd.DataFrame(columns=["invoice_id", "vendor", "invoice_date",
                                 "description", "quantity", "unit_price",
                                 "amount", "total"])
