from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from bond import DatedFixedRateBond, DayCount


def _to_date(x: Any) -> Optional[date]:
    if pd.isna(x) or x is None or str(x).strip() == "":
        return None
    return pd.to_datetime(x).date()


def _to_rate(x: Any) -> Optional[float]:
    """
    Accept either decimals (0.05) or percents (5).
    If value > 1, we assume it's percent and divide by 100.
    """
    if pd.isna(x) or x is None or str(x).strip() == "":
        return None
    v = float(x)
    return v / 100.0 if v > 1.0 else v


def _to_float(x: Any) -> Optional[float]:
    if pd.isna(x) or x is None or str(x).strip() == "":
        return None
    return float(x)


def _to_int(x: Any) -> Optional[int]:
    if pd.isna(x) or x is None or str(x).strip() == "":
        return None
    return int(x)


def compute_bond_row(row: pd.Series) -> Dict[str, Any]:
    """
    Required columns:
      bond_id, coupon_rate, frequency, maturity_date, settle_date

    Optional columns:
      face (default 100)
      quantity (default 1)
      day_count (default ACT/365F)
      issue_date
      ytm OR clean_price OR dirty_price  (at least one must be present)
      bump_bp (default 1)
    """
    out: Dict[str, Any] = {}

    bond_id = str(row.get("bond_id", "")).strip() or "UNKNOWN"

    face = _to_float(row.get("face")) or 100.0
    quantity = _to_float(row.get("quantity")) or 1.0  # ✅ FIX

    coupon_rate = _to_rate(row.get("coupon_rate"))
    frequency = _to_int(row.get("frequency"))
    maturity_date = _to_date(row.get("maturity_date"))
    settle_date = _to_date(row.get("settle_date"))
    issue_date = _to_date(row.get("issue_date"))

    day_count = str(row.get("day_count", DayCount.ACT_365F)).strip() or DayCount.ACT_365F
    if day_count not in (DayCount.ACT_365F, DayCount.THIRTY_360_US):
        # be forgiving with slight variations
        dc_up = day_count.upper().replace(" ", "")
        if dc_up in ("ACT/365", "ACT365", "ACT/365F"):
            day_count = DayCount.ACT_365F
        elif dc_up in ("30/360", "30/360US", "30360US"):
            day_count = DayCount.THIRTY_360_US
        else:
            raise ValueError(f"Unsupported day_count '{day_count}' (use 'ACT/365F' or '30/360 US')")

    ytm_in = _to_rate(row.get("ytm"))
    clean_in = _to_float(row.get("clean_price"))
    dirty_in = _to_float(row.get("dirty_price"))

    bump_bp = _to_float(row.get("bump_bp")) or 1.0

    if coupon_rate is None or frequency is None or maturity_date is None or settle_date is None:
        raise ValueError("Missing required fields: coupon_rate, frequency, maturity_date, settle_date")

    bond = DatedFixedRateBond(
        face=face,
        coupon_rate=coupon_rate,
        frequency=frequency,
        maturity_date=maturity_date,
        day_count=day_count,
        issue_date=issue_date,
    )

    # Decide yield:
    if ytm_in is not None:
        ytm = ytm_in
        ytm_source = "input_ytm"
    elif clean_in is not None:
        ytm = bond.ytm_from_clean_price(clean_in, settle_date)
        ytm_source = "implied_from_clean_price"
    elif dirty_in is not None:
        ytm = bond.ytm_from_dirty_price(dirty_in, settle_date)
        ytm_source = "implied_from_dirty_price"
    else:
        raise ValueError("Provide at least one of: ytm, clean_price, dirty_price")

    # Compute prices & risk
    ai = bond.accrued_interest(settle_date)
    dirty = bond.dirty_price_from_ytm(ytm, settle_date)
    clean = dirty - ai

    mac_dur = bond.macaulay_duration_years(ytm, settle_date)
    mod_dur = bond.modified_duration_years(ytm, settle_date)
    dv01 = bond.dv01(ytm, settle_date, bump_bp=bump_bp)
    conv = bond.convexity(ytm, settle_date, bump_bp=bump_bp)

    dp_25 = bond.price_change_parallel_shift(ytm, settle_date, shift_bp=25.0)
    dp_100 = bond.price_change_parallel_shift(ytm, settle_date, shift_bp=100.0)

    ttm = DayCount.year_fraction(settle_date, maturity_date, day_count)

    # Portfolio-scaled fields
    mv_dirty = dirty * quantity
    mv_clean = clean * quantity
    ai_total = ai * quantity
    dv01_total = dv01 * quantity

    out.update(
        {
            "status": "OK",
            "bond_id": bond_id,
            "face": face,
            "quantity": quantity,
            "coupon_rate": coupon_rate,
            "frequency": frequency,
            "day_count": day_count,
            "issue_date": issue_date,
            "settle_date": settle_date,
            "maturity_date": maturity_date,
            "time_to_maturity_years": ttm,
            "ytm_source": ytm_source,
            "ytm": ytm,
            "input_ytm": ytm_in,
            "input_clean_price": clean_in,
            "input_dirty_price": dirty_in,
            "accrued_interest": ai,
            "accrued_interest_total": ai_total,
            "clean_price": clean,
            "dirty_price": dirty,
            "mv_clean": mv_clean,
            "mv_dirty": mv_dirty,
            "macaulay_duration_years": mac_dur,
            "modified_duration_years": mod_dur,
            "dv01_per_100": dv01,
            "dv01_total": dv01_total,
            "convexity_num": conv,
            "price_change_+25bp": dp_25,
            "price_change_+100bp": dp_100,
            "bump_bp_used": bump_bp,
        }
    )
    return out


def run_batch(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    results = []
    for i in range(len(df)):
        row = df.iloc[i]
        try:
            results.append(compute_bond_row(row))
        except Exception as e:
            bond_id = str(row.get("bond_id", "")).strip() or f"ROW_{i}"
            results.append({"status": "ERROR", "bond_id": bond_id, "error": str(e)})

    return pd.DataFrame(results)
