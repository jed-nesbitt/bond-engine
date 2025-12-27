from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import calendar
from typing import List, Tuple, Optional


# -----------------------------
# Date helpers
# -----------------------------
def _last_day_of_month(y: int, m: int) -> int:
    return calendar.monthrange(y, m)[1]


def _add_months(d: date, months: int) -> date:
    """
    Add months to a date, clipping day to the end of the target month if needed.
    """
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    day = min(d.day, _last_day_of_month(y, m))
    return date(y, m, day)


# -----------------------------
# Day count conventions
# -----------------------------
class DayCount:
    ACT_365F = "ACT/365F"
    THIRTY_360_US = "30/360 US"

    @staticmethod
    def year_fraction(start: date, end: date, convention: str) -> float:
        if end < start:
            raise ValueError("end must be >= start")

        if convention == DayCount.ACT_365F:
            return (end - start).days / 365.0

        if convention == DayCount.THIRTY_360_US:
            # 30/360 US (NASD)
            d1 = start.day
            d2 = end.day
            m1 = start.month
            m2 = end.month
            y1 = start.year
            y2 = end.year

            if d1 == 31:
                d1 = 30
            if d2 == 31 and d1 in (30, 31):
                d2 = 30

            return (360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)) / 360.0

        raise ValueError(f"Unsupported day count: {convention}")


# -----------------------------
# Dated fixed-rate bond
# -----------------------------
@dataclass(frozen=True)
class DatedFixedRateBond:
    """
    Fixed-rate bullet bond valued on ANY settlement date.

    Key features:
    - Clean vs Dirty
    - Accrued interest
    - Works with issue_date (recommended for accurate AI early in life)

    Assumptions:
    - Regular coupon schedule (no stubs)
    - Coupon dates are aligned to maturity_date schedule
    - YTM uses periodic compounding at `frequency`
    """
    face: float
    coupon_rate: float
    frequency: int
    maturity_date: date
    day_count: str = DayCount.ACT_365F
    issue_date: Optional[date] = None  # optional but recommended

    def __post_init__(self) -> None:
        if self.face <= 0:
            raise ValueError("face must be > 0")
        if self.coupon_rate < 0:
            raise ValueError("coupon_rate must be >= 0")
        if self.frequency <= 0:
            raise ValueError("frequency must be >= 1")
        if 12 % self.frequency != 0:
            raise ValueError("frequency must divide 12 (e.g., 1,2,3,4,6,12)")
        if self.day_count not in (DayCount.ACT_365F, DayCount.THIRTY_360_US):
            raise ValueError("Unsupported day_count")
        if self.issue_date is not None and self.issue_date >= self.maturity_date:
            raise ValueError("issue_date must be < maturity_date")

    @property
    def coupon_per_period(self) -> float:
        return self.face * self.coupon_rate / self.frequency

    @property
    def coupon_interval_months(self) -> int:
        return 12 // self.frequency

    def coupon_schedule(self) -> List[date]:
        """
        Coupon payment dates (ascending), anchored off maturity_date.
        Does not include issue_date (issue is accrual start, not a payment date).
        """
        step = self.coupon_interval_months
        dates = [self.maturity_date]
        d = self.maturity_date

        # Step back from maturity until we reach issue_date boundary (if provided),
        # or until we go "far enough" (still safe for typical maturities).
        while True:
            d_prev = _add_months(d, -step)

            if self.issue_date is not None and d_prev <= self.issue_date:
                # d is the first coupon date AFTER issue_date
                break

            # stop if we somehow go too far back (safety guard)
            if len(dates) > 1000:
                raise RuntimeError("Coupon schedule too long; check inputs.")

            dates.append(d_prev)
            d = d_prev

            # If no issue_date, keep going back until we are comfortably before any realistic settle
            # (we will rely on settle to choose prev/next).
            if self.issue_date is None and len(dates) >= 400:
                break

        dates = sorted(set(dates))
        return dates

    def _prev_next_coupon(self, settle: date) -> Tuple[date, date]:
        if settle >= self.maturity_date:
            raise ValueError("settlement must be strictly before maturity_date")
        if self.issue_date is not None and settle < self.issue_date:
            raise ValueError("settlement must be on/after issue_date")

        cds = self.coupon_schedule()

        # next coupon is the first coupon date strictly AFTER settle
        nxt = None
        for d in cds:
            if d > settle:
                nxt = d
                break
        if nxt is None:
            raise RuntimeError("Could not find next coupon date. Check maturity/settle inputs.")

        # previous coupon is the last coupon date <= settle; if none, use issue_date if given
        prv = None
        for d in reversed(cds):
            if d <= settle:
                prv = d
                break

        if prv is None:
            if self.issue_date is None:
                # fallback: assume prev coupon is one interval before nxt
                prv = _add_months(nxt, -self.coupon_interval_months)
            else:
                prv = self.issue_date

        return prv, nxt

    def accrued_interest(self, settle: date) -> float:
        """
        Accrued interest from prev coupon (or issue_date) up to settlement.
        If settle is on the prev coupon date, AI = 0.
        """
        prv, nxt = self._prev_next_coupon(settle)

        if settle == prv:
            return 0.0

        num = DayCount.year_fraction(prv, settle, self.day_count)
        den = DayCount.year_fraction(prv, nxt, self.day_count)
        if den <= 0:
            raise RuntimeError("Invalid coupon period length")

        accrual_frac = num / den
        return self.coupon_per_period * accrual_frac

    def cashflows_from_settlement(self, settle: date) -> List[Tuple[date, float]]:
        """
        Future cashflows strictly AFTER settlement date.
        """
        cds = self.coupon_schedule()
        c = self.coupon_per_period

        flows: List[Tuple[date, float]] = []
        for pay in cds:
            if pay > settle:
                cf = c
                if pay == self.maturity_date:
                    cf += self.face
                flows.append((pay, cf))
        return flows

    def dirty_price_from_ytm(self, ytm: float, settle: date) -> float:
        """
        Dirty price using periodic compounding at `frequency`.
        We allow fractional exponents: exp = year_fraction * frequency.
        """
        m = self.frequency
        if ytm <= -m:
            raise ValueError("ytm too low; 1 + ytm/frequency must be > 0")

        per = 1.0 + ytm / m
        pv = 0.0
        for pay_date, cf in self.cashflows_from_settlement(settle):
            t = DayCount.year_fraction(settle, pay_date, self.day_count)
            exp = t * m
            pv += cf / (per ** exp)
        return pv

    def clean_price_from_ytm(self, ytm: float, settle: date) -> float:
        dirty = self.dirty_price_from_ytm(ytm, settle)
        ai = self.accrued_interest(settle)
        return dirty - ai

    def ytm_from_dirty_price(
        self,
        dirty_price: float,
        settle: date,
        *,
        guess: float = 0.05,
        tol: float = 1e-10,
        max_iter: int = 200
    ) -> float:
        if dirty_price <= 0:
            raise ValueError("dirty_price must be > 0")

        def f(y: float) -> float:
            return self.dirty_price_from_ytm(y, settle) - dirty_price

        lo = -0.99 * self.frequency
        hi = max(guess, 0.01)

        flo = f(lo)
        fhi = f(hi)

        steps = 0
        while flo * fhi > 0 and steps < 200:
            hi = hi * 1.5 + 0.01
            fhi = f(hi)
            steps += 1

        if flo * fhi > 0:
            raise RuntimeError("Failed to bracket YTM root. Check inputs (price vs terms).")

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            fmid = f(mid)
            if abs(fmid) < tol:
                return mid
            if flo * fmid <= 0:
                hi = mid
            else:
                lo = mid
                flo = fmid

        return 0.5 * (lo + hi)

    def ytm_from_clean_price(
        self,
        clean_price: float,
        settle: date,
        *,
        guess: float = 0.05,
        tol: float = 1e-10,
        max_iter: int = 200
    ) -> float:
        if clean_price <= 0:
            raise ValueError("clean_price must be > 0")
        dirty = clean_price + self.accrued_interest(settle)
        return self.ytm_from_dirty_price(dirty, settle, guess=guess, tol=tol, max_iter=max_iter)

    # -----------------------------
    # Risk metrics (settlement-date)
    # -----------------------------
    def macaulay_duration_years(self, ytm: float, settle: date) -> float:
        """
        Macaulay duration (years): sum(t * PV_cf) / price
        Here price is DIRTY (PV of future cashflows).
        """
        p = self.dirty_price_from_ytm(ytm, settle)
        if p <= 0:
            raise RuntimeError("Non-positive price; cannot compute duration.")
        m = 0.0
        per = 1.0 + ytm / self.frequency
        for pay_date, cf in self.cashflows_from_settlement(settle):
            t = DayCount.year_fraction(settle, pay_date, self.day_count)
            exp = t * self.frequency
            pv_cf = cf / (per ** exp)
            m += t * pv_cf
        return m / p

    def modified_duration_years(self, ytm: float, settle: date) -> float:
        """
        Modified duration (years): Macaulay / (1 + y/m)
        """
        return self.macaulay_duration_years(ytm, settle) / (1.0 + ytm / self.frequency)

    def dv01(self, ytm: float, settle: date, bump_bp: float = 1.0) -> float:
        """
        DV01 per face (e.g., per 100): price change for a +1bp increase in yield.
        (Positive number if price falls when yield rises.)
        """
        dy = bump_bp / 10000.0
        p_up = self.dirty_price_from_ytm(ytm + dy, settle)
        p_dn = self.dirty_price_from_ytm(ytm - dy, settle)
        return (p_dn - p_up) / 2.0

    def convexity(self, ytm: float, settle: date, bump_bp: float = 1.0) -> float:
        """
        Numerical convexity: (P_up + P_dn - 2P) / (P * dy^2)
        Units: 1/(decimal^2)
        """
        dy = bump_bp / 10000.0
        p = self.dirty_price_from_ytm(ytm, settle)
        p_up = self.dirty_price_from_ytm(ytm + dy, settle)
        p_dn = self.dirty_price_from_ytm(ytm - dy, settle)
        return (p_up + p_dn - 2.0 * p) / (p * (dy ** 2))

    def price_change_parallel_shift(self, ytm: float, settle: date, shift_bp: float) -> float:
        dy = shift_bp / 10000.0
        return self.dirty_price_from_ytm(ytm + dy, settle) - self.dirty_price_from_ytm(ytm, settle)
