from __future__ import annotations

from datetime import date
from typing import Optional

import yaml

from config import settings

_parsed: Optional[dict] = None


def _effective_date(block: dict, key: str) -> Optional[date]:
    """Actual date takes precedence over planned."""
    actual  = (block.get("actual")  or {}).get(key)
    planned = (block.get("planned") or {}).get(key)
    raw = actual or planned
    return date.fromisoformat(raw) if raw else None


def _infer_status(meta: dict, legs: list[dict], today: date) -> str:
    if override := meta.get("status_override"):
        return override
    first_arrival = _effective_date(legs[0], "arrival") if legs else None
    if first_arrival and today < first_arrival:
        return "pre_departure"
    all_departed = all(
        (leg.get("actual") or {}).get("departure") is not None
        for leg in legs
    )
    return "finished" if all_departed else "travelling"


def _infer_current_leg(meta: dict, legs: list[dict], today: date) -> Optional[dict]:
    if override := meta.get("current_leg_override"):
        return next((l for l in legs if l["id"] == override), None)
    for leg in legs:
        arrival   = _effective_date(leg, "arrival")
        departure = _effective_date(leg, "departure")
        if arrival and today >= arrival:
            if departure is None or today <= departure:
                return leg
    return None


def _fmt_date(d: Optional[date]) -> str:
    return d.strftime("%Y-%m-%d") if d else "TBC"


def _render(data: dict, today: date) -> str:
    meta = data.get("meta", {})
    legs = data.get("legs", [])

    status      = _infer_status(meta, legs, today)
    current_leg = _infer_current_leg(meta, legs, today)

    lines = [f"Trip status: {status}"]
    lines.append(f"Base: {meta.get('base_city')}, {meta.get('base_country')}.")

    if trip_start_raw := meta.get("trip_start"):
        trip_start = date.fromisoformat(trip_start_raw)
        if status == "pre_departure":
            lines.append(f"Departure: {trip_start} ({(trip_start - today).days} days away).")
        else:
            lines.append(f"Trip started: {trip_start}.")

    lines.append("")

    if current_leg:
        arrival    = _effective_date(current_leg, "arrival")
        departure  = _effective_date(current_leg, "departure")
        days_in    = (today - arrival).days if arrival else "?"
        lines.append(f"Current leg: {current_leg['name']} {current_leg.get('emoji', '')} (day {days_in}).")
        if departure:
            lines.append(f"Departing in: {(departure - today).days} days ({departure}).")
    elif status == "pre_departure" and legs:
        first         = legs[0]
        first_arrival = _effective_date(first, "arrival")
        if first_arrival:
            lines.append(
                f"Current leg: None (pre-departure). "
                f"Next: {first['name']} in {(first_arrival - today).days} days."
            )

    lines.append("")
    lines.append("Itinerary:")

    for i, leg in enumerate(legs, 1):
        arrival   = _effective_date(leg, "arrival")
        departure = _effective_date(leg, "departure")
        stopover  = " (stopover)" if leg.get("stopover") else ""
        lines.append(
            f"{i}. {leg['name']} {leg.get('emoji', '')} — "
            f"{_fmt_date(arrival)} → {_fmt_date(departure) if departure else 'open-ended'} "
            f"| {leg.get('visa', '')}{stopover}"
        )
        if leg.get("notes"):
            lines.append(f"   Notes: {leg['notes']}")
        for sl in leg.get("sub_legs") or []:
            a = _effective_date(sl, "arrival")
            d = _effective_date(sl, "departure")
            lines.append(
                f"   • {sl['name']}: "
                f"{a.strftime('%b %d') if a else 'TBC'} – {d.strftime('%b %d') if d else 'TBC'}"
            )

    return "\n".join(lines)


def _load_parsed() -> dict:
    global _parsed
    if _parsed is not None:
        return _parsed
    if not settings.travel_yml_path.exists():
        return {}
    with open(settings.travel_yml_path) as f:
        _parsed = yaml.safe_load(f)
    return _parsed

def parse_travel_yml(today: Optional[date] = None) -> str:
    """
    Render travel.yml as a plain-text prompt block.
    Cached after first load — restart Trevor to pick up changes.
    """
    data = _load_parsed()

    if not data:
        return """
            11 Jun 2026      Departs UK → Philadelphia, USA
            11-12 Jun 2026   Philadelphia, PA
            13-15 Jun 2026   Washington DC
            16 Jun-16 Aug    Warrensburg, NY (summer camp)
            16 Aug-~1 Sep    Warrensburg, NY (post-camp labouring)
            ~2 Sep 2026      Seattle, WA
            6 Sep 2026       Departs SFO
            8 Sep 2026       Arrives Nadi, Fiji
            12 Sep 2026      Departs Fiji → Melbourne, Australia
            Sep 2026-Feb 2027  Melbourne (base), with travel TBC
            Mid-Feb-early Mar 2027  East coast Australia road trip
            Mar-Jun 2027     Melbourne or Sydney
            Jun-Jul 2027     New Zealand
            Dec 2027-Apr 2028  SE Asia: Thailand, Cambodia, Laos, Vietnam
            Apr-Sep 2028     TBC (possibly UK / NZ / further SE Asia)
            Sep 2028-May 2029  Canada
            May 2029         Returns to UK"""

    return _render(data, today or date.today())