"""
MULTI-OBJECTIVE IMPACT SCORER
Evaluates the cost of a given CTOT assignment against three dimensions:
  1. Crew legality (FDP / FTL rules — EU OPS / FAR Part 117)
  2. Passenger connection loss (misconnect revenue)
  3. Fuel delta (accept-slot vs reroute vs hold)

Returns a composite ImpactScore used by the negotiation planner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Input data models (fed from OPS systems)
# ---------------------------------------------------------------------------

class CrewRecord(BaseModel):
    crew_id: str
    role: str                       # "CPT" | "FO" | "CC"
    duty_start: datetime            # When crew reported for duty (UTC)
    max_fdp_hours: float = 10.0     # Maximum FDP in hours (simplified)
    rest_required_hours: float = 11.0

    def fdp_at(self, block_on_time: datetime) -> float:
        """Flight Duty Period in hours from duty start to estimated block-on."""
        return (block_on_time - self.duty_start).total_seconds() / 3600.0

    def is_legal(self, block_on_time: datetime) -> bool:
        return self.fdp_at(block_on_time) <= self.max_fdp_hours


class Connection(BaseModel):
    pax_count: int
    connecting_flight: str          # e.g. "BA789"
    connecting_dep_time: datetime   # Scheduled departure of connecting flight
    min_connect_time_min: int = 45  # Minimum connection time in minutes
    avg_ticket_value_usd: float = 450.0
    has_same_day_backup: bool = True

    def misconnects_at(self, inbound_arr: datetime) -> bool:
        """True if pax miss their connection given this arrival time."""
        latest_inbound = self.connecting_dep_time - timedelta(minutes=self.min_connect_time_min)
        return inbound_arr > latest_inbound

    def revenue_at_risk(self, inbound_arr: datetime) -> float:
        if self.misconnects_at(inbound_arr):
            # Full revenue at risk if no backup, partial if backup exists
            multiplier = 1.0 if not self.has_same_day_backup else 0.15
            return self.pax_count * self.avg_ticket_value_usd * multiplier
        return 0.0


class FlightOperationalData(BaseModel):
    flight_number: str
    dep_airport: str
    arr_airport: str
    scheduled_off_block: datetime
    flight_time_min: int            # Planned block time in minutes
    taxi_out_min: int = 20
    taxi_in_min: int = 15
    crew: List[CrewRecord] = field(default_factory=list)
    connections: List[Connection] = field(default_factory=list)
    base_fuel_burn_kg: float = 8500.0
    fuel_price_usd_per_kg: float = 0.80
    apu_burn_kg_per_hour: float = 120.0
    alternate_route_extra_nm: float = 0.0    # 0 = no alternate route available
    alternate_route_extra_fuel_kg: float = 0.0

    class Config:
        arbitrary_types_allowed = True

    def estimated_block_on(self, ctot: datetime) -> datetime:
        """Calculate estimated arrival block-on time from a given CTOT."""
        airborne_min = self.flight_time_min
        return ctot + timedelta(minutes=self.taxi_out_min + airborne_min + self.taxi_in_min)


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

@dataclass
class ImpactScore:
    flight_number: str
    ctot: datetime

    # Crew
    crew_legal: bool = True
    crew_fdp_violation_hours: float = 0.0
    crew_affected: List[str] = field(default_factory=list)

    # Connections
    misconnecting_pax: int = 0
    connection_revenue_at_risk_usd: float = 0.0
    connection_details: List[dict] = field(default_factory=list)

    # Fuel
    delay_minutes: float = 0.0
    apu_extra_fuel_kg: float = 0.0
    apu_extra_cost_usd: float = 0.0
    reroute_available: bool = False
    reroute_extra_fuel_kg: float = 0.0
    reroute_extra_cost_usd: float = 0.0
    reroute_saves_usd: float = 0.0   # Positive = reroute is worth it

    # Composite
    composite_score: float = 0.0     # 0.0 (no impact) → 1.0 (catastrophic)
    priority: str = "LOW"            # LOW | MEDIUM | HIGH | CRITICAL
    recommendation: str = "ACCEPT"  # ACCEPT | SWAP | REROUTE | ESCALATE

    def summary(self) -> str:
        lines = [
            f"Flight {self.flight_number} | CTOT {self.ctot.strftime('%H:%M')}Z",
            f"  Priority    : {self.priority}",
            f"  Crew legal  : {'YES' if self.crew_legal else f'NO — {self.crew_fdp_violation_hours:.1f}h over limit'}",
            f"  Misconnects : {self.misconnecting_pax} pax | ${self.connection_revenue_at_risk_usd:,.0f} at risk",
            f"  APU cost    : ${self.apu_extra_cost_usd:,.0f} (hold {self.delay_minutes:.0f} min)",
            f"  Reroute     : {'saves $' + f'{self.reroute_saves_usd:,.0f}' if self.reroute_saves_usd > 0 else 'N/A'}",
            f"  Composite   : {self.composite_score:.2f}",
            f"  Decision    : {self.recommendation}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class ImpactScorer:
    """
    Evaluates the operational impact of a CTOT assignment on a given flight.
    All monetary values in USD for consistency.
    """

    # Scoring weights (tunable per airline policy)
    WEIGHT_CREW_ILLEGAL = 0.50      # Crew illegality is always highest priority
    WEIGHT_REVENUE_PER_1K = 0.02   # 0.02 score points per $1,000 revenue at risk
    WEIGHT_FUEL_PER_1K = 0.005     # Fuel cost is secondary

    CRITICAL_THRESHOLD = 0.70
    HIGH_THRESHOLD = 0.40
    MEDIUM_THRESHOLD = 0.15

    def score(self, data: FlightOperationalData, ctot: datetime) -> ImpactScore:
        result = ImpactScore(flight_number=data.flight_number, ctot=ctot)

        estimated_block_on = data.estimated_block_on(ctot)
        delay_min = (ctot - data.scheduled_off_block).total_seconds() / 60
        result.delay_minutes = max(delay_min, 0)

        # ------------------------------------------------------------------
        # 1. CREW LEGALITY CHECK
        # ------------------------------------------------------------------
        illegal_crew = []
        max_violation_h = 0.0

        for crew_member in data.crew:
            if not crew_member.is_legal(estimated_block_on):
                fdp = crew_member.fdp_at(estimated_block_on)
                violation_h = fdp - crew_member.max_fdp_hours
                illegal_crew.append(crew_member.crew_id)
                if violation_h > max_violation_h:
                    max_violation_h = violation_h

        result.crew_legal = len(illegal_crew) == 0
        result.crew_fdp_violation_hours = max_violation_h
        result.crew_affected = illegal_crew

        # ------------------------------------------------------------------
        # 2. CONNECTION IMPACT
        # ------------------------------------------------------------------
        total_misconnect_pax = 0
        total_revenue_at_risk = 0.0
        conn_details = []

        for conn in data.connections:
            missed = conn.misconnects_at(estimated_block_on)
            revenue = conn.revenue_at_risk(estimated_block_on)
            if missed:
                total_misconnect_pax += conn.pax_count
                total_revenue_at_risk += revenue
            conn_details.append({
                "connecting_flight": conn.connecting_flight,
                "pax": conn.pax_count,
                "missed": missed,
                "revenue_at_risk_usd": round(revenue, 2),
                "has_backup": conn.has_same_day_backup,
            })

        result.misconnecting_pax = total_misconnect_pax
        result.connection_revenue_at_risk_usd = total_revenue_at_risk
        result.connection_details = conn_details

        # ------------------------------------------------------------------
        # 3. FUEL COST ANALYSIS
        # ------------------------------------------------------------------
        # APU burn during gate hold
        hold_hours = result.delay_minutes / 60.0
        apu_extra_kg = data.apu_burn_kg_per_hour * hold_hours
        apu_extra_cost = apu_extra_kg * data.fuel_price_usd_per_kg
        result.apu_extra_fuel_kg = apu_extra_kg
        result.apu_extra_cost_usd = apu_extra_cost

        # Alternate reroute analysis
        if data.alternate_route_extra_nm > 0:
            result.reroute_available = True
            reroute_extra_cost = data.alternate_route_extra_fuel_kg * data.fuel_price_usd_per_kg
            result.reroute_extra_fuel_kg = data.alternate_route_extra_fuel_kg
            result.reroute_extra_cost_usd = reroute_extra_cost
            # Reroute saves: avoids misconnect revenue loss + APU, but costs extra fuel
            total_savings = total_revenue_at_risk + apu_extra_cost
            result.reroute_saves_usd = total_savings - reroute_extra_cost

        # ------------------------------------------------------------------
        # 4. COMPOSITE SCORE & RECOMMENDATION
        # ------------------------------------------------------------------
        score = 0.0

        if not result.crew_legal:
            score += self.WEIGHT_CREW_ILLEGAL + (max_violation_h * 0.05)

        score += (total_revenue_at_risk / 1000) * self.WEIGHT_REVENUE_PER_1K
        score += (apu_extra_cost / 1000) * self.WEIGHT_FUEL_PER_1K

        result.composite_score = min(score, 1.0)

        if result.composite_score >= self.CRITICAL_THRESHOLD:
            result.priority = "CRITICAL"
        elif result.composite_score >= self.HIGH_THRESHOLD:
            result.priority = "HIGH"
        elif result.composite_score >= self.MEDIUM_THRESHOLD:
            result.priority = "MEDIUM"
        else:
            result.priority = "LOW"

        # Recommend action
        if not result.crew_legal:
            result.recommendation = "ESCALATE"  # Must act — crew will go illegal
        elif result.reroute_saves_usd > 5000:
            result.recommendation = "REROUTE"
        elif result.composite_score >= self.MEDIUM_THRESHOLD:
            result.recommendation = "SWAP"
        else:
            result.recommendation = "ACCEPT"

        return result
