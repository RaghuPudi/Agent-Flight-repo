"""
NEGOTIATION PLANNER & COUNTER-PROPOSAL ENGINE
Implements the multi-round negotiation strategy:
  Round 1 → Accept (if low impact)
  Round 2 → Submit slot swap request via CFMU SSR
  Round 3 → Counter-CTOT with justification
  Round 4 → Escalate to human OCC coordinator
  Round 5 → Request regulatory exemption

Also implements the Swap Pool scanner for finding mutually beneficial exchanges.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel

from scoring.impact_scorer import ImpactScore


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class SwapPoolEntry(BaseModel):
    """Represents another airline's slot that may be available for swap."""
    entry_id: str
    airline: str
    flight_number: str
    regulation_id: str
    assigned_ctot: datetime
    desired_ctot: Optional[datetime] = None   # None = flexible
    priority_score: float = 0.0               # Low = willing to swap


class NegotiationProposal(BaseModel):
    proposal_id: str
    flight_number: str
    regulation_id: str
    round_number: int
    action: str           # "ACCEPT" | "SWAP_REQUEST" | "COUNTER_CTOT" | "ESCALATE" | "REROUTE" | "EXEMPTION"
    current_ctot: datetime
    requested_ctot: Optional[datetime] = None
    swap_with_flight: Optional[str] = None   # Partner flight for swap
    justification_codes: List[str] = field(default_factory=list)
    justification_text: str = ""
    estimated_benefit_usd: float = 0.0
    submitted_at: Optional[datetime] = None
    response: Optional[str] = None           # CFMU response text
    accepted: Optional[bool] = None

    class Config:
        arbitrary_types_allowed = True


@dataclass
class NegotiationState:
    flight_number: str
    regulation_id: str
    original_ctot: datetime
    current_ctot: datetime
    impact_score: ImpactScore
    proposals: List[NegotiationProposal] = field(default_factory=list)
    resolved: bool = False
    final_ctot: Optional[datetime] = None
    rounds_used: int = 0


# ---------------------------------------------------------------------------
# CFMU API Stub (simulates EUROCONTROL B2B calls)
# ---------------------------------------------------------------------------

class CFMUApiClient:
    """
    Stub for EUROCONTROL CFMU B2B Web Services.
    In production: HTTPS calls to EUROCONTROL NM B2B SOAP/REST endpoints.
    """

    def __init__(self, accept_rate: float = 0.65):
        self._accept_rate = accept_rate   # Simulated acceptance probability

    def submit_slot_swap(self, proposal: NegotiationProposal) -> Tuple[bool, str, Optional[datetime]]:
        """
        Submit a slot swap request (SSR) to CFMU.
        Returns: (accepted, message, new_ctot_if_accepted)
        """
        import random
        accepted = random.random() < self._accept_rate
        if accepted:
            # Simulate CFMU granting a slot 10-30 min better than requested
            offset = random.randint(0, 20)
            new_ctot = proposal.requested_ctot + timedelta(minutes=offset)
            return True, "SLOT_SWAP_ACCEPTED", new_ctot
        return False, "SLOT_SWAP_REJECTED: No compatible slot available", None

    def submit_counter_ctot(self, proposal: NegotiationProposal) -> Tuple[bool, str, Optional[datetime]]:
        """Submit a counter-CTOT request with justification codes."""
        import random
        # Higher chance of acceptance when crew is illegal (CFMU respects safety codes)
        has_safety_code = "FDP_VIOLATION" in proposal.justification_codes
        base_rate = 0.75 if has_safety_code else 0.45
        accepted = random.random() < base_rate
        if accepted:
            offset = random.randint(0, 15)
            new_ctot = proposal.requested_ctot + timedelta(minutes=offset)
            return True, "COUNTER_CTOT_ACCEPTED", new_ctot
        return False, "COUNTER_CTOT_REJECTED: Insufficient capacity", None

    def get_swap_pool(self, regulation_id: str) -> List[SwapPoolEntry]:
        """Fetch available slots in the swap pool for this regulation."""
        # Simulated swap pool entries from other airlines
        base_time = datetime(2026, 10, 15, 14, 0, tzinfo=timezone.utc)
        return [
            SwapPoolEntry(
                entry_id="POOL-001",
                airline="LH",
                flight_number="LH234",
                regulation_id=regulation_id,
                assigned_ctot=base_time + timedelta(minutes=55),
                priority_score=0.1,   # LH doesn't care about this slot
            ),
            SwapPoolEntry(
                entry_id="POOL-002",
                airline="AF",
                flight_number="AF891",
                regulation_id=regulation_id,
                assigned_ctot=base_time + timedelta(minutes=30),
                priority_score=0.05,
            ),
            SwapPoolEntry(
                entry_id="POOL-003",
                airline="IB",
                flight_number="IB456",
                regulation_id=regulation_id,
                assigned_ctot=base_time + timedelta(minutes=15),
                priority_score=0.3,   # IB wants this slot
            ),
        ]


# ---------------------------------------------------------------------------
# Negotiation Planner
# ---------------------------------------------------------------------------

class NegotiationPlanner:
    """
    Implements the 5-round negotiation strategy for a single regulated flight.
    Determines what action to take at each round based on impact score and
    what CFMU has responded with so far.
    """

    # Thresholds for deciding when to fight vs accept
    ACCEPT_SCORE_THRESHOLD = 0.15       # Below this → just accept
    SWAP_SCORE_THRESHOLD = 0.35         # Above this → try swap
    ESCALATE_SCORE_THRESHOLD = 0.70     # Above this → escalate to human

    # How far ahead in time to request a CTOT (target improvement in minutes)
    TARGET_IMPROVEMENT_MIN = 90

    def __init__(self, cfmu_client: Optional[CFMUApiClient] = None):
        self._cfmu = cfmu_client or CFMUApiClient()
        self._states: Dict[str, NegotiationState] = {}

    def negotiate(self, impact: ImpactScore, original_sobt: datetime) -> NegotiationState:
        """
        Run the full negotiation loop for a single flight.
        Returns the final NegotiationState after all rounds are exhausted or resolved.
        """
        state = NegotiationState(
            flight_number=impact.flight_number,
            regulation_id="REG-" + impact.flight_number,
            original_ctot=impact.ctot,
            current_ctot=impact.ctot,
            impact_score=impact,
        )
        self._states[impact.flight_number] = state

        # Round 1: Accept if low impact
        if impact.composite_score < self.ACCEPT_SCORE_THRESHOLD:
            return self._accept(state, reason="Low composite impact score.")

        # Round 2: Reroute if better economics
        if impact.recommendation == "REROUTE" and impact.reroute_saves_usd > 5000:
            return self._request_reroute(state)

        # Round 3: Try slot swap from pool
        swap_result = self._try_swap_pool(state)
        if swap_result:
            return state

        # Round 4: Submit counter-CTOT
        counter_result = self._submit_counter_ctot(state, original_sobt)
        if counter_result:
            return state

        # Round 5: Escalate to human
        return self._escalate(state)

    # ------------------------------------------------------------------
    # Round implementations
    # ------------------------------------------------------------------

    def _accept(self, state: NegotiationState, reason: str = "") -> NegotiationState:
        proposal = NegotiationProposal(
            proposal_id=str(uuid.uuid4()),
            flight_number=state.flight_number,
            regulation_id=state.regulation_id,
            round_number=1,
            action="ACCEPT",
            current_ctot=state.current_ctot,
            justification_text=reason,
            submitted_at=datetime.now(timezone.utc),
            accepted=True,
        )
        state.proposals.append(proposal)
        state.resolved = True
        state.final_ctot = state.current_ctot
        state.rounds_used = 1
        return state

    def _try_swap_pool(self, state: NegotiationState) -> bool:
        """
        Scan the CFMU swap pool for a compatible slot.
        Returns True if a swap was successfully negotiated.
        """
        state.rounds_used = max(state.rounds_used, 2)
        pool = self._cfmu.get_swap_pool(state.regulation_id)

        # Find entries with earlier CTOTs and low priority (willing to swap)
        candidates = [
            e for e in pool
            if e.assigned_ctot < state.current_ctot and e.priority_score < 0.25
        ]
        if not candidates:
            return False

        # Pick the best candidate (earliest CTOT among willing swappers)
        best = min(candidates, key=lambda e: e.assigned_ctot)

        proposal = NegotiationProposal(
            proposal_id=str(uuid.uuid4()),
            flight_number=state.flight_number,
            regulation_id=state.regulation_id,
            round_number=2,
            action="SWAP_REQUEST",
            current_ctot=state.current_ctot,
            requested_ctot=best.assigned_ctot,
            swap_with_flight=best.flight_number,
            justification_codes=self._build_justification_codes(state.impact_score),
            justification_text=(
                f"Requesting slot swap with {best.airline}/{best.flight_number}. "
                f"Current CTOT {state.current_ctot.strftime('%H:%M')}Z causes "
                f"{'crew FDP violation and ' if not state.impact_score.crew_legal else ''}"
                f"{state.impact_score.misconnecting_pax} misconnecting pax "
                f"(${state.impact_score.connection_revenue_at_risk_usd:,.0f} revenue at risk)."
            ),
            estimated_benefit_usd=(
                state.impact_score.connection_revenue_at_risk_usd
                + state.impact_score.apu_extra_cost_usd
            ),
            submitted_at=datetime.now(timezone.utc),
        )

        accepted, message, new_ctot = self._cfmu.submit_slot_swap(proposal)
        proposal.response = message
        proposal.accepted = accepted

        if accepted and new_ctot:
            proposal.requested_ctot = new_ctot
            state.current_ctot = new_ctot
            state.final_ctot = new_ctot
            state.resolved = True

        state.proposals.append(proposal)
        return accepted

    def _submit_counter_ctot(self, state: NegotiationState, original_sobt: datetime) -> bool:
        """Submit a counter-CTOT request with regulatory justification."""
        state.rounds_used = max(state.rounds_used, 3)

        # Request a CTOT close to original SOBT + realistic taxi time
        target = original_sobt + timedelta(minutes=20 + 10)  # SOBT + taxi estimate

        justification_codes = self._build_justification_codes(state.impact_score)

        proposal = NegotiationProposal(
            proposal_id=str(uuid.uuid4()),
            flight_number=state.flight_number,
            regulation_id=state.regulation_id,
            round_number=3,
            action="COUNTER_CTOT",
            current_ctot=state.current_ctot,
            requested_ctot=target,
            justification_codes=justification_codes,
            justification_text=self._build_justification_text(state.impact_score, target),
            estimated_benefit_usd=(
                state.impact_score.connection_revenue_at_risk_usd
                + state.impact_score.apu_extra_cost_usd
            ),
            submitted_at=datetime.now(timezone.utc),
        )

        accepted, message, new_ctot = self._cfmu.submit_counter_ctot(proposal)
        proposal.response = message
        proposal.accepted = accepted

        if accepted and new_ctot:
            proposal.requested_ctot = new_ctot
            state.current_ctot = new_ctot
            state.final_ctot = new_ctot
            state.resolved = True

        state.proposals.append(proposal)
        return accepted

    def _request_reroute(self, state: NegotiationState) -> NegotiationState:
        """Request an alternate route that avoids the regulation entirely."""
        state.rounds_used = max(state.rounds_used, 2)
        proposal = NegotiationProposal(
            proposal_id=str(uuid.uuid4()),
            flight_number=state.flight_number,
            regulation_id=state.regulation_id,
            round_number=2,
            action="REROUTE",
            current_ctot=state.current_ctot,
            justification_text=(
                f"Reroute via alternate waypoint saves "
                f"${state.impact_score.reroute_saves_usd:,.0f} net "
                f"vs accepting current CTOT."
            ),
            estimated_benefit_usd=state.impact_score.reroute_saves_usd,
            submitted_at=datetime.now(timezone.utc),
            accepted=True,
        )
        state.proposals.append(proposal)
        state.resolved = True
        state.final_ctot = state.original_ctot   # Reroute → depart on time
        return state

    def _escalate(self, state: NegotiationState) -> NegotiationState:
        """Escalate to human OCC coordinator with a full brief."""
        state.rounds_used = max(state.rounds_used, 4)
        brief = self._build_escalation_brief(state)
        proposal = NegotiationProposal(
            proposal_id=str(uuid.uuid4()),
            flight_number=state.flight_number,
            regulation_id=state.regulation_id,
            round_number=4,
            action="ESCALATE",
            current_ctot=state.current_ctot,
            justification_text=brief,
            estimated_benefit_usd=state.impact_score.connection_revenue_at_risk_usd,
            submitted_at=datetime.now(timezone.utc),
        )
        state.proposals.append(proposal)
        state.resolved = False   # Awaiting human decision
        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_justification_codes(self, impact: ImpactScore) -> List[str]:
        codes = []
        if not impact.crew_legal:
            codes.append("FDP_VIOLATION")
        if impact.misconnecting_pax > 20:
            codes.append("HIGH_PAX_MISCONNECT")
        if impact.connection_revenue_at_risk_usd > 10000:
            codes.append("SIGNIFICANT_REVENUE_IMPACT")
        if impact.apu_extra_cost_usd > 500:
            codes.append("FUEL_INEFFICIENCY")
        return codes or ["OPERATIONAL_IMPACT"]

    def _build_justification_text(self, impact: ImpactScore, requested_ctot: datetime) -> str:
        lines = [
            f"Requesting CTOT {requested_ctot.strftime('%H:%M')}Z for flight {impact.flight_number}.",
        ]
        if not impact.crew_legal:
            lines.append(
                f"Current CTOT causes FDP violation of {impact.crew_fdp_violation_hours:.1f}h "
                f"for crew members: {', '.join(impact.crew_affected)}."
            )
        if impact.misconnecting_pax > 0:
            lines.append(
                f"{impact.misconnecting_pax} passengers will misconnect, "
                f"placing ${impact.connection_revenue_at_risk_usd:,.0f} revenue at risk."
            )
        return " ".join(lines)

    def _build_escalation_brief(self, state: NegotiationState) -> str:
        impact = state.impact_score
        attempts = len(state.proposals)
        lines = [
            f"=== ESCALATION BRIEF: {state.flight_number} ===",
            f"Current CTOT  : {state.current_ctot.strftime('%H:%M')}Z",
            f"Priority      : {impact.priority}",
            f"Crew legal    : {'YES' if impact.crew_legal else f'NO — {impact.crew_fdp_violation_hours:.1f}h over limit'}",
            f"Misconnects   : {impact.misconnecting_pax} pax (${impact.connection_revenue_at_risk_usd:,.0f})",
            f"Agent rounds  : {attempts} attempts (swap + counter-CTOT both rejected)",
            f"Recommended   : Manual call to CFMU / request exemption",
        ]
        return "\n".join(lines)
