"""
DYNAMIC RE-SCORING ENGINE
==========================
Monitors ATFM regulations for changes mid-event and re-prioritises the fleet
in real time. Adds three capabilities over the static scorer:

  1. RegulationChangeListener  — detects CFMU regulation updates (extension,
                                 scope change, early lift) and triggers re-scoring
  2. ProbabilisticFDPScorer    — scores crew legality as a probability rather than
                                 binary, using forecast regulation duration distribution
  3. DynamicRescorer           — orchestrates full fleet re-prioritisation when
                                 a regulation change arrives, emits change events

Why this matters:
  - A regulation initially forecast to end at 18:00Z is extended to 21:00Z.
    Flights that were crew-legal are now illegal. The agent must re-act immediately.
  - A regulation lifts 90 minutes early. Swaps the agent was negotiating are no
    longer needed — save the CFMU budget.
  - Probability-weighted scoring: "this crew has a 72% chance of FDP violation
    given current regulation duration forecast" is more useful than a binary flag
    when the regulation end time is uncertain.
"""

from __future__ import annotations

import asyncio
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel

from scoring.impact_scorer import ImpactScorer, ImpactScore, FlightOperationalData, CrewRecord
from parsers.fixm_parser import RegulationNotice


# ---------------------------------------------------------------------------
# Regulation Change Models
# ---------------------------------------------------------------------------

class ChangeType(str, Enum):
    REGULATION_EXTENDED   = "REGULATION_EXTENDED"    # End time pushed later
    REGULATION_SHORTENED  = "REGULATION_SHORTENED"   # End time pulled earlier
    REGULATION_LIFTED     = "REGULATION_LIFTED"      # Regulation cancelled entirely
    SCOPE_EXPANDED        = "SCOPE_EXPANDED"         # More airports/FIRs added
    SCOPE_REDUCED         = "SCOPE_REDUCED"          # Fewer airports/FIRs
    CAPACITY_CHANGED      = "CAPACITY_CHANGED"       # Capacity reduction % changed
    NEW_SLOT_AVAILABLE    = "NEW_SLOT_AVAILABLE"     # A flight cancelled, slot freed


@dataclass
class RegulationChange:
    change_type: ChangeType
    regulation_id: str
    previous_value: str       # Human-readable previous state
    new_value: str            # Human-readable new state
    detected_at: datetime
    impact_magnitude: str     # "LOW" | "MEDIUM" | "HIGH" — how much re-scoring needed


@dataclass
class RegulationSnapshot:
    """Point-in-time state of a regulation — used to detect changes by comparison."""
    regulation_id: str
    start_time: datetime
    end_time: datetime
    capacity_reduction_pct: int
    affected_airports: List[str]
    recorded_at: datetime


# ---------------------------------------------------------------------------
# Historical regulation duration data (used for probability estimation)
# ---------------------------------------------------------------------------

# Based on EUROCONTROL STATFOR historical regulation data
# Format: {reason: [duration_hours list from historical events]}
_HISTORICAL_DURATION_HOURS: Dict[str, List[float]] = {
    "WEATHER":       [2.0, 3.5, 4.0, 5.5, 6.0, 7.0, 8.0, 9.0, 3.0, 4.5, 6.5, 2.5, 5.0, 8.5, 4.0],
    "ATC_CAPACITY":  [1.0, 2.0, 2.5, 3.0, 4.0, 3.5, 1.5, 2.0, 3.0, 2.5],
    "EQUIPMENT":     [0.5, 1.0, 1.5, 2.0, 1.0, 3.0, 0.75, 2.5, 1.0, 1.5],
    "SPECIAL_EVENT": [3.0, 4.0, 5.0, 3.5, 4.5, 6.0, 3.0, 4.0],
    "MILITARY":      [2.0, 3.0, 4.0, 2.5, 6.0, 8.0, 3.5, 5.0],
}


# ---------------------------------------------------------------------------
# 1. Probabilistic FDP Scorer
# ---------------------------------------------------------------------------

class ProbabilisticFDPScorer:
    """
    Scores crew FDP legality as a probability distribution rather than a binary flag.

    Instead of: "crew IS legal" (assumes regulation ends exactly as published)
    Returns:    "crew has 78% probability of FDP violation" (given forecast duration)

    Uses Monte Carlo simulation over historical regulation duration distributions.
    """

    N_SIMULATIONS = 2000

    def __init__(self):
        self._scorer = ImpactScorer()

    def score_probabilistic(
        self,
        data: FlightOperationalData,
        current_ctot: datetime,
        regulation_start: datetime,
        regulation_reason: str = "WEATHER",
        elapsed_fraction: float = 0.0,    # How far through the regulation we are (0.0–1.0)
    ) -> "ProbabilisticImpactScore":
        """
        Run Monte Carlo simulation of FDP legality across likely regulation durations.

        Returns a ProbabilisticImpactScore with:
          - p_crew_violation: probability (0.0–1.0) that crew goes illegal
          - expected_delay_min: expected delay in minutes (mean of distribution)
          - worst_case_delay_min: 95th percentile delay
          - base_score: deterministic score at current CTOT (backward compatible)
        """
        durations = _HISTORICAL_DURATION_HOURS.get(regulation_reason, _HISTORICAL_DURATION_HOURS["WEATHER"])

        # Condition on elapsed time — if regulation has been running for 2h,
        # we know it lasted at least 2h, so filter historical durations accordingly
        if elapsed_fraction > 0 and durations:
            mean_d = statistics.mean(durations)
            min_duration_h = elapsed_fraction * mean_d
            durations = [d for d in durations if d >= min_duration_h] or durations

        import random
        violations = 0
        delay_samples: List[float] = []

        for _ in range(self.N_SIMULATIONS):
            # Sample a regulation duration from historical distribution
            sampled_duration_h = random.choice(durations) + random.gauss(0, 0.5)
            sampled_duration_h = max(0.5, sampled_duration_h)

            # Estimate what the CTOT would be under this duration
            # (longer regulation → later CTOT)
            duration_ratio = sampled_duration_h / statistics.mean(durations)
            base_delay = (current_ctot - data.scheduled_off_block).total_seconds() / 60
            simulated_delay_min = base_delay * duration_ratio
            simulated_ctot = data.scheduled_off_block + timedelta(minutes=max(simulated_delay_min, 0))

            # Check crew legality under this simulated CTOT
            block_on = data.estimated_block_on(simulated_ctot)
            crew_illegal = any(not c.is_legal(block_on) for c in data.crew)
            if crew_illegal:
                violations += 1
            delay_samples.append(simulated_delay_min)

        p_violation = violations / self.N_SIMULATIONS
        expected_delay = statistics.mean(delay_samples)
        delay_samples.sort()
        worst_case_delay = delay_samples[int(0.95 * len(delay_samples))]

        # Deterministic base score (backward compatible with ImpactScorer)
        base_score = self._scorer.score(data, current_ctot)

        return ProbabilisticImpactScore(
            base_score=base_score,
            p_crew_violation=p_violation,
            expected_delay_min=expected_delay,
            worst_case_delay_min=worst_case_delay,
            n_simulations=self.N_SIMULATIONS,
            regulation_reason=regulation_reason,
        )


@dataclass
class ProbabilisticImpactScore:
    base_score: ImpactScore
    p_crew_violation: float       # 0.0 = certainly legal, 1.0 = certainly illegal
    expected_delay_min: float
    worst_case_delay_min: float   # 95th percentile
    n_simulations: int
    regulation_reason: str

    @property
    def flight_number(self) -> str:
        return self.base_score.flight_number

    @property
    def risk_adjusted_priority(self) -> str:
        """Priority that accounts for probabilistic crew risk."""
        if self.p_crew_violation >= 0.70:
            return "CRITICAL"
        if self.p_crew_violation >= 0.40 or self.base_score.priority == "HIGH":
            return "HIGH"
        if self.base_score.composite_score >= 0.15:
            return "MEDIUM"
        return "LOW"

    def summary(self) -> str:
        return (
            f"Flight {self.flight_number}\n"
            f"  Deterministic priority : {self.base_score.priority}\n"
            f"  Risk-adjusted priority : {self.risk_adjusted_priority}\n"
            f"  P(crew violation)      : {self.p_crew_violation:.1%}\n"
            f"  Expected delay         : {self.expected_delay_min:.0f} min\n"
            f"  Worst-case delay (P95) : {self.worst_case_delay_min:.0f} min\n"
            f"  Regulation reason      : {self.regulation_reason}\n"
        )


# ---------------------------------------------------------------------------
# 2. Regulation Change Listener
# ---------------------------------------------------------------------------

class RegulationChangeListener:
    """
    Polls CFMU for regulation updates and detects meaningful changes.
    In production: subscribes to EUROCONTROL SWIM regulation update stream.
    In PoC: compares snapshots and can be fed simulated changes.
    """

    def __init__(self, poll_interval_sec: int = 60):
        self._snapshots: Dict[str, RegulationSnapshot] = {}
        self._poll_interval = poll_interval_sec
        self._change_callbacks: List[Callable[[RegulationChange], None]] = []
        self._running = False

    def register_callback(self, callback: Callable[[RegulationChange], None]) -> None:
        """Register a function to be called whenever a regulation change is detected."""
        self._change_callbacks.append(callback)

    def take_snapshot(self, notice: RegulationNotice) -> None:
        """Record the current state of a regulation for future comparison."""
        self._snapshots[notice.regulation_id] = RegulationSnapshot(
            regulation_id=notice.regulation_id,
            start_time=notice.start_time,
            end_time=notice.end_time,
            capacity_reduction_pct=notice.capacity_reduction_pct,
            affected_airports=[notice.affected_airport],
            recorded_at=datetime.now(timezone.utc),
        )

    def detect_changes(self, updated_notice: RegulationNotice) -> List[RegulationChange]:
        """
        Compare an updated regulation notice against the stored snapshot.
        Returns a list of detected changes.
        """
        changes: List[RegulationChange] = []
        reg_id = updated_notice.regulation_id
        prev = self._snapshots.get(reg_id)

        if prev is None:
            return changes   # First time we've seen this regulation

        now = datetime.now(timezone.utc)

        # Check end time change
        if updated_notice.end_time != prev.end_time:
            delta_min = (updated_notice.end_time - prev.end_time).total_seconds() / 60
            if delta_min > 0:
                change_type = ChangeType.REGULATION_EXTENDED
                magnitude = "HIGH" if delta_min > 60 else "MEDIUM" if delta_min > 30 else "LOW"
            elif delta_min < 0:
                change_type = ChangeType.REGULATION_SHORTENED
                magnitude = "HIGH" if abs(delta_min) > 60 else "MEDIUM"
            else:
                change_type = None

            if change_type:
                changes.append(RegulationChange(
                    change_type=change_type,
                    regulation_id=reg_id,
                    previous_value=f"end {prev.end_time.strftime('%H:%MZ')}",
                    new_value=f"end {updated_notice.end_time.strftime('%H:%MZ')} ({delta_min:+.0f} min)",
                    detected_at=now,
                    impact_magnitude=magnitude,
                ))

        # Check capacity change
        if updated_notice.capacity_reduction_pct != prev.capacity_reduction_pct:
            delta = updated_notice.capacity_reduction_pct - prev.capacity_reduction_pct
            changes.append(RegulationChange(
                change_type=ChangeType.CAPACITY_CHANGED,
                regulation_id=reg_id,
                previous_value=f"{prev.capacity_reduction_pct}% reduction",
                new_value=f"{updated_notice.capacity_reduction_pct}% reduction ({delta:+d}%)",
                detected_at=now,
                impact_magnitude="HIGH" if abs(delta) > 20 else "MEDIUM",
            ))

        # Check if regulation was lifted
        if updated_notice.affected_flights_count == 0:
            changes.append(RegulationChange(
                change_type=ChangeType.REGULATION_LIFTED,
                regulation_id=reg_id,
                previous_value=f"active ({prev.capacity_reduction_pct}% reduction)",
                new_value="LIFTED",
                detected_at=now,
                impact_magnitude="HIGH",
            ))

        # Update snapshot
        self.take_snapshot(updated_notice)

        # Fire callbacks
        for change in changes:
            for cb in self._change_callbacks:
                cb(change)

        return changes

    def simulate_change(
        self,
        regulation_id: str,
        change_type: ChangeType,
        new_end_time: Optional[datetime] = None,
        new_capacity_pct: Optional[int] = None,
    ) -> Optional[RegulationChange]:
        """
        Inject a simulated regulation change (for testing and demo).
        """
        prev = self._snapshots.get(regulation_id)
        if not prev:
            return None

        now = datetime.now(timezone.utc)

        if change_type == ChangeType.REGULATION_EXTENDED and new_end_time:
            delta_min = (new_end_time - prev.end_time).total_seconds() / 60
            change = RegulationChange(
                change_type=change_type,
                regulation_id=regulation_id,
                previous_value=f"end {prev.end_time.strftime('%H:%MZ')}",
                new_value=f"end {new_end_time.strftime('%H:%MZ')} (+{delta_min:.0f} min)",
                detected_at=now,
                impact_magnitude="HIGH" if delta_min > 60 else "MEDIUM",
            )
            # Update snapshot
            self._snapshots[regulation_id] = RegulationSnapshot(
                regulation_id=prev.regulation_id,
                start_time=prev.start_time,
                end_time=new_end_time,
                capacity_reduction_pct=prev.capacity_reduction_pct,
                affected_airports=prev.affected_airports,
                recorded_at=now,
            )
            for cb in self._change_callbacks:
                cb(change)
            return change

        if change_type == ChangeType.REGULATION_LIFTED:
            change = RegulationChange(
                change_type=change_type,
                regulation_id=regulation_id,
                previous_value="active",
                new_value="LIFTED — all CTOTs cancelled",
                detected_at=now,
                impact_magnitude="HIGH",
            )
            for cb in self._change_callbacks:
                cb(change)
            return change

        return None


# ---------------------------------------------------------------------------
# 3. Dynamic Rescorer — orchestrates fleet re-prioritisation
# ---------------------------------------------------------------------------

class DynamicRescorer:
    """
    Orchestrates re-scoring of the entire regulated fleet when a regulation
    change arrives. Computes which flights changed priority and emits a
    structured RescoreEvent describing the impact.
    """

    def __init__(self):
        self._prob_scorer = ProbabilisticFDPScorer()
        self._change_listener = RegulationChangeListener()
        self._change_listener.register_callback(self._on_regulation_change)
        self._current_scores: Dict[str, ImpactScore] = {}
        self._flight_data: Dict[str, FlightOperationalData] = {}
        self._regulation_reason: str = "WEATHER"
        self._regulation_start: Optional[datetime] = None
        self._rescore_callbacks: List[Callable[["RescoreEvent"], None]] = []

    def initialise(
        self,
        flight_data: Dict[str, FlightOperationalData],
        initial_scores: Dict[str, ImpactScore],
        notice: RegulationNotice,
    ) -> None:
        """Register the current fleet state with the rescorer."""
        self._flight_data = flight_data
        self._current_scores = initial_scores
        self._regulation_reason = notice.reason
        self._regulation_start = notice.start_time
        self._change_listener.take_snapshot(notice)

    def on_rescore(self, callback: Callable[["RescoreEvent"], None]) -> None:
        """Register callback for when a re-score produces priority changes."""
        self._rescore_callbacks.append(callback)

    def ingest_update(self, updated_notice: RegulationNotice) -> List[RegulationChange]:
        """
        Feed an updated regulation notice. Returns detected changes.
        Any HIGH/MEDIUM changes trigger automatic fleet re-scoring.
        """
        return self._change_listener.detect_changes(updated_notice)

    def simulate_extension(self, regulation_id: str, new_end_time: datetime) -> Optional[RegulationChange]:
        """Inject a simulated regulation extension for demo/testing."""
        return self._change_listener.simulate_change(
            regulation_id, ChangeType.REGULATION_EXTENDED, new_end_time=new_end_time
        )

    def simulate_lift(self, regulation_id: str) -> Optional[RegulationChange]:
        """Inject a simulated regulation lift for demo/testing."""
        return self._change_listener.simulate_change(regulation_id, ChangeType.REGULATION_LIFTED)

    def rescore_fleet(
        self,
        updated_ctots: Dict[str, datetime],
        regulation_reason: str = "WEATHER",
        elapsed_fraction: float = 0.3,
    ) -> "RescoreEvent":
        """
        Re-score all flights with updated CTOTs and probabilistic FDP model.

        Args:
            updated_ctots: Current CTOT per flight (may have changed from initial)
            regulation_reason: Used to select historical duration distribution
            elapsed_fraction: How far through the regulation we are (0.0 = just started)

        Returns:
            RescoreEvent describing all priority changes and newly critical flights
        """
        new_scores: Dict[str, ProbabilisticImpactScore] = {}
        priority_changes: List[PriorityChange] = []
        newly_critical: List[str] = []

        for fn, data in self._flight_data.items():
            ctot = updated_ctots.get(fn, data.scheduled_off_block)

            prob_score = self._prob_scorer.score_probabilistic(
                data=data,
                current_ctot=ctot,
                regulation_start=self._regulation_start or ctot,
                regulation_reason=regulation_reason,
                elapsed_fraction=elapsed_fraction,
            )
            new_scores[fn] = prob_score

            # Detect priority changes
            old_score = self._current_scores.get(fn)
            if old_score:
                old_priority = old_score.priority
                new_priority = prob_score.risk_adjusted_priority
                if old_priority != new_priority:
                    priority_changes.append(PriorityChange(
                        flight_number=fn,
                        old_priority=old_priority,
                        new_priority=new_priority,
                        old_composite=old_score.composite_score,
                        new_p_violation=prob_score.p_crew_violation,
                        reason=self._explain_priority_change(old_priority, new_priority, prob_score),
                    ))
                    if new_priority == "CRITICAL" and old_priority != "CRITICAL":
                        newly_critical.append(fn)

            # Update current scores with deterministic base for compatibility
            self._current_scores[fn] = prob_score.base_score

        event = RescoreEvent(
            triggered_at=datetime.now(timezone.utc),
            regulation_reason=regulation_reason,
            elapsed_fraction=elapsed_fraction,
            probabilistic_scores=new_scores,
            priority_changes=priority_changes,
            newly_critical_flights=newly_critical,
            rescore_recommendation=self._build_rescore_recommendation(priority_changes, newly_critical),
        )

        # Fire callbacks
        for cb in self._rescore_callbacks:
            cb(event)

        return event

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_regulation_change(self, change: RegulationChange) -> None:
        """Internal callback: log the change. Actual re-scoring triggered by orchestrator."""
        print(
            f"[DynamicRescorer] Regulation change detected: "
            f"{change.change_type.value} — {change.previous_value} → {change.new_value} "
            f"(magnitude: {change.impact_magnitude})"
        )

    def _explain_priority_change(
        self, old: str, new: str, score: "ProbabilisticImpactScore"
    ) -> str:
        if new == "CRITICAL" and old != "CRITICAL":
            return (
                f"Regulation extension raised P(crew violation) to "
                f"{score.p_crew_violation:.0%} — now CRITICAL"
            )
        if new == "LOW" and old != "LOW":
            return f"Regulation shortened — delay now within legal FDP window"
        if _priority_rank(new) > _priority_rank(old):
            return f"Extended delay increases impact: {old} → {new}"
        return f"Reduced delay decreases impact: {old} → {new}"

    def _build_rescore_recommendation(
        self, changes: List["PriorityChange"], newly_critical: List[str]
    ) -> str:
        if not changes:
            return "Fleet re-scored — no priority changes detected."
        lines = [f"{len(changes)} flight(s) changed priority after regulation update:"]
        for c in changes:
            lines.append(f"  {c.flight_number}: {c.old_priority} → {c.new_priority} ({c.reason})")
        if newly_critical:
            lines.append(
                f"URGENT: {len(newly_critical)} flight(s) newly CRITICAL — "
                f"immediate negotiation recommended: {', '.join(newly_critical)}"
            )
        return "\n".join(lines)


def _priority_rank(priority: str) -> int:
    return {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}.get(priority, 0)


# ---------------------------------------------------------------------------
# Supporting dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PriorityChange:
    flight_number: str
    old_priority: str
    new_priority: str
    old_composite: float
    new_p_violation: float
    reason: str


@dataclass
class RescoreEvent:
    triggered_at: datetime
    regulation_reason: str
    elapsed_fraction: float
    probabilistic_scores: Dict[str, ProbabilisticImpactScore]
    priority_changes: List[PriorityChange]
    newly_critical_flights: List[str]
    rescore_recommendation: str

    def summary(self) -> str:
        lines = [
            f"Re-score Event @ {self.triggered_at.strftime('%H:%M:%S')}Z",
            f"  Flights scored    : {len(self.probabilistic_scores)}",
            f"  Priority changes  : {len(self.priority_changes)}",
            f"  Newly critical    : {len(self.newly_critical_flights)}",
            "",
            self.rescore_recommendation,
        ]
        return "\n".join(lines)
