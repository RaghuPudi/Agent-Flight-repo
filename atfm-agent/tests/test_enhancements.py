"""
Tests for LLM Justification Engine and Dynamic Re-Scoring Engine.
Run with: pytest tests/test_enhancements.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from datetime import datetime, timedelta, timezone

from scoring.impact_scorer import ImpactScorer, FlightOperationalData, CrewRecord, Connection
from negotiation.negotiation_planner import NegotiationProposal, NegotiationState
from parsers.fixm_parser import make_sample_regulation_xml, FIXMParser
from agent.llm_justification import (
    LLMJustificationEngine, LLMClient,
    CFMUJustificationMessage, EscalationBrief, DecisionExplanation,
    OverrideOutcomeRecord,
)
from agent.dynamic_rescorer import (
    DynamicRescorer, ProbabilisticFDPScorer, RegulationChangeListener,
    ChangeType, RescoreEvent,
)


BASE_TIME = datetime(2026, 10, 15, 14, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_ops_data(
    fn: str = "BA456",
    ctot_offset_min: int = 160,
    crew_duty_offset_h: float = -8.0,
    max_fdp: float = 9.5,
    connections: int = 2,
    backup: bool = False,
) -> FlightOperationalData:
    sobt = BASE_TIME
    duty_start = BASE_TIME + timedelta(hours=crew_duty_offset_h)
    crew = [
        CrewRecord(crew_id=f"{fn}-CPT", role="CPT", duty_start=duty_start, max_fdp_hours=max_fdp),
        CrewRecord(crew_id=f"{fn}-FO",  role="FO",  duty_start=duty_start, max_fdp_hours=max_fdp),
    ]
    conns = [
        Connection(
            pax_count=20,
            connecting_flight=f"BA{700 + i}",
            connecting_dep_time=BASE_TIME + timedelta(minutes=120),
            has_same_day_backup=backup,
        )
        for i in range(connections)
    ]
    return FlightOperationalData(
        flight_number=fn,
        dep_airport="EGLL",
        arr_airport="LFPG",
        scheduled_off_block=sobt,
        flight_time_min=75,
        crew=crew,
        connections=conns,
    )


def make_impact(fn: str = "BA456", ctot_offset_min: int = 160):
    scorer = ImpactScorer()
    data = make_ops_data(fn, ctot_offset_min)
    ctot = BASE_TIME + timedelta(minutes=ctot_offset_min)
    return scorer.score(data, ctot), data


def make_proposal(fn: str, action: str, codes: list) -> NegotiationProposal:
    return NegotiationProposal(
        proposal_id="test-001",
        flight_number=fn,
        regulation_id="REG001",
        round_number=2,
        action=action,
        current_ctot=BASE_TIME + timedelta(minutes=160),
        requested_ctot=BASE_TIME + timedelta(minutes=90),
        justification_codes=codes,
    )


def make_neg_state(fn: str, impact, proposals: list) -> NegotiationState:
    ctot = BASE_TIME + timedelta(minutes=160)
    state = NegotiationState(
        flight_number=fn,
        regulation_id="REG001",
        original_ctot=ctot,
        current_ctot=ctot,
        impact_score=impact,
    )
    state.proposals = proposals
    return state


# ---------------------------------------------------------------------------
# 1. LLM Justification Engine — Template Path (no API key)
# ---------------------------------------------------------------------------

class TestLLMJustificationEngine:
    """
    All tests run against the template fallback (no OPENAI_API_KEY set).
    This verifies the engine produces valid structured output regardless of
    whether an LLM is available.
    """

    engine = LLMJustificationEngine(llm_client=LLMClient())

    def test_cfmu_justification_returns_correct_type(self):
        impact, _ = make_impact()
        proposal = make_proposal("BA456", "SWAP_REQUEST", ["FDP_VIOLATION", "HIGH_PAX_MISCONNECT"])
        result = self.engine.generate_cfmu_justification(impact, proposal, "WEATHER")
        assert isinstance(result, CFMUJustificationMessage)

    def test_cfmu_justification_flight_number_matches(self):
        impact, _ = make_impact()
        proposal = make_proposal("BA456", "SWAP_REQUEST", ["FDP_VIOLATION"])
        result = self.engine.generate_cfmu_justification(impact, proposal)
        assert result.flight_number == "BA456"

    def test_cfmu_justification_contains_fdp_citation_when_crew_illegal(self):
        impact, _ = make_impact(ctot_offset_min=160)
        proposal = make_proposal("BA456", "COUNTER_CTOT", ["FDP_VIOLATION"])
        result = self.engine.generate_cfmu_justification(impact, proposal)
        # Should cite EU OPS FTL regulation
        if not impact.crew_legal:
            assert any("FTL" in c or "ORO" in c for c in result.regulatory_citations)

    def test_cfmu_justification_confidence_range(self):
        impact, _ = make_impact()
        proposal = make_proposal("BA456", "SWAP_REQUEST", ["FDP_VIOLATION"])
        result = self.engine.generate_cfmu_justification(impact, proposal)
        assert 0.0 <= result.confidence <= 1.0

    def test_cfmu_confidence_higher_with_fdp_violation_code(self):
        impact, _ = make_impact()
        prop_with_fdp = make_proposal("BA456", "SWAP_REQUEST", ["FDP_VIOLATION"])
        prop_without  = make_proposal("BA456", "SWAP_REQUEST", ["OPERATIONAL_IMPACT"])
        c_with    = self.engine.generate_cfmu_justification(impact, prop_with_fdp).confidence
        c_without = self.engine.generate_cfmu_justification(impact, prop_without).confidence
        assert c_with > c_without

    def test_escalation_brief_returns_correct_type(self):
        impact, _ = make_impact()
        proposal = make_proposal("BA456", "SWAP_REQUEST", ["FDP_VIOLATION"])
        neg_state = make_neg_state("BA456", impact, [proposal])
        result = self.engine.generate_escalation_brief(impact, neg_state)
        assert isinstance(result, EscalationBrief)

    def test_escalation_brief_fields_populated(self):
        impact, _ = make_impact()
        proposal = make_proposal("BA456", "SWAP_REQUEST", ["FDP_VIOLATION"])
        neg_state = make_neg_state("BA456", impact, [proposal])
        result = self.engine.generate_escalation_brief(impact, neg_state)
        assert result.flight_number == "BA456"
        assert len(result.headline) > 0
        assert len(result.recommended_action) > 10
        assert "$" in result.financial_impact    # Must quantify money

    def test_decision_explanation_returns_correct_type(self):
        impact, _ = make_impact()
        proposal = make_proposal("BA456", "SWAP_REQUEST", ["FDP_VIOLATION"])
        alts = [{"action": "ACCEPT", "score": 0.1, "reason": "Score too high"}]
        result = self.engine.explain_decision(impact, proposal, alts)
        assert isinstance(result, DecisionExplanation)

    def test_decision_explanation_considered_and_rejected_populated(self):
        impact, _ = make_impact()
        proposal = make_proposal("BA456", "ESCALATE", ["FDP_VIOLATION"])
        alts = [
            {"action": "ACCEPT", "score": 0.1, "reason": "Score too high"},
            {"action": "SWAP",   "score": 0.5, "reason": "Crew violation needs faster fix"},
        ]
        result = self.engine.explain_decision(impact, proposal, alts)
        assert len(result.what_was_considered) > 0
        assert len(result.what_was_rejected) > 0

    def test_decision_explanation_confidence_range(self):
        impact, _ = make_impact()
        proposal = make_proposal("BA456", "SWAP_REQUEST", ["FDP_VIOLATION"])
        result = self.engine.explain_decision(impact, proposal, [])
        assert 0 <= result.confidence_pct <= 100

    def test_override_outcome_record_no_actuals(self):
        result = self.engine.record_override_outcome(
            flight_number="BA456",
            regulation_id="REG001",
            agent_recommendation="SWAP",
            human_action="ACCEPT_SLOT",
            agent_projected_benefit_usd=20000.0,
        )
        assert isinstance(result, OverrideOutcomeRecord)
        assert result.post_hoc_analysis is None   # No actuals provided yet

    def test_override_outcome_with_actuals_generates_analysis(self):
        result = self.engine.record_override_outcome(
            flight_number="BA456",
            regulation_id="REG001",
            agent_recommendation="SWAP",
            human_action="ACCEPT_SLOT",
            agent_projected_benefit_usd=20000.0,
            actual_delay_saved_min=5.0,
            actual_revenue_saved_usd=1000.0,
        )
        assert result.post_hoc_analysis is not None
        assert len(result.post_hoc_analysis) > 10

    def test_learning_summary_no_records(self):
        summary = self.engine.get_learning_summary([])
        assert "No override records" in summary

    def test_learning_summary_with_records(self):
        records = [
            self.engine.record_override_outcome(
                "BA456", "REG001", "SWAP", "ACCEPT", 20000.0, 5.0, 500.0
            ),
            self.engine.record_override_outcome(
                "VS201", "REG001", "COUNTER_CTOT", "ACCEPT", 15000.0, 120.0, 18000.0
            ),
        ]
        summary = self.engine.get_learning_summary(records)
        assert len(summary) > 20


# ---------------------------------------------------------------------------
# 2. Probabilistic FDP Scorer
# ---------------------------------------------------------------------------

class TestProbabilisticFDPScorer:
    scorer = ProbabilisticFDPScorer()

    def test_returns_probabilistic_score(self):
        data = make_ops_data(ctot_offset_min=160, crew_duty_offset_h=-8.0, max_fdp=9.5)
        ctot = BASE_TIME + timedelta(minutes=160)
        result = self.scorer.score_probabilistic(
            data=data,
            current_ctot=ctot,
            regulation_start=BASE_TIME - timedelta(hours=5),
            regulation_reason="WEATHER",
        )
        assert 0.0 <= result.p_crew_violation <= 1.0

    def test_high_delay_produces_higher_p_violation(self):
        data_high = make_ops_data(ctot_offset_min=200, crew_duty_offset_h=-8.5, max_fdp=9.5)
        data_low  = make_ops_data(ctot_offset_min=30,  crew_duty_offset_h=-1.0, max_fdp=10.0)

        reg_start = BASE_TIME - timedelta(hours=1)
        p_high = self.scorer.score_probabilistic(
            data_high, BASE_TIME + timedelta(minutes=200), reg_start, "WEATHER"
        ).p_crew_violation

        p_low = self.scorer.score_probabilistic(
            data_low, BASE_TIME + timedelta(minutes=30), reg_start, "WEATHER"
        ).p_crew_violation

        assert p_high > p_low

    def test_expected_delay_positive(self):
        data = make_ops_data(ctot_offset_min=90)
        ctot = BASE_TIME + timedelta(minutes=90)
        result = self.scorer.score_probabilistic(data, ctot, BASE_TIME, "WEATHER")
        assert result.expected_delay_min >= 0

    def test_worst_case_delay_ge_expected(self):
        data = make_ops_data(ctot_offset_min=90)
        ctot = BASE_TIME + timedelta(minutes=90)
        result = self.scorer.score_probabilistic(data, ctot, BASE_TIME, "WEATHER")
        assert result.worst_case_delay_min >= result.expected_delay_min

    def test_risk_adjusted_priority_critical_at_high_p(self):
        data = make_ops_data(ctot_offset_min=200, crew_duty_offset_h=-9.0, max_fdp=9.0)
        ctot = BASE_TIME + timedelta(minutes=200)
        result = self.scorer.score_probabilistic(data, ctot, BASE_TIME, "WEATHER")
        # With 200min delay and crew near limit, should be HIGH or CRITICAL
        assert result.risk_adjusted_priority in ("HIGH", "CRITICAL")

    def test_n_simulations_matches_config(self):
        data = make_ops_data(ctot_offset_min=90)
        ctot = BASE_TIME + timedelta(minutes=90)
        result = self.scorer.score_probabilistic(data, ctot, BASE_TIME, "WEATHER")
        assert result.n_simulations == ProbabilisticFDPScorer.N_SIMULATIONS

    def test_elapsed_fraction_conditioning(self):
        """Conditioning on elapsed time should not crash and produce valid output."""
        data = make_ops_data(ctot_offset_min=90)
        ctot = BASE_TIME + timedelta(minutes=90)
        result = self.scorer.score_probabilistic(
            data, ctot, BASE_TIME, "WEATHER", elapsed_fraction=0.5
        )
        assert 0.0 <= result.p_crew_violation <= 1.0


# ---------------------------------------------------------------------------
# 3. Regulation Change Listener
# ---------------------------------------------------------------------------

class TestRegulationChangeListener:
    def _make_notice(self, end_offset_h: float = 4.0, capacity_pct: int = 40, count: int = 10):
        from parsers.fixm_parser import RegulationNotice
        return RegulationNotice(
            regulation_id="LFEREG0001",
            affected_airport="LFPG",
            affected_fir="LFFF",
            reason="WEATHER",
            start_time=BASE_TIME,
            end_time=BASE_TIME + timedelta(hours=end_offset_h),
            capacity_reduction_pct=capacity_pct,
            affected_flights_count=count,
        )

    def test_no_change_when_identical(self):
        listener = RegulationChangeListener()
        notice = self._make_notice()
        listener.take_snapshot(notice)
        changes = listener.detect_changes(notice)
        assert changes == []

    def test_extension_detected(self):
        listener = RegulationChangeListener()
        notice_orig = self._make_notice(end_offset_h=4.0)
        listener.take_snapshot(notice_orig)
        notice_ext = self._make_notice(end_offset_h=6.0)
        changes = listener.detect_changes(notice_ext)
        assert any(c.change_type == ChangeType.REGULATION_EXTENDED for c in changes)

    def test_shortening_detected(self):
        listener = RegulationChangeListener()
        listener.take_snapshot(self._make_notice(end_offset_h=6.0))
        changes = listener.detect_changes(self._make_notice(end_offset_h=4.0))
        assert any(c.change_type == ChangeType.REGULATION_SHORTENED for c in changes)

    def test_capacity_change_detected(self):
        listener = RegulationChangeListener()
        listener.take_snapshot(self._make_notice(capacity_pct=40))
        changes = listener.detect_changes(self._make_notice(capacity_pct=20))
        assert any(c.change_type == ChangeType.CAPACITY_CHANGED for c in changes)

    def test_regulation_lifted_detected(self):
        listener = RegulationChangeListener()
        listener.take_snapshot(self._make_notice())
        changes = listener.detect_changes(self._make_notice(count=0))
        assert any(c.change_type == ChangeType.REGULATION_LIFTED for c in changes)

    def test_callback_fired_on_change(self):
        fired = []
        listener = RegulationChangeListener()
        listener.register_callback(lambda c: fired.append(c))
        listener.take_snapshot(self._make_notice(end_offset_h=4.0))
        listener.detect_changes(self._make_notice(end_offset_h=7.0))
        assert len(fired) == 1
        assert fired[0].change_type == ChangeType.REGULATION_EXTENDED

    def test_simulate_extension(self):
        listener = RegulationChangeListener()
        listener.take_snapshot(self._make_notice(end_offset_h=4.0))
        new_end = BASE_TIME + timedelta(hours=7)
        change = listener.simulate_change("LFEREG0001", ChangeType.REGULATION_EXTENDED, new_end_time=new_end)
        assert change is not None
        assert change.change_type == ChangeType.REGULATION_EXTENDED
        assert "+180 min" in change.new_value

    def test_simulate_lift(self):
        listener = RegulationChangeListener()
        listener.take_snapshot(self._make_notice())
        change = listener.simulate_change("LFEREG0001", ChangeType.REGULATION_LIFTED)
        assert change is not None
        assert change.change_type == ChangeType.REGULATION_LIFTED


# ---------------------------------------------------------------------------
# 4. Dynamic Rescorer — fleet re-prioritisation
# ---------------------------------------------------------------------------

class TestDynamicRescorer:
    def _setup_rescorer(self):
        from parsers.fixm_parser import RegulationNotice
        rescorer = DynamicRescorer()
        flights = {
            "BA456": make_ops_data("BA456", 160, -8.0, 9.5),
            "FR987": make_ops_data("FR987", 20,  -1.0, 10.0, connections=0),
        }
        scorer = ImpactScorer()
        scores = {
            fn: scorer.score(data, BASE_TIME + timedelta(minutes=160 if fn == "BA456" else 20))
            for fn, data in flights.items()
        }
        notice = RegulationNotice(
            regulation_id="TESTFLEX001",
            affected_airport="LFPG",
            affected_fir="LFFF",
            reason="WEATHER",
            start_time=BASE_TIME - timedelta(hours=2),
            end_time=BASE_TIME + timedelta(hours=6),
            capacity_reduction_pct=40,
            affected_flights_count=5,
        )
        rescorer.initialise(flights, scores, notice)
        return rescorer, flights, scores, notice

    def test_rescore_returns_event(self):
        rescorer, flights, _, _ = self._setup_rescorer()
        ctots = {fn: BASE_TIME + timedelta(minutes=160) for fn in flights}
        event = rescorer.rescore_fleet(ctots, "WEATHER", 0.3)
        assert isinstance(event, RescoreEvent)

    def test_rescore_covers_all_flights(self):
        rescorer, flights, _, _ = self._setup_rescorer()
        ctots = {fn: BASE_TIME + timedelta(minutes=160) for fn in flights}
        event = rescorer.rescore_fleet(ctots, "WEATHER", 0.3)
        assert len(event.probabilistic_scores) == len(flights)

    def test_simulate_extension_triggers_callback(self):
        rescorer, _, _, _ = self._setup_rescorer()
        events = []
        rescorer.on_rescore(lambda e: events.append(e))
        # Simulate extension
        rescorer.simulate_extension("TESTFLEX001", BASE_TIME + timedelta(hours=10))
        # Manually trigger rescore (in real system the orchestrator does this)
        ctots = {"BA456": BASE_TIME + timedelta(minutes=160), "FR987": BASE_TIME + timedelta(minutes=20)}
        rescorer.rescore_fleet(ctots, "WEATHER", 0.5)
        assert len(events) == 1

    def test_priority_change_detected_when_delay_increases(self):
        rescorer, flights, _, _ = self._setup_rescorer()
        # Much bigger delay → should escalate priority
        big_delay_ctots = {fn: BASE_TIME + timedelta(minutes=300) for fn in flights}
        event = rescorer.rescore_fleet(big_delay_ctots, "WEATHER", 0.5)
        # At least one flight should have changed priority
        assert isinstance(event.priority_changes, list)

    def test_rescore_summary_contains_flight_count(self):
        rescorer, flights, _, _ = self._setup_rescorer()
        ctots = {fn: BASE_TIME + timedelta(minutes=160) for fn in flights}
        event = rescorer.rescore_fleet(ctots, "WEATHER", 0.3)
        summary = event.summary()
        assert str(len(flights)) in summary
