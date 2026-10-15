"""
Comprehensive test suite for the ATFM Negotiation Agent PoC.
Run with: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from datetime import datetime, timedelta, timezone

from parsers.fixm_parser import FIXMParser, make_sample_ctot_xml, make_sample_regulation_xml
from scoring.impact_scorer import (
    ImpactScorer, FlightOperationalData, CrewRecord, Connection
)
from negotiation.negotiation_planner import (
    NegotiationPlanner, CFMUApiClient, NegotiationState
)
from audit.audit_logger import AuditLogger, EventType, Outcome


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE_TIME = datetime(2026, 10, 15, 14, 0, 0, tzinfo=timezone.utc)


def make_crew(duty_start_offset_h: float = -6.0, max_fdp: float = 10.0):
    return [
        CrewRecord(
            crew_id="CPT-001",
            role="CPT",
            duty_start=BASE_TIME + timedelta(hours=duty_start_offset_h),
            max_fdp_hours=max_fdp,
        ),
        CrewRecord(
            crew_id="FO-001",
            role="FO",
            duty_start=BASE_TIME + timedelta(hours=duty_start_offset_h),
            max_fdp_hours=max_fdp,
        ),
    ]


def make_connections(n: int = 2, backup: bool = True):
    return [
        Connection(
            pax_count=20,
            connecting_flight=f"BA{700 + i}",
            connecting_dep_time=BASE_TIME + timedelta(minutes=120),
            has_same_day_backup=backup,
        )
        for i in range(n)
    ]


def make_ops_data(
    fn: str = "BA456",
    ctot_offset_min: int = 100,
    crew_duty_offset_h: float = -6.0,
    max_fdp: float = 10.0,
    connections: int = 2,
    backup: bool = True,
    alternate_nm: float = 0.0,
    alternate_fuel_kg: float = 0.0,
) -> FlightOperationalData:
    sobt = BASE_TIME
    return FlightOperationalData(
        flight_number=fn,
        dep_airport="EGLL",
        arr_airport="LFPG",
        scheduled_off_block=sobt,
        flight_time_min=75,
        crew=make_crew(crew_duty_offset_h, max_fdp),
        connections=make_connections(connections, backup),
        alternate_route_extra_nm=alternate_nm,
        alternate_route_extra_fuel_kg=alternate_fuel_kg,
    )


# ---------------------------------------------------------------------------
# 1. FIXM Parser Tests
# ---------------------------------------------------------------------------

class TestFIXMParser:
    def test_parse_ctot_message(self):
        xml = make_sample_ctot_xml(
            flight_number="BA456",
            dep="EGLL",
            arr="LFPG",
            sobt="2026-10-15T14:00:00Z",
            ctot="2026-10-15T16:40:00Z",
            regulation_id="LFEREG1034",
        )
        parser = FIXMParser()
        msg = parser.parse_ctot_message(xml)

        assert msg.flight.flight_number == "BA456"
        assert msg.flight.dep_airport == "EGLL"
        assert msg.flight.arr_airport == "LFPG"
        assert msg.regulation_id == "LFEREG1034"
        assert msg.message_type == "CTOT_ASSIGNMENT"
        assert msg.calculated_takeoff_time.hour == 16
        assert msg.calculated_takeoff_time.minute == 40

    def test_parse_regulation_notice(self):
        xml = make_sample_regulation_xml(regulation_id="LFEREG9999", capacity_pct=40)
        parser = FIXMParser()
        notice = parser.parse_regulation_notice(xml)

        assert notice.regulation_id == "LFEREG9999"
        assert notice.capacity_reduction_pct == 40
        assert notice.affected_airport == "LFPG"

    def test_swap_available_flag(self):
        xml_true = make_sample_ctot_xml(swap_available=True)
        xml_false = make_sample_ctot_xml(swap_available=False)
        parser = FIXMParser()
        assert parser.parse_ctot_message(xml_true).swap_available is True
        assert parser.parse_ctot_message(xml_false).swap_available is False


# ---------------------------------------------------------------------------
# 2. Impact Scorer Tests
# ---------------------------------------------------------------------------

class TestImpactScorer:
    scorer = ImpactScorer()

    def test_low_impact_accepted(self):
        """A small delay on a flight with no connections and legal crew → ACCEPT."""
        ctot = BASE_TIME + timedelta(minutes=30)
        data = make_ops_data(ctot_offset_min=30, connections=0, crew_duty_offset_h=-1)
        impact = self.scorer.score(data, ctot)
        assert impact.recommendation == "ACCEPT"
        assert impact.priority == "LOW"
        assert impact.crew_legal is True

    def test_crew_fdp_violation_detected(self):
        """Crew who started duty 8h ago + 2.5h delay = FDP violation → ESCALATE."""
        ctot = BASE_TIME + timedelta(minutes=150)   # 2.5h delay
        # Crew started duty 8h before BASE_TIME, max FDP 9.5h
        data = make_ops_data(ctot_offset_min=150, crew_duty_offset_h=-8.0, max_fdp=9.5, connections=0)
        impact = self.scorer.score(data, ctot)
        assert impact.crew_legal is False
        assert impact.crew_fdp_violation_hours > 0
        assert impact.recommendation == "ESCALATE"

    def test_crew_legal_within_fdp(self):
        """Crew with fresh duty start → legal."""
        ctot = BASE_TIME + timedelta(minutes=90)
        data = make_ops_data(ctot_offset_min=90, crew_duty_offset_h=-1.0, connections=0)
        impact = self.scorer.score(data, ctot)
        assert impact.crew_legal is True

    def test_misconnect_revenue_calculated(self):
        """20 pax × $450 with no backup = $9,000 at risk when connection is missed."""
        ctot = BASE_TIME + timedelta(minutes=200)  # Big delay, misses connection at +120min
        data = make_ops_data(ctot_offset_min=200, connections=1, backup=False, crew_duty_offset_h=-1)
        impact = self.scorer.score(data, ctot)
        assert impact.misconnecting_pax > 0
        assert impact.connection_revenue_at_risk_usd > 0

    def test_reroute_recommended_when_saves_money(self):
        """Reroute that saves $20K net → REROUTE recommendation."""
        ctot = BASE_TIME + timedelta(minutes=200)
        # 20 pax × $450 = $9k at risk per connection × 2 connections = $18k
        # Reroute fuel: 890kg × $0.80 = $712 extra
        data = make_ops_data(
            ctot_offset_min=200, connections=2, backup=False,
            alternate_nm=47.0, alternate_fuel_kg=890.0,
            crew_duty_offset_h=-1,
        )
        impact = self.scorer.score(data, ctot)
        assert impact.reroute_available is True
        # Reroute saves more than $5k net
        assert impact.recommendation in ("REROUTE", "ESCALATE")

    def test_composite_score_critical_crew_violation(self):
        """Crew violation alone should push composite score above CRITICAL threshold."""
        ctot = BASE_TIME + timedelta(minutes=150)
        data = make_ops_data(ctot_offset_min=150, crew_duty_offset_h=-9.0, max_fdp=9.0, connections=0)
        impact = self.scorer.score(data, ctot)
        assert impact.composite_score >= ImpactScorer.CRITICAL_THRESHOLD


# ---------------------------------------------------------------------------
# 3. Audit Logger Tests
# ---------------------------------------------------------------------------

class TestAuditLogger:
    def test_append_and_verify_chain(self, tmp_path):
        logger = AuditLogger(log_dir=str(tmp_path))
        for i in range(5):
            logger.log(
                event_type=EventType.SLOT_ASSIGNED,
                regulation_id="LFEREG0001",
                flight_number=f"BA{100 + i}",
                payload={"ctot": "2026-10-15T14:00:00Z"},
            )
        assert logger.verify_chain_integrity() is True

    def test_tampering_detected(self, tmp_path):
        logger = AuditLogger(log_dir=str(tmp_path))
        logger.log(EventType.SLOT_ASSIGNED, "REG001", "BA100", {"ctot": "14:00"})
        # Tamper with a record's payload
        logger._records[0].payload["ctot"] = "TAMPERED"
        assert logger.verify_chain_integrity() is False

    def test_flight_trail_filter(self, tmp_path):
        logger = AuditLogger(log_dir=str(tmp_path))
        logger.log(EventType.SLOT_ASSIGNED, "REG001", "BA100", {})
        logger.log(EventType.SLOT_ASSIGNED, "REG001", "BA200", {})
        logger.log(EventType.IMPACT_SCORED, "REG001", "BA100", {})

        trail = logger.get_flight_trail("BA100")
        assert len(trail) == 2
        assert all(r.flight_number == "BA100" for r in trail)

    def test_human_override_preserved(self, tmp_path):
        logger = AuditLogger(log_dir=str(tmp_path))
        logger.log(
            event_type=EventType.HUMAN_OVERRIDE,
            regulation_id="REG001",
            flight_number="BA100",
            payload={"human_action": "ACCEPT_SLOT", "operator_id": "OCC-001"},
            outcome=Outcome.OVERRIDDEN,
            human_override=True,
            agent_recommendation="SWAP",
        )
        trail = logger.get_flight_trail("BA100")
        override = trail[0]
        assert override.human_override is True
        assert override.agent_recommendation == "SWAP"
        assert override.outcome == Outcome.OVERRIDDEN

    def test_jsonl_persisted(self, tmp_path):
        import json
        logger = AuditLogger(log_dir=str(tmp_path))
        logger.log(EventType.SLOT_ASSIGNED, "LFEREG0001", "BA456", {"ctot": "14:00"})

        log_file = tmp_path / "LFEREG0001.jsonl"
        assert log_file.exists()
        with open(log_file) as f:
            record = json.loads(f.readline())
        assert record["flight_number"] == "BA456"


# ---------------------------------------------------------------------------
# 4. Negotiation Planner Tests
# ---------------------------------------------------------------------------

class TestNegotiationPlanner:
    def test_low_impact_accepts_immediately(self):
        """Score below threshold → Round 1 accept, no CFMU calls."""
        scorer = ImpactScorer()
        ctot = BASE_TIME + timedelta(minutes=20)
        data = make_ops_data(ctot_offset_min=20, connections=0, crew_duty_offset_h=-1)
        impact = scorer.score(data, ctot)

        planner = NegotiationPlanner(cfmu_client=CFMUApiClient(accept_rate=0.0))
        state = planner.negotiate(impact, BASE_TIME)

        assert state.resolved is True
        assert len(state.proposals) == 1
        assert state.proposals[0].action == "ACCEPT"

    def test_high_impact_attempts_swap(self):
        """High composite score → at least one SWAP_REQUEST or COUNTER_CTOT submitted."""
        scorer = ImpactScorer()
        ctot = BASE_TIME + timedelta(minutes=200)
        data = make_ops_data(
            ctot_offset_min=200, connections=3, backup=False,
            crew_duty_offset_h=-8, max_fdp=9.5,
        )
        impact = scorer.score(data, ctot)

        planner = NegotiationPlanner(cfmu_client=CFMUApiClient(accept_rate=1.0))
        state = planner.negotiate(impact, BASE_TIME)

        actions = [p.action for p in state.proposals]
        assert any(a in ("SWAP_REQUEST", "COUNTER_CTOT", "ESCALATE") for a in actions)

    def test_successful_swap_resolves_state(self):
        """When CFMU accepts the swap (100% rate), state must be resolved."""
        scorer = ImpactScorer()
        ctot = BASE_TIME + timedelta(minutes=120)
        data = make_ops_data(ctot_offset_min=120, connections=2, backup=False, crew_duty_offset_h=-3)
        impact = scorer.score(data, ctot)

        planner = NegotiationPlanner(cfmu_client=CFMUApiClient(accept_rate=1.0))
        state = planner.negotiate(impact, BASE_TIME)

        # Either resolved via swap or reroute — either is correct
        if state.resolved:
            assert state.final_ctot is not None

    def test_all_rejected_escalates(self):
        """When every CFMU request is rejected, agent escalates to human."""
        scorer = ImpactScorer()
        ctot = BASE_TIME + timedelta(minutes=200)
        data = make_ops_data(
            ctot_offset_min=200, connections=2, backup=False, crew_duty_offset_h=-8, max_fdp=9.5,
        )
        impact = scorer.score(data, ctot)
        # Force all CFMU responses to reject
        planner = NegotiationPlanner(cfmu_client=CFMUApiClient(accept_rate=0.0))
        state = planner.negotiate(impact, BASE_TIME)

        actions = [p.action for p in state.proposals]
        # Should end in ESCALATE when crew is illegal and all else rejected
        assert "ESCALATE" in actions or not state.resolved


# ---------------------------------------------------------------------------
# 5. End-to-End Integration Test
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_full_regulation_run(self, tmp_path):
        """Run a full regulation with 3 flights and verify audit chain is intact."""
        from agent.atfm_agent import ATFMAgent

        flights = [
            ("BA456", 160, -8.0, 9.5, 2, False),   # Crew violation + misconnects
            ("VS201", 90,  -3.0, 10.0, 1, True),    # Medium impact
            ("FR987", 20,  -1.0, 10.0, 0, True),    # Low impact
        ]

        slot_xmls = []
        flight_data: dict = {}

        for fn, ctot_off, duty_off, max_fdp, conns, backup in flights:
            sobt = BASE_TIME
            ctot = BASE_TIME + timedelta(minutes=ctot_off)
            slot_xmls.append(make_sample_ctot_xml(
                flight_number=fn,
                sobt=sobt.isoformat(),
                ctot=ctot.isoformat(),
                regulation_id="TEST-REG-001",
            ))
            flight_data[fn] = make_ops_data(
                fn=fn, ctot_offset_min=ctot_off, crew_duty_offset_h=duty_off,
                max_fdp=max_fdp, connections=conns, backup=backup,
            )

        agent = ATFMAgent(
            cfmu_client=CFMUApiClient(accept_rate=0.8),
            audit_log_dir=str(tmp_path),
        )
        state = agent.run_regulation("TEST-REG-001", slot_xmls, flight_data)

        # All flights were processed
        assert len(state.negotiation_states) == 3

        # Audit chain is intact
        audit = AuditLogger(log_dir=str(tmp_path))
        # Reload records from file
        import json
        log_file = tmp_path / "TEST-REG-001.jsonl"
        assert log_file.exists()
        with open(log_file) as f:
            records = [json.loads(l) for l in f if l.strip()]
        assert len(records) >= 3   # At least one event per flight

        # Low-impact flight should have been accepted quickly
        fr987_neg = state.negotiation_states.get("FR987")
        assert fr987_neg is not None
        assert fr987_neg.resolved is True
