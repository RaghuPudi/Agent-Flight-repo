"""
ATFM NEGOTIATION AGENT — MAIN ORCHESTRATOR
LangGraph-style stateful agent that:
  1. Receives FIXM slot messages (real or simulated)
  2. Scores the impact of each regulated flight
  2.5 Generates LLM-powered CFMU justification + probabilistic FDP scoring
  3. Runs the negotiation planner per flight (using LLM justification text)
  4. Logs every action to the immutable audit chain
  5. Outputs a post-ops summary report
  6. Supports mid-regulation dynamic re-scoring via DynamicRescorer

Entry point: ATFMAgent.run_regulation(regulation_id, slot_messages, flight_data)
Dynamic re-scoring: ATFMAgent.apply_regulation_change(updated_notice)
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from parsers.fixm_parser import FIXMParser, SlotMessage, RegulationNotice
from scoring.impact_scorer import ImpactScorer, ImpactScore, FlightOperationalData
from negotiation.negotiation_planner import NegotiationPlanner, NegotiationState, CFMUApiClient
from audit.audit_logger import AuditLogger, EventType, Outcome
from agent.llm_justification import LLMJustificationEngine, LLMClient
from agent.dynamic_rescorer import DynamicRescorer, RescoreEvent, RegulationChangeListener

console = Console()


# ---------------------------------------------------------------------------
# Agent State (the "graph state" in LangGraph terms)
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    regulation_id: str
    regulated_flights: List[SlotMessage] = field(default_factory=list)
    flight_data_map: Dict[str, FlightOperationalData] = field(default_factory=dict)
    impact_scores: Dict[str, ImpactScore] = field(default_factory=dict)
    negotiation_states: Dict[str, NegotiationState] = field(default_factory=dict)
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    # New: LLM-generated explanations per flight (decision card for OCC dashboard)
    decision_explanations: Dict[str, dict] = field(default_factory=dict)
    # New: Probabilistic FDP scores (p_crew_violation per flight)
    probabilistic_scores: Dict[str, dict] = field(default_factory=dict)
    # New: Dynamic rescore events that occurred mid-regulation
    rescore_events: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ATFM Agent
# ---------------------------------------------------------------------------

class ATFMAgent:
    """
    Autonomous ATFM Negotiation Agent.
    Manages the full lifecycle of negotiating ATFM slots for a regulated fleet.
    """

    def __init__(
        self,
        cfmu_client: Optional[CFMUApiClient] = None,
        audit_log_dir: str = "data/audit_logs",
        llm_client: Optional[LLMClient] = None,
    ):
        self._parser = FIXMParser()
        self._scorer = ImpactScorer()
        self._cfmu = cfmu_client or CFMUApiClient()
        self._planner = NegotiationPlanner(cfmu_client=self._cfmu)
        self._audit = AuditLogger(log_dir=audit_log_dir)
        self._llm = LLMJustificationEngine(llm_client=llm_client or LLMClient())
        self._rescorer = DynamicRescorer()
        self._current_state: Optional[AgentState] = None   # For mid-regulation access

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_regulation(
        self,
        regulation_id: str,
        slot_xml_messages: List[str],
        flight_data_map: Dict[str, FlightOperationalData],
    ) -> AgentState:
        """
        Process all regulated flights for a given ATFM regulation.

        Args:
            regulation_id: CFMU regulation identifier (e.g. "LFEREG1034")
            slot_xml_messages: List of FIXM CTOT_ASSIGNMENT XML strings
            flight_data_map: Operational data per flight number

        Returns:
            Final AgentState with all scores, negotiations, and outcomes
        """
        state = AgentState(regulation_id=regulation_id, flight_data_map=flight_data_map)
        self._current_state = state

        console.print(Panel(
            f"[bold cyan]ATFM Negotiation Agent Starting[/bold cyan]\n"
            f"Regulation: [yellow]{regulation_id}[/yellow]\n"
            f"Flights to process: [white]{len(slot_xml_messages)}[/white]\n"
            f"LLM backend: [dim]{self._llm._llm.backend}[/dim]",
            title="✈  ATFM Agent",
            border_style="cyan",
        ))

        # ------------------------------------------------------------------
        # Node 1: PARSE SLOT MESSAGES
        # ------------------------------------------------------------------
        console.print("\n[bold][ Node 1 ] Parsing FIXM slot messages...[/bold]")
        state.regulated_flights = self._node_parse_slots(regulation_id, slot_xml_messages)

        # ------------------------------------------------------------------
        # Node 2: SCORE IMPACT
        # ------------------------------------------------------------------
        console.print("\n[bold][ Node 2 ] Scoring operational impact...[/bold]")
        state.impact_scores = self._node_score_impacts(regulation_id, state)

        # ------------------------------------------------------------------
        # Node 2.5: LLM JUSTIFICATION + PROBABILISTIC FDP
        # ------------------------------------------------------------------
        console.print("\n[bold][ Node 2.5 ] Generating LLM justifications & probabilistic FDP scores...[/bold]")
        self._node_llm_enrich(regulation_id, state)

        # ------------------------------------------------------------------
        # Node 3: SORT BY PRIORITY & NEGOTIATE (with LLM justification text)
        # ------------------------------------------------------------------
        console.print("\n[bold][ Node 3 ] Running negotiation planner...[/bold]")
        state.negotiation_states = self._node_negotiate(regulation_id, state)

        # ------------------------------------------------------------------
        # Node 4: GENERATE POST-OPS REPORT
        # ------------------------------------------------------------------
        state.end_time = datetime.now(timezone.utc)
        console.print("\n[bold][ Node 4 ] Generating post-ops report...[/bold]")
        self._node_generate_report(state)

        return state

    def apply_regulation_change(
        self,
        updated_notice: RegulationNotice,
        updated_ctots: Optional[Dict[str, "datetime"]] = None,
    ) -> Optional[RescoreEvent]:
        """
        Called when CFMU sends a regulation update mid-event.
        Detects changes, re-scores the fleet probabilistically, and emits a RescoreEvent.

        Args:
            updated_notice: Updated RegulationNotice from CFMU
            updated_ctots: Current CTOT per flight (may have evolved from initial assignments)

        Returns:
            RescoreEvent if re-scoring was triggered, None if no significant change detected
        """
        if self._current_state is None:
            console.print("[yellow]⚠ apply_regulation_change called before run_regulation[/yellow]")
            return None

        state = self._current_state
        changes = self._rescorer.ingest_update(updated_notice)

        if not changes:
            console.print("[dim]  No regulation changes detected.[/dim]")
            return None

        significant = [c for c in changes if c.impact_magnitude in ("HIGH", "MEDIUM")]
        if not significant:
            return None

        console.print(Panel(
            "\n".join(f"[yellow]{c.change_type.value}[/yellow]: {c.previous_value} → {c.new_value}"
                      for c in significant),
            title="⚠  Regulation Change Detected — Re-scoring fleet",
            border_style="yellow",
        ))

        # Build current CTOT map from negotiation states
        if updated_ctots is None:
            updated_ctots = {
                fn: neg.final_ctot or neg.original_ctot
                for fn, neg in state.negotiation_states.items()
            }

        # Compute elapsed fraction
        now = datetime.now(timezone.utc)
        reg_duration = (updated_notice.end_time - updated_notice.start_time).total_seconds()
        elapsed = (now - updated_notice.start_time).total_seconds()
        elapsed_fraction = min(max(elapsed / reg_duration if reg_duration > 0 else 0, 0), 1)

        rescore_event = self._rescorer.rescore_fleet(
            updated_ctots=updated_ctots,
            regulation_reason=updated_notice.reason,
            elapsed_fraction=elapsed_fraction,
        )

        # Store in state
        state.rescore_events.append({
            "triggered_at": rescore_event.triggered_at.isoformat(),
            "changes": [c.change_type.value for c in changes],
            "priority_changes": len(rescore_event.priority_changes),
            "newly_critical": rescore_event.newly_critical_flights,
        })

        # Log to audit trail
        self._audit.log(
            event_type=EventType.REGULATION_RECEIVED,
            regulation_id=state.regulation_id,
            flight_number="FLEET",
            payload={
                "event": "DYNAMIC_RESCORE",
                "changes_detected": [c.change_type.value for c in significant],
                "priority_changes": len(rescore_event.priority_changes),
                "newly_critical": rescore_event.newly_critical_flights,
                "recommendation": rescore_event.rescore_recommendation,
            },
        )

        console.print(rescore_event.summary())
        return rescore_event

    # ------------------------------------------------------------------
    # Agent Nodes
    # ------------------------------------------------------------------

    def _node_parse_slots(
        self, regulation_id: str, xml_messages: List[str]
    ) -> List[SlotMessage]:
        """Parse all incoming FIXM XML messages."""
        parsed = []
        for xml in xml_messages:
            try:
                msg = self._parser.parse_ctot_message(xml)
                parsed.append(msg)
                self._audit.log(
                    event_type=EventType.SLOT_ASSIGNED,
                    regulation_id=regulation_id,
                    flight_number=msg.flight.flight_number,
                    payload={
                        "sobt": msg.scheduled_off_block.isoformat(),
                        "ctot": msg.calculated_takeoff_time.isoformat(),
                        "dep": msg.flight.dep_airport,
                        "arr": msg.flight.arr_airport,
                        "swap_available": msg.swap_available,
                    },
                )
                console.print(
                    f"  [green]✓[/green] {msg.flight.flight_number}: "
                    f"CTOT {msg.calculated_takeoff_time.strftime('%H:%M')}Z "
                    f"(SOBT {msg.scheduled_off_block.strftime('%H:%M')}Z, "
                    f"+{int((msg.calculated_takeoff_time - msg.scheduled_off_block).total_seconds()//60)} min)"
                )
            except Exception as e:
                console.print(f"  [red]✗ Parse error: {e}[/red]")
        return parsed

    def _node_score_impacts(
        self, regulation_id: str, state: AgentState
    ) -> Dict[str, ImpactScore]:
        """Score the operational impact of every regulated slot."""
        scores: Dict[str, ImpactScore] = {}
        for slot in state.regulated_flights:
            fn = slot.flight.flight_number
            ops_data = state.flight_data_map.get(fn)
            if ops_data is None:
                console.print(f"  [yellow]⚠[/yellow] No ops data for {fn}, skipping scoring.")
                continue

            impact = self._scorer.score(ops_data, slot.calculated_takeoff_time)
            scores[fn] = impact

            self._audit.log(
                event_type=EventType.IMPACT_SCORED,
                regulation_id=regulation_id,
                flight_number=fn,
                payload={
                    "composite_score": impact.composite_score,
                    "priority": impact.priority,
                    "crew_legal": impact.crew_legal,
                    "misconnecting_pax": impact.misconnecting_pax,
                    "revenue_at_risk_usd": impact.connection_revenue_at_risk_usd,
                    "recommendation": impact.recommendation,
                },
            )
            priority_color = {
                "CRITICAL": "red",
                "HIGH": "yellow",
                "MEDIUM": "cyan",
                "LOW": "green",
            }.get(impact.priority, "white")

            console.print(
                f"  [{priority_color}]{impact.priority:8}[/{priority_color}] "
                f"{fn}: score={impact.composite_score:.2f}, "
                f"crew={'✓' if impact.crew_legal else '✗'}, "
                f"misconnects={impact.misconnecting_pax} pax, "
                f"rec={impact.recommendation}"
            )

        return scores

    def _node_llm_enrich(self, regulation_id: str, state: AgentState) -> None:
        """
        Node 2.5: For each scored flight, generate:
          - LLM decision explanation (shown on OCC dashboard)
          - Probabilistic FDP score (P(crew violation) under regulation duration uncertainty)
        Stores results in state.decision_explanations and state.probabilistic_scores.
        """
        from agent.dynamic_rescorer import ProbabilisticFDPScorer
        prob_scorer = ProbabilisticFDPScorer()

        for fn, impact in state.impact_scores.items():
            ops_data = state.flight_data_map.get(fn)

            # --- Probabilistic FDP scoring ---
            if ops_data and ops_data.crew:
                slot = next((s for s in state.regulated_flights if s.flight.flight_number == fn), None)
                if slot:
                    prob_score = prob_scorer.score_probabilistic(
                        data=ops_data,
                        current_ctot=slot.calculated_takeoff_time,
                        regulation_start=slot.regulation_start,
                        regulation_reason=slot.regulation_reason,
                        elapsed_fraction=0.0,
                    )
                    state.probabilistic_scores[fn] = {
                        "p_crew_violation": round(prob_score.p_crew_violation, 3),
                        "expected_delay_min": round(prob_score.expected_delay_min, 1),
                        "worst_case_delay_min": round(prob_score.worst_case_delay_min, 1),
                        "risk_adjusted_priority": prob_score.risk_adjusted_priority,
                    }
                    p_color = "red" if prob_score.p_crew_violation > 0.6 else \
                              "yellow" if prob_score.p_crew_violation > 0.3 else "green"
                    console.print(
                        f"  [dim]{fn}[/dim] P(FDP violation)=[{p_color}]{prob_score.p_crew_violation:.0%}[/{p_color}] "
                        f"| expected delay={prob_score.expected_delay_min:.0f}min "
                        f"| risk-adj priority={prob_score.risk_adjusted_priority}"
                    )

            # --- LLM decision explanation ---
            from negotiation.negotiation_planner import NegotiationProposal
            from datetime import timedelta
            dummy_proposal = NegotiationProposal(
                proposal_id="preview",
                flight_number=fn,
                regulation_id=regulation_id,
                round_number=0,
                action=impact.recommendation,
                current_ctot=impact.ctot,
                requested_ctot=impact.ctot - timedelta(minutes=60) if impact.recommendation != "ACCEPT" else None,
                justification_codes=self._planner._build_justification_codes(impact),
            )
            alternatives = [
                {"action": "ACCEPT",    "score": 0.1,  "reason": "Score too high to absorb"} if impact.recommendation != "ACCEPT" else None,
                {"action": "REROUTE",   "score": impact.reroute_saves_usd / 10000 if impact.reroute_saves_usd > 0 else 0,
                 "reason": "Reroute fuel cost exceeds benefit"} if impact.recommendation != "REROUTE" else None,
                {"action": "SWAP",      "score": 0.5,  "reason": "Crew violation requires faster resolution"} if impact.recommendation == "ESCALATE" else None,
            ]
            alternatives = [a for a in alternatives if a]

            explanation = self._llm.explain_decision(impact, dummy_proposal, alternatives)
            state.decision_explanations[fn] = {
                "decision": explanation.decision,
                "why": explanation.why,
                "what_was_considered": explanation.what_was_considered,
                "what_was_rejected": explanation.what_was_rejected,
                "confidence_pct": explanation.confidence_pct,
                "generated_by": explanation.generated_by,
            }
            console.print(
                f"  [dim]{fn}[/dim] [{explanation.generated_by}] "
                f"{explanation.confidence_pct}% confidence → \"{explanation.why[:80]}...\""
                if len(explanation.why) > 80 else
                f"  [dim]{fn}[/dim] [{explanation.generated_by}] "
                f"{explanation.confidence_pct}% confidence → \"{explanation.why}\""
            )

    def _node_negotiate(
        self, regulation_id: str, state: AgentState
    ) -> Dict[str, NegotiationState]:
        """
        Run negotiations for all flights, prioritized by impact score.
        Highest-priority flights use the negotiation budget first.
        """
        # Sort by composite score descending — fight hardest for highest impact
        sorted_flights = sorted(
            state.impact_scores.items(),
            key=lambda kv: kv[1].composite_score,
            reverse=True,
        )

        negotiation_budget = 15  # Max swap/counter requests before CFMU flags us
        budget_used = 0
        results: Dict[str, NegotiationState] = {}

        for fn, impact in sorted_flights:
            slot = next((s for s in state.regulated_flights if s.flight.flight_number == fn), None)
            if slot is None:
                continue

            console.print(f"\n  ▸ Negotiating [bold]{fn}[/bold] (priority: {impact.priority})")

            self._audit.log(
                event_type=EventType.DECISION_MADE,
                regulation_id=regulation_id,
                flight_number=fn,
                payload={
                    "action": impact.recommendation,
                    "composite_score": impact.composite_score,
                    "budget_remaining": negotiation_budget - budget_used,
                },
                agent_recommendation=impact.recommendation,
            )

            # If we've exhausted the budget, force-accept remaining flights
            if budget_used >= negotiation_budget and impact.recommendation != "ESCALATE":
                console.print(
                    f"    [yellow]⚠ Budget exhausted — accepting slot for {fn}[/yellow]"
                )
                neg_state = NegotiationState(
                    flight_number=fn,
                    regulation_id=regulation_id,
                    original_ctot=impact.ctot,
                    current_ctot=impact.ctot,
                    impact_score=impact,
                    resolved=True,
                    final_ctot=impact.ctot,
                )
                results[fn] = neg_state
                continue

            neg_state = self._planner.negotiate(impact, slot.scheduled_off_block)
            results[fn] = neg_state

            # Count rounds that consumed CFMU budget
            budget_consuming_rounds = [
                p for p in neg_state.proposals
                if p.action in ("SWAP_REQUEST", "COUNTER_CTOT")
            ]
            budget_used += len(budget_consuming_rounds)

            # Log each proposal
            for proposal in neg_state.proposals:
                evt = EventType.SWAP_REQUEST_SUBMITTED if proposal.action == "SWAP_REQUEST" \
                    else EventType.COUNTER_PROPOSAL_SUBMITTED if proposal.action == "COUNTER_CTOT" \
                    else EventType.SLOT_ACCEPTED if proposal.action == "ACCEPT" \
                    else EventType.REROUTE_REQUESTED if proposal.action == "REROUTE" \
                    else EventType.NEGOTIATION_CLOSED

                outcome = Outcome.SUCCESS if proposal.accepted else \
                          Outcome.REJECTED if proposal.accepted is False else \
                          Outcome.PENDING

                self._audit.log(
                    event_type=evt,
                    regulation_id=regulation_id,
                    flight_number=fn,
                    payload={
                        "round": proposal.round_number,
                        "action": proposal.action,
                        "current_ctot": proposal.current_ctot.isoformat(),
                        "requested_ctot": proposal.requested_ctot.isoformat() if proposal.requested_ctot else None,
                        "swap_with": proposal.swap_with_flight,
                        "justification_codes": proposal.justification_codes,
                        "response": proposal.response,
                        "benefit_usd": proposal.estimated_benefit_usd,
                    },
                    outcome=outcome,
                    agent_recommendation=proposal.action,
                )

            # Print result
            if neg_state.resolved and neg_state.final_ctot:
                improvement = int(
                    (neg_state.original_ctot - neg_state.final_ctot).total_seconds() // 60
                )
                if improvement > 0:
                    console.print(
                        f"    [green]✓ Resolved: CTOT improved by {improvement} min → "
                        f"{neg_state.final_ctot.strftime('%H:%M')}Z[/green]"
                    )
                else:
                    console.print(
                        f"    [dim]✓ Accepted slot at {neg_state.final_ctot.strftime('%H:%M')}Z[/dim]"
                    )
            else:
                console.print(f"    [yellow]⚑ Escalated to human OCC coordinator[/yellow]")

        console.print(f"\n  CFMU budget used: {budget_used} / {negotiation_budget}")
        return results

    def _node_generate_report(self, state: AgentState) -> None:
        """Print the post-ops summary and write it to disk."""
        summary = self._audit.get_regulation_summary(state.regulation_id)

        # Calculate aggregate metrics
        total_delay_saved_min = 0
        total_revenue_protected = 0.0
        total_misconnects_prevented = 0
        crew_illegalities_prevented = 0

        for fn, neg in state.negotiation_states.items():
            if neg.final_ctot and neg.original_ctot:
                saved = max(0, int((neg.original_ctot - neg.final_ctot).total_seconds() // 60))
                total_delay_saved_min += saved

            impact = state.impact_scores.get(fn)
            if impact and neg.resolved and neg.final_ctot != neg.original_ctot:
                total_revenue_protected += impact.connection_revenue_at_risk_usd
                total_misconnects_prevented += impact.misconnecting_pax
                if not impact.crew_legal:
                    crew_illegalities_prevented += 1

        # Print summary table
        table = Table(
            title=f"Post-Ops Report — {state.regulation_id}",
            box=box.ROUNDED,
            border_style="cyan",
        )
        table.add_column("Metric", style="bold white")
        table.add_column("Value", style="green")

        elapsed = (state.end_time - state.start_time).total_seconds()
        table.add_row("Flights affected", str(summary["flights_affected"]))
        table.add_row("Agent actions taken", str(summary["agent_decisions"]))
        table.add_row("Slot swaps submitted", str(summary["swap_requests_submitted"]))
        table.add_row(
            "Swaps accepted",
            f"{summary['swap_requests_accepted']} / {summary['swap_requests_submitted']}"
        )
        table.add_row("Human overrides", str(summary["human_overrides"]))
        table.add_row("Delay recovered", f"{total_delay_saved_min} min")
        table.add_row("Revenue protected", f"${total_revenue_protected:,.0f}")
        table.add_row("Misconnects prevented", f"{total_misconnects_prevented} pax")
        table.add_row("Crew violations prevented", str(crew_illegalities_prevented))
        table.add_row("Agent processing time", f"{elapsed:.1f}s")
        table.add_row("Audit chain integrity", "✓ Verified" if self._audit.verify_chain_integrity() else "✗ COMPROMISED")

        console.print(table)

        # Write report to file
        report_path = f"data/audit_logs/{state.regulation_id}_report.txt"
        os.makedirs("data/audit_logs", exist_ok=True)
        with open(report_path, "w") as f:
            f.write(f"ATFM REGULATION POST-OPS REPORT\n")
            f.write(f"Regulation: {state.regulation_id}\n")
            f.write(f"Generated: {state.end_time.isoformat()}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Flights affected         : {summary['flights_affected']}\n")
            f.write(f"Swaps accepted           : {summary['swap_requests_accepted']} / {summary['swap_requests_submitted']}\n")
            f.write(f"Human overrides          : {summary['human_overrides']}\n")
            f.write(f"Delay recovered          : {total_delay_saved_min} min\n")
            f.write(f"Revenue protected        : ${total_revenue_protected:,.0f}\n")
            f.write(f"Misconnects prevented    : {total_misconnects_prevented} pax\n")
            f.write(f"Crew violations prevented: {crew_illegalities_prevented}\n\n")

            f.write("FLIGHT-BY-FLIGHT BREAKDOWN\n")
            f.write(f"{'='*60}\n")
            for fn, neg in state.negotiation_states.items():
                impact = state.impact_scores.get(fn)
                f.write(f"\n{fn}:\n")
                if impact:
                    f.write(f"  {impact.summary()}\n")
                f.write(f"  Rounds: {neg.rounds_used}\n")
                f.write(f"  Final CTOT: {neg.final_ctot.strftime('%H:%M') + 'Z' if neg.final_ctot else 'ESCALATED'}\n")
                for p in neg.proposals:
                    f.write(f"    Round {p.round_number}: {p.action} → {p.response or 'N/A'}\n")

        console.print(f"\n  [dim]Report written to {report_path}[/dim]")
