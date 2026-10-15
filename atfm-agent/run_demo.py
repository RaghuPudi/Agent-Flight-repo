"""
STANDALONE DEMO RUNNER
Runs a complete simulated ATFM regulation without needing the API server.
Now demonstrates all three agent capabilities:
  1. Core negotiation (original)
  2. LLM justification layer — CFMU messages, escalation briefs, decision cards
  3. Dynamic re-scoring — mid-regulation extension triggers fleet re-prioritisation

Usage: python run_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from datetime import datetime, timedelta, timezone

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from agent.atfm_agent import ATFMAgent
from agent.llm_justification import LLMJustificationEngine, LLMClient
from parsers.fixm_parser import make_sample_ctot_xml, make_sample_regulation_xml, FIXMParser
from scoring.impact_scorer import FlightOperationalData, CrewRecord, Connection, ImpactScorer
from negotiation.negotiation_planner import CFMUApiClient, NegotiationProposal
from agent.dynamic_rescorer import DynamicRescorer

console = Console()

BASE_TIME = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)


def build_demo_scenario():
    """
    Build a realistic simulation with 5 flights covering all priority levels
    and negotiation scenarios.
    """
    scenarios = [
        # (fn,  dep,    arr,    sobt_off, ctot_off, duty_off, max_fdp, conns, backup, alt_nm, alt_kg)
        ("BA456", "EGLL", "LFPG",  0,   160,   -8.0,  9.5,  2, False, 0.0,    0.0),    # CRITICAL: crew violation + misconnects
        ("VS201", "EGLL", "EHAM",  30,  120,   -4.0,  10.0, 1, True,  0.0,    0.0),    # HIGH: significant delay
        ("BA112", "EGLL", "EDDF",  15,  90,    -3.0,  10.0, 0, True,  47.0,  890.0),   # MEDIUM: reroute candidate
        ("EI445", "EIDW", "LFPG",  45,  60,    -2.0,  10.0, 0, True,  0.0,    0.0),    # MEDIUM: moderate delay
        ("FR987", "EGSS", "LFPG",  60,  25,    -1.0,  10.0, 0, True,  0.0,    0.0),    # LOW: small delay, accept
    ]

    slot_xmls = []
    flight_data = {}

    for fn, dep, arr, sobt_off, ctot_off, duty_off, max_fdp, n_conn, backup, alt_nm, alt_kg in scenarios:
        sobt = BASE_TIME + timedelta(minutes=sobt_off)
        ctot = BASE_TIME + timedelta(minutes=ctot_off)

        slot_xmls.append(make_sample_ctot_xml(
            flight_number=fn,
            dep=dep,
            arr=arr,
            sobt=sobt.isoformat(),
            ctot=ctot.isoformat(),
            regulation_id="LFEREG1034",
            swap_available=True,
        ))

        duty_start = BASE_TIME + timedelta(hours=duty_off)
        crew = [
            CrewRecord(crew_id=f"{fn}-CPT", role="CPT", duty_start=duty_start, max_fdp_hours=max_fdp),
            CrewRecord(crew_id=f"{fn}-FO",  role="FO",  duty_start=duty_start, max_fdp_hours=max_fdp),
        ]

        connections = [
            Connection(
                pax_count=20,
                connecting_flight=f"BA{700 + i}",
                connecting_dep_time=BASE_TIME + timedelta(minutes=120),
                has_same_day_backup=backup,
                avg_ticket_value_usd=520.0,
            )
            for i in range(n_conn)
        ]

        flight_data[fn] = FlightOperationalData(
            flight_number=fn,
            dep_airport=dep,
            arr_airport=arr,
            scheduled_off_block=sobt,
            flight_time_min=75,
            crew=crew,
            connections=connections,
            alternate_route_extra_nm=alt_nm,
            alternate_route_extra_fuel_kg=alt_kg,
        )

    return slot_xmls, flight_data


def demo_llm_justification(flight_data: dict) -> None:
    """
    Showcase 2: LLM Justification Layer
    Generates CFMU messages, escalation briefs, and decision explanation cards
    for the most critical flight.
    """
    console.print(Panel(
        "[bold]Showcase 2: LLM Justification Layer[/bold]\n"
        "Generating ICAO-standard CFMU messages, escalation briefs, and OCC decision cards.",
        border_style="magenta",
    ))

    engine = LLMJustificationEngine(llm_client=LLMClient())
    scorer = ImpactScorer()

    # Score BA456 (most critical)
    data = flight_data["BA456"]
    ctot = BASE_TIME + timedelta(minutes=160)
    impact = scorer.score(data, ctot)

    # Generate CFMU justification message
    proposal = NegotiationProposal(
        proposal_id="demo-001",
        flight_number="BA456",
        regulation_id="LFEREG1034",
        round_number=2,
        action="SWAP_REQUEST",
        current_ctot=ctot,
        requested_ctot=BASE_TIME + timedelta(minutes=90),
        justification_codes=["FDP_VIOLATION", "HIGH_PAX_MISCONNECT"],
        swap_with_flight="LH234",
    )

    cfmu_msg = engine.generate_cfmu_justification(impact, proposal, "WEATHER")
    console.print("\n[bold cyan]── CFMU B2B Submission Message ──[/bold cyan]")
    console.print(Panel(cfmu_msg.formal_text, title=f"BA456 → CFMU [{cfmu_msg.generated_by.upper()}]", border_style="cyan"))
    console.print(f"  Regulatory citations: {', '.join(cfmu_msg.regulatory_citations)}")
    console.print(f"  Agent confidence: [green]{cfmu_msg.confidence:.0%}[/green]")

    # Generate escalation brief
    from negotiation.negotiation_planner import NegotiationState
    neg_state = NegotiationState(
        flight_number="BA456",
        regulation_id="LFEREG1034",
        original_ctot=ctot,
        current_ctot=ctot,
        impact_score=impact,
        proposals=[proposal],
    )
    brief = engine.generate_escalation_brief(impact, neg_state)
    console.print("\n[bold yellow]── OCC Escalation Brief ──[/bold yellow]")
    console.print(Panel(
        f"[bold]{brief.headline}[/bold]\n\n"
        f"{brief.situation}\n\n"
        f"[green]RECOMMENDED ACTION:[/green] {brief.recommended_action}\n\n"
        f"[yellow]FINANCIAL IMPACT:[/yellow] {brief.financial_impact}",
        title=f"Escalation Brief — BA456 [{brief.generated_by.upper()}]",
        border_style="yellow",
    ))

    # Generate OCC dashboard decision card
    alts = [{"action": "ACCEPT", "score": 0.1, "reason": "Score too high to absorb"}]
    explanation = engine.explain_decision(impact, proposal, alts)
    console.print("\n[bold]── OCC Dashboard Decision Card ──[/bold]")
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Field", style="bold dim", width=22)
    table.add_column("Value")
    table.add_row("Flight", "BA456")
    table.add_row("Decision", f"[bold]{explanation.decision}[/bold]")
    table.add_row("Why", explanation.why)
    table.add_row("Confidence", f"[green]{explanation.confidence_pct}%[/green]")
    table.add_row("Generated by", explanation.generated_by)
    for item in explanation.what_was_considered:
        table.add_row("Considered", f"• {item}")
    for item in explanation.what_was_rejected:
        table.add_row("Rejected", f"✗ {item}")
    console.print(table)


def demo_dynamic_rescore(agent: ATFMAgent, final_state) -> None:
    """
    Showcase 3: Dynamic Re-Scoring
    Simulates a regulation extension (18:00Z → 21:00Z) and shows
    which flights change priority and why.
    """
    console.print(Panel(
        "[bold]Showcase 3: Mid-Regulation Dynamic Re-Scoring[/bold]\n"
        "Simulating regulation extension: LFEREG1034 end time 18:00Z → 21:00Z (+3h).\n"
        "Agent re-scores entire fleet probabilistically.",
        border_style="yellow",
    ))

    # Build an updated RegulationNotice with extended end time
    parser = FIXMParser()
    extended_notice_xml = make_sample_regulation_xml(
        regulation_id="LFEREG1034",
        airport="LFPG",
        fir="LFFF",
        reason="WEATHER",
        start="2026-10-15T09:00:00Z",
        end="2026-10-15T21:00:00Z",   # Extended from 18:00Z to 21:00Z
        capacity_pct=40,
        affected_count=47,
    )
    updated_notice = parser.parse_regulation_notice(extended_notice_xml)

    # Initialise the rescorer with original state
    from agent.dynamic_rescorer import DynamicRescorer
    from parsers.fixm_parser import RegulationNotice

    original_notice = parser.parse_regulation_notice(make_sample_regulation_xml(
        regulation_id="LFEREG1034",
        end="2026-10-15T18:00:00Z",
        affected_count=47,
    ))

    agent._rescorer.initialise(
        flight_data=final_state.flight_data_map,
        initial_scores=final_state.impact_scores,
        notice=original_notice,
    )

    # Apply the change
    rescore_event = agent.apply_regulation_change(updated_notice)

    if rescore_event:
        console.print("\n[bold cyan]── Re-Score Results ──[/bold cyan]")
        table = Table(box=box.ROUNDED, border_style="cyan")
        table.add_column("Flight", style="bold")
        table.add_column("P(FDP Violation)")
        table.add_column("Expected Delay")
        table.add_column("P95 Delay")
        table.add_column("Risk-Adj Priority")

        for fn, ps in rescore_event.probabilistic_scores.items():
            p_viol = ps.p_crew_violation
            p_color = "red" if p_viol > 0.6 else "yellow" if p_viol > 0.3 else "green"
            table.add_row(
                fn,
                f"[{p_color}]{p_viol:.0%}[/{p_color}]",
                f"{ps.expected_delay_min:.0f} min",
                f"{ps.worst_case_delay_min:.0f} min",
                f"[{p_color}]{ps.risk_adjusted_priority}[/{p_color}]",
            )
        console.print(table)

        if rescore_event.priority_changes:
            console.print("\n[bold yellow]Priority Changes:[/bold yellow]")
            for change in rescore_event.priority_changes:
                console.print(
                    f"  [bold]{change.flight_number}[/bold]: "
                    f"{change.old_priority} → [bold red]{change.new_priority}[/bold red] "
                    f"— {change.reason}"
                )

        if rescore_event.newly_critical_flights:
            console.print(
                f"\n  [bold red]⚠ URGENT: {len(rescore_event.newly_critical_flights)} flight(s) "
                f"newly CRITICAL: {', '.join(rescore_event.newly_critical_flights)}[/bold red]"
            )
    else:
        console.print("[dim]  No significant re-score changes detected.[/dim]")


if __name__ == "__main__":
    console.print(Panel(
        "[bold cyan]ATFM NEGOTIATION AGENT — ENHANCED DEMO[/bold cyan]\n"
        "Regulation: LFEREG1034 | Paris CDG Weather Event\n\n"
        "This demo covers:\n"
        "  1. Core fleet negotiation\n"
        "  2. LLM-powered CFMU justification messages\n"
        "  3. Mid-regulation dynamic re-scoring",
        title="✈  Demo Runner",
        border_style="cyan",
    ))

    slot_xmls, flight_data = build_demo_scenario()

    agent = ATFMAgent(
        cfmu_client=CFMUApiClient(accept_rate=0.65),
        audit_log_dir="data/audit_logs",
    )

    # ── Showcase 1: Core Regulation Run ──
    console.rule("[bold]Showcase 1: Core Regulation Negotiation[/bold]")
    final_state = agent.run_regulation(
        regulation_id="LFEREG1034",
        slot_xml_messages=slot_xmls,
        flight_data_map=flight_data,
    )

    # ── Showcase 2: LLM Justification ──
    console.rule("[bold]Showcase 2: LLM Justification Layer[/bold]")
    demo_llm_justification(flight_data)

    # ── Showcase 3: Dynamic Re-Scoring ──
    console.rule("[bold]Showcase 3: Dynamic Re-Scoring[/bold]")
    demo_dynamic_rescore(agent, final_state)

    console.print(Panel(
        "[green]✓ All showcases complete[/green]\n\n"
        "  data/audit_logs/LFEREG1034.jsonl       — full audit chain\n"
        "  data/audit_logs/LFEREG1034_report.txt  — post-ops report\n\n"
        "Run [bold]pytest tests/ -v[/bold] to execute the full test suite.",
        title="Demo Complete",
        border_style="green",
    ))
