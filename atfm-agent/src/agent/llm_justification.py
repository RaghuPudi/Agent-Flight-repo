"""
LLM JUSTIFICATION & REASONING LAYER
====================================
Replaces templated justification strings with GPT-4o / Claude generated text
that mirrors how an experienced EUROCONTROL Network Manager writes ATFM messages.

Three capabilities:
  1. generate_cfmu_justification()  — ICAO-standard slot swap / counter-CTOT
                                      message for submission to CFMU B2B API
  2. generate_escalation_brief()    — Natural language brief for OCC coordinator,
                                      includes regulatory citation and recommended action
  3. explain_decision()             — Plain-English reasoning card shown on OCC dashboard
                                      so coordinators can trust or override confidently

Override learning:
  4. record_override_outcome()      — Stores human override + actual outcome
  5. get_learning_summary()         — Surfaces patterns where agent was wrong

In production: uses openai or anthropic SDK.
In PoC: falls back to a high-quality template engine if no API key is set,
        so the system runs without credentials while still showing realistic output.
"""

from __future__ import annotations

import os
import json
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from scoring.impact_scorer import ImpactScore
from negotiation.negotiation_planner import NegotiationState, NegotiationProposal


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

@dataclass
class CFMUJustificationMessage:
    """ICAO-standard message for CFMU B2B submission."""
    flight_number: str
    regulation_id: str
    requested_ctot: datetime
    justification_codes: List[str]
    formal_text: str           # Submitted to CFMU API (terse, coded)
    plain_english: str         # Shown to OCC coordinator
    regulatory_citations: List[str]
    confidence: float          # 0.0–1.0 agent confidence in this request
    generated_by: str          # "llm" | "template"


@dataclass
class EscalationBrief:
    """Natural language brief for OCC coordinator requiring human action."""
    flight_number: str
    headline: str              # One-line summary
    situation: str             # What happened, what was tried
    recommended_action: str    # Specific next step
    regulatory_context: str    # Which rules apply and why
    time_critical_by: Optional[datetime]   # When decision must be made
    financial_impact: str      # Dollar-quantified consequence of inaction
    generated_by: str


@dataclass
class DecisionExplanation:
    """Plain-English reasoning card for OCC dashboard."""
    flight_number: str
    decision: str              # What the agent decided
    why: str                   # Primary reason in one sentence
    what_was_considered: List[str]   # Bullet-point alternatives
    what_was_rejected: List[str]     # Alternatives and why they lost
    confidence_pct: int        # 0–100
    generated_by: str


@dataclass
class OverrideOutcomeRecord:
    """Stores agent recommendation vs human action vs actual outcome for learning."""
    flight_number: str
    regulation_id: str
    timestamp: datetime
    agent_recommendation: str
    human_action: str
    agent_projected_benefit_usd: float
    actual_outcome_delay_saved_min: Optional[float] = None
    actual_outcome_revenue_saved_usd: Optional[float] = None
    post_hoc_analysis: Optional[str] = None


# ---------------------------------------------------------------------------
# Regulation & justification context maps
# ---------------------------------------------------------------------------

_REGULATION_REASON_CONTEXT = {
    "WEATHER":       "Meteorological conditions reducing ATC capacity",
    "ATC_CAPACITY":  "Air Traffic Control staffing or sector capacity constraints",
    "EQUIPMENT":     "ATC equipment failure or system degradation",
    "SPECIAL_EVENT": "Special event generating abnormal traffic density",
    "MILITARY":      "Military activity restricting airspace availability",
}

_JUSTIFICATION_CODE_DESCRIPTIONS = {
    "FDP_VIOLATION":             "Flight Duty Period will exceed legal maximum under EU OPS ORO.FTL.205",
    "HIGH_PAX_MISCONNECT":       "High number of passengers will miss onward connections",
    "SIGNIFICANT_REVENUE_IMPACT": "Significant commercial revenue impact to airline network",
    "FUEL_INEFFICIENCY":         "Excessive APU fuel burn during extended ground hold",
    "OPERATIONAL_IMPACT":        "General operational impact to airline schedule integrity",
}

_REGULATORY_CITATIONS = {
    "FDP_VIOLATION": [
        "EU OPS ORO.FTL.205 — Maximum Flight Duty Period",
        "EU OPS ORO.FTL.235 — Rest Period requirements",
        "CAT.OP.MPA.190 — Crew complement requirements",
    ],
    "HIGH_PAX_MISCONNECT": [
        "EC Regulation 261/2004 — Passenger rights and compensation",
        "IATA Resolution 735d — Involuntary rerouting obligations",
    ],
    "FUEL_INEFFICIENCY": [
        "EUROCONTROL Performance Scheme — Fuel efficiency KPI",
        "EU ETS Directive 2003/87/EC — Carbon cost obligations",
    ],
}


# ---------------------------------------------------------------------------
# LLM Client wrapper (with template fallback)
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Wraps OpenAI / Anthropic SDK.
    Falls back to high-quality template generation if no API key is set.
    """

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.2):
        self._model = model
        self._temperature = temperature
        self._api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        self._use_llm = self._api_key is not None

        if self._use_llm:
            try:
                import openai
                self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                self._backend = "openai"
            except ImportError:
                try:
                    import anthropic
                    self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
                    self._backend = "anthropic"
                    self._model = "claude-3-5-sonnet-20241022"
                except ImportError:
                    self._use_llm = False
                    self._backend = "template"
        else:
            self._backend = "template"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM or fall back to returning the user_prompt as-is (template path)."""
        if not self._use_llm:
            return "__TEMPLATE__"

        try:
            if self._backend == "openai":
                response = self._client.chat.completions.create(
                    model=self._model,
                    temperature=self._temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    max_tokens=600,
                )
                return response.choices[0].message.content.strip()

            elif self._backend == "anthropic":
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=600,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                return response.content[0].text.strip()

        except Exception as e:
            # Graceful degradation — never crash the agent due to LLM unavailability
            return f"__TEMPLATE__ [LLM error: {e}]"

    @property
    def backend(self) -> str:
        return self._backend


# ---------------------------------------------------------------------------
# Justification Engine
# ---------------------------------------------------------------------------

class LLMJustificationEngine:
    """
    Generates three types of language output for the ATFM agent:
      1. CFMU submission messages (terse, ICAO-standard)
      2. OCC coordinator escalation briefs (natural language, actionable)
      3. Decision explanation cards (plain English, dashboard-ready)
    """

    _CFMU_SYSTEM_PROMPT = textwrap.dedent("""
        You are an expert EUROCONTROL Network Manager with 20 years of ATFM experience.
        You write slot swap and counter-CTOT justification messages for CFMU B2B submission.
        Your messages are terse, precise, use ICAO phraseology, reference specific regulatory
        articles, and are structured to maximise CFMU acceptance likelihood.
        Never use marketing language. Be factual, specific, and cite regulation numbers.
        Keep formal messages under 120 words.
    """).strip()

    _ESCALATION_SYSTEM_PROMPT = textwrap.dedent("""
        You are an airline Operations Control Centre (OCC) AI assistant.
        You write escalation briefs for human OCC coordinators who need to make urgent decisions.
        Your briefs are clear, structured, time-aware, and always end with a specific recommended
        action. You quantify financial consequences. You cite regulatory obligations when relevant.
        Write for an experienced coordinator — no jargon explanation needed.
    """).strip()

    _EXPLANATION_SYSTEM_PROMPT = textwrap.dedent("""
        You are the ATFM Negotiation Agent explaining your own decision to an OCC coordinator.
        Write in first person. Be honest about uncertainty. Show what you considered and why
        you chose what you chose. Use plain English — no ICAO codes in this output.
        Keep it under 100 words for the main reasoning.
    """).strip()

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self._llm = llm_client or LLMClient()

    # ------------------------------------------------------------------
    # 1. CFMU Justification Message
    # ------------------------------------------------------------------

    def generate_cfmu_justification(
        self,
        impact: ImpactScore,
        proposal: NegotiationProposal,
        regulation_reason: str = "WEATHER",
    ) -> CFMUJustificationMessage:
        """
        Generate an ICAO-standard justification message for CFMU submission.
        """
        codes = proposal.justification_codes
        reg_context = _REGULATION_REASON_CONTEXT.get(regulation_reason, regulation_reason)
        citations = []
        for code in codes:
            citations.extend(_REGULATORY_CITATIONS.get(code, []))

        user_prompt = textwrap.dedent(f"""
            Flight: {impact.flight_number}
            Regulation: {proposal.regulation_id} ({regulation_reason} — {reg_context})
            Current CTOT: {proposal.current_ctot.strftime('%H:%MZ')}
            Requested CTOT: {proposal.requested_ctot.strftime('%H:%MZ') if proposal.requested_ctot else 'N/A'}
            Action type: {proposal.action}
            Swap partner: {proposal.swap_with_flight or 'N/A'}

            Impact facts:
            - Crew FDP legal: {impact.crew_legal} {f'(violation: {impact.crew_fdp_violation_hours:.1f}h over limit, crew: {", ".join(impact.crew_affected)})' if not impact.crew_legal else ''}
            - Misconnecting pax: {impact.misconnecting_pax}
            - Revenue at risk: ${impact.connection_revenue_at_risk_usd:,.0f}
            - Justification codes: {', '.join(codes)}

            Write the formal CFMU B2B justification message for this {proposal.action} request.
        """).strip()

        raw = self._llm.complete(self._CFMU_SYSTEM_PROMPT, user_prompt)

        if raw.startswith("__TEMPLATE__"):
            formal_text = self._template_cfmu_message(impact, proposal, regulation_reason)
            generated_by = "template"
        else:
            formal_text = raw
            generated_by = "llm"

        plain_english = self._build_plain_cfmu_summary(impact, proposal)

        return CFMUJustificationMessage(
            flight_number=impact.flight_number,
            regulation_id=proposal.regulation_id,
            requested_ctot=proposal.requested_ctot or proposal.current_ctot,
            justification_codes=codes,
            formal_text=formal_text,
            plain_english=plain_english,
            regulatory_citations=list(set(citations)),
            confidence=self._estimate_confidence(impact, proposal),
            generated_by=generated_by,
        )

    # ------------------------------------------------------------------
    # 2. Escalation Brief
    # ------------------------------------------------------------------

    def generate_escalation_brief(
        self,
        impact: ImpactScore,
        neg_state: NegotiationState,
        time_critical_by: Optional[datetime] = None,
    ) -> EscalationBrief:
        """
        Generate a natural language escalation brief for the OCC coordinator.
        """
        attempts_summary = "; ".join(
            f"Round {p.round_number} {p.action} → {p.response or 'N/A'}"
            for p in neg_state.proposals
            if p.action not in ("ACCEPT", "ESCALATE")
        )

        user_prompt = textwrap.dedent(f"""
            Flight {impact.flight_number} needs immediate human intervention.

            Current situation:
            - Assigned CTOT: {impact.ctot.strftime('%H:%MZ')}
            - Priority: {impact.priority} (score: {impact.composite_score:.2f})
            - Crew legal: {impact.crew_legal}{f' — VIOLATION {impact.crew_fdp_violation_hours:.1f}h over limit' if not impact.crew_legal else ''}
            - Misconnecting pax: {impact.misconnecting_pax} | Revenue at risk: ${impact.connection_revenue_at_risk_usd:,.0f}
            - APU hold cost: ${impact.apu_extra_cost_usd:,.0f}

            What the agent tried:
            {attempts_summary or 'All automated rounds attempted — all rejected by CFMU'}

            Write an escalation brief with: headline, situation summary, specific recommended
            action, regulatory context (if crew issue), financial impact of inaction.
        """).strip()

        raw = self._llm.complete(self._ESCALATION_SYSTEM_PROMPT, user_prompt)

        if raw.startswith("__TEMPLATE__"):
            headline, situation, recommended_action, regulatory_context, financial_impact = \
                self._template_escalation(impact, neg_state)
            generated_by = "template"
        else:
            # Parse LLM output into structured fields
            headline, situation, recommended_action, regulatory_context, financial_impact = \
                self._parse_escalation_output(raw, impact, neg_state)
            generated_by = "llm"

        return EscalationBrief(
            flight_number=impact.flight_number,
            headline=headline,
            situation=situation,
            recommended_action=recommended_action,
            regulatory_context=regulatory_context,
            time_critical_by=time_critical_by,
            financial_impact=financial_impact,
            generated_by=generated_by,
        )

    # ------------------------------------------------------------------
    # 3. Decision Explanation (dashboard card)
    # ------------------------------------------------------------------

    def explain_decision(
        self,
        impact: ImpactScore,
        chosen_proposal: NegotiationProposal,
        alternatives_considered: List[Dict[str, Any]],
    ) -> DecisionExplanation:
        """
        Generate a plain-English explanation of why the agent made a particular decision.
        Shown on the OCC dashboard so coordinators can understand and trust (or override) it.
        """
        alts_text = "\n".join(
            f"- {a['action']}: score {a.get('score', 'N/A')}, rejected because: {a.get('reason', 'N/A')}"
            for a in alternatives_considered
        )

        user_prompt = textwrap.dedent(f"""
            I (the ATFM agent) decided to {chosen_proposal.action} for flight {impact.flight_number}.

            Key facts I used:
            - Crew legal: {impact.crew_legal}{f', violating FDP by {impact.crew_fdp_violation_hours:.1f}h' if not impact.crew_legal else ''}
            - {impact.misconnecting_pax} passengers will miss connections (${impact.connection_revenue_at_risk_usd:,.0f} at risk)
            - Composite impact score: {impact.composite_score:.2f} ({impact.priority})
            - Expected benefit: ${chosen_proposal.estimated_benefit_usd:,.0f}

            Alternatives I considered:
            {alts_text}

            Explain my decision in plain English for the OCC coordinator dashboard.
            Format: one-sentence 'why', then bullet points for considered vs rejected.
        """).strip()

        raw = self._llm.complete(self._EXPLANATION_SYSTEM_PROMPT, user_prompt)

        if raw.startswith("__TEMPLATE__"):
            why, considered, rejected = self._template_explanation(impact, chosen_proposal, alternatives_considered)
            generated_by = "template"
        else:
            why, considered, rejected = self._parse_explanation_output(raw, impact, chosen_proposal, alternatives_considered)
            generated_by = "llm"

        confidence_pct = int(self._estimate_confidence(impact, chosen_proposal) * 100)

        return DecisionExplanation(
            flight_number=impact.flight_number,
            decision=chosen_proposal.action,
            why=why,
            what_was_considered=considered,
            what_was_rejected=rejected,
            confidence_pct=confidence_pct,
            generated_by=generated_by,
        )

    # ------------------------------------------------------------------
    # 4. Override Learning
    # ------------------------------------------------------------------

    def record_override_outcome(
        self,
        flight_number: str,
        regulation_id: str,
        agent_recommendation: str,
        human_action: str,
        agent_projected_benefit_usd: float,
        actual_delay_saved_min: Optional[float] = None,
        actual_revenue_saved_usd: Optional[float] = None,
    ) -> OverrideOutcomeRecord:
        """
        Record a human override and its actual outcome for post-hoc analysis.
        After 24h, compares agent projection vs reality.
        """
        post_hoc = None

        if actual_delay_saved_min is not None:
            comparison_prompt = textwrap.dedent(f"""
                Flight {flight_number} — post-hoc override analysis:
                Agent recommended: {agent_recommendation} (projected ${agent_projected_benefit_usd:,.0f} benefit)
                Human did: {human_action}
                Actual outcome: {actual_delay_saved_min:.0f} min saved, ${actual_revenue_saved_usd or 0:,.0f} revenue saved

                Write a 2-sentence post-hoc analysis: was the human right to override?
                What can the agent learn from this? Be honest and specific.
            """).strip()

            raw = self._llm.complete(
                "You are an aviation operations analyst reviewing AI agent override decisions.",
                comparison_prompt,
            )

            if not raw.startswith("__TEMPLATE__"):
                post_hoc = raw
            else:
                agent_value = agent_projected_benefit_usd
                actual_value = (actual_revenue_saved_usd or 0) + (actual_delay_saved_min or 0) * 100
                if actual_value >= agent_value * 0.9:
                    post_hoc = (
                        f"Human override was justified — actual outcome (${actual_value:,.0f}) "
                        f"matched or exceeded agent projection (${agent_value:,.0f}). "
                        f"Agent should review its swap pool scoring for this route."
                    )
                else:
                    post_hoc = (
                        f"Agent recommendation was superior — actual outcome (${actual_value:,.0f}) "
                        f"underperformed agent projection (${agent_value:,.0f}) by "
                        f"${agent_value - actual_value:,.0f}. "
                        f"Consider reducing override frequency for {agent_recommendation} decisions."
                    )

        return OverrideOutcomeRecord(
            flight_number=flight_number,
            regulation_id=regulation_id,
            timestamp=datetime.now(timezone.utc),
            agent_recommendation=agent_recommendation,
            human_action=human_action,
            agent_projected_benefit_usd=agent_projected_benefit_usd,
            actual_outcome_delay_saved_min=actual_delay_saved_min,
            actual_outcome_revenue_saved_usd=actual_revenue_saved_usd,
            post_hoc_analysis=post_hoc,
        )

    def get_learning_summary(self, override_records: List[OverrideOutcomeRecord]) -> str:
        """
        Summarise patterns in human override outcomes to surface tuning recommendations.
        """
        if not override_records:
            return "No override records yet."

        resolved = [r for r in override_records if r.actual_outcome_delay_saved_min is not None]
        if not resolved:
            return f"{len(override_records)} overrides recorded — awaiting outcome data (check back in 24h)."

        agent_wins = sum(
            1 for r in resolved
            if (r.actual_outcome_revenue_saved_usd or 0) < r.agent_projected_benefit_usd * 0.9
        )
        human_wins = len(resolved) - agent_wins

        prompt = textwrap.dedent(f"""
            Override learning summary from {len(resolved)} resolved override events:
            - Agent recommendation was better: {agent_wins} times
            - Human override was better: {human_wins} times
            - Override types: {[r.agent_recommendation for r in resolved]}

            Write a 3-bullet actionable learning summary for the airline's AI governance team.
        """).strip()

        raw = self._llm.complete(
            "You are an AI performance analyst for an airline operations system.",
            prompt,
        )

        if raw.startswith("__TEMPLATE__"):
            return (
                f"Override analysis ({len(resolved)} events):\n"
                f"  • Agent recommendation was superior in {agent_wins}/{len(resolved)} cases\n"
                f"  • Human override was superior in {human_wins}/{len(resolved)} cases\n"
                f"  • {'Consider reviewing swap acceptance threshold — agent recommendations being overridden successfully.' if human_wins > agent_wins else 'Agent performing well — coordinate with OCC to reduce unnecessary overrides.'}"
            )
        return raw

    # ------------------------------------------------------------------
    # Template fallbacks (high quality, no LLM needed)
    # ------------------------------------------------------------------

    def _template_cfmu_message(
        self, impact: ImpactScore, proposal: NegotiationProposal, reason: str
    ) -> str:
        lines = [
            f"REF: {proposal.regulation_id} / {impact.flight_number}",
            f"REQUEST: {proposal.action.replace('_', ' ')} — CTOT {proposal.requested_ctot.strftime('%H%MZ') if proposal.requested_ctot else 'TBD'}",
        ]
        if not impact.crew_legal:
            lines.append(
                f"SAFETY: FDP EXCEEDANCE {impact.crew_fdp_violation_hours:.1f}H — "
                f"EU OPS ORO.FTL.205 / CREW {', '.join(impact.crew_affected)}."
            )
        if impact.misconnecting_pax > 0:
            lines.append(
                f"COMMERCIAL: {impact.misconnecting_pax} PAX MISCONNECT — "
                f"LAST BANK DEPARTURE — EC 261/2004 COMPENSATION EXPOSURE USD{impact.connection_revenue_at_risk_usd:,.0f}."
            )
        if proposal.swap_with_flight:
            lines.append(f"SWAP PARTNER: {proposal.swap_with_flight} — MUTUALLY BENEFICIAL EXCHANGE CONFIRMED.")
        lines.append("REQUEST PRIORITY CONSIDERATION. OPS CTRL AVAILABLE.")
        return "\n".join(lines)

    def _template_escalation(
        self, impact: ImpactScore, neg_state: NegotiationState
    ):
        attempts = len([p for p in neg_state.proposals if p.action not in ("ESCALATE",)])
        headline = (
            f"{'🔴 SAFETY: Crew FDP violation imminent — ' if not impact.crew_legal else ''}"
            f"{impact.flight_number}: {attempts} automated rounds exhausted — manual intervention required"
        )
        situation = (
            f"Flight {impact.flight_number} was assigned CTOT {impact.ctot.strftime('%H:%MZ')} "
            f"under regulation {neg_state.regulation_id}. "
            f"{'Crew will exceed FDP limit by ' + f'{impact.crew_fdp_violation_hours:.1f}h — this is a safety violation. ' if not impact.crew_legal else ''}"
            f"The agent attempted {attempts} round(s) of automated negotiation "
            f"(slot swap + counter-CTOT) — all were rejected by CFMU."
        )
        recommended_action = (
            f"Call EUROCONTROL Network Manager directly on the dedicated airline hotline. "
            f"{'Cite FDP_VIOLATION under EU OPS ORO.FTL.205 — CFMU gives priority treatment to safety-coded exemption requests. ' if not impact.crew_legal else ''}"
            f"Request CTOT {(neg_state.original_ctot).strftime('%H:%MZ')} or nearest available. "
            f"If unavailable, initiate crew change procedure immediately."
        )
        regulatory_context = (
            "EU OPS ORO.FTL.205 prohibits operating flight if FDP would be exceeded. "
            "EC Regulation 261/2004 requires compensation for misconnecting passengers. "
            "CAT.OP.MPA.190 requires minimum crew complement at all times."
        ) if not impact.crew_legal else (
            "EC Regulation 261/2004 may require compensation for misconnecting passengers "
            f"if delay exceeds threshold. {impact.misconnecting_pax} passengers affected."
        )
        financial_impact = (
            f"Inaction cost: ${impact.connection_revenue_at_risk_usd:,.0f} misconnect revenue + "
            f"${impact.apu_extra_cost_usd:,.0f} APU hold fuel + potential EC 261/2004 compensation. "
            f"Total exposure: ${impact.connection_revenue_at_risk_usd + impact.apu_extra_cost_usd:,.0f}+"
        )
        return headline, situation, recommended_action, regulatory_context, financial_impact

    def _template_explanation(
        self, impact: ImpactScore, proposal: NegotiationProposal, alternatives: List[Dict]
    ):
        why = (
            f"I chose {proposal.action} for {impact.flight_number} because "
            f"{'the crew would go illegal at the assigned CTOT — this is a safety obligation, not a preference. ' if not impact.crew_legal else ''}"
            f"{'the swap pool had a willing partner with an earlier slot, giving the best chance of recovery. ' if proposal.action == 'SWAP_REQUEST' else ''}"
            f"{'a reroute via alternate waypoint saves more than the extra fuel cost. ' if proposal.action == 'REROUTE' else ''}"
            f"{'the delay impact is low enough that consuming CFMU budget here would be wasteful. ' if proposal.action == 'ACCEPT' else ''}"
        ).strip()

        considered = [
            f"Composite impact score: {impact.composite_score:.2f} ({impact.priority} priority)",
            f"Crew legality: {'Legal ✓' if impact.crew_legal else f'ILLEGAL — {impact.crew_fdp_violation_hours:.1f}h over limit ✗'}",
            f"{impact.misconnecting_pax} misconnecting passengers (${impact.connection_revenue_at_risk_usd:,.0f} at risk)",
            f"CFMU budget position: negotiation budget available",
        ]
        if impact.reroute_available:
            considered.append(f"Reroute option: saves ${impact.reroute_saves_usd:,.0f} net vs holding")

        rejected = [
            f"{a['action']}: {a.get('reason', 'lower expected outcome')}"
            for a in alternatives
            if a.get("action") != proposal.action
        ]
        if not rejected:
            rejected = [f"ACCEPT: score too high ({impact.composite_score:.2f}) to absorb without negotiation"]

        return why, considered, rejected

    def _parse_escalation_output(self, raw: str, impact: ImpactScore, neg_state: NegotiationState):
        """Best-effort parse of LLM escalation output into structured fields."""
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        headline = lines[0] if lines else f"Manual intervention required — {impact.flight_number}"
        body = "\n".join(lines[1:]) if len(lines) > 1 else raw
        _, situation, recommended_action, regulatory_context, financial_impact = \
            self._template_escalation(impact, neg_state)
        return headline, body or situation, recommended_action, regulatory_context, financial_impact

    def _parse_explanation_output(self, raw: str, impact: ImpactScore, proposal: NegotiationProposal, alternatives: List[Dict]):
        """Best-effort parse of LLM explanation output."""
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        why = lines[0] if lines else ""
        _, considered, rejected = self._template_explanation(impact, proposal, alternatives)
        return why or considered[0], considered, rejected

    def _build_plain_cfmu_summary(self, impact: ImpactScore, proposal: NegotiationProposal) -> str:
        return (
            f"Requesting CTOT improvement for {impact.flight_number} from "
            f"{proposal.current_ctot.strftime('%H:%MZ')} to "
            f"{proposal.requested_ctot.strftime('%H:%MZ') if proposal.requested_ctot else 'TBD'}. "
            f"{'Crew will exceed legal FDP if current slot holds. ' if not impact.crew_legal else ''}"
            f"{impact.misconnecting_pax} passengers risk misconnecting (${impact.connection_revenue_at_risk_usd:,.0f})."
        )

    def _estimate_confidence(self, impact: ImpactScore, proposal: NegotiationProposal) -> float:
        """Estimate agent confidence in its recommendation (0.0–1.0)."""
        score = 0.5
        # Higher confidence when safety code present (CFMU historically responds well)
        if "FDP_VIOLATION" in proposal.justification_codes:
            score += 0.25
        # Higher confidence when swap partner available
        if proposal.swap_with_flight:
            score += 0.15
        # Lower confidence when only operational code
        if proposal.justification_codes == ["OPERATIONAL_IMPACT"]:
            score -= 0.15
        return min(max(score, 0.1), 0.95)
