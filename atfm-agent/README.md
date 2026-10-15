# ✈ ATFM Negotiation Agent — Proof of Concept

An autonomous AI agent that negotiates ATFM departure slots on behalf of an
airline's entire fleet, replacing the slow, manual OCC coordinator workflow
with a principled multi-round negotiation engine.

---

## Project Structure

```
atfm-agent/
├── run_demo.py                    ← Standalone demo (no server needed)
├── requirements.txt
├── src/
│   ├── parsers/
│   │   └── fixm_parser.py         ← FIXM 4.2 XML message parser
│   ├── scoring/
│   │   └── impact_scorer.py       ← Crew legality + connection + fuel scorer
│   ├── negotiation/
│   │   └── negotiation_planner.py ← 5-round negotiation strategy engine
│   ├── audit/
│   │   └── audit_logger.py        ← Immutable hash-chained audit log
│   ├── agent/
│   │   └── atfm_agent.py          ← Main orchestrator (LangGraph-style nodes)
│   └── api/
│       └── server.py              ← FastAPI + WebSocket OCC dashboard API
├── tests/
│   └── test_atfm_agent.py         ← Full test suite (parser/scorer/audit/planner/e2e)
└── data/
    └── audit_logs/                ← JSONL audit logs + post-ops reports (auto-created)
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the standalone demo
```bash
python run_demo.py
```
Simulates ATFM regulation `LFEREG1034` with 5 flights covering all scenarios
(crew violation, misconnects, reroute candidate, medium and low impact).
Outputs a rich terminal report and writes audit logs + post-ops report to
`data/audit_logs/`.

### 3. Run the test suite
```bash
pytest tests/ -v
```

### 4. Start the OCC Dashboard API
```bash
# Always use the launch script — it sets up all paths correctly:
python start_server.py

# Options:
python start_server.py --port 8080 --reload

# Then open:  http://127.0.0.1:8000
# API docs:   http://127.0.0.1:8000/docs
```

> ⚠️ Do NOT run `uvicorn src.api.server:app` directly — it will fail with
> `ModuleNotFoundError` because uvicorn won't resolve the internal imports.
> Always use `python start_server.py`.

---

## Core Components

### `FIXMParser` — [`src/parsers/fixm_parser.py`](src/parsers/fixm_parser.py)
Decodes EUROCONTROL CFMU B2B XML messages (FIXM 4.2 schema subset).
- `parse_ctot_message(xml)` → `SlotMessage`
- `parse_regulation_notice(xml)` → `RegulationNotice`
- Sample XML generators for testing without a live CFMU connection

### `ImpactScorer` — [`src/scoring/impact_scorer.py`](src/scoring/impact_scorer.py)
Multi-objective impact evaluator with three dimensions:

| Dimension | Rule | Weight |
|---|---|---|
| **Crew legality** | EU OPS FDP limit (configurable per crew) | 0.50 |
| **Connection loss** | Misconnecting pax × ticket value × backup factor | 0.02 per $1K |
| **Fuel delta** | APU hold cost vs reroute cost | 0.005 per $1K |

Outputs `ImpactScore` with composite score (0–1), priority, and recommendation.

### `NegotiationPlanner` — [`src/negotiation/negotiation_planner.py`](src/negotiation/negotiation_planner.py)
5-round negotiation strategy:

```
Round 1 → Accept            (composite score < 0.15)
Round 2 → Reroute request   (if reroute_saves > $5,000)
Round 2 → Slot swap (SSR)   (scan pool, find willing partner)
Round 3 → Counter-CTOT      (with justification codes)
Round 4 → Escalate to human (with full brief)
```

Fleet-level optimization: highest-priority flights consume the CFMU request
budget first (15 requests per regulation event).

### `AuditLogger` — [`src/audit/audit_logger.py`](src/audit/audit_logger.py)
Append-only, SHA-256 hash-chained event log.
- Every event: `SLOT_ASSIGNED → IMPACT_SCORED → DECISION_MADE → SWAP_REQUEST → RESPONSE`
- Human overrides logged separately — agent's original recommendation is never erased
- `verify_chain_integrity()` detects any tampering
- Persists to JSONL files (one file per regulation)
- `get_regulation_summary()` generates aggregate post-ops metrics

### `ATFMAgent` — [`src/agent/atfm_agent.py`](src/agent/atfm_agent.py)
Main orchestrator with four sequential nodes:

```
Node 1: Parse FIXM messages
Node 2: Score operational impact (all flights in parallel context)
Node 3: Sort by priority, run negotiation planner (budget-aware)
Node 4: Generate post-ops report + verify audit chain
```

---

## Negotiation Budget Management

CFMU monitors airlines that submit excessive swap/counter requests.
The agent has a configurable budget (default: 15 CFMU calls per regulation).
It allocates this budget to flights in descending composite score order.
Low-priority flights that exceed the budget are force-accepted.

---

## Extending to Production

| Component | PoC | Production |
|---|---|---|
| CFMU connection | Stub (`CFMUApiClient`) | EUROCONTROL B2B SOAP/REST (SWIM) |
| Crew legality | Simplified FDP rule | Full EU OPS / FAR Part 117 rules engine |
| OPS data | Hardcoded in demo | Real-time feed from OPS system (ACARS/AIMS) |
| Audit persistence | JSONL on disk | Apache Kafka + immutable object store (S3/COS) |
| Fleet optimizer | Sequential scoring | OR-Tools multi-objective integer program |
| LLM reasoning | Plain text justification | GPT-4o / Claude for natural language briefings |
| Dashboard | Inline HTML | Full React app with WebSocket live updates |

---

## Sample Post-Ops Report Output

```
ATFM REGULATION POST-OPS REPORT
Regulation: LFEREG1034
Generated: 2026-10-15T14:32:11Z
============================================================

Flights affected         : 5
Swaps accepted           : 2 / 3
Human overrides          : 0
Delay recovered          : 140 min
Revenue protected        : $41,600
Misconnects prevented    : 40 pax
Crew violations prevented: 1

FLIGHT-BY-FLIGHT BREAKDOWN
============================================================

BA456:
  Flight BA456 | CTOT 16:40Z
  Priority    : CRITICAL
  Crew legal  : NO — 1.8h over limit
  Misconnects : 40 pax | $41,600 at risk
  APU cost    : $224 (hold 160 min)
  Reroute     : N/A
  Composite   : 0.93
  Decision    : ESCALATE
  Rounds: 4
  Final CTOT: ESCALATED
    Round 2: SWAP_REQUEST → SLOT_SWAP_ACCEPTED
    Round 3: COUNTER_CTOT → COUNTER_CTOT_REJECTED: Insufficient capacity
    Round 4: ESCALATE → N/A
```
