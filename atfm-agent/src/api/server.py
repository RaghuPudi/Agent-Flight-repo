"""
OCC DASHBOARD API
FastAPI server that exposes:
  GET  /regulations/{id}/status   — Current state of a regulation
  GET  /regulations/{id}/flights  — All flight scores & negotiation states
  GET  /regulations/{id}/audit    — Full audit trail
  GET  /regulations/{id}/report   — Post-ops report
  POST /flights/{fn}/override     — Human OCC override
  POST /simulate                  — Trigger a simulated regulation event
  WS   /ws/live                   — Real-time event stream to dashboard
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agent.atfm_agent import ATFMAgent, AgentState
from audit.audit_logger import AuditLogger, EventType, Outcome
from parsers.fixm_parser import make_sample_ctot_xml
from scoring.impact_scorer import (
    FlightOperationalData, CrewRecord, Connection
)
from negotiation.negotiation_planner import CFMUApiClient

app = FastAPI(
    title="ATFM Negotiation Agent — OCC Dashboard API",
    description="Autonomous ATFM slot negotiation for airline Operations Control",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory state store (production: Redis / PostgreSQL)
_agent_states: Dict[str, AgentState] = {}
_audit_loggers: Dict[str, AuditLogger] = {}
_ws_clients: List[WebSocket] = []


# ---------------------------------------------------------------------------
# WebSocket broadcast
# ---------------------------------------------------------------------------

async def broadcast(event: Dict[str, Any]) -> None:
    """Broadcast a JSON event to all connected OCC dashboard clients."""
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()   # Keep-alive ping
    except WebSocketDisconnect:
        _ws_clients.remove(websocket)


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------

class OverrideRequest(BaseModel):
    human_action: str       # e.g. "ACCEPT_SLOT" | "MANUAL_CALL_CFMU"
    reason: str
    operator_id: str

class SimulateRequest(BaseModel):
    regulation_id: str = "LFEREG1034"
    num_flights: int = 5


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/simulate")
async def simulate_regulation(req: SimulateRequest):
    """
    Trigger a simulated ATFM regulation with synthetic flight data.
    Runs the agent and returns the full result.
    """
    regulation_id = req.regulation_id
    base_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    # Build synthetic slot messages and operational data
    flight_configs = [
        {
            "fn": "BA456", "dep": "EGLL", "arr": "LFPG",
            "sobt_offset_min": 0, "ctot_offset_min": 160,
            "crew_duty_start_offset_h": -6, "max_fdp_h": 9.5,   # Will be illegal
            "connections": [
                {"flight": "BA789", "dep_offset_min": 240, "pax": 34, "backup": False},
                {"flight": "BA901", "dep_offset_min": 300, "pax": 12, "backup": True},
            ],
        },
        {
            "fn": "VS201", "dep": "EGLL", "arr": "EHAM",
            "sobt_offset_min": 30, "ctot_offset_min": 120,
            "crew_duty_start_offset_h": -4, "max_fdp_h": 10.0,
            "connections": [
                {"flight": "KL302", "dep_offset_min": 200, "pax": 22, "backup": True},
            ],
        },
        {
            "fn": "BA112", "dep": "EGLL", "arr": "EDDF",
            "sobt_offset_min": 15, "ctot_offset_min": 90,
            "crew_duty_start_offset_h": -3, "max_fdp_h": 10.0,
            "connections": [],
            "alternate_nm": 47.0, "alternate_fuel_kg": 890.0,
        },
        {
            "fn": "EI445", "dep": "EIDW", "arr": "LFPG",
            "sobt_offset_min": 45, "ctot_offset_min": 60,
            "crew_duty_start_offset_h": -2, "max_fdp_h": 10.0,
            "connections": [],
        },
        {
            "fn": "FR987", "dep": "EGSS", "arr": "LFPG",
            "sobt_offset_min": 60, "ctot_offset_min": 40,   # Low impact
            "crew_duty_start_offset_h": -1, "max_fdp_h": 10.0,
            "connections": [],
        },
    ][: req.num_flights]

    slot_xmls = []
    flight_data_map: Dict[str, FlightOperationalData] = {}

    for cfg in flight_configs:
        sobt = base_time + timedelta(minutes=cfg["sobt_offset_min"])
        ctot = base_time + timedelta(minutes=cfg["ctot_offset_min"])

        slot_xmls.append(make_sample_ctot_xml(
            flight_number=cfg["fn"],
            dep=cfg["dep"],
            arr=cfg["arr"],
            sobt=sobt.isoformat(),
            ctot=ctot.isoformat(),
            regulation_id=regulation_id,
            swap_available=True,
        ))

        duty_start = base_time + timedelta(hours=cfg["crew_duty_start_offset_h"])
        crew = [
            CrewRecord(
                crew_id=f"{cfg['fn']}-CPT",
                role="CPT",
                duty_start=duty_start,
                max_fdp_hours=cfg["max_fdp_h"],
            ),
            CrewRecord(
                crew_id=f"{cfg['fn']}-FO",
                role="FO",
                duty_start=duty_start,
                max_fdp_hours=cfg["max_fdp_h"],
            ),
        ]

        connections = [
            Connection(
                pax_count=c["pax"],
                connecting_flight=c["flight"],
                connecting_dep_time=base_time + timedelta(minutes=c["dep_offset_min"]),
                has_same_day_backup=c["backup"],
            )
            for c in cfg.get("connections", [])
        ]

        flight_data_map[cfg["fn"]] = FlightOperationalData(
            flight_number=cfg["fn"],
            dep_airport=cfg["dep"],
            arr_airport=cfg["arr"],
            scheduled_off_block=sobt,
            flight_time_min=75,
            crew=crew,
            connections=connections,
            alternate_route_extra_nm=cfg.get("alternate_nm", 0.0),
            alternate_route_extra_fuel_kg=cfg.get("alternate_fuel_kg", 0.0),
        )

    # Run the agent
    agent = ATFMAgent(
        cfmu_client=CFMUApiClient(accept_rate=0.65),
        audit_log_dir="data/audit_logs",
    )
    agent_state = agent.run_regulation(regulation_id, slot_xmls, flight_data_map)
    _agent_states[regulation_id] = agent_state

    # Broadcast to WebSocket clients
    await broadcast({
        "event": "REGULATION_COMPLETE",
        "regulation_id": regulation_id,
        "flights_processed": len(agent_state.negotiation_states),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    summary = agent._audit.get_regulation_summary(regulation_id)
    return {"status": "complete", "regulation_id": regulation_id, "summary": summary}


@app.get("/regulations/{regulation_id}/status")
async def get_regulation_status(regulation_id: str):
    state = _agent_states.get(regulation_id)
    if not state:
        raise HTTPException(status_code=404, detail="Regulation not found")
    return {
        "regulation_id": regulation_id,
        "start_time": state.start_time.isoformat(),
        "end_time": state.end_time.isoformat() if state.end_time else None,
        "flights_affected": len(state.negotiation_states),
    }


@app.get("/regulations/{regulation_id}/flights")
async def get_flight_states(regulation_id: str):
    state = _agent_states.get(regulation_id)
    if not state:
        raise HTTPException(status_code=404, detail="Regulation not found")

    result = []
    for fn, neg in state.negotiation_states.items():
        impact = state.impact_scores.get(fn)
        result.append({
            "flight_number": fn,
            "original_ctot": neg.original_ctot.isoformat(),
            "final_ctot": neg.final_ctot.isoformat() if neg.final_ctot else None,
            "resolved": neg.resolved,
            "rounds_used": neg.rounds_used,
            "priority": impact.priority if impact else "UNKNOWN",
            "composite_score": impact.composite_score if impact else 0.0,
            "crew_legal": impact.crew_legal if impact else True,
            "misconnecting_pax": impact.misconnecting_pax if impact else 0,
            "revenue_at_risk_usd": impact.connection_revenue_at_risk_usd if impact else 0.0,
            "recommendation": impact.recommendation if impact else "ACCEPT",
            "proposals": [p.model_dump() for p in neg.proposals],
        })
    return result


@app.get("/regulations/{regulation_id}/audit")
async def get_audit_trail(regulation_id: str, flight: Optional[str] = None):
    log_file = f"data/audit_logs/{regulation_id}.jsonl"
    if not os.path.exists(log_file):
        raise HTTPException(status_code=404, detail="Audit log not found")

    records = []
    with open(log_file, "r") as f:
        for line in f:
            record = json.loads(line.strip())
            if flight is None or record.get("flight_number") == flight:
                records.append(record)
    return {"regulation_id": regulation_id, "records": records, "count": len(records)}


@app.get("/regulations/{regulation_id}/report")
async def get_report(regulation_id: str):
    report_path = f"data/audit_logs/{regulation_id}_report.txt"
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not found — run simulation first")
    with open(report_path, "r") as f:
        return {"regulation_id": regulation_id, "report": f.read()}


@app.post("/flights/{flight_number}/override")
async def human_override(flight_number: str, req: OverrideRequest, regulation_id: str):
    """
    OCC coordinator overrides the agent's decision.
    The agent's original recommendation is preserved in the audit trail.
    """
    state = _agent_states.get(regulation_id)
    if not state:
        raise HTTPException(status_code=404, detail="Regulation not found")

    neg = state.negotiation_states.get(flight_number)
    if not neg:
        raise HTTPException(status_code=404, detail="Flight not found in this regulation")

    # Find the regulation's audit logger and log the override
    log_file = f"data/audit_logs/{regulation_id}.jsonl"
    logger = AuditLogger(log_dir="data/audit_logs")

    agent_rec = state.impact_scores[flight_number].recommendation if flight_number in state.impact_scores else "UNKNOWN"

    logger.log(
        event_type=EventType.HUMAN_OVERRIDE,
        regulation_id=regulation_id,
        flight_number=flight_number,
        payload={
            "human_action": req.human_action,
            "reason": req.reason,
            "operator_id": req.operator_id,
        },
        outcome=Outcome.OVERRIDDEN,
        human_override=True,
        agent_recommendation=agent_rec,
    )

    await broadcast({
        "event": "HUMAN_OVERRIDE",
        "flight_number": flight_number,
        "operator_id": req.operator_id,
        "action": req.human_action,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return {
        "status": "override_logged",
        "flight_number": flight_number,
        "agent_original_recommendation": agent_rec,
        "human_action": req.human_action,
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Minimal OCC dashboard HTML (production: full React app)."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ATFM Negotiation Agent — OCC Dashboard</title>
<style>
  body { font-family: -apple-system, Segoe UI, sans-serif; background: #0d1117; color: #e6edf3; margin: 0; padding: 20px; }
  h1 { color: #58a6ff; } h2 { color: #79c0ff; border-bottom: 1px solid #30363d; padding-bottom: 8px; }
  button { background: #1f6feb; color: #fff; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; font-size: 14px; }
  button:hover { background: #388bfd; }
  #log { background: #161b22; border: 1px solid #30363d; padding: 16px; border-radius: 8px; height: 300px; overflow-y: auto; font-family: monospace; font-size: 13px; }
  #flights { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; margin-top: 16px; }
  .flight-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; }
  .CRITICAL { border-left: 4px solid #f85149; }
  .HIGH { border-left: 4px solid #e3b341; }
  .MEDIUM { border-left: 4px solid #58a6ff; }
  .LOW { border-left: 4px solid #3fb950; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }
  .badge-CRITICAL { background: #f85149; } .badge-HIGH { background: #e3b341; color: #000; }
  .badge-MEDIUM { background: #58a6ff; } .badge-LOW { background: #3fb950; color: #000; }
</style>
</head>
<body>
<h1>✈ ATFM Negotiation Agent</h1>
<h2>OCC Dashboard</h2>
<button onclick="simulate()">▶ Simulate Regulation (LFEREG1034)</button>
<span id="status" style="margin-left: 16px; color: #8b949e;"></span>
<div id="flights"></div>
<h2 style="margin-top: 24px;">Live Event Log</h2>
<div id="log"></div>

<script>
const ws = new WebSocket(`ws://${location.host}/ws/live`);
ws.onmessage = e => {
  const data = JSON.parse(e.data);
  const log = document.getElementById('log');
  log.innerHTML += `<div style="color:#58a6ff">[${new Date().toISOString()}]</div>`;
  log.innerHTML += `<pre style="margin:2px 0 8px 12px;color:#e6edf3">${JSON.stringify(data, null, 2)}</pre>`;
  log.scrollTop = log.scrollHeight;
};

async function simulate() {
  document.getElementById('status').textContent = 'Running simulation...';
  document.getElementById('flights').innerHTML = '';
  const res = await fetch('/simulate', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({regulation_id:'LFEREG1034', num_flights:5})});
  const data = await res.json();
  document.getElementById('status').textContent = `Done — ${data.summary.flights_affected} flights processed`;
  loadFlights();
}

async function loadFlights() {
  const res = await fetch('/regulations/LFEREG1034/flights');
  const flights = await res.json();
  const container = document.getElementById('flights');
  container.innerHTML = flights.map(f => `
    <div class="flight-card ${f.priority}">
      <strong>${f.flight_number}</strong>
      <span class="badge badge-${f.priority}" style="margin-left:8px">${f.priority}</span>
      <div style="margin-top:8px;font-size:13px;color:#8b949e">
        <div>CTOT: ${new Date(f.original_ctot).toISOString().substr(11,5)}Z → ${f.final_ctot ? new Date(f.final_ctot).toISOString().substr(11,5)+'Z' : '⚑ ESCALATED'}</div>
        <div>Crew legal: ${f.crew_legal ? '✓' : '✗ VIOLATION'}</div>
        <div>Misconnects: ${f.misconnecting_pax} pax</div>
        <div>Revenue at risk: $${f.revenue_at_risk_usd.toLocaleString()}</div>
        <div>Decision: <strong>${f.recommendation}</strong> (rounds: ${f.rounds_used})</div>
      </div>
    </div>`).join('');
}
</script>
</body>
</html>"""
