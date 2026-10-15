"""
IMMUTABLE AUDIT LOGGER
Every negotiation event is logged as an append-only record.
Each record is SHA-256 hashed and chained to the previous record
(blockchain-style integrity) so tampering is detectable.

In production: persists to Apache Kafka + immutable object store (S3/COS).
For PoC: in-memory chain + JSONL file on disk.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Event Types
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    REGULATION_RECEIVED     = "REGULATION_RECEIVED"
    SLOT_ASSIGNED           = "SLOT_ASSIGNED"
    IMPACT_SCORED           = "IMPACT_SCORED"
    DECISION_MADE           = "DECISION_MADE"
    SWAP_REQUEST_SUBMITTED  = "SWAP_REQUEST_SUBMITTED"
    SWAP_RESPONSE_RECEIVED  = "SWAP_RESPONSE_RECEIVED"
    COUNTER_PROPOSAL_SUBMITTED = "COUNTER_PROPOSAL_SUBMITTED"
    REROUTE_REQUESTED       = "REROUTE_REQUESTED"
    SLOT_ACCEPTED           = "SLOT_ACCEPTED"
    HUMAN_OVERRIDE          = "HUMAN_OVERRIDE"
    SHADOW_RECOMMENDATION   = "SHADOW_RECOMMENDATION"   # Agent rec when human overrides
    NEGOTIATION_CLOSED      = "NEGOTIATION_CLOSED"


class Outcome(str, Enum):
    PENDING   = "PENDING"
    SUCCESS   = "SUCCESS"
    REJECTED  = "REJECTED"
    EXPIRED   = "EXPIRED"
    OVERRIDDEN = "OVERRIDDEN"


# ---------------------------------------------------------------------------
# Audit Record
# ---------------------------------------------------------------------------

class AuditRecord(BaseModel):
    event_id: str
    sequence: int                    # Monotonically increasing
    timestamp: datetime
    event_type: EventType
    regulation_id: str
    flight_number: str
    payload: Dict[str, Any]
    outcome: Outcome = Outcome.PENDING
    human_override: bool = False
    agent_recommendation: Optional[str] = None  # What agent wanted (even if overridden)
    previous_hash: str = ""          # Hash of previous record (chain integrity)
    record_hash: str = ""            # SHA-256 of this record's content

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of the record content (excluding record_hash itself)."""
        content = {
            "event_id": self.event_id,
            "sequence": self.sequence,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "regulation_id": self.regulation_id,
            "flight_number": self.flight_number,
            "payload": self.payload,
            "outcome": self.outcome,
            "human_override": self.human_override,
            "previous_hash": self.previous_hash,
        }
        raw = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Audit Logger
# ---------------------------------------------------------------------------

class AuditLogger:
    """
    Append-only, hash-chained audit log for all agent negotiation events.
    Thread-safe for single-process PoC. In production: Kafka producer.
    """

    def __init__(self, log_dir: str = "data/audit_logs"):
        self._records: List[AuditRecord] = []
        self._sequence: int = 0
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core logging
    # ------------------------------------------------------------------

    def log(
        self,
        event_type: EventType,
        regulation_id: str,
        flight_number: str,
        payload: Dict[str, Any],
        outcome: Outcome = Outcome.PENDING,
        human_override: bool = False,
        agent_recommendation: Optional[str] = None,
    ) -> AuditRecord:
        """Append a new event to the audit chain."""
        self._sequence += 1
        prev_hash = self._records[-1].record_hash if self._records else ""

        record = AuditRecord(
            event_id=f"EVT-{regulation_id}-{flight_number}-{self._sequence:04d}",
            sequence=self._sequence,
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            regulation_id=regulation_id,
            flight_number=flight_number,
            payload=payload,
            outcome=outcome,
            human_override=human_override,
            agent_recommendation=agent_recommendation,
            previous_hash=prev_hash,
        )
        record.record_hash = record.compute_hash()

        self._records.append(record)
        self._persist(record)
        return record

    def update_outcome(self, event_id: str, outcome: Outcome) -> None:
        """Update the outcome of an existing record (creates a new resolution record)."""
        for r in self._records:
            if r.event_id == event_id:
                self.log(
                    event_type=EventType.NEGOTIATION_CLOSED,
                    regulation_id=r.regulation_id,
                    flight_number=r.flight_number,
                    payload={"resolved_event_id": event_id, "final_outcome": outcome.value},
                    outcome=outcome,
                )
                return

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_chain_integrity(self) -> bool:
        """Verify the hash chain has not been tampered with."""
        prev_hash = ""
        for record in self._records:
            expected_hash = record.compute_hash()
            if record.record_hash != expected_hash:
                print(f"[INTEGRITY FAIL] Record {record.event_id} hash mismatch.")
                return False
            if record.previous_hash != prev_hash:
                print(f"[INTEGRITY FAIL] Record {record.event_id} chain broken.")
                return False
            prev_hash = record.record_hash
        return True

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_flight_trail(self, flight_number: str) -> List[AuditRecord]:
        """Return full negotiation history for a specific flight."""
        return [r for r in self._records if r.flight_number == flight_number]

    def get_regulation_summary(self, regulation_id: str) -> Dict[str, Any]:
        """Aggregate stats for an entire regulation event."""
        records = [r for r in self._records if r.regulation_id == regulation_id]
        flights = set(r.flight_number for r in records)
        human_overrides = [r for r in records if r.human_override]
        decisions = [r for r in records if r.event_type == EventType.DECISION_MADE]
        swaps_submitted = [r for r in records if r.event_type == EventType.SWAP_REQUEST_SUBMITTED]
        swaps_accepted = [
            r for r in records
            if r.event_type == EventType.SWAP_RESPONSE_RECEIVED
            and r.outcome == Outcome.SUCCESS
        ]

        return {
            "regulation_id": regulation_id,
            "total_events": len(records),
            "flights_affected": len(flights),
            "agent_decisions": len(decisions),
            "swap_requests_submitted": len(swaps_submitted),
            "swap_requests_accepted": len(swaps_accepted),
            "human_overrides": len(human_overrides),
            "override_details": [
                {
                    "event_id": r.event_id,
                    "flight": r.flight_number,
                    "agent_recommendation": r.agent_recommendation,
                    "human_action": r.payload.get("human_action"),
                }
                for r in human_overrides
            ],
        }

    def get_all_records(self) -> List[Dict[str, Any]]:
        return [r.model_dump() for r in self._records]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self, record: AuditRecord) -> None:
        """Append record to a JSONL file (one JSON object per line)."""
        log_file = self._log_dir / f"{record.regulation_id}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")
