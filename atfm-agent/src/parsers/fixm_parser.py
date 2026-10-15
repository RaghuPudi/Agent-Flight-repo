"""
FIXM 4.2 / EUROCONTROL CFMU SLOT MESSAGE PARSER
Decodes incoming ATFM regulation and CTOT assignment messages.
In production this connects to EUROCONTROL B2B Web Services (SWIM).
For the PoC we parse realistic XML samples.
"""

from __future__ import annotations

import xmltodict
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class FlightIdentifier(BaseModel):
    flight_number: str          # e.g. "BA456"
    aircraft_reg: str           # e.g. "G-EUPT"
    aircraft_type: str          # e.g. "A320"
    dep_airport: str            # ICAO e.g. "EGLL"
    arr_airport: str            # ICAO e.g. "LFPG"


class SlotMessage(BaseModel):
    message_id: str
    message_type: str           # "CTOT_ASSIGNMENT" | "SLOT_SWAP_RESPONSE" | "REGULATION_NOTICE"
    regulation_id: str          # e.g. "LFEREG1034"
    regulation_reason: str      # e.g. "WEATHER" | "ATC_CAPACITY" | "EQUIPMENT"
    regulation_start: datetime
    regulation_end: datetime
    flight: FlightIdentifier
    scheduled_off_block: datetime   # SOBT
    calculated_takeoff_time: datetime   # CTOT assigned by CFMU
    most_penalising_regulation: str
    flight_state: str           # "REGULATED" | "EXEMPTED" | "SUSPENDED"
    swap_available: bool = False
    raw_xml: Optional[str] = None


class RegulationNotice(BaseModel):
    regulation_id: str
    affected_airport: str
    affected_fir: str
    reason: str
    start_time: datetime
    end_time: datetime
    capacity_reduction_pct: int    # 0-100
    affected_flights_count: int


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class FIXMParser:
    """
    Parses EUROCONTROL CFMU B2B XML messages (FIXM 4.2 schema subset).
    Real production parser would validate against full FIXM XSD schema.
    """

    def parse_ctot_message(self, xml_str: str) -> SlotMessage:
        """Parse a CTOT_ASSIGNMENT XML message from CFMU."""
        doc = xmltodict.parse(xml_str)
        root = doc.get("cfmu:SlotMessage", doc.get("SlotMessage", {}))

        flight_node = root["flight"]
        ident_node = flight_node["flightIdentification"]
        reg_node = root["regulation"]

        flight = FlightIdentifier(
            flight_number=ident_node["aircraftIdentification"],
            aircraft_reg=ident_node.get("aircraftRegistration", "UNKNOWN"),
            aircraft_type=ident_node.get("aircraftType", "UNKNOWN"),
            dep_airport=flight_node["departure"]["aerodrome"]["designator"],
            arr_airport=flight_node["arrival"]["aerodrome"]["designator"],
        )

        return SlotMessage(
            message_id=root["messageId"],
            message_type=root["messageType"],
            regulation_id=reg_node["regulationId"],
            regulation_reason=reg_node.get("regulationReason", "UNKNOWN"),
            regulation_start=_parse_dt(reg_node["regulationStart"]),
            regulation_end=_parse_dt(reg_node["regulationEnd"]),
            flight=flight,
            scheduled_off_block=_parse_dt(flight_node["departure"]["scheduledOffBlockTime"]),
            calculated_takeoff_time=_parse_dt(flight_node["departure"]["calculatedTakeOffTime"]),
            most_penalising_regulation=reg_node.get("mostPenalisingRegulation", reg_node["regulationId"]),
            flight_state=root.get("flightState", "REGULATED"),
            swap_available=root.get("slotSwapAvailable", "false").lower() == "true",
            raw_xml=xml_str,
        )

    def parse_regulation_notice(self, xml_str: str) -> RegulationNotice:
        """Parse a REGULATION_NOTICE XML broadcast from CFMU."""
        doc = xmltodict.parse(xml_str)
        root = doc.get("cfmu:RegulationNotice", doc.get("RegulationNotice", {}))

        return RegulationNotice(
            regulation_id=root["regulationId"],
            affected_airport=root.get("affectedAerodrome", {}).get("designator", "UNKNOWN"),
            affected_fir=root.get("affectedFIR", "UNKNOWN"),
            reason=root["regulationReason"],
            start_time=_parse_dt(root["regulationStart"]),
            end_time=_parse_dt(root["regulationEnd"]),
            capacity_reduction_pct=int(root.get("capacityReductionPercent", 0)),
            affected_flights_count=int(root.get("affectedFlightsCount", 0)),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_dt(value: str) -> datetime:
    """Parse ISO-8601 datetime strings from FIXM messages."""
    from dateutil import parser as dp
    dt = dp.parse(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Sample XML generators (for testing without live CFMU connection)
# ---------------------------------------------------------------------------

def make_sample_ctot_xml(
    flight_number: str = "BA456",
    dep: str = "EGLL",
    arr: str = "LFPG",
    sobt: str = "2026-10-15T14:00:00Z",
    ctot: str = "2026-10-15T16:40:00Z",
    regulation_id: str = "LFEREG1034",
    swap_available: bool = True,
) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<SlotMessage>
  <messageId>MSG-{regulation_id}-{flight_number}-001</messageId>
  <messageType>CTOT_ASSIGNMENT</messageType>
  <flightState>REGULATED</flightState>
  <slotSwapAvailable>{"true" if swap_available else "false"}</slotSwapAvailable>
  <regulation>
    <regulationId>{regulation_id}</regulationId>
    <regulationReason>WEATHER</regulationReason>
    <regulationStart>2026-10-15T09:00:00Z</regulationStart>
    <regulationEnd>2026-10-15T18:00:00Z</regulationEnd>
    <mostPenalisingRegulation>{regulation_id}</mostPenalisingRegulation>
  </regulation>
  <flight>
    <flightIdentification>
      <aircraftIdentification>{flight_number}</aircraftIdentification>
      <aircraftRegistration>G-EUPT</aircraftRegistration>
      <aircraftType>A320</aircraftType>
    </flightIdentification>
    <departure>
      <aerodrome><designator>{dep}</designator></aerodrome>
      <scheduledOffBlockTime>{sobt}</scheduledOffBlockTime>
      <calculatedTakeOffTime>{ctot}</calculatedTakeOffTime>
    </departure>
    <arrival>
      <aerodrome><designator>{arr}</designator></aerodrome>
    </arrival>
  </flight>
</SlotMessage>"""


def make_sample_regulation_xml(
    regulation_id: str = "LFEREG1034",
    airport: str = "LFPG",
    fir: str = "LFFF",
    reason: str = "WEATHER",
    start: str = "2026-10-15T09:00:00Z",
    end: str = "2026-10-15T18:00:00Z",
    capacity_pct: int = 40,
    affected_count: int = 47,
) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<RegulationNotice>
  <regulationId>{regulation_id}</regulationId>
  <affectedAerodrome><designator>{airport}</designator></affectedAerodrome>
  <affectedFIR>{fir}</affectedFIR>
  <regulationReason>{reason}</regulationReason>
  <regulationStart>{start}</regulationStart>
  <regulationEnd>{end}</regulationEnd>
  <capacityReductionPercent>{capacity_pct}</capacityReductionPercent>
  <affectedFlightsCount>{affected_count}</affectedFlightsCount>
</RegulationNotice>"""
