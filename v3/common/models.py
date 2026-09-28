"""
Shared shapes for a demand item as it moves API -> Redis Stream -> worker,
plus the (de)serialization helpers. Redis Streams only store flat
string->string field maps, so we serialize resource demand as plain
top-level fields rather than nested JSON to keep XADD/XREADGROUP simple
and keep every field visible with `XRANGE`/`redis-cli` during debugging.
"""
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DemandItem:
    demand_id: str
    buyer_id: str
    app_type: str
    lease_duration: float
    cpu: float
    mem: float
    gpu: float
    arrival_seq: int
    storage: float = 0.0
    arrival_ts: float = field(default_factory=time.time)
    ip: Optional[str] = None

    def demand(self) -> dict:
        return {"cpu": self.cpu, "mem": self.mem, "gpu": self.gpu, "storage": self.storage}

    def to_stream_fields(self) -> dict:
        return {
            "demand_id": self.demand_id,
            "buyer_id": self.buyer_id,
            "app_type": self.app_type or "",
            "lease_duration": str(self.lease_duration),
            "cpu": str(self.cpu),
            "mem": str(self.mem),
            "gpu": str(self.gpu),
            "storage": str(self.storage),
            "arrival_seq": str(self.arrival_seq),
            "arrival_ts": str(self.arrival_ts),
            "ip": self.ip or "",
        }

    @staticmethod
    def from_stream_fields(fields: dict) -> "DemandItem":
        return DemandItem(
            demand_id=fields["demand_id"],
            buyer_id=fields["buyer_id"],
            app_type=fields.get("app_type", ""),
            lease_duration=float(fields.get("lease_duration", 0) or 0),
            cpu=float(fields.get("cpu", 0) or 0),
            mem=float(fields.get("mem", 0) or 0),
            gpu=float(fields.get("gpu", 0) or 0),
            storage=float(fields.get("storage", 0) or 0),
            arrival_seq=int(fields.get("arrival_seq", 0) or 0),
            arrival_ts=float(fields.get("arrival_ts", 0) or 0),
            ip=fields.get("ip") or None,
        )
