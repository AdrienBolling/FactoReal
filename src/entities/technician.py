import dataclasses

@dataclasses.dataclass
class Technician:
    id: int
    name: str
    ai_trust: float
    knowledge: dict
    status: str = "idle"
    available: bool = True
    unavailability_length: int = 0
    fatigue: float = 0.0
