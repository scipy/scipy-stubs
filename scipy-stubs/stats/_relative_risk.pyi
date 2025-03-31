from dataclasses import dataclass

from ._common import ConfidenceInterval

@dataclass
class RelativeRiskResult:
    relative_risk: float | int | bool
    exposed_cases: int | bool
    exposed_total: int | bool
    control_cases: int | bool
    control_total: int | bool
    def confidence_interval(self, /, confidence_level: float | int | bool = 0.95) -> ConfidenceInterval: ...

def relative_risk(
    exposed_cases: int | bool, exposed_total: int | bool, control_cases: int | bool, control_total: int | bool
) -> RelativeRiskResult: ...
