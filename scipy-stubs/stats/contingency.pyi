from scipy._typing import Untyped

from ._crosstab import crosstab as crosstab
from ._odds_ratio import odds_ratio as odds_ratio
from ._relative_risk import relative_risk as relative_risk
from ._stats_py import power_divergence as power_divergence

def margins(a) -> Untyped: ...
def expected_freq(observed) -> Untyped: ...

Chi2ContingencyResult: Untyped

def chi2_contingency(observed, correction: bool = True, lambda_: Untyped | None = None) -> Untyped: ...
def association(observed, method: str = "cramer", correction: bool = False, lambda_: Untyped | None = None) -> Untyped: ...
