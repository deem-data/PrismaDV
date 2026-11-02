"""A fix for gepa 0.0.7 (the GEPA package that dspy.GEPA runs): GEPAState keeps the base program's val subscores and
the Pareto front as one list, and updates the Pareto front in place (gepa/core/state.py), so the base program's
recorded subscores become the best score per val example over all candidates. Its aggregate score is computed
before that and stays right, and with merge off nothing else reads the subscores, but the logged val_subscores[0] is
wrong. apply() gives the Pareto front its own copy. Our SIFTA fork has the same fix in sifta/core/state.py.
"""
from __future__ import annotations

_applied = False


def apply() -> None:
    global _applied
    if _applied:
        return
    from gepa.core.state import GEPAState

    init = GEPAState.__init__

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.pareto_front_valset = list(self.pareto_front_valset)

    GEPAState.__init__ = __init__
    _applied = True
