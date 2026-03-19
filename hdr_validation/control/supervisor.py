"""
Extended Supervisor
==============================
Implements the 9-branch supervisor logic from Table 2/3 in v7.0.
"""
from __future__ import annotations

import numpy as np


class ExtendedSupervisor:
    """Implements the 9-branch supervisor logic from Table 2/3 in v7.0:

    1. ICI violated -> Mode C (identification)
    2. Parameter drift exceeds ISS margin -> Mode C (re-identification)
    2b. Eigenvalue of A_hat crosses unit circle -> Mode C (basin-count re-identification)
    3. Unstable basin (z_t in K_u) -> Mode B (escape)
    4. Jump-risk threshold exceeded -> Mode B (prophylactic)
    5. All Mode B criteria met (stable maladaptive) -> Mode B (escape)
    6. Irreversible approaching absorbing boundary -> Mode B (rate reduction)
    7. Default (stable, healthy) -> Mode A (stabilise)
    8. Adverse marker or infeasible -> Rollback to Mode A

    Priority: C > B > A, with rollback override.
    """

    def __init__(
        self,
        config: dict,
        ici_checker=None,
        basin_classifier=None,
        drift_detector=None,
        jump_monitor=None,
    ):
        self.config = config
        self.ici_checker = ici_checker
        self.basin_classifier = basin_classifier
        self.drift_detector = drift_detector
        self.jump_monitor = jump_monitor
        self.jump_risk_threshold = float(config.get("jump_risk_threshold", 0.3))
        self.irr_boundary_threshold = float(config.get("irr_boundary_threshold", 0.9))
        self.xi_max = float(config.get("xi_max", 100.0))

    def select_mode(self, state: dict) -> str:
        """Select operating mode based on state dict.

        Parameters
        ----------
        state : dict with keys:
            - ici_violated : bool (ICI conditions triggered)
            - drift_exceeded : bool (parameter drift > ISS margin)
            - basin_idx : int (current basin index)
            - basin_stability : str ('stable' or 'unstable')
            - jump_risk : float (catastrophe probability)
            - mode_b_eligible : bool (standard Mode B entry conditions met)
            - irr_fraction : float (xi / xi_max, how close to absorbing boundary)
            - adverse_marker : bool (adverse event detected)
            - infeasible : bool (MPC infeasible)

        Returns
        -------
        'A', 'B', or 'C'
        """
        # Branch 8: Rollback override (highest priority safety net)
        if state.get("adverse_marker", False) or state.get("infeasible", False):
            return "A"

        # Branch 1: ICI violated -> Mode C
        if state.get("ici_violated", False):
            return "C"

        # Branch 2: Drift exceeded -> Mode C (re-identification)
        if state.get("drift_exceeded", False):
            return "C"

        # Branch 2b: Eigenvalue crossing unit circle -> Mode C (basin-count re-ID)
        if state.get("eigenvalue_crossing", False):
            return "C"

        # Branch 3: Unstable basin -> Mode B (escape)
        if state.get("basin_stability", "stable") == "unstable":
            return "B"

        # Branch 4: Jump-risk exceeded -> Mode B (prophylactic)
        jump_risk = float(state.get("jump_risk", 0.0))
        if jump_risk > self.jump_risk_threshold:
            return "B"

        # Branch 5: Standard Mode B eligible -> Mode B (escape)
        if state.get("mode_b_eligible", False):
            return "B"

        # Branch 6: Irreversible approaching absorbing boundary -> Mode B
        irr_fraction = float(state.get("irr_fraction", 0.0))
        if irr_fraction > self.irr_boundary_threshold:
            return "B"

        # Branch 7: Default -> Mode A
        return "A"
