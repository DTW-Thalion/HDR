from __future__ import annotations

from ..model.hsmm import DwellModel, hazard_at


def sample_duration(dwell_model: DwellModel, rng):
    return dwell_model.sample(rng)


def hazard(dwell_model: DwellModel, length: int) -> float:
    return hazard_at(dwell_model, length)
