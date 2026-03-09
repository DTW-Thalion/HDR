from __future__ import annotations

import numpy as np


def pulse(T: int, start: int, width: int, amplitude: float, dims: list[int], state_dim: int) -> np.ndarray:
    out = np.zeros((T, state_dim))
    end = min(start + width, T)
    for t in range(start, end):
        frac = (t - start) / max(width, 1)
        shape = amplitude * np.exp(-3.0 * frac)
        out[t, dims] += shape
    return out


def orthostatic_like(T: int, state_dim: int) -> np.ndarray:
    return pulse(T, start=max(2, T // 6), width=max(4, T // 18), amplitude=0.7, dims=[6, 5], state_dim=state_dim)


def mixed_meal_like(T: int, state_dim: int) -> np.ndarray:
    return pulse(T, start=max(3, T // 4), width=max(6, T // 12), amplitude=0.8, dims=[1, 3], state_dim=state_dim)


def exercise_like(T: int, state_dim: int) -> np.ndarray:
    return pulse(T, start=max(4, T // 2), width=max(8, T // 10), amplitude=-0.6, dims=[1, 7, 6], state_dim=state_dim)


def acute_stress_like(T: int, state_dim: int) -> np.ndarray:
    return pulse(T, start=max(4, (2 * T) // 3), width=max(4, T // 20), amplitude=0.9, dims=[0, 6, 5], state_dim=state_dim)


def compose_challenges(T: int, state_dim: int, include_stress: bool = True) -> tuple[np.ndarray, dict[str, int]]:
    out = np.zeros((T, state_dim))
    markers = {}
    for name, fn in [
        ("orthostatic", orthostatic_like),
        ("meal", mixed_meal_like),
        ("exercise", exercise_like),
    ]:
        arr = fn(T, state_dim)
        out += arr
        idx = int(np.argmax(np.linalg.norm(arr, axis=1)))
        markers[name] = idx
    if include_stress:
        arr = acute_stress_like(T, state_dim)
        out += arr
        markers["stress"] = int(np.argmax(np.linalg.norm(arr, axis=1)))
    return out, markers
