"""
Multi-rate observation and delay augmentation — HDR v7.3
========================================================
Implements Def 5.20 (multi-rate observation) and Def 5.21 (delay augmentation).
"""
from __future__ import annotations

import numpy as np


class MultiRateObserver:
    """Multi-rate observation (Def 5.20): C_t = diag(C^(1),
    1[t mod c_2=0]*C^(2), ...). Handles time-varying missingness.

    Parameters
    ----------
    C_tiers : list of observation matrices, one per tier
    c_factors : list of int, observation cadence for each tier.
                c_factors[0] should be 1 (fastest tier, observed every step).
    """

    def __init__(self, C_tiers: list[np.ndarray], c_factors: list[int]):
        assert len(C_tiers) == len(c_factors)
        self.C_tiers = [np.asarray(C, dtype=float) for C in C_tiers]
        self.c_factors = list(c_factors)
        # Total obs dim = sum of rows in each tier
        self.obs_dim = sum(C.shape[0] for C in self.C_tiers)
        self.state_dim = self.C_tiers[0].shape[1]

    def C_at(self, t: int) -> np.ndarray:
        """Return the effective observation matrix at time t.

        Tiers that are not active at time t have their rows zeroed out.
        """
        blocks = []
        for C_tier, c_factor in zip(self.C_tiers, self.c_factors):
            if t % c_factor == 0:
                blocks.append(C_tier.copy())
            else:
                blocks.append(np.zeros_like(C_tier))
        return np.vstack(blocks)


class DelayAugmentedState:
    """Delay augmentation (Def 5.21): x_tilde includes delay buffer.
    Implements companion-matrix structure for delayed coupling.

    The augmented state is: x_tilde = [x_t, x_{t-1}, ..., x_{t-h+1}]
    of dimension n * h (where h is the delay in steps).

    For the j-th coupling channel (dimension n_j), the delayed effect
    appears as A_delay @ x_{t-h} in the dynamics.

    Parameters
    ----------
    n : state dimension
    n_j : coupling channel dimension (number of delayed state components)
    h : delay in time steps
    """

    def __init__(self, n: int, n_j: int, h: int):
        self.n = n
        self.n_j = n_j
        self.h = max(h, 1)
        self.augmented_dim = n * self.h

    def augment(self, x: np.ndarray, buffer: list[np.ndarray]) -> np.ndarray:
        """Create augmented state from current x and delay buffer.

        Parameters
        ----------
        x : current state (n,)
        buffer : list of past states [x_{t-1}, ..., x_{t-h+1}], each (n,)
                 If shorter than h-1, pads with zeros.

        Returns
        -------
        x_tilde : augmented state (n*h,)
        """
        parts = [np.asarray(x, dtype=float)]
        for i in range(self.h - 1):
            if i < len(buffer):
                parts.append(np.asarray(buffer[i], dtype=float))
            else:
                parts.append(np.zeros(self.n))
        return np.concatenate(parts)

    def augmented_dynamics(self, A_k: np.ndarray, A_delay: np.ndarray) -> np.ndarray:
        """Build augmented dynamics matrix for delay system.

        The augmented dynamics are:
        x_tilde_{t+1} = A_aug @ x_tilde_t

        where A_aug is a companion-like matrix:
        [[A_k,  0, ..., 0, A_delay],
         [ I,   0, ..., 0,   0    ],
         [ 0,   I, ..., 0,   0    ],
         ...
         [ 0,   0, ..., I,   0    ]]

        Parameters
        ----------
        A_k : (n, n) current dynamics
        A_delay : (n, n) delayed coupling matrix

        Returns
        -------
        A_aug : (n*h, n*h) augmented dynamics matrix
        """
        n = self.n
        h = self.h
        dim = n * h
        A_aug = np.zeros((dim, dim))

        # Top-left block: A_k
        A_aug[:n, :n] = A_k

        # Top-right block: A_delay (coupling from x_{t-h+1})
        if h > 1:
            A_aug[:n, (h-1)*n:h*n] = A_delay

        # Shift blocks: identity matrices on sub-diagonal
        for i in range(1, h):
            A_aug[i*n:(i+1)*n, (i-1)*n:i*n] = np.eye(n)

        return A_aug

    def check_delay_lmi(self, A_k: np.ndarray, A_delay: np.ndarray) -> bool:
        """Check if the augmented system is stable (Prop 5.22).

        The delay-augmented system is stable iff rho(A_aug) < 1.
        This is equivalent to the LMI feasibility condition.

        Returns True if the augmented system is stable.
        """
        A_aug = self.augmented_dynamics(A_k, A_delay)
        eigvals = np.linalg.eigvals(A_aug)
        rho = float(np.max(np.abs(eigvals)))
        return rho < 1.0
