"""
Misra-Gries Algorithm for Frequency Estimation.

Extended to support the Strict Turnstile Model with α-Bounded Deletion property.

Reference:
    Misra, J., & Gries, D. (1982). Finding repeated elements.
    Science of Computer Programming, 2(2), 143-152.

In the original insertion-only model, Misra-Gries maintains at most k counters.
When a new item arrives and all k slots are full, every counter is decremented by 1
and zero-counters are evicted.

Extension for Strict Turnstile + α-Bounded Deletion:
    - Positive updates (insertions): standard Misra-Gries logic.
    - Negative updates (deletions): if the item is monitored, decrement its counter
      by |delta|. If the item is NOT monitored, we apply a "spread decrement" approach
      inspired by SpaceSaving±: we decrement the minimum counter, since the unmonitored
      item's true frequency is at most the current minimum error.
    - The α-property guarantees total deletions are bounded, so the additional error
      introduced is bounded by ε * F1 when using O(α/ε) counters.

Guarantee:
    |f̂(x) - f(x)| <= F1 / k  where F1 = sum of final frequencies, k = number of counters.
    Setting k = ceil(α / ε) gives error <= ε * F1.
"""

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple


class MisraGries:
    """
    Misra-Gries frequency estimation for Strict Turnstile + α-Bounded Deletion.

    Parameters
    ----------
    k : int
        Number of counters (space budget). For error ε with α-bounded deletion,
        set k = ceil(α / ε).
    alpha : float, optional
        The α parameter of the bounded deletion property. Used only for
        informational / space-computation purposes. Default 1.0 (insertion-only).
    """

    def __init__(self, k: int, alpha: float = 1.0):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.alpha = alpha
        # item -> estimated count
        self.counters: Dict = {}
        # Track total positive weight and total negative weight for F1 computation
        self._total_positive = 0
        self._total_negative = 0
        # Number of times we performed a decrement sweep (for error bound)
        self._decrement_total = 0
        # Track update count for timing
        self._num_updates = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def update(self, item, delta: int = 1) -> None:
        """
        Process a stream update (item, delta).

        Parameters
        ----------
        item : hashable
            Element from the universe.
        delta : int
            Update weight. Positive for insertion, negative for deletion.
        """
        self._num_updates += 1

        if delta > 0:
            self._update_insert(item, delta)
        elif delta < 0:
            self._update_delete(item, -delta)
        # delta == 0 is a no-op

    def query(self, item) -> int:
        """
        Return the estimated frequency of *item*.

        The estimate is always a lower bound in the insertion-only case;
        under turnstile it may be off by at most F1/k.
        """
        return self.counters.get(item, 0)

    def heavy_hitters(self, phi: float) -> List[Tuple]:
        """
        Return items whose estimated frequency exceeds phi * F1.

        Parameters
        ----------
        phi : float
            Threshold fraction in (0, 1).

        Returns
        -------
        list of (item, estimated_frequency) tuples.
        """
        f1 = self.get_F1()
        threshold = phi * f1
        return [(item, cnt) for item, cnt in self.counters.items() if cnt >= threshold]

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------

    def get_F1(self) -> int:
        """Return F1 = total positive weight - total negative weight (net weight)."""
        return self._total_positive - self._total_negative

    def get_space(self) -> int:
        """Return current number of active counters."""
        return len(self.counters)

    def get_max_space(self) -> int:
        """Return the configured maximum number of counters k."""
        return self.k

    def get_error_bound(self) -> float:
        """Return the theoretical error bound F1 / k."""
        f1 = self.get_F1()
        if f1 == 0:
            return 0.0
        return f1 / self.k

    # ------------------------------------------------------------------
    # Internal logic
    # ------------------------------------------------------------------

    def _update_insert(self, item, weight: int) -> None:
        """Handle a positive update (insertion)."""
        self._total_positive += weight

        if item in self.counters:
            self.counters[item] += weight
        elif len(self.counters) < self.k:
            # Free slot available
            self.counters[item] = weight
        else:
            # No free slot – add then perform decrement sweep
            # We decrement every counter by *weight* (generalised for weighted streams)
            # but the classic approach is per-unit; we iterate weight units.
            # For efficiency with large weights we do it in bulk.
            self.counters[item] = weight
            self._decrement_all(weight)

    def _update_delete(self, item, weight: int) -> None:
        """
        Handle a negative update (deletion of *weight* units).

        If the item is monitored, subtract directly.
        If not monitored, decrement the counter with the smallest value
        (this is the "max-error decrement" heuristic for unmonitored deletes).
        """
        self._total_negative += weight

        if item in self.counters:
            self.counters[item] -= weight
            if self.counters[item] <= 0:
                del self.counters[item]
        else:
            # Unmonitored deletion: decrement the minimum-counter item
            if self.counters:
                min_item = min(self.counters, key=self.counters.get)
                self.counters[min_item] -= weight
                if self.counters[min_item] <= 0:
                    del self.counters[min_item]

    def _decrement_all(self, amount: int) -> None:
        """
        Decrement every counter by *amount* and evict zeros/negatives.
        """
        self._decrement_total += amount
        to_delete = []
        for item in self.counters:
            self.counters[item] -= amount
            if self.counters[item] <= 0:
                to_delete.append(item)
        for item in to_delete:
            del self.counters[item]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MisraGries(k={self.k}, alpha={self.alpha}, "
            f"active_counters={len(self.counters)}, F1={self.get_F1()})"
        )

    @classmethod
    def from_epsilon_alpha(cls, epsilon: float, alpha: float) -> "MisraGries":
        """
        Convenience constructor: create a MisraGries instance sized so that
        the deterministic error bound is at most ε * F1 under α-bounded deletion.

        Space = ceil(α / ε).
        """
        k = math.ceil(alpha / epsilon)
        return cls(k=k, alpha=alpha)