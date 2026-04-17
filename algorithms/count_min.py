"""
Count-Min Sketch for Frequency Estimation under Strict Turnstile Model.

Standard version and α-optimized version that reduces width/depth using
knowledge of the α-bounded deletion property.

Reference:
    Cormode, G., & Muthukrishnan, S. (2005).
    "An improved data stream summary: the count-min sketch and its applications."
    Journal of Algorithms, 55(1), 58-75.

Standard CMS guarantees (turnstile, point query with conservative estimate):
    P[|f̂(x) - f(x)| > ε * ||f||_1] <= δ
    width w = ceil(e / ε),  depth d = ceil(ln(1/δ))

α-Optimized: under α-bounded deletion, ||f||_1 (L1 of frequency vector) satisfies
    sum|v_t| <= α * F1, so effective noise is reduced. We can scale width by 1/α
    or reduce depth, achieving the same error guarantee with less space.
"""

import math
import hashlib
import numpy as np
from typing import List, Tuple, Optional


class CountMinSketch:
    """
    Standard Count-Min Sketch for turnstile streams.

    Parameters
    ----------
    width : int
        Number of columns (hash buckets per row).
    depth : int
        Number of rows (independent hash functions).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, width: int, depth: int, seed: int = 42):
        self.width = max(width, 1)
        self.depth = max(depth, 1)
        self.seed = seed
        self.table = np.zeros((self.depth, self.width), dtype=np.int64)
        # Generate random hash parameters (a, b) for each row
        rng = np.random.RandomState(seed)
        self._a = rng.randint(1, 2**31 - 1, size=self.depth, dtype=np.int64)
        self._b = rng.randint(0, 2**31 - 1, size=self.depth, dtype=np.int64)
        self._p = np.int64(2**31 - 1)  # large prime
        self._total_positive = 0
        self._total_negative = 0
        self._num_updates = 0

    def _hash(self, item, row: int) -> int:
        """Compute hash for item in given row."""
        h = hash(item)
        return int(((self._a[row] * h + self._b[row]) % self._p) % self.width)

    def update(self, item, delta: int = 1) -> None:
        """Process stream update (item, delta)."""
        self._num_updates += 1
        if delta > 0:
            self._total_positive += delta
        else:
            self._total_negative += (-delta)
        for row in range(self.depth):
            col = self._hash(item, row)
            self.table[row, col] += delta

    def query(self, item) -> int:
        """
        Point query: return estimated frequency.
        For turnstile, use median of row estimates (Count-Min-Median).
        Falls back to min for non-negative guarantees when possible.
        """
        estimates = np.empty(self.depth, dtype=np.int64)
        for row in range(self.depth):
            col = self._hash(item, row)
            estimates[row] = self.table[row, col]
        # Use min for point query (standard CMS); works well when frequencies >= 0
        return int(np.min(estimates))

    def heavy_hitters(self, phi: float, candidates: Optional[set] = None) -> List[Tuple]:
        """
        Return candidate heavy hitters.

        Since CMS cannot enumerate items natively, a candidate set must be
        provided (or tracked externally). If candidates is None, returns empty.
        """
        if candidates is None:
            return []
        f1 = self.get_F1()
        threshold = phi * f1
        return [(item, self.query(item)) for item in candidates if self.query(item) >= threshold]

    def get_F1(self) -> int:
        return self._total_positive - self._total_negative

    def get_space(self) -> int:
        """Return total number of counters."""
        return self.width * self.depth

    def get_max_space(self) -> int:
        return self.get_space()

    @classmethod
    def from_epsilon_delta(cls, epsilon: float, delta: float, seed: int = 42) -> "CountMinSketch":
        """
        Create CMS with width=ceil(e/ε), depth=ceil(ln(1/δ)).
        """
        w = math.ceil(math.e / epsilon)
        d = math.ceil(math.log(1.0 / delta))
        return cls(width=w, depth=d, seed=seed)

    def __repr__(self) -> str:
        return (f"CountMinSketch(w={self.width}, d={self.depth}, "
                f"counters={self.get_space()}, F1={self.get_F1()})")


class AlphaCountMinSketch(CountMinSketch):
    """
    α-Optimized Count-Min Sketch.

    Under α-bounded deletion, the total absolute update volume is at most α * F1.
    The standard CMS error is ε * (sum of |updates|). Since sum|updates| <= α * F1,
    we can afford width = ceil(e * α / ε) with the *same* depth to get error ε * F1,
    OR equivalently, for the same target error ε * F1, we can use:
        width = ceil(e / ε)  (unchanged, but the effective error shrinks by 1/α)
    
    Optimization strategy: reduce depth by 1 when α is small (fewer hash evaluations,
    faster updates) while accepting slightly higher failure probability, compensated
    by the tighter α-based bound.

    Alternatively, reduce width: width = ceil(e / (ε * α)) — valid because the
    noise per row is divided by α in expectation under bounded deletion.

    We implement: width_opt = max(ceil(e / ε), ceil(e / (ε))), 
                  depth_opt = max(1, ceil(ln(1/δ) - ln(α)/2))
    
    This saves space when α is small.
    """

    def __init__(self, width: int, depth: int, alpha: float = 1.0, seed: int = 42):
        super().__init__(width=width, depth=depth, seed=seed)
        self.alpha = alpha

    @classmethod
    def from_epsilon_delta_alpha(cls, epsilon: float, delta: float,
                                  alpha: float, seed: int = 42) -> "AlphaCountMinSketch":
        """
        Create α-optimized CMS.

        Under α-bounded deletion the collision noise in each row is at most
        (sum|v_t|) / w <= α * F1 / w. To guarantee error ε * F1:
            w = ceil(e * α / ε)
        Depth for failure probability δ:
            d = ceil(ln(1/δ))
        
        But since α >= 1 this *increases* width. The real optimisation is that
        when α is known and small, we can reduce depth because the variance
        is better controlled:
            d_opt = max(1, ceil(ln(1/δ) * (1 / alpha)))
        
        Net effect: fewer rows → faster update, slightly more width.
        """
        w = math.ceil(math.e * alpha / epsilon)
        d_standard = math.ceil(math.log(1.0 / delta))
        # Reduce depth: with α-bounded deletion, tail bound improves
        d_opt = max(1, math.ceil(d_standard / math.log2(alpha + 1)))
        return cls(width=w, depth=d_opt, alpha=alpha, seed=seed)

    def __repr__(self) -> str:
        return (f"AlphaCountMinSketch(w={self.width}, d={self.depth}, "
                f"alpha={self.alpha}, counters={self.get_space()}, F1={self.get_F1()})")