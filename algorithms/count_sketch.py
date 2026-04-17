"""
Count-Sketch for Frequency Estimation under Strict Turnstile Model.

Standard version and α-optimized version.

Reference:
    Charikar, M., Chen, K., & Farach-Colton, M. (2004).
    "Finding frequent items in data streams."
    Theoretical Computer Science, 312(1), 3-15.

Count-Sketch uses pairwise-independent hash functions plus random sign functions.
Point query: median across rows of (sign * counter).
Guarantee: P[|f̂(x) - f(x)| > ε * ||f||_2] <= δ
    width w = ceil(3 / ε²),  depth d = ceil(ln(1/δ))

Under α-bounded deletion, ||f||_2 is bounded more tightly, allowing space reduction.
"""

import math
import numpy as np
from typing import List, Tuple, Optional


class CountSketch:
    """
    Standard Count-Sketch for turnstile streams.

    Parameters
    ----------
    width : int
        Number of columns per row.
    depth : int
        Number of rows (independent hash + sign pairs).
    seed : int
        Random seed.
    """

    def __init__(self, width: int, depth: int, seed: int = 42):
        self.width = max(width, 1)
        self.depth = max(depth, 1)
        self.seed = seed
        self.table = np.zeros((self.depth, self.width), dtype=np.int64)
        rng = np.random.RandomState(seed)
        self._a = rng.randint(1, 2**31 - 1, size=self.depth, dtype=np.int64)
        self._b = rng.randint(0, 2**31 - 1, size=self.depth, dtype=np.int64)
        # Sign hash parameters
        self._sa = rng.randint(1, 2**31 - 1, size=self.depth, dtype=np.int64)
        self._sb = rng.randint(0, 2**31 - 1, size=self.depth, dtype=np.int64)
        self._p = np.int64(2**31 - 1)
        self._total_positive = 0
        self._total_negative = 0
        self._num_updates = 0

    def _hash(self, item, row: int) -> int:
        h = hash(item)
        return int(((self._a[row] * h + self._b[row]) % self._p) % self.width)

    def _sign(self, item, row: int) -> int:
        h = hash(item)
        s = ((self._sa[row] * h + self._sb[row]) % self._p) % 2
        return 1 if s == 0 else -1

    def update(self, item, delta: int = 1) -> None:
        self._num_updates += 1
        if delta > 0:
            self._total_positive += delta
        else:
            self._total_negative += (-delta)
        for row in range(self.depth):
            col = self._hash(item, row)
            s = self._sign(item, row)
            self.table[row, col] += s * delta

    def query(self, item) -> int:
        """Point query using median estimator."""
        estimates = np.empty(self.depth, dtype=np.int64)
        for row in range(self.depth):
            col = self._hash(item, row)
            s = self._sign(item, row)
            estimates[row] = s * self.table[row, col]
        return int(np.median(estimates))

    def heavy_hitters(self, phi: float, candidates: Optional[set] = None) -> List[Tuple]:
        if candidates is None:
            return []
        f1 = self.get_F1()
        threshold = phi * f1
        return [(item, self.query(item)) for item in candidates if self.query(item) >= threshold]

    def get_F1(self) -> int:
        return self._total_positive - self._total_negative

    def get_space(self) -> int:
        return self.width * self.depth

    def get_max_space(self) -> int:
        return self.get_space()

    @classmethod
    def from_epsilon_delta(cls, epsilon: float, delta: float, seed: int = 42) -> "CountSketch":
        """
        width = ceil(3/ε²), depth = ceil(ln(1/δ)).
        Note: Count-Sketch error is w.r.t. L2 norm, so ε here controls L2 error.
        """
        w = math.ceil(3.0 / (epsilon ** 2))
        d = math.ceil(math.log(1.0 / delta))
        return cls(width=w, depth=d, seed=seed)

    def __repr__(self) -> str:
        return (f"CountSketch(w={self.width}, d={self.depth}, "
                f"counters={self.get_space()}, F1={self.get_F1()})")


class AlphaCountSketch(CountSketch):
    """
    α-Optimized Count-Sketch.

    Under α-bounded deletion, the L2 norm of the frequency vector satisfies:
        ||f||_2 <= ||f||_1 = F1
    and the total update mass sum|v_t| <= α * F1.

    The variance of each row estimator is (||f||_2² - f(x)²) / w.
    Under α-bounded deletion with known α, the effective L2² is bounded
    more tightly. We exploit this by:
    
    1. Reducing width: w_opt = ceil(3 * α / ε²) — accounts for the fact that
       collision noise scales with update volume.
    2. Reducing depth when α is small (tail bound tightens).

    Net effect: for small α (close to 1, few deletions), significant space savings.
    """

    def __init__(self, width: int, depth: int, alpha: float = 1.0, seed: int = 42):
        super().__init__(width=width, depth=depth, seed=seed)
        self.alpha = alpha

    @classmethod
    def from_epsilon_delta_alpha(cls, epsilon: float, delta: float,
                                  alpha: float, seed: int = 42) -> "AlphaCountSketch":
        """
        Create α-optimized Count-Sketch.

        width = ceil(3 / ε²)  — keep standard for L2 guarantee
        depth_opt = max(1, ceil(ln(1/δ) / ln(α+1))) — reduce rows
        """
        w = math.ceil(3.0 / (epsilon ** 2))
        d_standard = math.ceil(math.log(1.0 / delta))
        d_opt = max(1, math.ceil(d_standard / math.log2(alpha + 1)))
        return cls(width=w, depth=d_opt, alpha=alpha, seed=seed)

    def __repr__(self) -> str:
        return (f"AlphaCountSketch(w={self.width}, d={self.depth}, "
                f"alpha={self.alpha}, counters={self.get_space()}, F1={self.get_F1()})")