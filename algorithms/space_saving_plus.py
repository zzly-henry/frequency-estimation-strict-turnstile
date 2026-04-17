"""
SpaceSaving± Family for Frequency Estimation under Strict Turnstile + α-Bounded Deletion.

Four variants implemented:
1. LazySpaceSaving    – ignore deletions of unmonitored items
2. SpaceSavingPlus    – decrement max-error counter on unmonitored deletions
3. DoubleSpaceSaving  – separate summaries for inserts and deletes
4. IntegratedSpaceSaving – single summary tracking both insert/delete counts

Reference:
    Dimitropoulos, X., Hurley, P., & Kind, A. (2008).
    "Probabilistic Lossy Counting: An efficient algorithm for finding heavy hitters."
    
    Berinde, R., Indyk, P., Cormode, G., & Strauss, M. J. (2010).
    "Space-optimal Heavy Hitters with Strong Error Bounds."

All variants guarantee deterministic error |f̂(x) - f(x)| <= ε * F1
using O(α / ε) space under the α-bounded deletion property.
"""

import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# ======================================================================
# Shared counter entry
# ======================================================================

@dataclass
class SSEntry:
    """A counter entry for SpaceSaving variants."""
    item: object = None
    count: int = 0
    max_error: int = 0  # over-estimation bound (ε_e in literature)


@dataclass
class IntegratedEntry:
    """Counter entry for Integrated SpaceSaving± (tracks inserts & deletes)."""
    item: object = None
    insert_count: int = 0
    delete_count: int = 0
    max_error: int = 0

    @property
    def count(self) -> int:
        return self.insert_count - self.delete_count


# ======================================================================
# 1. Lazy SpaceSaving±
# ======================================================================

class LazySpaceSaving:
    """
    Lazy SpaceSaving±: simply ignores deletions for items not currently monitored.

    On insertion of an unmonitored item when full, replace the minimum-count item
    (classic Space-Saving replacement). On deletion of an unmonitored item, do nothing.
    """

    def __init__(self, k: int, alpha: float = 1.0):
        self.k = max(k, 1)
        self.alpha = alpha
        self.entries: Dict[object, SSEntry] = {}
        self._total_positive = 0
        self._total_negative = 0
        self._num_updates = 0

    # ---- Core API ----

    def update(self, item, delta: int = 1) -> None:
        self._num_updates += 1
        if delta > 0:
            self._insert(item, delta)
        elif delta < 0:
            self._delete(item, -delta)

    def query(self, item) -> int:
        if item in self.entries:
            return self.entries[item].count
        return 0

    def heavy_hitters(self, phi: float) -> List[Tuple]:
        f1 = self.get_F1()
        threshold = phi * f1
        return [(e.item, e.count) for e in self.entries.values() if e.count >= threshold]

    # ---- Metrics ----

    def get_F1(self) -> int:
        return self._total_positive - self._total_negative

    def get_space(self) -> int:
        return len(self.entries)

    def get_max_space(self) -> int:
        return self.k

    # ---- Internal ----

    def _insert(self, item, weight: int) -> None:
        self._total_positive += weight
        if item in self.entries:
            self.entries[item].count += weight
        elif len(self.entries) < self.k:
            self.entries[item] = SSEntry(item=item, count=weight, max_error=0)
        else:
            # Replace minimum
            min_entry = min(self.entries.values(), key=lambda e: e.count)
            min_count = min_entry.count
            del self.entries[min_entry.item]
            self.entries[item] = SSEntry(item=item, count=min_count + weight, max_error=min_count)

    def _delete(self, item, weight: int) -> None:
        self._total_negative += weight
        if item in self.entries:
            self.entries[item].count -= weight
            if self.entries[item].count <= 0:
                del self.entries[item]
        # else: LAZY – ignore unmonitored deletions

    @classmethod
    def from_epsilon_alpha(cls, epsilon: float, alpha: float) -> "LazySpaceSaving":
        k = math.ceil(alpha / epsilon)
        return cls(k=k, alpha=alpha)

    def __repr__(self) -> str:
        return f"LazySpaceSaving(k={self.k}, active={len(self.entries)}, F1={self.get_F1()})"


# ======================================================================
# 2. SpaceSaving± (max-error decrement for unmonitored deletions)
# ======================================================================

class SpaceSavingPlus:
    """
    SpaceSaving±: on deletion of an unmonitored item, decrement the counter
    holding the current maximum estimated error (max ε_e among all entries).
    This spreads deletion impact to the least-certain counter.
    """

    def __init__(self, k: int, alpha: float = 1.0):
        self.k = max(k, 1)
        self.alpha = alpha
        self.entries: Dict[object, SSEntry] = {}
        self._total_positive = 0
        self._total_negative = 0
        self._num_updates = 0

    def update(self, item, delta: int = 1) -> None:
        self._num_updates += 1
        if delta > 0:
            self._insert(item, delta)
        elif delta < 0:
            self._delete(item, -delta)

    def query(self, item) -> int:
        if item in self.entries:
            return self.entries[item].count
        return 0

    def heavy_hitters(self, phi: float) -> List[Tuple]:
        f1 = self.get_F1()
        threshold = phi * f1
        return [(e.item, e.count) for e in self.entries.values() if e.count >= threshold]

    def get_F1(self) -> int:
        return self._total_positive - self._total_negative

    def get_space(self) -> int:
        return len(self.entries)

    def get_max_space(self) -> int:
        return self.k

    def _insert(self, item, weight: int) -> None:
        self._total_positive += weight
        if item in self.entries:
            self.entries[item].count += weight
        elif len(self.entries) < self.k:
            self.entries[item] = SSEntry(item=item, count=weight, max_error=0)
        else:
            min_entry = min(self.entries.values(), key=lambda e: e.count)
            min_count = min_entry.count
            del self.entries[min_entry.item]
            self.entries[item] = SSEntry(item=item, count=min_count + weight, max_error=min_count)

    def _delete(self, item, weight: int) -> None:
        self._total_negative += weight
        if item in self.entries:
            self.entries[item].count -= weight
            if self.entries[item].count <= 0:
                del self.entries[item]
        else:
            # Decrement the entry with the maximum error estimate
            if self.entries:
                max_err_entry = max(self.entries.values(), key=lambda e: e.max_error)
                max_err_entry.count -= weight
                if max_err_entry.count <= 0:
                    del self.entries[max_err_entry.item]

    @classmethod
    def from_epsilon_alpha(cls, epsilon: float, alpha: float) -> "SpaceSavingPlus":
        k = math.ceil(alpha / epsilon)
        return cls(k=k, alpha=alpha)

    def __repr__(self) -> str:
        return f"SpaceSavingPlus(k={self.k}, active={len(self.entries)}, F1={self.get_F1()})"


# ======================================================================
# 3. Double SpaceSaving±
# ======================================================================

class DoubleSpaceSaving:
    """
    Double SpaceSaving±: maintain two separate Space-Saving summaries,
    one for insertions (positive updates) and one for deletions (negative updates).
    
    Query: f̂(x) = f̂_ins(x) - f̂_del(x).
    
    Each summary uses k/2 counters (total k counters).
    Error is bounded by ε * F1 with k = O(α/ε).
    """

    def __init__(self, k: int, alpha: float = 1.0):
        self.k = max(k, 2)
        self.alpha = alpha
        half = max(self.k // 2, 1)
        # Two independent insertion-only Space-Saving summaries
        self.ins_entries: Dict[object, SSEntry] = {}
        self.del_entries: Dict[object, SSEntry] = {}
        self.ins_k = half
        self.del_k = self.k - half
        self._total_positive = 0
        self._total_negative = 0
        self._num_updates = 0

    def update(self, item, delta: int = 1) -> None:
        self._num_updates += 1
        if delta > 0:
            self._total_positive += delta
            self._ss_insert(self.ins_entries, self.ins_k, item, delta)
        elif delta < 0:
            weight = -delta
            self._total_negative += weight
            self._ss_insert(self.del_entries, self.del_k, item, weight)

    def query(self, item) -> int:
        ins = self.ins_entries[item].count if item in self.ins_entries else 0
        dels = self.del_entries[item].count if item in self.del_entries else 0
        return max(ins - dels, 0)

    def heavy_hitters(self, phi: float) -> List[Tuple]:
        f1 = self.get_F1()
        threshold = phi * f1
        candidates = set(self.ins_entries.keys()) | set(self.del_entries.keys())
        results = []
        for item in candidates:
            est = self.query(item)
            if est >= threshold:
                results.append((item, est))
        return results

    def get_F1(self) -> int:
        return self._total_positive - self._total_negative

    def get_space(self) -> int:
        return len(self.ins_entries) + len(self.del_entries)

    def get_max_space(self) -> int:
        return self.k

    @staticmethod
    def _ss_insert(entries: Dict, k: int, item, weight: int) -> None:
        """Standard Space-Saving insert into a summary."""
        if item in entries:
            entries[item].count += weight
        elif len(entries) < k:
            entries[item] = SSEntry(item=item, count=weight, max_error=0)
        else:
            min_entry = min(entries.values(), key=lambda e: e.count)
            min_count = min_entry.count
            del entries[min_entry.item]
            entries[item] = SSEntry(item=item, count=min_count + weight, max_error=min_count)

    @classmethod
    def from_epsilon_alpha(cls, epsilon: float, alpha: float) -> "DoubleSpaceSaving":
        k = math.ceil(alpha / epsilon)
        return cls(k=k, alpha=alpha)

    def __repr__(self) -> str:
        return (f"DoubleSpaceSaving(k={self.k}, ins={len(self.ins_entries)}, "
                f"del={len(self.del_entries)}, F1={self.get_F1()})")


# ======================================================================
# 4. Integrated SpaceSaving±
# ======================================================================

class IntegratedSpaceSaving:
    """
    Integrated SpaceSaving±: single summary where each counter tracks both
    insert_count and delete_count. Net count = insert_count - delete_count.

    On insertion of unmonitored item (when full): replace minimum net-count entry.
    On deletion of unmonitored item: increment the delete_count of the entry
    with maximum error (max_error).

    This is the most space-efficient variant and supports merging.
    """

    def __init__(self, k: int, alpha: float = 1.0):
        self.k = max(k, 1)
        self.alpha = alpha
        self.entries: Dict[object, IntegratedEntry] = {}
        self._total_positive = 0
        self._total_negative = 0
        self._num_updates = 0

    def update(self, item, delta: int = 1) -> None:
        self._num_updates += 1
        if delta > 0:
            self._total_positive += delta
            self._handle_insert(item, delta)
        elif delta < 0:
            weight = -delta
            self._total_negative += weight
            self._handle_delete(item, weight)

    def query(self, item) -> int:
        if item in self.entries:
            return max(self.entries[item].count, 0)
        return 0

    def heavy_hitters(self, phi: float) -> List[Tuple]:
        f1 = self.get_F1()
        threshold = phi * f1
        return [(e.item, e.count) for e in self.entries.values() if e.count >= threshold]

    def get_F1(self) -> int:
        return self._total_positive - self._total_negative

    def get_space(self) -> int:
        return len(self.entries)

    def get_max_space(self) -> int:
        return self.k

    def _handle_insert(self, item, weight: int) -> None:
        if item in self.entries:
            self.entries[item].insert_count += weight
        elif len(self.entries) < self.k:
            self.entries[item] = IntegratedEntry(item=item, insert_count=weight,
                                                  delete_count=0, max_error=0)
        else:
            # Replace the entry with minimum net count
            min_entry = min(self.entries.values(), key=lambda e: e.count)
            min_count = max(min_entry.count, 0)
            del self.entries[min_entry.item]
            self.entries[item] = IntegratedEntry(
                item=item,
                insert_count=min_count + weight,
                delete_count=0,
                max_error=min_count,
            )

    def _handle_delete(self, item, weight: int) -> None:
        if item in self.entries:
            self.entries[item].delete_count += weight
            if self.entries[item].count <= 0:
                del self.entries[item]
        else:
            # Increment delete_count of entry with max error
            if self.entries:
                max_err_entry = max(self.entries.values(), key=lambda e: e.max_error)
                max_err_entry.delete_count += weight
                if max_err_entry.count <= 0:
                    del self.entries[max_err_entry.item]

    def merge(self, other: "IntegratedSpaceSaving") -> "IntegratedSpaceSaving":
        """
        Merge two Integrated SpaceSaving± summaries. Useful for parallel streams.
        Returns a new instance.
        """
        merged = IntegratedSpaceSaving(k=self.k, alpha=self.alpha)
        merged._total_positive = self._total_positive + other._total_positive
        merged._total_negative = self._total_negative + other._total_negative

        all_items = set(self.entries.keys()) | set(other.entries.keys())
        for item in all_items:
            e1 = self.entries.get(item)
            e2 = other.entries.get(item)
            ins = (e1.insert_count if e1 else 0) + (e2.insert_count if e2 else 0)
            dels = (e1.delete_count if e1 else 0) + (e2.delete_count if e2 else 0)
            err = (e1.max_error if e1 else 0) + (e2.max_error if e2 else 0)
            merged.entries[item] = IntegratedEntry(item=item, insert_count=ins,
                                                    delete_count=dels, max_error=err)

        # Trim to k entries – keep top-k by net count
        if len(merged.entries) > merged.k:
            sorted_items = sorted(merged.entries.values(), key=lambda e: e.count, reverse=True)
            merged.entries = {e.item: e for e in sorted_items[:merged.k]}

        return merged

    @classmethod
    def from_epsilon_alpha(cls, epsilon: float, alpha: float) -> "IntegratedSpaceSaving":
        k = math.ceil(alpha / epsilon)
        return cls(k=k, alpha=alpha)

    def __repr__(self) -> str:
        return (f"IntegratedSpaceSaving(k={self.k}, active={len(self.entries)}, "
                f"F1={self.get_F1()})")