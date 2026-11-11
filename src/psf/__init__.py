"""Reference implementation of the leakage-safe, slot-aware predictive scheduling framework.

Only synthetic data is generated and used here; no operator records are included.
"""

SLOTS = ("Morning", "Noon", "Afternoon", "Evening")
OPERATIONAL_SLOTS = (0, 1, 2)  # Evening (index 3) is the unbounded overtime slot
