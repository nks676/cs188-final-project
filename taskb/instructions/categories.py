"""Category enum and metadata for Task B evaluation."""
from enum import Enum


class Category(str, Enum):
    A = "A"  # Direct Placement
    B = "B"  # Stacking
    C = "C"  # Spatial Arrangements
    D = "D"  # Conditional / Relational
    E = "E"  # Composite Multi-Step
    F = "F"  # Rejection (unsupported operations)


CATEGORY_LABELS = {
    Category.A: "Direct Placement",
    Category.B: "Stacking",
    Category.C: "Spatial Arrangements",
    Category.D: "Conditional/Relational",
    Category.E: "Composite Multi-Step",
    Category.F: "Rejection",
}
