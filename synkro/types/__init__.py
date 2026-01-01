"""Type definitions for Synkro.

Usage:
    from synkro.types import DatasetType, Message, Trace
"""

from synkro.types.core import (
    Role,
    Message,
    Scenario,
    Trace,
    GradeResult,
    Plan,
    Category,
)
from synkro.types.dataset_type import DatasetType

__all__ = [
    # Dataset type
    "DatasetType",
    # Core types
    "Role",
    "Message",
    "Scenario",
    "Trace",
    "GradeResult",
    "Plan",
    "Category",
]
