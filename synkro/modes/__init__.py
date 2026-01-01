"""Mode configurations for different dataset types."""

from synkro.modes.config import ModeConfig, get_mode_config
from synkro.modes.qa import QA_CONFIG
from synkro.modes.sft import SFT_CONFIG

__all__ = [
    "ModeConfig",
    "get_mode_config",
    "QA_CONFIG",
    "SFT_CONFIG",
]

