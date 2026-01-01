"""SFT mode configuration."""

from synkro.modes.config import ModeConfig
from synkro.prompts.templates import (
    SCENARIO_GENERATOR_PROMPT,
    SINGLE_RESPONSE_PROMPT,
    SINGLE_GRADE_PROMPT,
    BATCHED_REFINER_PROMPT,
)

SFT_CONFIG = ModeConfig(
    scenario_prompt=SCENARIO_GENERATOR_PROMPT,
    response_prompt=SINGLE_RESPONSE_PROMPT,
    grade_prompt=SINGLE_GRADE_PROMPT,
    refine_prompt=BATCHED_REFINER_PROMPT,
    output_description="Chat messages: {messages: [system, user, assistant]}",
)

