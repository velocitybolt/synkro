"""QA mode configuration."""

from synkro.modes.config import ModeConfig
from synkro.prompts.qa_templates import (
    QA_SCENARIO_PROMPT,
    QA_RESPONSE_PROMPT,
    QA_GRADE_PROMPT,
    QA_REFINE_PROMPT,
)

QA_CONFIG = ModeConfig(
    scenario_prompt=QA_SCENARIO_PROMPT,
    response_prompt=QA_RESPONSE_PROMPT,
    grade_prompt=QA_GRADE_PROMPT,
    refine_prompt=QA_REFINE_PROMPT,
    output_description="Question-Answer pairs: {question, answer, context}",
)

