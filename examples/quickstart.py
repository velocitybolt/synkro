"""
Synkro Quickstart - RLAIF Data Generation
==========================================

Generate high-quality SFT (supervised fine-tuning) datasets:
- LLM generates diverse scenarios from your policy 
- Strong model creates expert responses 
- AI grader filters by quality against policy rules
- Output: Ready-to-train JSONL with only passing traces

Dataset types: SFT (chat format), QA (question-answer), DPO (preference pairs)

See other examples for advanced features.
"""

from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from synkro.pipelines import create_pipeline
from synkro.models.google import Google
from synkro.types import DatasetType
from synkro.examples import EXPENSE_POLICY

# Create a pipeline: generates scenarios, responses, and grades them
# - model: Used for scenario and response generation (fast model recommended)
# - grading_model: Used for quality grading (stronger model recommended for better filtering)
# - dataset_type: SFT = chat format, QA = question-answer, DPO = preference pairs
pipeline = create_pipeline(
    model=Google.GEMINI_25_FLASH,       # Fast generation
    grading_model=Google.GEMINI_25_FLASH, # Quality grading
    #grading_model=Google.GEMINI_25_PRO, # Quality grading (stronger = better filtering)
    dataset_type=DatasetType.SFT,       # Chat format for fine-tuning
    max_iterations=3,                   # Max refinement iterations per trace
    #skip_grading=True,                  # Skip grading phase for faster generation
)

# Output: Only traces that passed quality checks
dataset = pipeline.generate(EXPENSE_POLICY, traces=20)

# Filter to only passing traces (removes failed grading)
passing = dataset.filter(passed=True)

# Save to JSONL file (ready for training)
passing.save()

# View summary with quality metrics
print(passing.summary())