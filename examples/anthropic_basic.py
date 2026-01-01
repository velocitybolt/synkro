"""
Anthropic Basic Example - Dataset Generation
=============================================

Generate SFT datasets using Anthropic Claude models:
- Claude 4.5 Haiku for fast generation
- Claude 4.5 Sonnet for quality grading

Requires: ANTHROPIC_API_KEY environment variable
"""

from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from synkro.pipelines import create_pipeline
from synkro.models.anthropic import Anthropic
from synkro.types import DatasetType
from synkro.examples import EXPENSE_POLICY

# Create pipeline with Anthropic models
# - model: Used for scenario and response generation
# - grading_model: Used for quality grading (stronger = better filtering)
pipeline = create_pipeline(
    model=Anthropic.CLAUDE_45_HAIKU,      # Fast, cost-effective generation
    grading_model=Anthropic.CLAUDE_45_SONNET,  # High-quality grading
    dataset_type=DatasetType.SFT,         # Chat format for fine-tuning
    max_iterations=3,                     # Max refinement attempts per trace
)

# Generate dataset from policy
dataset = pipeline.generate(EXPENSE_POLICY, traces=20)

# Filter to only passing traces
passing = dataset.filter(passed=True)

# Save to JSONL file
passing.save("anthropic_sft.jsonl")

# View summary
print(passing.summary())

