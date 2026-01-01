"""Pydantic schemas for structured LLM outputs and validation."""

from typing import Literal
from pydantic import BaseModel, Field


# =============================================================================
# SCENARIO SCHEMAS
# =============================================================================


class ScenarioOutput(BaseModel):
    """Output schema for scenario generation."""

    scenario: str = Field(description="Detailed scenario description")
    context: str = Field(description="Relevant background information")


class ScenariosArray(BaseModel):
    """Array of generated scenarios."""

    scenarios: list[ScenarioOutput]


# =============================================================================
# POLICY ANALYSIS SCHEMAS
# =============================================================================


class PolicyComplexity(BaseModel):
    """Policy complexity analysis for auto-detecting optimal turns."""

    variable_count: int = Field(
        description="Number of variables/conditions in the policy (rules, exceptions, conditions)"
    )
    complexity_level: Literal["simple", "conditional", "complex"] = Field(
        description="Overall complexity: simple (1 var), conditional (2-3 vars), complex (4+ vars)"
    )
    recommended_turns: int = Field(
        ge=1, le=6, description="Recommended conversation turns based on complexity"
    )
    reasoning: str = Field(description="Brief explanation of the complexity assessment")


class PlanCategory(BaseModel):
    """A category in the generation plan."""

    name: str = Field(description='Short category name (e.g., "Consent Violations", "Edge Cases")')
    description: str = Field(description="What this category tests")
    traces: int = Field(ge=1, description="Number of traces to generate for this category")


class PolicyPlan(BaseModel):
    """LLM-generated plan for dataset generation."""

    categories: list[PlanCategory] = Field(
        min_length=2, max_length=10, description="Scenario categories with trace allocations"
    )
    reasoning: str = Field(
        description="Explanation of why these categories were chosen based on policy content"
    )


# =============================================================================
# CHAT MESSAGE SCHEMAS
# =============================================================================


class ChatMessage(BaseModel):
    """A single chat message in OpenAI format."""

    role: Literal["system", "user", "assistant"] = Field(description="Message role")
    content: str = Field(description="Message content")


class ConversationOutput(BaseModel):
    """Output from response generation - a complete conversation."""

    index: int = Field(description="Scenario index (0-based)")
    messages: list[ChatMessage] = Field(
        description="Full conversation with system, user, and assistant messages"
    )


class BatchedConversations(BaseModel):
    """Batch of generated conversations."""

    conversations: list[ConversationOutput]


# =============================================================================
# GRADING SCHEMAS
# =============================================================================


class GradeOutput(BaseModel):
    """Grading result for a single response."""

    index: int = Field(description="Scenario index (0-based)")
    passed: bool = Field(
        alias="pass", description="Is the response FULLY correct, policy-compliant, and format-valid?"
    )
    policy_violations: list[str] = Field(
        default_factory=list,
        description="Specific policy rules that were violated or misinterpreted",
    )
    missing_citations: list[str] = Field(
        default_factory=list,
        description="Policy sections that should have been cited but were not",
    )
    incomplete_reasoning: list[str] = Field(
        default_factory=list, description="Logical gaps or missing steps in the chain of thought"
    )
    vague_recommendations: list[str] = Field(
        default_factory=list,
        description="Recommendations that need to be more specific or actionable",
    )
    feedback: str = Field(description="Summary of how to fix the issues")

    class Config:
        populate_by_name = True


class BatchedGrades(BaseModel):
    """Batch of grading results."""

    grades: list[GradeOutput]


# =============================================================================
# SINGLE-ITEM SCHEMAS (for parallel generation)
# =============================================================================


class SingleResponse(BaseModel):
    """Single response output for parallel generation."""

    messages: list[ChatMessage] = Field(
        min_length=3, max_length=3, description="Exactly 3 messages: system, user, assistant"
    )


class SingleGrade(BaseModel):
    """Single grade output for parallel generation."""

    passed: bool = Field(
        alias="pass", description="Is the response FULLY correct, policy-compliant, and format-valid?"
    )
    policy_violations: list[str] = Field(
        default_factory=list, description="Specific policy rules that were violated"
    )
    missing_citations: list[str] = Field(
        default_factory=list, description="Policy sections that should have been cited"
    )
    incomplete_reasoning: list[str] = Field(
        default_factory=list, description="Logical gaps or missing reasoning steps"
    )
    vague_recommendations: list[str] = Field(
        default_factory=list, description="Recommendations that need to be more specific"
    )
    feedback: str = Field(description='Summary of issues or "Correct" if passing')

    class Config:
        populate_by_name = True


# =============================================================================
# MULTI-TURN SCHEMAS
# =============================================================================


class FollowUpQuestion(BaseModel):
    """A follow-up question for multi-turn conversations."""

    index: int = Field(description="Scenario index")
    question: str = Field(description="Follow-up question from the user")
    question_type: Literal["clarification", "edge_case", "what_if", "specificity", "challenge"] = (
        Field(description="Type of follow-up")
    )


class TurnGrade(BaseModel):
    """Grade for a single turn in a multi-turn conversation."""

    turn_index: int = Field(description="Which turn (0-based, only assistant turns)")
    passed: bool = Field(alias="pass", description="Does this turn pass all criteria?")
    policy_violations: list[str] = Field(
        default_factory=list, description="Policy violations in this turn"
    )
    missing_citations: list[str] = Field(
        default_factory=list, description="Missing citations in this turn"
    )
    incomplete_reasoning: list[str] = Field(
        default_factory=list, description="Reasoning gaps in this turn"
    )
    vague_recommendations: list[str] = Field(
        default_factory=list, description="Vague recommendations in this turn"
    )
    feedback: str = Field(description="Specific feedback for this turn")

    class Config:
        populate_by_name = True


class ConversationGrade(BaseModel):
    """Full grading for a multi-turn conversation."""

    index: int = Field(description="Scenario index")
    overall_pass: bool = Field(description="Does the ENTIRE conversation pass?")
    turn_grades: list[TurnGrade] = Field(description="Grade for each assistant turn")
    coherence_pass: bool = Field(
        description="Is the conversation coherent with no contradictions?"
    )
    coherence_issues: list[str] = Field(
        default_factory=list, description="Any contradictions or incoherence across turns"
    )
    progressive_depth: bool = Field(
        description="Does each turn build on previous context appropriately?"
    )
    overall_feedback: str = Field(
        description="Summary of what needs to be fixed across the conversation"
    )


# =============================================================================
# AGENTIC SCHEMAS
# =============================================================================


class ToolCall(BaseModel):
    """A tool call in an agentic trace."""

    tool_name: str = Field(description="Name of the tool to call")
    arguments: dict[str, str] = Field(description="Arguments to pass to the tool")


class AgenticStep(BaseModel):
    """A single step in an agentic trace."""

    reasoning: str = Field(description="Reasoning before tool call")
    tool_name: str = Field(description="Tool to call")
    tool_args: dict = Field(description="Tool arguments")


class AgenticTrace(BaseModel):
    """Complete agentic trace with tool usage."""

    index: int = Field(description="Scenario index")
    steps: list[AgenticStep] = Field(description="Steps of tool usage")
    final_answer: str = Field(description="Final comprehensive answer")

