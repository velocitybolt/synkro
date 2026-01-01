"""Core Pydantic models for Synkro."""

from typing import Literal
from pydantic import BaseModel, Field


Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    """A single message in a conversation."""

    role: Role
    content: str


class Scenario(BaseModel):
    """A test scenario for trace generation."""

    description: str = Field(description="The scenario description")
    context: str = Field(description="Additional context and background")
    category: str | None = Field(default=None, description="Category this scenario belongs to")


class GradeResult(BaseModel):
    """Result of grading a trace."""

    passed: bool = Field(description="Whether the trace passes quality checks")
    issues: list[str] = Field(default_factory=list, description="List of issues found")
    feedback: str = Field(default="", description="Summary feedback for improvement")


class Trace(BaseModel):
    """A complete training trace with messages and metadata."""

    messages: list[Message] = Field(description="The conversation messages")
    scenario: Scenario = Field(description="The scenario this trace was generated from")
    grade: GradeResult | None = Field(default=None, description="Grading result if graded")

    @property
    def system_message(self) -> str | None:
        """Get the system message content."""
        for m in self.messages:
            if m.role == "system":
                return m.content
        return None

    @property
    def user_message(self) -> str:
        """Get the first user message content."""
        for m in self.messages:
            if m.role == "user":
                return m.content
        return ""

    @property
    def assistant_message(self) -> str:
        """Get the last assistant message content."""
        for m in reversed(self.messages):
            if m.role == "assistant":
                return m.content
        return ""


class Category(BaseModel):
    """A category for organizing scenarios."""

    name: str = Field(description="Category name")
    description: str = Field(description="What this category tests")
    count: int = Field(description="Number of traces to generate for this category")


class Plan(BaseModel):
    """A generation plan with categories."""

    categories: list[Category] = Field(description="Categories with trace allocations")
    reasoning: str = Field(description="Explanation of why these categories were chosen")

