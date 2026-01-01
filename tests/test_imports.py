"""Test that all public imports work correctly."""

import pytest


def test_main_imports():
    """Test importing from synkro package."""
    from synkro import (
        # Quick function
        generate,
        # Core classes
        Policy,
        Dataset,
        Trace,
        Scenario,
        Message,
        GradeResult,
        Plan,
        Category,
        # Generation
        Generator,
        ScenarioGenerator,
        ResponseGenerator,
        Planner,
        # Quality
        Grader,
        Refiner,
        # LLM
        LLM,
        # Prompts
        SystemPrompt,
        ScenarioPrompt,
        ResponsePrompt,
        GradePrompt,
        # Formatters
        SFTFormatter,
        QAFormatter,
        # Model enums (OpenAI, Anthropic, Google supported)
        OpenAI,
        Anthropic,
        Google,
    )

    # Verify they're all importable
    assert generate is not None
    assert Policy is not None
    assert Generator is not None
    assert OpenAI.GPT_4O_MINI.value == "gpt-4o-mini"


def test_model_enums():
    """Test model enum values."""
    from synkro import OpenAI, Anthropic, Google

    assert OpenAI.GPT_5_MINI.value == "gpt-5-mini"
    assert OpenAI.GPT_52.value == "gpt-5.2"
    assert Anthropic.CLAUDE_45_SONNET.value == "claude-sonnet-4-5-20250601"
    assert Google.GEMINI_25_FLASH.value == "gemini/gemini-2.5-flash"


def test_policy_from_text():
    """Test creating Policy from text."""
    from synkro import Policy

    text = "All expenses over $50 require approval."
    policy = Policy(text=text)

    assert policy.text == text
    assert policy.word_count == 6


def test_trace_properties():
    """Test Trace message extraction."""
    from synkro import Trace, Scenario, Message

    trace = Trace(
        messages=[
            Message(role="system", content="You are helpful."),
            Message(role="user", content="What is the policy?"),
            Message(role="assistant", content="The policy states..."),
        ],
        scenario=Scenario(description="Test", context="Context"),
    )

    assert trace.system_message == "You are helpful."
    assert trace.user_message == "What is the policy?"
    assert trace.assistant_message == "The policy states..."


def test_dataset_filter():
    """Test Dataset filtering."""
    from synkro import Dataset, Trace, Scenario, Message, GradeResult

    traces = [
        Trace(
            messages=[
                Message(role="user", content="Q1"),
                Message(role="assistant", content="A1"),
            ],
            scenario=Scenario(description="S1", context="C1", category="A"),
            grade=GradeResult(passed=True, issues=[], feedback=""),
        ),
        Trace(
            messages=[
                Message(role="user", content="Q2"),
                Message(role="assistant", content="A2"),
            ],
            scenario=Scenario(description="S2", context="C2", category="B"),
            grade=GradeResult(passed=False, issues=["Issue"], feedback="Fix it"),
        ),
    ]

    dataset = Dataset(traces=traces)
    assert len(dataset) == 2

    passing = dataset.filter(passed=True)
    assert len(passing) == 1

    failing = dataset.filter(passed=False)
    assert len(failing) == 1

    cat_a = dataset.filter(category="A")
    assert len(cat_a) == 1


def test_formatter_output():
    """Test SFT formatter output."""
    from synkro import SFTFormatter, Trace, Scenario, Message

    trace = Trace(
        messages=[
            Message(role="system", content="System"),
            Message(role="user", content="User"),
            Message(role="assistant", content="Assistant"),
        ],
        scenario=Scenario(description="Test", context="Context"),
    )

    formatter = SFTFormatter()
    output = formatter.format([trace])

    assert len(output) == 1
    assert "messages" in output[0]
    assert len(output[0]["messages"]) == 3

