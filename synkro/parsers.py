"""Response parsing functions for LLM outputs with robust JSON extraction."""

import json
import re
from typing import Any

from synkro.schemas import (
    ScenarioOutput,
    GradeOutput,
    SingleGrade,
    SingleResponse,
    PolicyComplexity,
    PolicyPlan,
    ChatMessage,
)
from synkro.prompts.templates import SYSTEM_PROMPT


def strip_markdown_fences(content: str) -> str:
    """Strip markdown code fences from content."""
    # Remove ```json ... ``` blocks, keeping just the content
    content = re.sub(r'```json\s*', '', content)
    content = re.sub(r'```\s*', '', content)
    return content.strip()


def extract_json(content: str, start_char: str = "[") -> str | None:
    """
    Extract JSON from a string that may contain other text.

    Args:
        content: Raw content that may contain JSON
        start_char: Starting character to look for ('[' for arrays, '{' for objects)

    Returns:
        Extracted JSON string or None if not found
    """
    end_char = "]" if start_char == "[" else "}"
    start = content.find(start_char)
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(content)):
        char = content[i]

        if escape:
            escape = False
            continue

        if char == "\\" and in_string:
            escape = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == start_char:
            depth += 1
        if char == end_char:
            depth -= 1

        if depth == 0:
            return content[start : i + 1]

    return None


def extract_content(response: Any) -> str:
    """
    Extract text content from various LLM response formats.

    Args:
        response: Raw response from an LLM

    Returns:
        Extracted text content
    """
    try:
        if isinstance(response, str):
            return response

        # Gemini format
        if isinstance(response, dict):
            if "candidates" in response:
                return response["candidates"][0]["content"]["parts"][0]["text"]

            # OpenAI format
            if "choices" in response:
                return response["choices"][0]["message"]["content"]

            # Simple content field
            if "content" in response:
                return response["content"]

            if "text" in response:
                return response["text"]

            if "output" in response:
                return response["output"]

        return json.dumps(response)
    except Exception:
        return str(response)


def parse_scenarios(response: Any, expected_count: int) -> list[ScenarioOutput]:
    """
    Parse scenario output from LLM response.

    Args:
        response: Raw LLM response
        expected_count: Number of scenarios expected

    Returns:
        List of parsed scenarios
    """
    try:
        content = extract_content(response)
        json_str = extract_json(content, "[")

        if json_str:
            parsed = json.loads(json_str)

            if isinstance(parsed, list):
                scenarios = []
                for s in parsed[:expected_count]:
                    scenarios.append(
                        ScenarioOutput(
                            scenario=s.get("scenario", s.get("description", "")),
                            context=s.get("context", s.get("background", "")),
                        )
                    )
                return scenarios
    except Exception:
        pass  # Fallback handles this

    # Fallback: generate placeholder scenarios
    return [
        ScenarioOutput(
            scenario=f"Policy compliance scenario {i + 1}",
            context="General policy application context",
        )
        for i in range(expected_count)
    ]


def parse_batched_responses(
    response: Any, expected_count: int, scenarios: list[ScenarioOutput]
) -> list[dict]:
    """
    Parse batched response output from LLM.

    Args:
        response: Raw LLM response
        expected_count: Number of responses expected
        scenarios: Original scenarios for fallback

    Returns:
        List of response dicts with 'index' and 'messages'
    """
    try:
        content = extract_content(response)
        json_str = extract_json(content, "[")

        if json_str:
            parsed = json.loads(json_str)

            if isinstance(parsed, list):
                results = []
                for r in parsed:
                    index = r.get("index", 0)

                    if isinstance(r.get("messages"), list) and len(r["messages"]) > 0:
                        results.append(
                            {
                                "index": index,
                                "messages": [
                                    ChatMessage(role=m["role"], content=m.get("content", ""))
                                    for m in r["messages"]
                                ],
                            }
                        )
                    else:
                        # Fallback: construct messages from old format
                        scenario = scenarios[index] if index < len(scenarios) else scenarios[0]
                        results.append(
                            {
                                "index": index,
                                "messages": [
                                    ChatMessage(role="system", content=SYSTEM_PROMPT),
                                    ChatMessage(
                                        role="user",
                                        content=f"Scenario: {scenario.scenario}\n\nContext: {scenario.context}",
                                    ),
                                    ChatMessage(
                                        role="assistant", content=r.get("response", "")
                                    ),
                                ],
                            }
                        )
                return results
    except Exception:
        pass  # Fallback handles this

    # Fallback
    return [
        {
            "index": i,
            "messages": [
                ChatMessage(role="system", content=SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=f"Scenario: {scenarios[i].scenario}\n\nContext: {scenarios[i].context}",
                ),
                ChatMessage(role="assistant", content="Unable to generate response"),
            ],
        }
        for i in range(min(expected_count, len(scenarios)))
    ]


def parse_batched_grades(response: Any) -> list[GradeOutput]:
    """
    Parse grading output from LLM response.

    Args:
        response: Raw LLM response

    Returns:
        List of parsed grades
    """
    try:
        content = extract_content(response)
        json_str = extract_json(content, "[")

        if json_str:
            parsed = json.loads(json_str)

            if isinstance(parsed, list):
                grades = []
                for g in parsed:
                    grades.append(
                        GradeOutput(
                            index=g.get("index", 0),
                            passed=g.get("pass", False),
                            policy_violations=g.get("policy_violations", []),
                            missing_citations=g.get("missing_citations", []),
                            incomplete_reasoning=g.get("incomplete_reasoning", []),
                            vague_recommendations=g.get("vague_recommendations", []),
                            feedback=g.get("feedback", ""),
                        )
                    )
                return grades
    except Exception:
        pass  # Return empty list below

    return []


def parse_single_response(response: Any) -> SingleResponse | None:
    """
    Parse a single response from parallel generation.

    Args:
        response: Raw LLM response for a single scenario

    Returns:
        Parsed SingleResponse or None if parsing failed
    """
    try:
        content = extract_content(response)
        # Strip markdown fences first
        content = strip_markdown_fences(content)
        
        # Try to find and parse valid JSON objects with messages
        remaining = content
        while remaining:
            json_str = extract_json(remaining, "{")
            if not json_str:
                break
                
            try:
                parsed = json.loads(json_str)
                
                # Validate it has the expected structure
                if isinstance(parsed.get("messages"), list) and len(parsed["messages"]) >= 1:
                    messages = []
                    valid = True
                    
                    for m in parsed["messages"]:
                        if not isinstance(m, dict) or "role" not in m or "content" not in m:
                            valid = False
                            break
                        
                        msg_content = m.get("content", "")
                        # Reject if content contains refinement prompt leak markers
                        if "GRADER FEEDBACK" in msg_content or "Generate an IMPROVED response" in msg_content:
                            valid = False
                            break
                            
                        messages.append(ChatMessage(role=m["role"], content=msg_content))
                    
                    if valid and len(messages) >= 1:
                        return SingleResponse(messages=messages)
                        
            except json.JSONDecodeError:
                pass
            
            # Move past this JSON object and try to find another
            end_pos = remaining.find(json_str) + len(json_str)
            remaining = remaining[end_pos:]
            
    except Exception:
        pass  # Caller handles None with fallback

    return None


def parse_single_grade(response: Any) -> SingleGrade | None:
    """
    Parse a single grade from parallel grading.

    Args:
        response: Raw LLM response for a single grade

    Returns:
        Parsed SingleGrade or None if parsing failed
    """
    try:
        content = extract_content(response)
        json_str = extract_json(content, "{")

        if json_str:
            parsed = json.loads(json_str)
            return SingleGrade(
                passed=parsed.get("pass", False),
                policy_violations=parsed.get("policy_violations", []),
                missing_citations=parsed.get("missing_citations", []),
                incomplete_reasoning=parsed.get("incomplete_reasoning", []),
                vague_recommendations=parsed.get("vague_recommendations", []),
                feedback=parsed.get("feedback", ""),
            )
    except Exception:
        pass  # Caller handles None with fallback

    return None


def parse_policy_complexity(response: Any) -> PolicyComplexity:
    """
    Parse policy complexity analysis from LLM response.

    Args:
        response: Raw LLM response

    Returns:
        Parsed PolicyComplexity with defaults if parsing fails
    """
    try:
        content = extract_content(response)
        json_str = extract_json(content, "{")

        if json_str:
            parsed = json.loads(json_str)
            return PolicyComplexity(
                variable_count=parsed.get("variable_count", 2),
                complexity_level=parsed.get("complexity_level", "conditional"),
                recommended_turns=parsed.get("recommended_turns", 3),
                reasoning=parsed.get("reasoning", "Defaulting to conditional complexity"),
            )
    except Exception:
        pass  # Fallback handles this

    # Default fallback
    return PolicyComplexity(
        variable_count=2,
        complexity_level="conditional",
        recommended_turns=3,
        reasoning="Unable to analyze policy, defaulting to conditional complexity with 3 turns",
    )


def parse_policy_plan(response: Any, target_traces: int) -> PolicyPlan:
    """
    Parse policy planning output from LLM response.

    Args:
        response: Raw LLM response
        target_traces: Target number of traces for fallback

    Returns:
        Parsed PolicyPlan with defaults if parsing fails
    """
    try:
        content = extract_content(response)
        json_str = extract_json(content, "{")

        if json_str:
            parsed = json.loads(json_str)

            categories = []
            for cat in parsed.get("categories", []):
                categories.append(
                    {
                        "name": cat.get("name", "General"),
                        "description": cat.get("description", "General scenarios"),
                        "traces": cat.get("traces", target_traces // 3),
                    }
                )

            if categories:
                return PolicyPlan(
                    categories=categories,
                    reasoning=parsed.get("reasoning", ""),
                )
    except Exception:
        pass  # Fallback handles this

    # Default fallback plan
    third = target_traces // 3
    remainder = target_traces - (third * 3)
    return PolicyPlan(
        categories=[
            {"name": "Happy Path", "description": "Clear success cases", "traces": third},
            {"name": "Edge Cases", "description": "Ambiguous situations", "traces": third},
            {
                "name": "Violations",
                "description": "Clear failure cases",
                "traces": third + remainder,
            },
        ],
        reasoning="Default plan - unable to parse LLM response",
    )

