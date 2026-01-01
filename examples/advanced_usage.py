"""
Advanced Usage Example
======================

Comprehensive example demonstrating all Synkro features working together:
- Multiple dataset types (SFT, QA)
- Custom grader selection
- Evaluation workflow
- Custom pipeline components
- Quality filtering and analysis
- Real-world workflow from start to finish
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

import asyncio
from synkro import (
    Policy,
    Dataset,
    DatasetType,
    Generator,
    Grader,
    Refiner,
    Planner,
    ScenarioGenerator,
    ResponseGenerator,
    LLM,
    OpenAI,
    Anthropic,
)
from synkro.examples import EXPENSE_POLICY


async def main():
    print("=" * 80)
    print("Synkro Advanced Usage - Complete Workflow")
    print("=" * 80)
    print()

    # ============================================================================
    # Part 1: Load Policy
    # ============================================================================

    print("Part 1: Loading Policy")
    print("-" * 80)
    print()

    # Load from built-in example (or use Policy.from_file("handbook.pdf"))
    policy = Policy(text=EXPENSE_POLICY)
    print(f"Loaded policy: {policy.word_count} words")
    print()

    # ============================================================================
    # Part 2: Generate Multiple Dataset Types
    # ============================================================================

    print("Part 2: Generating Multiple Dataset Types")
    print("-" * 80)
    print()

    # Configure generator with custom models
    # Fast/cheap for generation, high-quality for grading
    generator = Generator(
        generation_model=OpenAI.GPT_4O_MINI,  # Fast, cheap
        grading_model=OpenAI.GPT_4O,          # High quality
        max_iterations=3,
    )

    # Generate SFT format (default)
    print("Generating SFT dataset...")
    dataset_sft = generator.generate(policy, traces=10)
    print(f"  Generated {len(dataset_sft)} traces, pass rate: {dataset_sft.passing_rate:.1%}")
    dataset_sft.save("advanced_sft.jsonl", format="sft")
    print()

    # Generate QA format
    print("Generating QA dataset...")
    generator_qa = Generator(
        dataset_type=DatasetType.QA,
        generation_model=OpenAI.GPT_4O_MINI,
        grading_model=OpenAI.GPT_4O,
    )
    dataset_qa = generator_qa.generate(policy, traces=10)
    print(f"  Generated {len(dataset_qa)} traces, pass rate: {dataset_qa.passing_rate:.1%}")
    dataset_qa.save("advanced_qa.jsonl", format="qa")
    print()

    # ============================================================================
    # Part 3: Custom Grader Selection
    # ============================================================================

    print("Part 3: Custom Grader Selection")
    print("-" * 80)
    print()

    # Re-grade with different grader model
    print("Re-grading SFT dataset with Claude grader...")
    claude_grader = Grader(model=Anthropic.CLAUDE_35_SONNET)

    for trace in dataset_sft:
        new_grade = await claude_grader.grade(trace, policy.text)
        trace.grade = new_grade

    # Compare pass rates
    claude_passed = sum(1 for t in dataset_sft if t.grade and t.grade.passed)
    print(f"  GPT-4o grader: {dataset_sft.passing_rate:.1%} passed")
    print(f"  Claude grader: {claude_passed/len(dataset_sft):.1%} passed")
    print("  (Different graders may have different evaluation criteria)")
    print()

    # ============================================================================
    # Part 4: Custom Pipeline with Individual Components
    # ============================================================================

    print("Part 4: Custom Pipeline with Individual Components")
    print("-" * 80)
    print()

    # Create LLM clients for different tasks
    generation_llm = LLM(model=OpenAI.GPT_4O_MINI, temperature=0.8)
    grading_llm = LLM(model=OpenAI.GPT_4O, temperature=0.3)

    # Step 1: Plan categories
    print("Step 1: Planning categories...")
    planner = Planner(llm=grading_llm)
    plan = await planner.plan(policy.text, target_traces=20)
    print(f"  Created plan with {len(plan.categories)} categories:")
    for cat in plan.categories:
        print(f"    - {cat.name}: {cat.count} traces")
    print()

    # Step 2: Generate scenarios
    print("Step 2: Generating scenarios...")
    scenario_gen = ScenarioGenerator(llm=generation_llm)
    all_scenarios = []
    for category in plan.categories:
        scenarios = await scenario_gen.generate(
            policy.text,
            count=category.count,
            category=category,
        )
        all_scenarios.extend(scenarios)
    print(f"  Generated {len(all_scenarios)} scenarios")
    print()

    # Step 3: Generate responses
    print("Step 3: Generating responses...")
    response_gen = ResponseGenerator(llm=generation_llm)
    traces = await response_gen.generate(policy.text, all_scenarios)
    print(f"  Generated {len(traces)} responses")
    print()

    # Step 4: Grade responses
    print("Step 4: Grading responses...")
    grader = Grader(llm=grading_llm)
    grades = await grader.grade_batch(traces, policy.text)
    
    for trace, grade in zip(traces, grades):
        trace.grade = grade

    passed = sum(1 for g in grades if g.passed)
    print(f"  {passed}/{len(grades)} passed ({passed/len(grades):.1%})")
    print()

    # Step 5: Refine failed responses
    print("Step 5: Refining failed responses...")
    refiner = Refiner(llm=generation_llm)
    refined_traces = []
    
    for trace, grade in zip(traces, grades):
        if grade.passed:
            refined_traces.append(trace)
        else:
            refined = await refiner.refine(trace, grade, policy.text)
            # Re-grade refined response
            new_grade = await grader.grade(refined, policy.text)
            refined.grade = new_grade
            refined_traces.append(refined)

    final_passed = sum(1 for t in refined_traces if t.grade and t.grade.passed)
    print(f"  After refinement: {final_passed}/{len(refined_traces)} passed ({final_passed/len(refined_traces):.1%})")
    print()

    # Create dataset from custom pipeline
    custom_dataset = Dataset(traces=refined_traces)
    print()

    # ============================================================================
    # Part 5: Quality Analysis and Filtering
    # ============================================================================

    print("Part 5: Quality Analysis and Filtering")
    print("-" * 80)
    print()

    # Overall statistics
    print("Overall statistics:")
    print(f"  Total traces: {len(custom_dataset)}")
    print(f"  Pass rate: {custom_dataset.passing_rate:.1%}")
    print(f"  Categories: {len(custom_dataset.categories)}")
    print()

    # Filter by quality
    print("Filtering by quality...")
    passing = custom_dataset.filter(passed=True)
    failing = custom_dataset.filter(passed=False)
    print(f"  Passing traces: {len(passing)}")
    print(f"  Failing traces: {len(failing)}")
    print()

    # Filter by category
    if custom_dataset.categories:
        category = custom_dataset.categories[0]
        category_traces = custom_dataset.filter(category=category)
        print(f"  Traces in '{category}' category: {len(category_traces)}")
        print()

    # Filter by multiple criteria
    high_quality = custom_dataset.filter(
        passed=True,
        min_length=200,  # Minimum response length
    )
    print(f"  High-quality traces (passed, min 200 chars): {len(high_quality)}")
    print()

    # Analyze issues
    print("Issue analysis:")
    all_issues = []
    for trace in custom_dataset:
        if trace.grade and trace.grade.issues:
            all_issues.extend(trace.grade.issues)

    if all_issues:
        from collections import Counter
        issue_counts = Counter(all_issues)
        print("  Most common issues:")
        for issue, count in issue_counts.most_common(3):
            print(f"    - {issue} ({count}x)")
    else:
        print("  No issues found")
    print()

    # ============================================================================
    # Part 6: Export Results
    # ============================================================================

    print("Part 6: Exporting Results")
    print("-" * 80)
    print()

    # Save all traces
    custom_dataset.save("advanced_custom_pipeline.jsonl", format="sft")
    print("  Saved all traces: advanced_custom_pipeline.jsonl")

    # Save only passing traces
    if len(passing) > 0:
        passing.save("advanced_high_quality.jsonl", format="sft")
        print("  Saved passing traces: advanced_high_quality.jsonl")

    # Save in different formats
    custom_dataset.save("advanced_custom_qa.jsonl", format="qa")
    print("  Saved as QA format: advanced_custom_qa.jsonl")
    print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print("This example demonstrated:")
    print("  ✓ Generating multiple dataset types (SFT, QA)")
    print("  ✓ Custom grader selection with different models")
    print("  ✓ Custom pipeline using individual components")
    print("  ✓ Evaluation workflow with grading and refinement")
    print("  ✓ Quality analysis and filtering")
    print("  ✓ Exporting datasets in different formats")
    print()
    print("Files created:")
    print("  - advanced_sft.jsonl")
    print("  - advanced_qa.jsonl")
    print("  - advanced_custom_pipeline.jsonl")
    print("  - advanced_high_quality.jsonl")
    print("  - advanced_custom_qa.jsonl")
    print()


if __name__ == "__main__":
    asyncio.run(main())

