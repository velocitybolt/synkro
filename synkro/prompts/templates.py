"""Universal prompt templates for dataset generation across ANY domain."""

# =============================================================================
# POLICY ANALYSIS PROMPTS
# =============================================================================

POLICY_COMPLEXITY_PROMPT = """You are an expert at analyzing policy documents to determine their complexity.

Analyze the given policy and determine the optimal number of conversation turns needed to properly test understanding.

Guidelines:
- **Simple (1-2 turns)**: Policy has 1 clear variable/rule. Single query → Straight answer.
  Example: "All data must be encrypted" - just one rule to check.
  
- **Conditional (3 turns)**: Policy has 2-3 variables/conditions. Query → Clarification → Verdict.
  Example: "Data can be shared IF consent is given AND purpose is specified" - needs clarification.
  
- **Complex (5+ turns)**: Policy has 4+ nested variables, exceptions, or conditions. 
  Multiple rounds of validation before final sign-off.
  Example: "Data retention varies by type, region, consent status, and business need" - needs deep exploration.

Count the following as "variables":
- Distinct rules or requirements
- Conditional branches (if/then/else)
- Exceptions to rules
- Categories or types that affect decisions
- Time-based conditions
- Role-based permissions

Respond with your analysis."""

POLICY_PLANNING_PROMPT = """You are an expert at creating training data plans for AI models across ANY domain.

Given a task description, policy, or domain specification and a target number of traces, analyze the content and create an optimal plan for generating training data.

Your task:
1. Deeply analyze the domain/task to understand its core concepts, rules, processes, and challenges
2. Identify distinct SCENARIO CATEGORIES that test different aspects of the domain
3. Distribute the target traces across categories based on complexity and importance
4. Ensure coverage of: clear violations/errors, edge cases, happy paths, real-world constraints, and domain-specific challenges

Guidelines for dynamic category creation:
- **Analyze the domain deeply**: Understand the core rules, processes, stakeholders, and common challenges
- **Create domain-specific categories**: Base categories on the actual content, not generic assumptions
- **Balance complexity**: Allocate based on domain complexity (simple domains: 60% happy paths, complex domains: 40% edge cases)
- **Ensure comprehensive coverage**: Every major aspect of the domain should be tested
- **Consider domain-specific challenges**: Time pressure in trading, regulatory changes in finance, technical failures in engineering, etc.

For each category, provide:
- name: Short descriptive name specific to the domain
- description: What this category tests, including specific domain concepts and challenges
- traces: Number of traces to generate (must sum to target)

Provide detailed reasoning explaining:
1. Your analysis of the domain's core concepts and challenges
2. Why you chose these specific categories for this domain
3. How the category distribution reflects the domain's complexity and real-world usage patterns"""

# =============================================================================
# SCENARIO GENERATION PROMPTS
# =============================================================================

SCENARIO_GENERATOR_PROMPT = """You are an expert at creating realistic scenarios for ANY domain or task.

Given a task description, policy, or domain specification, first deeply analyze the domain to understand:
- Core concepts, rules, and processes
- Key stakeholders and their roles
- Common challenges and failure modes
- Domain-specific terminology and workflows

Then generate diverse scenarios that thoroughly test understanding of the domain:

1. **Clear Success/Failure Cases** - Obvious correct/incorrect applications of domain rules
2. **Edge Cases** - Ambiguous situations with multiple valid interpretations
3. **Multi-Step Processes** - Complex scenarios requiring sequential reasoning
4. **Real-World Constraints** - Practical limitations like time pressure, incomplete info, resource constraints
5. **Domain-Specific Challenges** - Scenarios that test unique aspects of this particular domain
6. **Stakeholder Interactions** - Situations involving coordination between different parties
7. **Exception Handling** - Scenarios requiring deviation from standard processes

Make each scenario:
- Deeply grounded in the specific domain's concepts and terminology
- Realistic and challenging for someone working in that domain
- Specific with concrete details that reflect actual domain practices
- Varied in complexity and stakeholder perspectives
- Designed to reveal both expert and novice understanding gaps

Focus on creating "golden traces" - perfect examples that demonstrate deep domain mastery."""

CATEGORY_SCENARIO_PROMPT = """You are an expert at creating realistic scenarios for ANY domain or task.

Generate scenarios specifically for the following CATEGORY within the given domain:
**Category Name**: {CATEGORY_NAME}
**Category Description**: {CATEGORY_DESCRIPTION}

First, deeply understand:
- How this category fits into the broader domain
- What specific skills or knowledge this category tests
- The real-world contexts where this category applies
- Common mistakes or misconceptions in this category

All generated scenarios MUST:
- Perfectly fit this specific category's focus and objectives
- Demonstrate deep understanding of the category's role in the domain
- Test the exact skills and knowledge described in the category
- Be realistic and occur in actual domain practice

Make each scenario:
- Highly specific with concrete details that reflect domain expertise
- Challenging and nuanced - not simplistic examples
- Varied in stakeholder perspectives, contexts, and complexity levels
- Different from each other (no duplicates) - explore different facets of the category
- Include domain-specific terminology, processes, and challenges
- Designed as "golden traces" that showcase expert-level understanding

Focus on creating scenarios that would distinguish between novice and expert performance in this category."""

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a domain expert. When given a scenario and context, provide comprehensive, expert-level guidance.

IMPORTANT: Always show your reasoning process using <reasoning> tags before giving your answer.

Your responses must:
- Start with <reasoning> tags showing step-by-step analysis
- Cite specific domain concepts, rules, or processes that apply
- Give specific, actionable recommendations grounded in domain best practices
- Address all aspects of the scenario from multiple stakeholder perspectives
- Acknowledge edge cases, exceptions, and potential complications
- Consider contemporary challenges and modern practices in the domain

Vary your response style while maintaining expertise:
- For concise responses: Direct, focused guidance with key domain principles
- For detailed responses: Comprehensive analysis with structured breakdowns and examples
- For practical responses: Step-by-step implementation guides and checklists
- For complex responses: Thorough exploration of trade-offs and alternative approaches

Always prioritize accuracy, clarity, and deep domain understanding in your guidance."""

# =============================================================================
# BATCHED PROMPTS (for batch generation)
# =============================================================================

BATCHED_RESPONSE_PROMPT = """You are generating training data for a domain expert model.

For EACH scenario below, create a complete training example in CHAT MESSAGES FORMAT.

Each training example must have exactly 3 messages:
1. "system" - The system prompt defining the assistant's role
2. "user" - The scenario and context as the user's question
3. "assistant" - Your expert guidance response

The assistant response must:
- Cite specific policy sections that apply
- Explain reasoning step-by-step
- Give specific, actionable recommendations
- Address all aspects of the scenario
- Acknowledge edge cases and complications

Respond with a JSON array where each object has:
- "index": the scenario number (0-based)
- "messages": array of 3 message objects with "role" and "content" fields"""

BATCHED_GRADER_PROMPT = """You are a strict policy compliance evaluator. Your job is to determine if EACH response is FULLY CORRECT.

A response PASSES only if ALL of the following are true:
1. **Policy Compliant** - Every recommendation follows the policy exactly. No violations.
2. **Fully Supported** - Every claim is backed by a specific policy section. Nothing made up.
3. **Properly Cited** - All relevant policy sections are explicitly referenced.
4. **Complete Reasoning** - The chain of thought is complete with no gaps or skipped steps.
5. **Actionable & Specific** - All recommendations are concrete and implementable, not vague.

If ANY of these fail, the response does NOT pass. Be strict - only mark "pass": true for perfect responses.

For each response, provide structured feedback:
- "policy_violations": List any rules misinterpreted or violated
- "missing_citations": List policy sections that should have been cited
- "incomplete_reasoning": List logical gaps or missing reasoning steps
- "vague_recommendations": List recommendations that need to be more specific
- "feedback": Summary of what needs to be fixed

Respond with a JSON array where each object has:
- "index": the scenario number (0-based)
- "pass": boolean (true ONLY if response is fully correct)
- "policy_violations": array of violations
- "missing_citations": array of missing citations
- "incomplete_reasoning": array of reasoning gaps
- "vague_recommendations": array of vague items
- "feedback": summary of how to fix"""

BATCHED_REFINER_PROMPT = """You are improving training data for a domain expert model based on grader feedback.

For EACH scenario with feedback below, fix ALL issues while keeping what was correct.

You will receive structured feedback with:
- policy_violations: Rules you violated or misinterpreted - FIX THESE
- missing_citations: Policy sections you should cite - ADD THESE  
- incomplete_reasoning: Gaps in your logic - FILL THESE IN
- vague_recommendations: Things that need to be more specific - MAKE CONCRETE

Requirements:
1. Fix every policy violation - ensure recommendations follow the policy exactly
2. Add citations for every missing policy section mentioned
3. Complete any incomplete reasoning chains with step-by-step logic
4. Replace vague language with specific, actionable recommendations
5. Keep the parts that were already correct

Output in CHAT MESSAGES FORMAT with exactly 3 messages:
1. "system" - The system prompt defining the assistant's role
2. "user" - The scenario and context as the user's question  
3. "assistant" - Your IMPROVED guidance

Respond with a JSON array where each object has:
- "index": the scenario number (0-based)
- "messages": array of 3 message objects with "role" and "content" fields"""

# =============================================================================
# SINGLE PROMPTS (for parallel high-concurrency generation)
# =============================================================================

SINGLE_RESPONSE_PROMPT = """You are a domain expert generating a training example.

Given the scenario and policy below, create a complete training example.

Your response must be a JSON object with exactly 3 messages:
{{
  "messages": [
    {{"role": "system", "content": "<system prompt defining expert role>"}},
    {{"role": "user", "content": "<the scenario as a user question>"}},
    {{"role": "assistant", "content": "<your expert response>"}}
  ]
}}

The assistant response must:
- Start with <reasoning> tags showing your thought process
- Cite specific policy sections that apply
- Give specific, actionable recommendations
- Address all aspects of the scenario
- Acknowledge edge cases and complications

SCENARIO:
{scenario}

CONTEXT:
{context}

POLICY:
{policy}

Respond with ONLY the JSON object, no additional text."""

SINGLE_GRADE_PROMPT = """You are a strict evaluator. Grade this response.

A response PASSES only if ALL are true:
1. Policy Compliant - Every recommendation follows the policy exactly
2. Fully Supported - Every claim backed by specific policy section
3. Properly Cited - All relevant policy sections referenced
4. Complete Reasoning - Chain of thought has no gaps
5. Actionable & Specific - Recommendations are concrete, not vague

SCENARIO:
{scenario}

POLICY:
{policy}

RESPONSE TO GRADE:
{response}

Respond with ONLY a JSON object:
{{
  "pass": <true/false>,
  "policy_violations": ["<violation 1>", ...],
  "missing_citations": ["<missing 1>", ...],
  "incomplete_reasoning": ["<gap 1>", ...],
  "vague_recommendations": ["<vague 1>", ...],
  "feedback": "<summary of issues or 'Correct'>"
}}"""

