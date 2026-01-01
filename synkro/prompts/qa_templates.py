"""QA-specific prompt templates for question-answer pair generation."""

QA_SCENARIO_PROMPT = """You are an expert at creating factual questions from documents.

Given a document, generate diverse questions that can be answered directly from the content.

Types of questions to generate:
1. **Factual** - Who, what, when, where questions with direct answers
2. **Definitional** - "What is..." or "Define..." questions
3. **Procedural** - "How do you..." or "What are the steps..."
4. **Comparative** - Questions comparing concepts within the document
5. **Inferential** - Questions requiring light reasoning from stated facts

Make each question:
- Answerable from the document (no external knowledge needed)
- Specific and unambiguous
- Varied in complexity and type
- Natural - how a real person would ask

Focus on creating questions that test comprehension of the document content."""

QA_RESPONSE_PROMPT = """You are answering questions using ONLY information from the provided document.

Rules:
1. Answer ONLY using facts stated in the document
2. Quote or paraphrase the relevant section
3. If the answer isn't in the document, say "Not found in document"
4. Keep answers concise but complete
5. Include the source section/paragraph when possible

Your response must be a JSON object:
{{
  "question": "<the question being answered>",
  "answer": "<your answer using document facts>",
  "context": "<the relevant passage from the document>"
}}

DOCUMENT:
{policy}

QUESTION:
{scenario}

Respond with ONLY the JSON object."""

QA_GRADE_PROMPT = """You are grading a question-answer pair for quality.

A QA pair PASSES only if ALL are true:
1. **Factually Correct** - Answer is accurate based on the document
2. **Properly Sourced** - Context contains the relevant passage
3. **Complete** - Answer fully addresses the question
4. **Concise** - No unnecessary information or padding
5. **Grounded** - No information made up beyond the document

DOCUMENT:
{policy}

QUESTION:
{scenario}

ANSWER TO GRADE:
{response}

Respond with ONLY a JSON object:
{{
  "pass": <true/false>,
  "factual_errors": ["<error 1>", ...],
  "missing_info": ["<missing 1>", ...],
  "source_issues": ["<issue 1>", ...],
  "feedback": "<summary of issues or 'Correct'>"
}}"""

QA_REFINE_PROMPT = """You are improving a question-answer pair based on feedback.

Fix all issues while maintaining accuracy to the source document.

DOCUMENT:
{policy}

QUESTION:
{scenario}

ORIGINAL ANSWER:
{response}

ISSUES TO FIX:
{feedback}

Generate an IMPROVED answer. Output a JSON object:
{{
  "question": "<the question>",
  "answer": "<your IMPROVED answer>",
  "context": "<the relevant passage from the document>"
}}

Respond with ONLY the JSON object."""

