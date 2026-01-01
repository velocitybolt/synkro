"""QA (Question-Answer) formatter."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synkro.types.core import Trace


class QAFormatter:
    """
    Format traces for Question-Answer datasets.

    QA format is simple question/answer pairs with optional context,
    suitable for RAG training and knowledge extraction.

    Example output:
        {"question": "...", "answer": "...", "context": "..."}
        {"question": "...", "answer": "...", "context": "..."}
    """

    def __init__(self, include_context: bool = True):
        """
        Initialize the QA formatter.

        Args:
            include_context: If True, include source context in output
        """
        self.include_context = include_context

    def format(self, traces: list["Trace"]) -> list[dict]:
        """
        Format traces as QA pairs.

        Args:
            traces: List of traces to format

        Returns:
            List of QA examples (dicts with 'question', 'answer', optionally 'context')
        """
        examples = []

        for trace in traces:
            example = {
                "question": trace.user_message,
                "answer": trace.assistant_message,
            }

            if self.include_context:
                # Use scenario context or the source section if available
                example["context"] = trace.scenario.context or ""

            examples.append(example)

        return examples

    def save(self, traces: list["Trace"], path: str | Path) -> None:
        """
        Save formatted traces to a JSONL file.

        Args:
            traces: List of traces to save
            path: Output file path
        """
        path = Path(path)
        examples = self.format(traces)

        with open(path, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")

    def to_jsonl(self, traces: list["Trace"]) -> str:
        """
        Convert traces to JSONL string.

        Args:
            traces: List of traces to convert

        Returns:
            JSONL formatted string
        """
        examples = self.format(traces)
        return "\n".join(json.dumps(e) for e in examples)

