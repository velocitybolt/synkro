"""Dataset class for managing generated traces."""

import json
from datetime import datetime
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, Field
from rich.console import Console

from synkro.types.core import Trace

console = Console()


class Dataset(BaseModel):
    """
    A collection of generated training traces.

    Provides methods for filtering, saving, and exporting traces
    in various formats.

    Examples:
        >>> dataset = generator.generate(policy, traces=100)

        >>> # Filter to only passing traces
        >>> passing = dataset.filter(passed=True)

        >>> # Save to JSONL
        >>> dataset.save("training.jsonl")

        >>> # Push to HuggingFace
        >>> dataset.to_huggingface().push_to_hub("my-org/dataset")
    """

    traces: list[Trace] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def __len__(self) -> int:
        return len(self.traces)

    def __iter__(self) -> Iterator[Trace]:
        return iter(self.traces)

    def __getitem__(self, idx: int) -> Trace:
        return self.traces[idx]

    def filter(
        self,
        passed: bool | None = None,
        category: str | None = None,
        min_length: int | None = None,
    ) -> "Dataset":
        """
        Filter traces by criteria.

        Args:
            passed: Filter by grade pass/fail status
            category: Filter by scenario category
            min_length: Minimum response length in characters

        Returns:
            New Dataset with filtered traces
        """
        filtered = self.traces

        if passed is not None:
            filtered = [
                t for t in filtered if t.grade and t.grade.passed == passed
            ]

        if category is not None:
            filtered = [
                t for t in filtered if t.scenario.category == category
            ]

        if min_length is not None:
            filtered = [
                t for t in filtered if len(t.assistant_message) >= min_length
            ]

        return Dataset(traces=filtered)

    @property
    def passing_rate(self) -> float:
        """Get the percentage of traces that passed grading."""
        if not self.traces:
            return 0.0

        passed = sum(1 for t in self.traces if t.grade and t.grade.passed)
        return passed / len(self.traces)

    @property
    def categories(self) -> list[str]:
        """Get unique categories in the dataset."""
        return list(set(t.scenario.category for t in self.traces if t.scenario.category))

    def save(self, path: str | Path | None = None, format: str = "sft") -> "Dataset":
        """
        Save dataset to a JSONL file.

        Args:
            path: Output file path (auto-generated if not provided)
            format: Output format - "sft" or "qa"

        Returns:
            Self for method chaining

        Example:
            >>> dataset.save()  # Auto-names: synkro_sft_2024-01-15.jsonl
            >>> dataset.save("training.jsonl")
            >>> dataset.save("qa_data.jsonl", format="qa")
        """
        from synkro.formatters import SFTFormatter, QAFormatter

        # Auto-generate filename if not provided
        if path is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
            path = f"synkro_{format}_{timestamp}.jsonl"
        
        path = Path(path)

        if format == "sft":
            SFTFormatter().save(self.traces, path)
        elif format == "qa":
            QAFormatter().save(self.traces, path)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'sft' or 'qa'")
        
        # Print confirmation
        file_size = path.stat().st_size
        size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024 * 1024 else f"{file_size / 1024 / 1024:.1f} MB"
        console.print(f"[green]📁 Saved:[/green] {path} ({size_str})")
        
        return self

    def to_jsonl(self, format: str = "sft") -> str:
        """
        Convert dataset to JSONL string.

        Args:
            format: Output format - "sft" or "qa"

        Returns:
            JSONL formatted string
        """
        from synkro.formatters import SFTFormatter, QAFormatter

        if format == "sft":
            return SFTFormatter().to_jsonl(self.traces)
        elif format == "qa":
            return QAFormatter().to_jsonl(self.traces)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'sft' or 'qa'")

    def to_huggingface(self):
        """
        Convert to HuggingFace Dataset.

        Returns:
            HuggingFace Dataset object

        Example:
            >>> hf_dataset = dataset.to_huggingface()
            >>> hf_dataset.push_to_hub("my-org/policy-traces")
        """
        try:
            from datasets import Dataset as HFDataset

            # Convert to SFT format for HF
            from synkro.formatters import SFTFormatter

            examples = SFTFormatter(include_metadata=True).format(self.traces)
            return HFDataset.from_list(examples)
        except ImportError:
            raise ImportError(
                "datasets is required for HuggingFace export. "
                "Install with: pip install datasets"
            )

    def to_dict(self) -> dict:
        """
        Convert dataset to a dictionary.

        Returns:
            Dictionary with trace data
        """
        return {
            "traces": [t.model_dump() for t in self.traces],
            "stats": {
                "total": len(self.traces),
                "passing_rate": self.passing_rate,
                "categories": self.categories,
            },
        }

    def summary(self) -> str:
        """
        Get a summary of the dataset.

        Returns:
            Human-readable summary string
        """
        lines = [
            f"Dataset Summary",
            f"===============",
            f"Total traces: {len(self.traces)}",
            f"Passing rate: {self.passing_rate:.1%}",
            f"Categories: {len(self.categories)}",
        ]

        if self.categories:
            lines.append("")
            lines.append("By category:")
            for cat in self.categories:
                count = sum(1 for t in self.traces if t.scenario.category == cat)
                lines.append(f"  - {cat}: {count}")

        return "\n".join(lines)

    def __str__(self) -> str:
        return f"Dataset(traces={len(self.traces)}, passing={self.passing_rate:.1%})"

    def __repr__(self) -> str:
        return self.__str__()

