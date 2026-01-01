"""Policy document handling with multi-format support."""

from pathlib import Path

from pydantic import BaseModel, Field

from synkro.errors import FileNotFoundError as SynkroFileNotFoundError, PolicyTooShortError


MIN_POLICY_WORDS = 10  # Minimum words for meaningful generation


class Policy(BaseModel):
    """
    A policy document to generate training data from.

    Supports loading from multiple formats:
    - Plain text (.txt, .md)
    - PDF documents (.pdf) - via marker-pdf
    - Word documents (.docx) - via mammoth

    Examples:
        >>> # From text
        >>> policy = Policy(text="All expenses over $50 require approval")

        >>> # From file
        >>> policy = Policy.from_file("compliance.pdf")

        >>> # From URL
        >>> policy = Policy.from_url("https://example.com/policy")
    """

    text: str = Field(description="Full policy text in markdown format")
    source: str | None = Field(default=None, description="Source file path or URL")

    @classmethod
    def from_file(cls, path: str | Path) -> "Policy":
        """
        Load policy from a file.

        Supports: .txt, .md, .pdf, .docx

        Args:
            path: Path to the policy file

        Returns:
            Policy object with extracted text

        Example:
            >>> policy = Policy.from_file("compliance.pdf")
            >>> len(policy.text) > 0
            True
        """
        path = Path(path)

        if not path.exists():
            # Find similar files to suggest
            similar = []
            if path.parent.exists():
                for ext in [".txt", ".md", ".pdf", ".docx"]:
                    similar.extend(path.parent.glob(f"*{ext}"))
            similar_names = [str(f.name) for f in similar[:5]]
            raise SynkroFileNotFoundError(str(path), similar_names if similar_names else None)

        suffix = path.suffix.lower()

        if suffix in (".txt", ".md"):
            return cls(text=path.read_text(), source=str(path))

        if suffix == ".pdf":
            return cls._from_pdf(path)

        if suffix == ".docx":
            return cls._from_docx(path)

        raise ValueError(f"Unsupported file type: {suffix}. Use .txt, .md, .pdf, or .docx")

    @classmethod
    def _from_pdf(cls, path: Path) -> "Policy":
        """
        Parse PDF to markdown using marker-pdf.

        Args:
            path: Path to PDF file

        Returns:
            Policy with extracted markdown text
        """
        try:
            from marker.convert import convert_single_pdf
            from marker.models import load_all_models

            models = load_all_models()
            markdown, _, _ = convert_single_pdf(str(path), models)
            return cls(text=markdown, source=str(path))
        except ImportError:
            raise ImportError(
                "marker-pdf is required for PDF support. "
                "Install with: pip install marker-pdf"
            )

    @classmethod
    def _from_docx(cls, path: Path) -> "Policy":
        """
        Parse DOCX to markdown using mammoth.

        Args:
            path: Path to DOCX file

        Returns:
            Policy with extracted markdown text
        """
        try:
            import mammoth

            with open(path, "rb") as f:
                result = mammoth.convert_to_markdown(f)
                return cls(text=result.value, source=str(path))
        except ImportError:
            raise ImportError(
                "mammoth is required for DOCX support. "
                "Install with: pip install mammoth"
            )

    @classmethod
    def from_url(cls, url: str) -> "Policy":
        """
        Fetch and parse policy from a URL.

        Extracts main content and converts to markdown.

        Args:
            url: URL to fetch

        Returns:
            Policy with extracted content

        Example:
            >>> policy = Policy.from_url("https://example.com/terms")
        """
        try:
            import httpx
            from bs4 import BeautifulSoup
            import html2text

            response = httpx.get(url, follow_redirects=True)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove scripts, styles, nav, footer
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            # Convert to markdown
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = True
            markdown = h.handle(str(soup))

            return cls(text=markdown, source=url)
        except ImportError as e:
            missing = str(e).split("'")[1] if "'" in str(e) else "required packages"
            raise ImportError(
                f"{missing} is required for URL support. "
                "This should be installed automatically with synkro. "
                "If you see this error, try: pip install --upgrade synkro"
            )

    @property
    def word_count(self) -> int:
        """Get the word count of the policy."""
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        """Get the character count of the policy."""
        return len(self.text)

    def validate_length(self) -> None:
        """
        Validate that the policy has enough content for meaningful generation.
        
        Raises:
            PolicyTooShortError: If policy is too short
        """
        if self.word_count < MIN_POLICY_WORDS:
            raise PolicyTooShortError(self.word_count)

    def __str__(self) -> str:
        """String representation showing source and length."""
        source = self.source or "inline"
        return f"Policy(source={source}, words={self.word_count})"

    def __repr__(self) -> str:
        return self.__str__()

