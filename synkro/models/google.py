"""Google Gemini models.

Updated based on: https://ai.google.dev/gemini-api/docs/models#model-versions
"""

from enum import Enum


class Google(str, Enum):
    """Google Gemini models."""

    GEMINI_3_PRO = "gemini/gemini-3-pro"
    GEMINI_3_FLASH = "gemini/gemini-3-flash"

    GEMINI_25_FLASH = "gemini/gemini-2.5-flash"
    GEMINI_25_PRO = "gemini/gemini-2.5-pro"

    GEMINI_2_FLASH = "gemini/gemini-2.0-flash"
    GEMINI_2_FLASH_LITE = "gemini/gemini-2.0-flash-lite"