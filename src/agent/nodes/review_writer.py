"""Review Writer node — writes one section at a time using gathered papers."""
from __future__ import annotations

import json
import re
import time
from typing import Any

from src.config import ModelTier, settings
from src.llm import chat


_SYSTEM = """You are a literature review WRITER. Write a single section of a \\
structured literature review with inline citations [S1], [S2], etc.

Rules:
- Write 3-8 paragraphs synthesizing the papers found for this section
- Cite every factual claim with the [S#] tag from the evidence
- Compare and contrast papers where they differ
- Identify trends, disagreements, or gaps in the literature
- Use academic tone but clear, direct prose
- Each paragraph should synthesize MULTIPLE sources, not just summarize one paper

Output ONLY the section body as plain text with [S#] citations. No section heading,
no preamble, no markdown fence."""


def write_section(
    section_heading: str,
    section_description: str,
    evidence_block: str,
) -> str:
    """Write one section given its heading, description, and evidence text."""
    user = f"""Section to write: {section_heading}
{section_description}

Evidence sources for this section:
{evidence_block}

Write the section now with [S#] citations."""

    t0 = time.perf_counter()
    raw = chat(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
        tier=ModelTier.REASONING,
        temperature=0.2,
        max_tokens=1500,
    )
    dur_ms = int((time.perf_counter() - t0) * 1000)

    return raw.strip()