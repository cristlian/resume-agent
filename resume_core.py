#!/usr/bin/env python3
"""Core deterministic logic for the resume optimizer."""

from __future__ import annotations

from collections.abc import Mapping
from html import unescape
from pathlib import Path
from typing import Any
import re

import yaml


CRITIC_SECTIONS = ("A", "B", "C", "D")


def check_for_markdown(text: str) -> list[str]:
    """Check for markdown-like patterns in text. Returns list of warnings."""
    warnings: list[str] = []
    lines = text.split("\n")

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if re.match(r"^#{1,6}\s", stripped):
            warnings.append(f"Line {i}: Markdown heading detected (starts with #)")
        if stripped.startswith("```"):
            warnings.append(f"Line {i}: Code block marker detected (```)")
        if re.search(r"\*\*[^*]+\*\*", stripped) or re.search(r"__[^_]+__", stripped):
            warnings.append(f"Line {i}: Bold markdown detected (**text** or __text__)")

    return warnings


def enforce_plain_text(text: str) -> str:
    """Normalize common markdown syntax into ATS-friendly plain text."""
    normalized_lines: list[str] = []

    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        line = re.sub(r"^\s*#{1,6}\s*", "", line)
        line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
        line = re.sub(r"__([^_]+)__", r"\1", line)
        line = re.sub(r"^\s*[*+]\s+", "- ", line)
        normalized_lines.append(line.rstrip())

    normalized = "\n".join(normalized_lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def extract_role_from_jd_filename(jd_path: Path) -> str:
    """Extract a clean role name from the JD filename."""
    name = jd_path.stem
    for suffix in ["_Job_Description", "_JD", "-JD", "_job_description", " Job Description"]:
        name = name.replace(suffix, "")
    name = re.sub(r"[_-]+", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def extract_length_constraint(question: str) -> str:
    """Extract length constraints from a question."""
    q_lower = question.lower()

    sentence_match = re.search(r"(?:in\s+)?(\d+|one|two|three|four|five)\s+sentences?", q_lower)
    if sentence_match:
        num = sentence_match.group(1)
        num_map = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}
        num = num_map.get(num, num)
        return f"exactly {num} sentence(s)"

    word_match = re.search(r"(?:in\s+|max(?:imum)?\s+|under\s+|within\s+)?(\d+)\s+words?", q_lower)
    if word_match:
        num = word_match.group(1)
        if "max" in q_lower or "under" in q_lower:
            return f"maximum {num} words"
        return f"approximately {num} words"

    char_match = re.search(r"(?:max(?:imum)?\s+|under\s+)?(\d+)\s+characters?", q_lower)
    if char_match:
        return f"maximum {char_match.group(1)} characters"

    para_match = re.search(r"(?:in\s+)?(\d+|one|two|three)\s+paragraphs?", q_lower)
    if para_match:
        num = para_match.group(1)
        num_map = {"one": "1", "two": "2", "three": "3"}
        num = num_map.get(num, num)
        return f"exactly {num} paragraph(s)"

    if re.search(r"\bbrief(?:ly)?\b", q_lower):
        return "2-3 sentences (keep it brief)"

    if re.search(r"\bconcise(?:ly)?\b", q_lower):
        return "3-4 sentences (be concise)"

    return "max 2 short paragraphs"


def build_critic_prompt(draft_resume: str, profile_facts_text: str) -> str:
    """Build a structured critic prompt with deterministic categories."""
    return f"""RESUME TO EVALUATE:

{draft_resume}

---

PROFILE FACTS (authoritative user-provided facts; do NOT mark these as missing):
{profile_facts_text}

---

Evaluate this resume from a top-tier recruiter perspective with strict structure.

Output ONLY plain text using exactly these sections:
A) Needs user input (blocking)
B) Auto-fixable
C) Already covered (Profile Facts / evidence)
D) Optional

Rules:
- Put each finding on its own line prefixed with "- "
- Section A must contain only truly blocking missing information
- If a section has no findings, write "- None"
- No flattery, no reassurance, no markdown
"""


def _is_heading_description(text: str) -> bool:
    """Return True when *text* is just a section heading description, not a real finding."""
    normalised = text.lower().strip().rstrip(".")
    # Known heading descriptions that the critic prompt itself contains.
    known = {
        "needs user input",
        "needs user input (blocking)",
        "blocking",
        "auto-fixable",
        "auto fixable",
        "already covered",
        "already covered (profile facts / evidence)",
        "already covered (profile facts/evidence)",
        "optional",
    }
    return normalised in known


def parse_structured_critic(critique_text: str) -> dict[str, list[str]]:
    """Parse structured critic text into category buckets A/B/C/D."""
    parsed: dict[str, list[str]] = {section: [] for section in CRITIC_SECTIONS}
    current_section: str | None = None
    found_heading = False

    for raw_line in critique_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        heading_match = re.match(r"^([ABCD])\s*[\)\].:-]\s*(.*)$", line)
        if heading_match:
            current_section = heading_match.group(1)
            found_heading = True
            remainder = heading_match.group(2).strip()
            if remainder and remainder.lower() != "none" and not _is_heading_description(remainder):
                parsed[current_section].append(remainder)
            continue

        if current_section is None:
            continue

        cleaned = re.sub(r"^\s*[-*]\s*", "", line).strip()
        if cleaned.lower() == "none":
            continue
        parsed[current_section].append(cleaned)

    if critique_text.strip() and not found_heading:
        parsed["A"].append("Critic output was unstructured; manual review required.")

    return parsed


def has_blocking_critic_issues(parsed_critique: Mapping[str, list[str]]) -> bool:
    """Return True when category A contains blocking issues."""
    return bool(parsed_critique.get("A"))


def load_profile_facts(path: Path | None) -> dict[str, Any]:
    """Load profile facts from YAML. Missing file returns empty dict."""
    if path is None or not path.exists():
        return {}

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("profile_facts.yaml must have a top-level mapping.")
    return data


def _flatten_facts(prefix: str, value: Any, lines: list[str]) -> None:
    if isinstance(value, dict):
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else key
            _flatten_facts(child_prefix, value[key], lines)
        return

    if isinstance(value, list):
        if not value:
            lines.append(f"- {prefix}: []")
            return
        for item in value:
            if isinstance(item, (dict, list)):
                _flatten_facts(prefix, item, lines)
            else:
                lines.append(f"- {prefix}: {item}")
        return

    lines.append(f"- {prefix}: {value}")


def format_profile_facts_for_prompt(profile_facts: Mapping[str, Any]) -> str:
    """Flatten profile facts into deterministic plain text for prompts."""
    if not profile_facts:
        return "- None provided"

    lines: list[str] = []
    for key in sorted(profile_facts):
        _flatten_facts(key, profile_facts[key], lines)
    return "\n".join(lines)


def extract_readable_text_from_html(html: str) -> str:
    """Extract simplified readable text from raw HTML."""
    content = re.sub(r"<script\b[^>]*>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    content = re.sub(r"<style\b[^>]*>.*?</style>", " ", content, flags=re.IGNORECASE | re.DOTALL)
    content = re.sub(r"</(p|div|li|h[1-6]|br)>", "\n", content, flags=re.IGNORECASE)
    content = re.sub(r"<[^>]+>", " ", content)
    content = unescape(content)
    content = content.replace("\xa0", " ")
    content = re.sub(r"[ \t]+", " ", content)
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()
