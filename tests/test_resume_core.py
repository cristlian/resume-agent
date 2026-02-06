from pathlib import Path

import pytest

from resume_core import (
    build_critic_prompt,
    check_for_markdown,
    enforce_plain_text,
    extract_length_constraint,
    extract_readable_text_from_html,
    extract_role_from_jd_filename,
    format_profile_facts_for_prompt,
    has_blocking_critic_issues,
    load_profile_facts,
    parse_structured_critic,
)


def test_check_for_markdown_detects_patterns() -> None:
    text = "# Header\n```python\nx=1\n```\nNormal\n**Bold** and __also__"
    warnings = check_for_markdown(text)
    assert len(warnings) == 4
    assert "heading" in warnings[0].lower()


def test_enforce_plain_text_removes_markdown_and_normalizes() -> None:
    text = "# Title\r\n\r\n* item\n```code```\n**Bold** __Text__\n\n\nTail"
    cleaned = enforce_plain_text(text)
    assert "```" not in cleaned
    assert "# " not in cleaned
    assert "**" not in cleaned
    assert "__" not in cleaned
    assert cleaned.startswith("Title")
    assert "- item" in cleaned
    assert "\n\n\n" not in cleaned


def test_extract_role_from_jd_filename_cleans_suffixes() -> None:
    path = Path("Forward_Deployed_Engineer_Job_Description.pdf")
    assert extract_role_from_jd_filename(path) == "Forward Deployed Engineer"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Tell me in two sentences why", "exactly 2 sentence(s)"),
        ("Please answer in 120 words", "approximately 120 words"),
        ("Keep it max 80 words", "maximum 80 words"),
        ("Maximum 200 characters please", "maximum 200 characters"),
        ("Describe in one paragraph", "exactly 1 paragraph(s)"),
        ("Briefly describe this", "2-3 sentences (keep it brief)"),
        ("Explain concisely", "3-4 sentences (be concise)"),
        ("Why this role?", "max 2 short paragraphs"),
    ],
)
def test_extract_length_constraint(question: str, expected: str) -> None:
    assert extract_length_constraint(question) == expected


def test_build_critic_prompt_contains_required_sections() -> None:
    prompt = build_critic_prompt("DRAFT", "- location: London")
    assert "A) Needs user input (blocking)" in prompt
    assert "B) Auto-fixable" in prompt
    assert "C) Already covered" in prompt
    assert "D) Optional" in prompt
    assert "DRAFT" in prompt
    assert "location: London" in prompt


def test_parse_structured_critic_with_headings_and_none() -> None:
    critique = """
intro line ignored
A) Missing graduation month
- Need exact month
B) - tighten ATS keywords
- None
C) none
D) Optional extras
* Add volunteering section
"""
    parsed = parse_structured_critic(critique)
    assert parsed["A"] == ["Missing graduation month", "Need exact month"]
    assert parsed["B"] == ["- tighten ATS keywords"]
    assert parsed["C"] == []
    assert parsed["D"] == ["Optional extras", "Add volunteering section"]


def test_parse_structured_critic_heading_descriptions_stripped() -> None:
    """Heading descriptions like 'Needs user input (blocking)' must not become findings."""
    critique = """A) Needs user input (blocking)
- None

B) Auto-fixable
- Fix some keywords

C) Already covered (Profile Facts / evidence)
- Contact info

D) Optional
- Add volunteering
"""
    parsed = parse_structured_critic(critique)
    assert parsed["A"] == []
    assert has_blocking_critic_issues(parsed) is False
    assert parsed["B"] == ["Fix some keywords"]
    assert parsed["C"] == ["Contact info"]
    assert parsed["D"] == ["Add volunteering"]


def test_parse_structured_critic_unstructured_falls_back_to_blocking() -> None:
    parsed = parse_structured_critic("No valid headings at all")
    assert parsed["A"] == ["Critic output was unstructured; manual review required."]
    assert has_blocking_critic_issues(parsed) is True


def test_has_blocking_critic_issues_false_when_empty() -> None:
    assert has_blocking_critic_issues({"A": []}) is False
    assert has_blocking_critic_issues({"B": ["x"]}) is False


def test_load_profile_facts_missing_or_none(tmp_path: Path) -> None:
    assert load_profile_facts(None) == {}

    missing = tmp_path / "missing.yaml"
    assert load_profile_facts(missing) == {}

    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert load_profile_facts(empty) == {}


def test_load_profile_facts_invalid_top_level_raises(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("- item\n- item2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_profile_facts(invalid)


def test_load_profile_facts_valid_mapping(tmp_path: Path) -> None:
    valid = tmp_path / "valid.yaml"
    valid.write_text("location: London\nlinks:\n  github: github.com/me\n", encoding="utf-8")
    data = load_profile_facts(valid)
    assert data["location"] == "London"
    assert data["links"]["github"] == "github.com/me"


def test_format_profile_facts_for_prompt_empty_and_nested() -> None:
    assert format_profile_facts_for_prompt({}) == "- None provided"

    facts = {
        "links": ["github.com/x", {"portfolio": "site.dev"}],
        "location": "London",
        "skills": [],
    }
    formatted = format_profile_facts_for_prompt(facts)
    assert "- links: github.com/x" in formatted
    assert "- links.portfolio: site.dev" in formatted
    assert "- location: London" in formatted
    assert "- skills: []" in formatted


def test_extract_readable_text_from_html_strips_noise() -> None:
    html = """
<html>
  <head>
    <style>.x{color:red}</style>
    <script>console.log('x')</script>
  </head>
  <body>
    <h1>Senior Engineer&nbsp;</h1>
    <p>Build systems &amp; infrastructure.</p>
    <div>Remote friendly</div>
  </body>
</html>
"""
    text = extract_readable_text_from_html(html)
    assert "console.log" not in text
    assert ".x{color:red}" not in text
    assert "Senior Engineer" in text
    assert "Build systems & infrastructure." in text
    assert "Remote friendly" in text
