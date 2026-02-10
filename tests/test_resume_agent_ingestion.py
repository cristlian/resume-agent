import resume_agent
import pytest


def test_normalize_pasted_jd_text_preserves_internal_blank_lines() -> None:
    raw = "\r\n  Title Line\r\n\r\nRequirement A\r\nRequirement B\r\n\r\n"
    normalized = resume_agent._normalize_pasted_jd_text(raw)
    assert normalized == "Title Line\n\nRequirement A\nRequirement B"


def test_prompt_for_jd_text_terminal_accepts_empty_lines(monkeypatch) -> None:
    entries = iter(
        [
            "Senior ML Engineer",
            "",
            "Responsibilities",
            "- Build systems",
            "END_JD",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda: next(entries))

    pasted = resume_agent.prompt_for_jd_text_terminal()
    assert pasted == "Senior ML Engineer\n\nResponsibilities\n- Build systems"


def test_prompt_for_jd_text_auto_prefers_web(monkeypatch) -> None:
    monkeypatch.setattr(resume_agent, "prompt_for_jd_text_web", lambda source_url=None: "JD from web")
    monkeypatch.setattr(resume_agent, "prompt_for_jd_text_terminal", lambda end_marker="END_JD": "JD from terminal")

    pasted = resume_agent.prompt_for_jd_text("auto", source_url="https://example.com/job")
    assert pasted == "JD from web"


def test_prompt_for_jd_text_auto_falls_back_to_terminal(monkeypatch) -> None:
    monkeypatch.setattr(resume_agent, "prompt_for_jd_text_web", lambda source_url=None: "")
    monkeypatch.setattr(resume_agent, "prompt_for_jd_text_terminal", lambda end_marker="END_JD": "JD from terminal")

    pasted = resume_agent.prompt_for_jd_text("auto")
    assert pasted == "JD from terminal"


def test_prompt_for_jd_text_web_mode_does_not_fallback(monkeypatch) -> None:
    monkeypatch.setattr(resume_agent, "prompt_for_jd_text_web", lambda source_url=None: "")
    monkeypatch.setattr(resume_agent, "prompt_for_jd_text_terminal", lambda end_marker="END_JD": "JD from terminal")

    pasted = resume_agent.prompt_for_jd_text("web")
    assert pasted == ""


def test_prompt_for_jd_text_invalid_mode_raises() -> None:
    with pytest.raises(ValueError):
        resume_agent.prompt_for_jd_text("invalid")
