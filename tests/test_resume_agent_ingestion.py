import json

import resume_agent


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_extract_ashby_job_text_via_posting_api_prefers_plain_text(monkeypatch) -> None:
    def fake_urlopen(request, timeout=15):  # noqa: ARG001
        assert request.full_url == "https://api.ashbyhq.com/posting-api/job-board/recraft"
        payload = {
            "jobs": [
                {
                    "title": "ML Engineer",
                    "location": "Remote",
                    "jobUrl": "https://jobs.ashbyhq.com/recraft/f9c15249-88f1-4e68-8eaf-03fff97971e5",
                    "descriptionPlain": (
                        "Build production-grade generative image systems. "
                        "Collaborate with product, infra, and applied research to improve quality and speed."
                    ),
                }
            ]
        }
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(resume_agent, "urlopen", fake_urlopen)

    text = resume_agent._extract_ashby_job_text_via_posting_api(
        "https://jobs.ashbyhq.com/recraft/f9c15249-88f1-4e68-8eaf-03fff97971e5"
    )
    assert "Title: ML Engineer" in text
    assert "Build production-grade generative image systems." in text


def test_extract_ashby_job_text_via_posting_api_falls_back_to_html(monkeypatch) -> None:
    def fake_urlopen(request, timeout=15):  # noqa: ARG001
        assert request.full_url == "https://api.ashbyhq.com/posting-api/job-board/recraft"
        payload = {
            "jobs": [
                {
                    "title": "Platform Engineer",
                    "applyUrl": "https://jobs.ashbyhq.com/recraft/f9c15249-88f1-4e68-8eaf-03fff97971e5/apply",
                    "descriptionPlain": "Short",
                    "descriptionHtml": (
                        "<h1>Platform Engineer</h1><p>Build distributed systems with strong reliability focus, "
                        "ship production features, and collaborate across teams.</p>"
                    ),
                }
            ]
        }
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(resume_agent, "urlopen", fake_urlopen)

    text = resume_agent._extract_ashby_job_text_via_posting_api(
        "https://jobs.ashbyhq.com/recraft/f9c15249-88f1-4e68-8eaf-03fff97971e5"
    )
    assert "Platform Engineer" in text
    assert "Build distributed systems with strong reliability focus" in text


def test_fetch_jd_text_from_url_uses_ashby_api_first(monkeypatch) -> None:
    called_urls: list[str] = []

    def fake_urlopen(request, timeout=15):  # noqa: ARG001
        called_urls.append(request.full_url)
        if request.full_url == "https://api.ashbyhq.com/posting-api/job-board/recraft":
            payload = {
                "jobs": [
                    {
                        "jobUrl": "https://jobs.ashbyhq.com/recraft/f9c15249-88f1-4e68-8eaf-03fff97971e5",
                        "descriptionPlain": (
                            "Design, build, and ship product-quality machine learning workflows in production. "
                            "Partner with engineering and design to iterate quickly."
                        ),
                    }
                ]
            }
            return _FakeResponse(json.dumps(payload).encode("utf-8"))
        raise AssertionError("HTML fallback should not be called when Ashby API succeeds.")

    monkeypatch.setattr(resume_agent, "urlopen", fake_urlopen)

    text = resume_agent.fetch_jd_text_from_url(
        "https://jobs.ashbyhq.com/recraft/f9c15249-88f1-4e68-8eaf-03fff97971e5"
    )
    assert "Design, build, and ship product-quality machine learning workflows" in text
    assert called_urls == ["https://api.ashbyhq.com/posting-api/job-board/recraft"]
