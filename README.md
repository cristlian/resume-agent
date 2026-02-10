# Resume Tailoring Agent

A CLI tool that optimizes your resume for a target job description using Gemini AI.

## Features

- Fast default run path:
  - `python resume_agent.py` (uses default merged resume PDF and JD paste UI)
- Phase 0-aligned JD ingestion:
  - Local web paste form (default; terminal fallback)
  - `--jd-url` (context-only metadata for labeling/role naming; no fetch)
  - `--jd-text` (direct inline JD text)
  - `--jd-pdf` (optional PDF alternative)
  - `--jd-input-mode` (`auto`, `web`, `terminal`)
- Runtime reliability hardening:
  - Accepts `GEMINI_API_KEY` or `GOOGLE_API_KEY`
  - Sanitizes broken loopback proxy env vars for Gemini API calls in-process
- Dual-agent architecture: Architect (writer) and Critic (strict reviewer)
- Structured Critic categories:
  - `A) Needs user input (blocking)`
  - `B) Auto-fixable`
  - `C) Already covered`
  - `D) Optional`
- Conditional user pause: prompts for clarification only when section `A` is non-empty
- Profile facts support via `profile_facts.yaml`
- Plain-text enforcement for saved outputs (markdown-like formatting is normalized)

## Requirements

- Python 3.10+
- Gemini API key

## Installation

1. Navigate to this folder:
   ```
   cd C:\Users\cristlian\resume-agent
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set your Gemini API key:

   PowerShell:
   ```
   $env:GEMINI_API_KEY = "your-api-key-here"
   ```
   or
   ```
   $env:GOOGLE_API_KEY = "your-api-key-here"
   ```

   Or pass directly:
   ```
   python resume_agent.py --api-key "your-api-key-here" ...
   ```

## Usage

```bash
python resume_agent.py
```

### Examples

```bash
python resume_agent.py
python resume_agent.py --jd-input-mode terminal --resume-pdf "Yifei-Lian-Resume-merged.pdf"
python resume_agent.py --jd-text "We are hiring a robotics ML engineer..." --resume-pdf "Yifei-Lian-Resume-merged.pdf"
python resume_agent.py --jd-pdf "JD/VSIM_Job_Description.pdf" --resume-pdf "Yifei-Lian-Resume-merged.pdf"
python resume_agent.py --jd-url "https://jobs.gem.com/..." --resume-pdf "Yifei-Lian-Resume-merged.pdf"
```

### JD Input UX

- Default behavior opens a temporary local web page (`127.0.0.1`) with a large textarea for multi-line paste.
- Empty lines are preserved in web mode.
- If web mode is unavailable (or times out in `auto` mode), the script falls back to terminal mode.
- Terminal mode uses an explicit end marker: type `END_JD` on its own line to submit.
- `--jd-url` is optional metadata only; JD content still comes from paste text unless `--jd-text` or `--jd-pdf` is provided.

## Workflow

1. Draft: Architect tailors a one-page resume from JD + resume + profile facts.
2. Critique: Critic outputs structured `A/B/C/D` findings.
3. Conditional clarification: user input is requested only if section `A` has blocking issues.
4. Finalize: Architect rewrites with critique + optional user input.
5. Artifacts: text and docx outputs are saved.

## Output Files

- `Final_Resume.txt` - final tailored resume (plain text)
- `Draft_Resume.txt` - initial draft
- `Critique.txt` - raw critic output
- `Critique_Structured.json` - parsed `A/B/C/D` categories
- `tailored_resumes/<role>_<date>.docx` - Word artifact

## Profile Facts

If `profile_facts.yaml` exists in repo root, it is auto-loaded.
You can also set a custom path:

```bash
python resume_agent.py --profile-facts path/to/profile_facts.yaml ...
```

## Notes

- Default resume path is `Yifei-Lian-Resume-merged.pdf` in repo root.
- To preserve loopback proxies for debugging, set `RESUME_AGENT_ALLOW_LOOPBACK_PROXY=1`.
