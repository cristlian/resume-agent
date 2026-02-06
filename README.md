# Resume Tailoring Agent

A CLI tool that optimizes your resume for a target job description using Gemini AI.

## Features

- Phase 0-aligned JD ingestion:
  - `--jd-url` (fetch HTML and extract text, fallback to pasted text)
  - `--jd-text` (direct inline JD text)
  - `--jd-pdf` (existing PDF flow)
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

   Or pass directly:
   ```
   python resume_agent.py --api-key "your-api-key-here" ...
   ```

## Usage

```bash
python resume_agent.py --jd-url "<job-posting-url>" --resume-pdf <path-to-your-resume.pdf>
```

### Examples

```bash
python resume_agent.py --jd-pdf "JD/VSIM_Job_Description.pdf" --resume-pdf "Yifei-Lian-Resume-merged.pdf"
python resume_agent.py --jd-text "We are hiring a robotics ML engineer..." --resume-pdf "Yifei-Lian-Resume-merged.pdf"
python resume_agent.py --jd-url "https://example.com/job/123" --resume-pdf "Yifei-Lian-Resume-merged.pdf"
```

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
