# Resume Tailoring Agent

A CLI tool that optimizes your resume for a target job description using Gemini AI.

## Features

- **4-Phase Workflow**: Draft -> Critique -> Human Input -> Final Resume
- **PDF Input**: Accepts job description and resume as PDF files
- **Dual-Agent Architecture**: Architect (optimizer) and Critic (brutal reviewer) with isolated contexts
- **ATS Optimization**: Keyword extraction and alignment for Applicant Tracking Systems
- **Plain Text Output**: Clean, ATS-friendly format without markdown

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

   **PowerShell:**
   ```
   $env:GEMINI_API_KEY = "your-api-key-here"
   ```

   **Or pass directly:**
   ```
   python resume_agent.py --api-key "your-api-key-here" ...
   ```

## Usage

```
python resume_agent.py --jd-pdf <path-to-job-description.pdf> --resume-pdf <path-to-your-resume.pdf>
```

### Example

```
python resume_agent.py --jd-pdf "Job_Description.pdf" --resume-pdf "My_Resume.pdf"
```

## Workflow

1. **Phase 1 - Draft**: The Architect agent analyzes both PDFs and creates an initial tailored resume
2. **Phase 2 - Critique**: The Critic agent identifies 5 specific problems with the draft
3. **Phase 3 - Human Input**: You can optionally provide clarifications or additional instructions
4. **Phase 4 - Final**: The Architect incorporates feedback and produces the final optimized resume

## Output Files

- Final_Resume.txt - Your tailored resume (main output)
- Draft_Resume.txt - Initial draft from Phase 1
- Critique.txt - Feedback from Phase 2
