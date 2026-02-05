#!/usr/bin/env python3
"""
Resume Tailoring Agent - Optimizes resumes for target job descriptions using Gemini.

Usage:
    python resume_agent.py [--jd-pdf JD.pdf] [--resume-pdf Resume.pdf] [--api-key KEY]

By default:
    - JD: fetches the latest PDF from ./JD folder (by creation time)
    - Resume: uses ./Yifei-Lian-Resume-merged.pdf
"""

import argparse
import os
import re
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

from google import genai
from google.genai import types
import pypdf
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
import datetime

# === CONFIGURATION ===
MODEL_NAME = "gemini-3-pro-preview"
WORKSPACE_DIR = Path(__file__).parent.resolve()
DEFAULT_RESUME = WORKSPACE_DIR / "Yifei-Lian-Resume-merged.pdf"
JD_FOLDER = WORKSPACE_DIR / "JD"
TAILORED_RESUMES_FOLDER = WORKSPACE_DIR / "tailored_resumes"
APPLICATION_RESPONSES_FOLDER = WORKSPACE_DIR / "application_question_response"
MAX_RETRIES = 4
RETRY_BASE_SECONDS = 2
RETRY_MAX_SECONDS = 20

# === PERFORMANCE UTILITIES ===
@contextmanager
def timed_section(name: str):
    """Context manager to time and report section duration."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"  [TIMING] {name}: {elapsed:.2f}s")


class Spinner:
    """Minimal CLI spinner for long-running operations."""
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    
    def __init__(self, message: str = "Processing"):
        self.message = message
        self._stop_event = threading.Event()
        self._thread = None
    
    def _spin(self):
        idx = 0
        while not self._stop_event.is_set():
            frame = self.FRAMES[idx % len(self.FRAMES)]
            print(f"\r  {frame} {self.message}...", end="", flush=True)
            idx += 1
            time.sleep(0.1)
        # Clear spinner line
        print(f"\r" + " " * (len(self.message) + 10) + "\r", end="", flush=True)
    
    def __enter__(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self
    
    def __exit__(self, *args):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.5)


def parallel_extract_pdfs(jd_path: Path, resume_path: Path) -> tuple[str | types.Part, str | types.Part]:
    """Extract text from both PDFs in parallel using thread pool."""
    results = {}
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(load_pdf_content, jd_path): 'jd',
            executor.submit(load_pdf_content, resume_path): 'resume'
        }
        for future in as_completed(futures):
            key = futures[future]
            results[key] = future.result()
    
    return results['jd'], results['resume']


# === SYSTEM INSTRUCTIONS ===
ARCHITECT_SYSTEM_INSTRUCTION = """<role>

You are Gemini 3, a specialized assistant for Resume Optimization and ATS Alignment.

You are precise, analytical, and persistent.

</role>



<instructions>

1. **Plan**: Analyze the job description and resume variants to determine the best alignment.

2. **Execute**: Select and rephrase the most relevant achievements, skills, and experiences from the provided resume(s) to match the target role.

3. **Validate**: Ensure all extracted information is factual, user-provided, and directly matches job requirements without exaggeration or inference.

4. **Format**: Output a clean, well-structured one-page resume with standardized sections, appropriate formatting, and professional clarity.

</instructions>



<constraints>

- Verbosity: Medium

- Tone: Formal, Technical

- OUTPUT FORMAT: Plain text only. NO markdown syntax whatsoever.
   - Use UPPERCASE for section headers
   - Use simple dashes or asterisks for bullet points, not markdown-style
   - No # symbols for headings
   - No ``` code blocks
   - No **bold** or *italic* formatting

</constraints>



<output_format>

Structure your response as follows:

1. **Executive Summary**: Brief overview of tailoring logic-why certain experiences were selected and how they match the target role.

2. **Tailored Resume**:

   - Name / Phone / Email / GitHub

   - Work Experience (chronological, most relevant content prioritized)

   - Project Experience (concise, impact-focused)

   - Education (institution, degree, dates)

   - Professional Skills (tools, languages, systems)

</output_format>"""


def get_latest_jd_pdf() -> Path | None:
    """Find the most recently created PDF in the JD folder."""
    if not JD_FOLDER.exists():
        return None
    
    pdf_files = list(JD_FOLDER.glob("*.pdf"))
    if not pdf_files:
        return None
    
    # Sort by creation time (st_ctime), most recent first
    latest = max(pdf_files, key=lambda p: p.stat().st_ctime)
    return latest


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tailor a resume to a job description using Gemini AI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Defaults:
  --jd-pdf      Latest PDF from {JD_FOLDER}
  --resume-pdf  {DEFAULT_RESUME}
""",
    )
    parser.add_argument(
        "--jd-pdf",
        required=False,
        help="Path to the job description PDF (default: latest from JD folder)",
    )
    parser.add_argument(
        "--resume-pdf",
        required=False,
        help=f"Path to the merged resume PDF (default: {DEFAULT_RESUME.name})",
    )
    parser.add_argument(
        "--api-key",
        help="Gemini API key (overrides GEMINI_API_KEY env var)",
    )
    return parser.parse_args()


def get_api_key(cli_key: str | None) -> str:
    """Get API key from CLI arg or environment variable."""
    if cli_key:
        return cli_key
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key
    print("ERROR: No API key provided.", file=sys.stderr)
    print("Set GEMINI_API_KEY environment variable or use --api-key argument.", file=sys.stderr)
    sys.exit(1)


def validate_pdf_path(path: Path, label: str) -> Path:
    """Validate that the path exists, is a PDF, and is non-empty."""
    if not path.exists():
        print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
        sys.exit(1)
    if not path.is_file():
        print(f"ERROR: {label} is not a file: {path}", file=sys.stderr)
        sys.exit(1)
    if path.suffix.lower() != ".pdf":
        print(f"ERROR: {label} must be a PDF file (got: {path.suffix})", file=sys.stderr)
        sys.exit(1)
    if path.stat().st_size == 0:
        print(f"ERROR: {label} file is empty: {path}", file=sys.stderr)
        sys.exit(1)
    return path


def extract_text_from_pdf(path: Path) -> str:
    """Extract text content from a PDF file."""
    try:
        reader = pypdf.PdfReader(path)
        text = []
        for page in reader.pages:
            text.append(page.extract_text() or "")
        return "\n".join(text)
    except Exception as e:
        print(f"WARNING: Failed to extract text from {path.name}: {e}\nFalling back to PDF input.", file=sys.stderr)
        return None

def load_pdf_content(path: Path) -> str | types.Part:
    """Load PDF content: try text extraction first, fall back to Part."""
    text_content = extract_text_from_pdf(path)
    if text_content and len(text_content.strip()) > 50:
        return f"--- DOCUMENT: {path.name} ---\n{text_content}\n--- END DOCUMENT ---\n"
    
    # Fallback to binary Part if extraction fails or is empty
    with open(path, "rb") as f:
        pdf_bytes = f.read()
    return types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")


def check_for_markdown(text: str) -> list[str]:
    """Check for markdown-like patterns in text. Returns list of warnings."""
    warnings = []
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


def send_with_retry(chat: object, message: object, label: str) -> object:
    """Send a message with retry/backoff on quota or rate-limit errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return chat.send_message(message)
        except Exception as e:
            error_msg = str(e).lower()
            is_rate_limit = any(token in error_msg for token in ["quota", "rate", "429", "resource_exhausted"])
            if not is_rate_limit or attempt == MAX_RETRIES:
                raise
            sleep_seconds = min(RETRY_BASE_SECONDS ** attempt, RETRY_MAX_SECONDS)
            print(
                f"WARNING: {label} hit a rate/quota limit. Retrying in {sleep_seconds}s..."
            )
            time.sleep(sleep_seconds)


def print_section(title: str, content: str) -> None:
    """Print a clearly labeled section to terminal."""
    print(f"\n{'=' * 60}")
    print(f"=== {title} ===")
    print("=" * 60)
    print(content)
    print()


def run_architect_draft(
    client: genai.Client,
    jd_content: str | types.Part,
    resume_content: str | types.Part,
) -> tuple[object, str]:
    """Run the Architect agent to create initial draft. Returns (chat, draft_text)."""
    
    draft_prompt = """<context>

You will receive two inputs:

1. A document containing multiple tailored versions of my resume-each optimized for roles in data engineering, DevOps, AI/ML, and general software engineering.

2. A target job description I am applying to.



Each resume includes only real, previously written content about my academic, project, and internship experience. No fictional elements are allowed.



</context>



<task>

Your task is to:

- Parse and analyze the job description for keywords, required qualifications, and core responsibilities.

- Cross-reference this with the content in my resume versions.

- Extract the most relevant and high-impact items (work, projects, skills) that align with the job.

- Rewrite and reformat them into a **single one-page resume** tailored to the target role.



You may:

- Rephrase or restructure bullet points for clarity and relevance

- Merge elements from different resume versions

- Adjust section ordering for stronger alignment



You may NOT:

- Fabricate or infer experience

- Add skills, tools, or projects that were not explicitly mentioned in the original documents



The final resume must include:

- Name / Phone / Email / GitHub

- Work Experience

- Project Experience

- Education

- Professional Skills



Make it ready to pass both ATS filters and human recruiters.



</task>



<final_instruction>

Think step-by-step before answering. Prioritize precision, factual relevance, and clear structure.

Output in PLAIN TEXT format only - no markdown syntax.

</final_instruction>"""

    chat = client.chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(
            system_instruction=ARCHITECT_SYSTEM_INSTRUCTION,
            temperature=0.3,
            top_p=0.95,
            max_output_tokens=8192,
        ),
    )
    
    response = send_with_retry(chat, [jd_content, resume_content, draft_prompt], "Architect draft")
    
    return chat, response.text


def run_critic(client: genai.Client, draft_resume: str) -> str:
    """Run the Critic agent to brutally evaluate the draft resume."""
    
    critic_prompt = f"""RESUME TO EVALUATE:

{draft_resume}

---

Evaluate this resume from the perspective of a top-tier recruiting expert. Your task:

- Provide 5 real and brutally honest reasons why this resume would fail screening

- Explain the cause and consequence of each issue

- Provide concrete and actionable directions for improvement

If additional information is required to improve the resume:

- Explicitly state what is missing

- Explain why it is necessary

! No flattery. No reassurance. No agreement.

Facts, logic, and judgment only.

Output in PLAIN TEXT format only - no markdown syntax."""

    chat = client.chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(
            temperature=0.4,
            top_p=0.95,
            max_output_tokens=4096,
        ),
    )
    
    response = send_with_retry(chat, critic_prompt, "Critic review")
    return response.text


def run_architect_final(
    architect_chat: object,
    critique: str,
    human_input: str,
) -> str:
    """Run the Architect agent to produce final resume incorporating feedback."""
    
    additional_info_section = ""
    if human_input.strip():
        additional_info_section = f"""
Additional information (if any):

{human_input}"""
    else:
        additional_info_section = "\nAdditional information (if any):\n\n[None provided]"
    
    final_prompt = f"""CRITIQUE FROM RECRUITER:

{critique}

---

Based on all feedback and critiques provided above, rewrite this resume.

Objectives:

- Fully ATS-compliant

- Incorporate role-relevant keywords from the JD

- Clearly reflect required hard skills and soft skills

- Quantify outcomes wherever possible

Rules:

- Do NOT add unverified experiences

- Do NOT soften issues

- Avoid vague or generic phrasing
{additional_info_section}

Output ONLY the final resume in PLAIN TEXT format - no markdown syntax, no explanations, no preamble."""

    response = send_with_retry(architect_chat, final_prompt, "Architect final")
    return response.text


def get_human_input() -> str:
    """Prompt for and collect human clarifications."""
    print("\n" + "=" * 60)
    print("=== USER INPUT REQUIRED ===")
    print("=" * 60)
    print("Review the draft and critique above.")
    print("Enter any clarifications or additional instructions for the final resume.")
    print("(Press Enter twice to submit, or just Enter to skip)")
    print("-" * 40)
    
    lines = []
    empty_count = 0
    
    try:
        while True:
            line = input()
            if line == "":
                empty_count += 1
                if empty_count >= 1 and not lines:
                    break
                if empty_count >= 1 and lines:
                    break
                lines.append("")
            else:
                empty_count = 0
                lines.append(line)
    except EOFError:
        pass
    
    return "\n".join(lines).strip()


def save_output(filename: str, content: str) -> None:
    """Save content to a file in the workspace directory."""
    path = WORKSPACE_DIR / filename
    path.write_text(content, encoding="utf-8")
    print(f"Saved: {path}")


def extract_role_from_jd_filename(jd_path: Path) -> str:
    """Extract a clean role name from the JD filename."""
    # Remove extension and common suffixes
    name = jd_path.stem
    # Remove common suffixes like _Job_Description, -JD, etc.
    for suffix in ["_Job_Description", "_JD", "-JD", "_job_description", " Job Description"]:
        name = name.replace(suffix, "")
    # Replace underscores and multiple spaces with single space
    name = re.sub(r"[_-]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def save_resume_as_word(content: str, role_name: str) -> Path:
    """Convert plain text resume to Word document and save to tailored_resumes folder."""
    # Ensure folder exists
    TAILORED_RESUMES_FOLDER.mkdir(exist_ok=True)
    
    # Create Word document
    doc = Document()
    
    # Set default font
    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(11)
    
    # Parse and add content
    lines = content.split('\n')
    for line in lines:
        stripped = line.strip()
        if not stripped:
            # Empty line - add paragraph break
            doc.add_paragraph()
        elif stripped.isupper() and len(stripped) > 2:
            # Section header (all caps)
            p = doc.add_paragraph()
            run = p.add_run(stripped)
            run.bold = True
            run.font.size = Pt(12)
        elif stripped.startswith('- '):
            # Bullet point
            doc.add_paragraph(stripped[2:], style='List Bullet')
        elif '|' in stripped and lines.index(line) < 5:
            # Contact info line (name/email/phone)
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(stripped)
            if lines.index(line) == 0 or (len(stripped) > 10 and stripped[0].isupper()):
                run.bold = True
                run.font.size = Pt(14)
        else:
            # Regular text
            doc.add_paragraph(stripped)
    
    # Generate filename with role and date
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    safe_role = re.sub(r'[^\w\s-]', '', role_name).strip().replace(' ', '_')
    filename = f"{safe_role}_{date_str}.docx"
    filepath = TAILORED_RESUMES_FOLDER / filename
    
    # Save document
    doc.save(filepath)
    print(f"Saved Word document: {filepath}")
    return filepath


def get_application_questions() -> list[str]:
    """Prompt user to enter application questions. Returns list of questions."""
    print("\n" + "=" * 60)
    print("=== APPLICATION QUESTIONS ===")
    print("=" * 60)
    print("Enter your application questions (one per line).")
    print("Press Enter on an empty line when done, or type 'skip' to skip.")
    print("-" * 40)
    
    questions = []
    question_num = 1
    while True:
        try:
            line = input(f"Q{question_num}: ").strip()
        except EOFError:
            break
        
        if line.lower() == 'skip':
            return []
        if not line:
            if questions:  # Only break if we have at least one question
                break
            continue  # Ignore empty first line
        
        questions.append(line)
        question_num += 1
    
    return questions


def run_application_response(
    client: genai.Client,
    jd_content: str,
    resume: str,
    questions: list[str],
) -> str:
    """Generate high-quality responses to application questions."""
    
    questions_formatted = "\n".join([f"[Question {i+1}] {q}" for i, q in enumerate(questions)])
    
    prompt = f"""Based on the latest resume and JD, draft high quality responses to these job application questions/cover letter:

{questions_formatted}

You must strictly follow the below instructions:
- Tone: passionate, sincere, natural, fluent, persuasive, confident
- Length: max 2 short paragraphs per question
- Output format: Plain text only (NO markdown, NO asterisks, NO bullet points)
- Structure each response with a clear header like "QUESTION 1: [question text]" followed by the response

---

JOB DESCRIPTION:
{jd_content}

---

RESUME:
{resume}
"""
    
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=4096,
        ),
    )
    
    return response.text


def save_application_responses(content: str, role_name: str) -> Path:
    """Save application question responses to a txt file."""
    # Ensure folder exists
    APPLICATION_RESPONSES_FOLDER.mkdir(exist_ok=True)
    
    # Generate filename with role and date
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    safe_role = re.sub(r'[^\w\s-]', '', role_name).strip().replace(' ', '_')
    filename = f"{safe_role}_responses_{date_str}.txt"
    filepath = APPLICATION_RESPONSES_FOLDER / filename
    
    # Add header for easy copy-paste
    header = f"""{'=' * 60}
APPLICATION RESPONSES
Role: {role_name}
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
{'=' * 60}

"""
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(header + content)
    
    print(f"Saved responses: {filepath}")
    return filepath


def main() -> None:
    """Main entry point."""
    total_start = time.perf_counter()
    
    args = parse_args()
    
    # Get and validate API key
    api_key = get_api_key(args.api_key)
    
    # Resolve JD path
    if args.jd_pdf:
        jd_path = Path(args.jd_pdf)
    else:
        jd_path = get_latest_jd_pdf()
        if jd_path is None:
            print(f"ERROR: No PDF files found in JD folder: {JD_FOLDER}", file=sys.stderr)
            print("Either add a JD PDF to the folder or use --jd-pdf argument.", file=sys.stderr)
            sys.exit(1)
        print(f"Auto-detected latest JD: {jd_path.name}")
    
    # Resolve resume path
    if args.resume_pdf:
        resume_path = Path(args.resume_pdf)
    else:
        resume_path = DEFAULT_RESUME
        print(f"Using default resume: {resume_path.name}")
    
    # Validate paths
    jd_path = validate_pdf_path(jd_path, "Job Description")
    resume_path = validate_pdf_path(resume_path, "Resume")
    
    print("-" * 60)
    print(f"Job Description: {jd_path}")
    print(f"Resume: {resume_path}")
    print(f"Model: {MODEL_NAME}")
    print("-" * 60)
    
    # Load PDFs in PARALLEL (optimized)
    print("Loading PDF files (parallel extraction)...")
    with timed_section("PDF extraction"):
        jd_content, resume_content = parallel_extract_pdfs(jd_path, resume_path)
    
    # Initialize Gemini client
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"ERROR: Failed to initialize Gemini client: {e}", file=sys.stderr)
        sys.exit(1)
    
    # === PHASE 1: Architect Draft ===
    print("\n[Phase 1/4] Architect creating initial draft...")
    with Spinner("Gemini drafting resume"), timed_section("Architect Draft API"):
        try:
            architect_chat, draft_response = run_architect_draft(client, jd_content, resume_content)
        except Exception as e:
            error_msg = str(e).lower()
            if "api key" in error_msg or "authentication" in error_msg or "401" in error_msg:
                print(f"ERROR: Authentication failed. Check your API key.", file=sys.stderr)
            elif "quota" in error_msg or "429" in error_msg:
                print(f"ERROR: API quota exceeded. Try again later.", file=sys.stderr)
            elif "model" in error_msg or "404" in error_msg:
                print(f"ERROR: Model '{MODEL_NAME}' not found or unavailable.", file=sys.stderr)
            else:
                print(f"ERROR: API call failed: {e}", file=sys.stderr)
            sys.exit(1)
    
    print_section("DRAFT RESUME", draft_response)
    
    # Check for markdown in draft
    md_warnings = check_for_markdown(draft_response)
    if md_warnings:
        print("WARNING: Markdown-like formatting detected in draft:")
        for w in md_warnings[:5]:
            print(f"  - {w}")
        print()
    
    # Save draft
    save_output("Draft_Resume.txt", draft_response)
    
    # === PHASE 2: Critic Review ===
    print("\n[Phase 2/4] Critic performing brutal review...")
    with Spinner("Critic analyzing draft"), timed_section("Critic Review API"):
        try:
            critique = run_critic(client, draft_response)
        except Exception as e:
            print(f"ERROR: Critic API call failed: {e}", file=sys.stderr)
            sys.exit(1)
    
    print_section("CRITIQUE", critique)
    
    # Save critique
    save_output("Critique.txt", critique)
    
    # === PHASE 3: Human Input ===
    human_input = get_human_input()
    if human_input:
        print(f"\nReceived clarifications ({len(human_input)} chars)")
    else:
        print("\nNo clarifications provided, proceeding with critique only.")
    
    # === PHASE 4: Final Resume ===
    print("\n[Phase 4/4] Architect producing final resume...")
    with Spinner("Gemini finalizing resume"), timed_section("Architect Final API"):
        try:
            final_resume = run_architect_final(architect_chat, critique, human_input)
        except Exception as e:
            print(f"ERROR: Final resume API call failed: {e}", file=sys.stderr)
            sys.exit(1)
    
    print_section("FINAL RESUME", final_resume)
    
    # Check for markdown in final
    md_warnings = check_for_markdown(final_resume)
    if md_warnings:
        print("WARNING: Markdown-like formatting detected in final resume:")
        for w in md_warnings[:5]:
            print(f"  - {w}")
        print()
    
    # Save final resume (txt for reference)
    save_output("Final_Resume.txt", final_resume)
    
    # === PHASE 5: Convert to Word and save to tailored_resumes folder ===
    print("\n[Phase 5/6] Converting to Word format...")
    with timed_section("Word conversion"):
        role_name = extract_role_from_jd_filename(jd_path)
        word_path = save_resume_as_word(final_resume, role_name)
    
    # === PHASE 6: Application Question Responses ===
    print("\n[Phase 6/6] Application Question Responses (Optional)")
    questions = get_application_questions()
    
    responses_path = None
    if questions:
        print(f"\nGenerating responses for {len(questions)} question(s)...")
        with Spinner("Crafting responses"), timed_section("Application Responses API"):
            try:
                # Use text content for JD if it was extracted
                jd_text = jd_content if isinstance(jd_content, str) else "[JD content from PDF]"
                responses = run_application_response(client, jd_text, final_resume, questions)
            except Exception as e:
                print(f"ERROR: Failed to generate responses: {e}", file=sys.stderr)
                responses = None
        
        if responses:
            print_section("APPLICATION RESPONSES", responses)
            responses_path = save_application_responses(responses, role_name)
    else:
        print("Skipped application questions.")
    
    # Total time
    total_elapsed = time.perf_counter() - total_start
    
    print("\n" + "=" * 60)
    print(f"COMPLETE! Your tailored resume is saved to:")
    print(f"  - Word: {word_path}")
    print(f"  - Text: {WORKSPACE_DIR / 'Final_Resume.txt'}")
    if responses_path:
        print(f"  - Responses: {responses_path}")
    print(f"  - Total time: {total_elapsed:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()