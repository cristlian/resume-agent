# Resume Optimizer — High-Level Product Roadmap

## 0) Product goal and scope

**Goal:** Generate a **trustworthy, ready-to-use, one-page tailored resume** (plain text output, optionally converted to Word) from:
- a Job Description (prefer **URL fetch**, fallback to **user-pasted text**), and
- a user "source of truth" about experience (moving toward a **KB + retrieval** approach).

**Out of scope (for this roadmap):**
- Job discovery, auto-apply, submission automation, multi-agent job hunting.
- Application Q&A automation beyond noting future integration (resume optimizer only).

---

## 1) Current state (already exists)

### Existing pipeline (single-run CLI tool)
- **Inputs:** JD PDF + merged resume PDF via CLI paths  
- **Preprocess:** PDF → extracted text (token saving)
- **LLM:** Gemini 3 Pro (via API key) runs:
  1) Architect draft + Critic critique
  2) Pause for user clarifications
  3) Architect finalization
- **Postprocess:** plain text → Word conversion; save locally

### Current "resume source" strategy
- "Merged resume" assembled from **5 older tailored versions** (robotics, AI, SWE, data, devops).
- Works mainly by "exposing projects," not by providing meaningful structured evidence.

---

## 2) Target architecture direction

### External interface: Skill
Expose the whole resume optimizer as a single callable capability (tool/skill):
- `tailor_resume(jd_input, kb_or_resume_source, options) -> final_resume_text (+ artifacts)`

### Internal behavior: Agentic workflow with sub-agents (bounded)
- **Architect (writer/synthesizer)**
- **Critic (brutal screening, no flattery)**

"Agentic" should be **bounded and pragmatic**:
- Avoid open-ended loops.
- Prefer deterministic steps + strict guardrails over "cool autonomy."
---

## 3) Recommended path (practical hybrid, minimal overengineering)

### Core idea
1) Move experience input toward **KB chunks** (projects/bullets as atomic facts).
2) Add **retrieval** to feed only relevant chunks into the LLM.
3) Add **lightweight verification** that minimizes rewrite loops.

**Key principle:** Retrieval should be **cheap/deterministic by default**. Gemini 3 Pro is for synthesis, not scanning a giant KB every time.

---

## 4) Roadmap phases (priority-ordered)

### Phase 0 — Stabilize the MVP (P0: high impact, low/medium effort) [Completed]
**Objective:** Make output consistent + reduce false "missing info" critiques.
**Status:** Finished.

1) **JD ingestion change**
   - Add `--jd-url` option:
     - Try fetch HTML → extract readable text
     - If fail, prompt user to paste JD text
   - Keep existing PDF JD input as optional fallback if desired.

2) **Plain-text output guarantee**
   - Ensure final resume is **plain text** (no Markdown syntax).
   - Keep docx conversion as a post-step.

3) **Add stable "Profile Facts" input**
   - Introduce `profile_facts.yaml` (grad date, predicted grade phrasing, location, links, etc.)
   - Feed it to both Architect and Critic.
   - This prevents Critic repeatedly flagging known stable facts.

4) **Critic output becomes structured**
   - Replace freeform critique with categories:
     - A) Needs user input (blocking)
     - B) Auto-fixable (wording/ATS)
     - C) Already covered (Profile Facts / evidence)
     - D) Optional
   - Pipeline pauses for human input **only if A is non-empty**.

**Deliverable:** A "trustable" single-run tailor that usually doesn't require manual clarifications.

---

### Phase 1 — Replace merged resume with KB + retrieval (P1: highest ROI)
**Objective:** Better coverage, lower tokens, better grounding.

1) **KB format + chunking**
   - Single expanding KB file (e.g., `kb.jsonl` or `kb.yaml`) with atomic chunks:
     - `chunk_id`, `text`, `tags/skills`, `project`, `role`, `dates`, `links`, optional metrics fields.
   - Include "active projects" not present in old resumes.

2) **Retrieval layer (non-LLM first)**
   - Build query from JD text (keywords/responsibilities).
   - Retrieve top-K chunks using BM25/keyword (optionally embeddings later).
   - Feed only top-K chunks to Architect/Critic.

3) **Rerank (optional, cheap)**
   - Only if retrieval quality is weak:
     - rerank top-30 candidates to top-10/15
   - Prefer cheap model or lightweight heuristics; avoid Gemini 3 Pro on full KB.

4) **Project date normalization via KB canonicalization**
   - Fix project start/end dates by treating date resolution as a KB responsibility.
   - Use cross-reference resumes from `submitted-resume/` to match projects that have different names but same/similar content.
   - In KB, each project must have one unique canonical name/category.
   - Store per-resume project-name aliases that map to the canonical project.
   - Canonical project dates are shared across aliases for consistent start/end dates.

5) **Hard rule: work experience entries do not require project names**
   - Add a pipeline-wide hard rule: work experience bullets/jobs should not require project name fields.
   - If selected for inclusion, map the following into work experience:
     - UCL Computer Science Dept project -> `Research Assistant`
     - Hybrid Body Lab project -> `Research Intern`
     - UCL Surgical Robot Vision Group project -> `Research Intern`
   - Apply this rule across selection, retrieval, drafting, critique, and verification stages.

**Deliverable:** Evidence-limited context that improves precision + reduces hallucination risk.

---

### Phase 2 — Lightweight "strict verification" (P2: improves trust without big complexity)
**Objective:** Anti-hallucination via structure, but keep loops bounded.

1) **Internal evidence mapping**
   - Architect produces bullets with internal `source_chunk_ids`.
   - Mapping is kept in logs/debug artifacts, stripped from final resume text.

2) **Verifier (deterministic)**
   - Checks each bullet has valid chunk references.
   - If missing: **drop bullet** or request a single rewrite pass.

3) **Bounded repair loop**
   - Max 1 retry.
   - "Fail-closed": prioritize trust over completeness.

**Deliverable:** Strong grounding with minimal "agentic loop" complexity.

---

### Phase 3 — Performance + batch mode (P3: optional, situational)
**Objective:** Throughput improvements; does not change conceptual architecture.

1) **Batch processing**
   - `--batch manifest.jsonl` where each entry is one JD+settings.
   - Draft+critique in parallel; finalize in second pass (since human input is conditional).

2) **Concurrency controls**
   - Rate limiting, retries, caching.

**Deliverable:** Scalable infra features; only needed for multi-run sessions.

---

## 5) Separate but related: KB updater from GitHub (recommended as an on-demand skill)

### Concept
A separate tool that:
- scans GitHub repos (or a provided repo list),
- drafts KB entries from README/docs/reports,
- asks for your approval,
- writes KB patches.

### Recommended implementation approach
- **On-demand skill** (run manually weekly / before application pushes).
- Deterministic triggers first (no always-on LLM):
  - only process repos with meaningful changes (README/docs/report updates, releases)
  - "maturity gating" to avoid WIP repo noise

---

## 6) Model strategy (cost-aware, quality-first)

- **Gemini 3 Pro Preview**: Architect drafting + finalization; Critic critique.
- **Cheaper model (optional)**: query rewriting / rerank / cleanup.
- **Retrieval should be non-LLM by default**.

---

## 7) Quality guardrails (must-have for "trustable resume")

- **No fabrication policy** reinforced by:
  - retrieval-limited evidence set
  - profile facts input
  - verifier requiring chunk references (Phase 2)
- **Tone discipline (optional)**
  - "No flattery / technical-doc tone" is useful mainly for Critic.
  - Not a substitute for grounding.

- **Work-experience normalization hard rule**
  - No project name is required for work experience entries.
  - If selected, map UCL Computer Science Dept -> Research Assistant; Hybrid Body Lab -> Research Intern; UCL Surgical Robot Vision Group -> Research Intern.
  - Enforce this rule across retrieval, drafting, critique, and verification.
---

## 8) Feature backlog with priority and difficulty

### P0 (completed)
- JD URL fetch with fallback paste
- `profile_facts.yaml`
- structured Critic categories; pause only if blocking issues exist
- strict plain-text output + stable file outputs/logs

**Difficulty:** Low–Medium

### P1 (highest ROI)
- KB format + chunking
- BM25 retrieval + top-K evidence feed
- optional rerank (cheap)
- project date normalization using canonical KB project identity + alias mapping from `submitted-resume/` cross-references
- hard rule: no project name required for work experience; map UCL CS Dept -> `Research Assistant`, Hybrid Body Lab -> `Research Intern`, UCL Surgical Robot Vision Group -> `Research Intern` when included

**Difficulty:** Medium

### P2 (trust upgrade)
- Architect bullets include chunk IDs internally
- deterministic verifier + single retry; drop ungrounded bullets

**Difficulty:** Medium–High (but bounded)

### P3 (infra scaling)
- batch mode + concurrency
- caching/rate limiting/robust job runner

**Difficulty:** Medium

### Separate tool (recommended)
- GitHub → KB updater as on-demand skill with human approval + maturity gating

**Difficulty:** Medium

---

## 9) Guidance for the coding agent after scanning the workspace

1) Identify what's already implemented:
   - CLI args, PDF extraction, Gemini calls, pauses, docx conversion.
2) Confirm Phase 0 remains stable (completed baseline):
   - JD URL ingestion + fallback text
   - profile facts integration
   - structured Critic output + conditional pause
   - plain text enforcement
3) Implement Phase 1 (KB + retrieval), including:
   - canonical project identity + alias mapping for cross-resume date consistency
   - work-experience hard rule (no project-name requirement, with required UCL role mappings)
4) Only then add Phase 2 verifier (bounded, fail-closed).

---

## 10) Highest leverage upgrades (to make this project shine)

These upgrades turn "useful tool" into a **credible RAG 0→1 portfolio project** without bloating scope.

### A) Add an evaluation harness (makes it "real RAG")
- Create a small benchmark set (e.g., 30–50 JDs).
- For each JD, label which KB chunks are relevant (even approximate).
- Track retrieval + pipeline metrics:
  - **recall@k / precision@k** (retrieval quality)
  - **% bullets grounded** (trust)
  - latency + cost per run

### B) Make grounding auditable (strong trust signal)
- Keep internal mapping: resume bullet → `source_chunk_ids`
- Save artifacts:
  - `Final_Resume.txt`
  - `Evidence_Map.json` (bullet text, chunk IDs, chunk snippets/links)

### C) Add infra-minded logging (minimal but meaningful)
- Per-stage latency and token/cost estimates
- Retrieval diagnostics: top-k chunk IDs + scores
- Cache:
  - JD fetch results
  - KB embedding/index build artifacts (if used)

### D) OSS polish (what reviewers notice)
- Clean CLI + good README
- Example KB + example JD + one "demo run"- Basic tests for:
  - chunk parsing
  - retrieval returning expected chunks on a tiny fixture
  - verifier rules (drop ungrounded bullets)

### E) Optional "wow" feature that stays controlled
- Two retrievers behind flags:
  - BM25 baseline
  - embeddings retrieval (optional)
- Include a short comparison report in README (cost/latency/quality tradeoffs)

---

## 11) Success definition (what "done" looks like)
- Produces a **plain-text, one-page** tailored resume with minimal user intervention.
- Uses retrieval to limit context and reduce cost.
- Provides at least lightweight grounding assurance (evidence map; bounded verifier).
- Includes an evaluation harness demonstrating retrieval quality and trust metrics.





