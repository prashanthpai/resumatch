# app.py — ResuMatch
# ------------------------------------------------------------
# Logic:
# - Upload: one Job Description (JD) + multiple resumes (PDF/DOCX/TXT)
# - Extracts text locally (no DB needed)
# - Sends compact structured prompts to Claude Sonnet 4.5 for scoring per candidate
# - Returns a ranked shortlist with strengths, gaps, risks, and tailored interview questions
# - Lets you tweak priorities (e.g., "prioritize leadership, Python, fintech domain")
# - Exports results to CSV
# ------------------------------------------------------------

import os
import io
import re
import json
import zipfile
from typing import Optional, Dict, Any
import pandas as pd
from rapidfuzz import fuzz
import streamlit as st
from anthropic import Anthropic

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    import docx2txt
except Exception:
    docx2txt = None

# ----------------------------
# Config & helpers
# ----------------------------
MODEL = "claude-sonnet-4-5-20250929"
LOGO_PATH = "logo.png"

st.set_page_config(page_title="ResuMatch: Recruiter Copilot", layout="wide")
st.title("ResuMatch: Recruiter Copilot")
st.markdown(
    "<p style='font-size:1.1rem;'>ResuMeh → ResuYay: Filtering out the meh, spotlighting the yay with Claude Sonnet 4.5</p>",
    unsafe_allow_html=True,
)
st.caption(
    "Upload a job description and multiple resumes. Get a ranked shortlist + interview kits."
)

# Sidebar environment checks
with st.sidebar:
    st.markdown("## ResuMatch")
    try:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)
        else:
            st.info(f"Place your logo image at: {LOGO_PATH}")
    except Exception:
        # In very restricted environments os.path may not be available as expected
        st.info("Logo not displayed — unable to access file path.")

    st.markdown("### Environment checks")
    st.write("pdfplumber:", "✅" if pdfplumber else "❌ not installed")
    st.write("pypdf:", "✅" if PdfReader else "❌ not installed")
    st.write("docx2txt:", "✅" if docx2txt else "❌ not installed (fallback used)")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("ANTHROPIC_API_KEY is not set — API calls will fail.")


def _extract_pdf_text(file_bytes: bytes) -> str:
    """Extract text from PDF using available backends. Empty string on failure."""
    # Try pdfplumber first
    if pdfplumber:
        try:
            text_parts = []
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    text_parts.append(t)
            return "\n".join(text_parts).strip()
        except Exception as e:  # pragma: no cover
            st.info(f"pdfplumber failed, falling back: {e}")
    # Fall back to pypdf
    if PdfReader:
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            out = []
            for p in reader.pages:
                try:
                    out.append(p.extract_text() or "")
                except Exception:
                    out.append("")
            return "\n".join(out).strip()
        except Exception as e:
            st.info(f"pypdf failed: {e}")

    st.error(
        "No PDF text extractor available. Install `pdfplumber` or `pypdf`, or upload DOCX/TXT instead."
    )
    return ""


def _extract_docx_text_with_zip(file_bytes: bytes) -> str:
    """Minimal DOCX reader via zipfile -> word/document.xml stripping tags (no external deps)."""
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
        # Replace paragraph/line separators with newlines for readability
        xml = xml.replace("</w:p>", "\n").replace("</w:tr>", "\n")
        # Strip xml tags
        text = re.sub(r"<[^>]+>", "", xml)
        return re.sub(r"\n{3,}", "\n\n", text).strip()
    except Exception:
        return ""


@st.cache_data(show_spinner=False)
def extract_text(file_bytes: bytes, filename: str) -> str:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return _extract_pdf_text(file_bytes)
    if name.endswith(".docx"):
        # Prefer docx2txt if available
        if docx2txt:
            tmp = io.BytesIO(file_bytes)
            tmp_path = "_tmp_upload.docx"
            with open(tmp_path, "wb") as f:
                f.write(tmp.getbuffer())
            try:
                return (docx2txt.process(tmp_path) or "").strip()
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        # Fallback: zipfile-based reader
        text = _extract_docx_text_with_zip(file_bytes)
        if text:
            return text
        st.error("Could not read DOCX. Install `docx2txt` or upload as TXT.")
        return ""
    if name.endswith(".txt"):
        try:
            return file_bytes.decode("utf-8", errors="ignore").strip()
        except Exception:
            return file_bytes.decode("latin-1", errors="ignore").strip()
    st.warning("Unsupported file type. Please upload PDF, DOCX, or TXT.")
    return ""


# Lightweight heuristic score to complement LLM scoring (for speed & stability)
def heuristic_score(jd_text: str, resume_text: str) -> float:
    # crude skill overlap using fuzzy partial ratios on top tokens from JD
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#\.\-]{2,}", jd_text)
    uniq = []
    for t in tokens:
        t_low = t.lower()
        if t_low not in uniq:
            uniq.append(t_low)
        if len(uniq) >= 100:
            break
    if not uniq:
        return 0.0
    step = max(1, len(uniq) // 20)
    sample = uniq[::step][:20]
    score = 0
    for w in sample:
        score += fuzz.partial_ratio(w, resume_text.lower())
    return score / (len(sample) or 1)


def parse_candidate_json(content_text: str, fallback_filename: str) -> Dict[str, Any]:
    """Parse single-candidate JSON from model response. Tolerant to pre/post text."""
    data: Optional[Dict[str, Any]] = None
    # direct parse
    try:
        data = json.loads(content_text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    # extract first {...} block
    m = re.search(r"\{[\s\S]*\}", content_text)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    # fallback minimal structure
    return {
        "candidate_name": fallback_filename,
        "overall": 0,
        "subscores": {},
        "strengths": [],
        "risks": ["Could not parse model output"],
        "interview_questions": [],
        "explanations": content_text[:500],
    }


anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """
You are Recruiter Copilot, a concise and rigorous evaluator of candidates against a single Job Description (JD).
Return JSON only — no markdown. Be fair, avoid demographic bias, and base claims only on the provided resume.
Scoring rubric (0–10 each): role_fit, core_skills, seniority, domain_match, communication.
Weights (default): 0.30, 0.30, 0.15, 0.15, 0.10. You may adjust based on `priority_notes`.
Output fields: {
  "candidate_name": str,
  "overall": float (0–100),
  "subscores": {"role_fit": int, "core_skills": int, "seniority": int, "domain_match": int, "communication": int},
  "strengths": [str],
  "risks": [str],
  "interview_questions": [str],
  "explanations": str
}
If name unavailable, infer a short identifier from resume filename when provided.
If information is missing, say so explicitly (do not guess).
"""

EVAL_PROMPT = """
JD:
<<<JD>>>

PRIORITY_NOTES (optional, may adjust weights and emphasis):
<<<PRIORITY_NOTES>>>

RESUME (plain text):
<<<RESUME>>>

Task: Evaluate this candidate strictly against the JD using the rubric. Provide JSON only as specified.
Filename (for identifier fallback): <<<FILENAME>>>.
"""

REFINE_PROMPT = """
You previously evaluated multiple candidates for a JD. The user has provided new priorities.
Re-rank the following candidate JSON results accordingly.

NEW PRIORITY NOTES:
<<<PRIORITY_NOTES>>>

CANDIDATE_RESULTS_JSON:
<<<RESULTS_JSON>>>

Task: Return JSON array of candidates re-ranked, each with updated `overall` if appropriate, and a short `why_changed` field.
Return JSON only.
"""

# ----------------------------
# Built-in self checks (simple tests)
# ----------------------------
with st.sidebar:
    if st.button("Run self‑checks"):
        try:
            # 1) TXT extraction
            txt = extract_text("hello world".encode("utf-8"), "sample.txt")
            assert txt == "hello world"
            # 2) DOCX fallback (zip-based) minimal docx
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr("[Content_Types].xml", "<Types></Types>")
                zf.writestr("_rels/.rels", "")
                zf.writestr("word/_rels/document.xml.rels", "")
                zf.writestr(
                    "word/document.xml",
                    "<w:document><w:body><w:p><w:r><w:t>Hi Docx</w:t></w:r></w:p></w:body></w:document>",
                )
            docx_text = extract_text(buf.getvalue(), "sample.docx")
            assert "Hi Docx" in docx_text
            # 3) JSON tolerance
            messy = (
                "Here are the results:\n{"
                + '"candidate_name":"A","overall":80,"subscores":{}}'
                + "\nEnd"
            )
            parsed = parse_candidate_json(messy, "fallback")
            assert parsed.get("candidate_name") == "A" and parsed.get("overall") == 80
            st.success("Self‑checks passed: TXT, DOCX fallback, JSON parser ✅")
        except AssertionError as e:
            st.error(f"Self‑check failed: {e}")
        except Exception as e:  # pragma: no cover
            st.error(f"Unexpected error in self‑checks: {e}")

# ----------------------------
# UI
# ----------------------------
col1, col2 = st.columns([1, 1])
with col1:
    jd_file = st.file_uploader(
        "Upload Job Description (PDF/DOCX/TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=False,
    )
with col2:
    resume_files = st.file_uploader(
        "Upload Resumes (PDF/DOCX/TXT) — multiple",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

priority_notes = st.text_input(
    "Optional: Prioritize (comma‑separated). Examples: leadership, Python, fintech domain, customer empathy",
    value="",
    placeholder="leadership, Python, fintech",
)

run_btn = st.button("Run Screening")

if run_btn:
    if not jd_file or not resume_files:
        st.warning("Please upload a JD and at least one resume.")
    else:
        with st.spinner("Extracting text…"):
            jd_text = extract_text(jd_file.read(), jd_file.name)
            resumes = []
            for f in resume_files:
                txt = extract_text(f.read(), f.name)
                if not txt.strip():
                    st.info(f"No text extracted from {f.name}. Skipping.")
                    continue
                resumes.append({"filename": f.name, "text": txt})

        if not resumes:
            st.error("No readable resumes found.")
        else:
            results = []
            for r in resumes:
                # quick heuristic score (helps sort if API rate limits)
                h = heuristic_score(jd_text, r["text"])

                user_msg = (
                    EVAL_PROMPT.replace("<<<JD>>>", jd_text[:15000])
                    .replace("<<<PRIORITY_NOTES>>>", priority_notes or "(none)")
                    .replace("<<<RESUME>>>", r["text"][:15000])
                    .replace("<<<FILENAME>>>", r["filename"])
                )
                try:
                    resp = anthropic_client.messages.create(
                        model=MODEL,
                        max_tokens=1200,
                        temperature=0.2,
                        system=SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": user_msg}],
                    )
                    content_text = "".join(
                        [c.text for c in resp.content if hasattr(c, "text")]
                    )
                    data = parse_candidate_json(content_text, r["filename"])
                except Exception as e:
                    data = {
                        "candidate_name": r["filename"],
                        "overall": 0,
                        "subscores": {},
                        "strengths": [],
                        "risks": [f"API error: {e}"],
                        "interview_questions": [],
                        "explanations": "",
                    }

                data["filename"] = r["filename"]
                data["heuristic"] = round(h, 2)
                # combine scores: 85% model, 15% heuristic (scaled)
                try:
                    model_score = float(data.get("overall", 0))
                except Exception:
                    model_score = 0.0
                combined = 0.85 * model_score + 0.15 * (h / 100.0 * 100)
                data["combined_score"] = round(combined, 2)
                results.append(data)

            # Initial ranking
            df = pd.DataFrame(results)
            df = df.sort_values(by=["combined_score"], ascending=False)

            st.subheader("Shortlist (ranked)")
            st.dataframe(
                df[
                    [
                        "candidate_name",
                        "combined_score",
                        "overall",
                        "heuristic",
                        "filename",
                    ]
                ],
                use_container_width=True,
            )

            # Explain top 5
            st.markdown("### Top 5: Strengths, Risks, Questions")
            top5 = df.head(5).to_dict(orient="records")
            for i, row in enumerate(top5, start=1):
                with st.expander(
                    f"#{i} — {row.get('candidate_name')} (Score: {row.get('combined_score')})"
                ):
                    strengths = row.get("strengths", [])
                    risks = row.get("risks", [])
                    questions = row.get("interview_questions", [])
                    st.markdown("**Strengths**")
                    for s in strengths or []:
                        st.write("• ", s)
                    st.markdown("**Risks / Gaps**")
                    for rsk in risks or []:
                        st.write("• ", rsk)
                    st.markdown("**Tailored Interview Questions**")
                    for q in questions or []:
                        st.write("• ", q)

            # Download CSV
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results as CSV", csv, "shortlist.csv", mime="text/csv"
            )

            # Re‑rank by new priorities
            st.markdown("---")
            st.subheader("Re‑rank by new priorities (context editing demo)")
            new_priorities = st.text_input(
                "Enter new priority notes to re‑rank (e.g., leadership, fintech, Python)",
                key="reprioritize",
            )
            if st.button("Re‑rank"):
                try:
                    results_json = df.to_json(orient="records")
                    user_msg = REFINE_PROMPT.replace(
                        "<<<PRIORITY_NOTES>>>", new_priorities or "(none)"
                    ).replace("<<<RESULTS_JSON>>>", results_json)
                    resp = anthropic_client.messages.create(
                        model=MODEL,
                        max_tokens=1200,
                        temperature=0,
                        messages=[{"role": "user", "content": user_msg}],
                    )
                    content_text = "".join(
                        [c.text for c in resp.content if hasattr(c, "text")]
                    )
                    new_df = pd.read_json(io.StringIO(content_text))
                    st.dataframe(
                        new_df[
                            ["candidate_name", "overall", "why_changed", "filename"]
                        ],
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Could not re‑rank: {e}")

st.markdown(
    """
---
**Notes, limitations/TODOs:**
- This demo keeps everything in memory; add a DB for persistence.
- Heuristic score is intentionally simple; can be swapped for embeddings & semantic similarity later.
- Ensure we have rights to process resumes; avoid storing PII unless necessary, and add consent/bias checks for production.
- For large PDFs, we truncate to keep prompts small for speed/cost. Will consider chunking + retrieval for bigger builds.
"""
)
