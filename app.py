# Custom CSS
import streamlit as st   # 👈 this is required

st.markdown(
    """
    <style>
    /* Background */
    .stApp {
        background-color: #0E1117;
    }

    /* Title */
    h1 {
        color: #FF4B4B;
        text-align: center;
    }

    /* Buttons */
    .stButton > button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        font-size: 16px;
        padding: 8px 20px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #FF1C1C;
    }

    /* Text inputs */
    textarea, input {
        border: 2px solid #FF4B4B !important;
        border-radius: 6px !important;
    }

    /* Table */
    .stDataFrame {
        border: 2px solid #FF4B4B;
        border-radius: 8px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div style="padding:15px; background:#262730; border-radius:10px; margin-bottom:20px;">
        <h2 style="color:#FF4B4B; text-align:center;">🔥 AI Resume Scanner</h2>
        <p style="color:#FFFFFF; text-align:center;">Upload resumes and match them with your Job Description.</p>
    </div>
    """,
    unsafe_allow_html=True
)

from docx import Document
import io
import re
import sys
import json
import base64
import pandas as pd
import numpy as np
import streamlit as st

# Text extraction deps
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import docx2txt
except Exception:
    docx2txt = None

from docx import Document  # NEW

# NLP / similarity deps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional: sentence-transformers for embeddings
_EMBEDDINGS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer("all-MiniLM-L6-v2")
    _EMBEDDINGS_AVAILABLE = True
except Exception:
    _model = None

st.set_page_config(page_title="AI Resume Scanner", page_icon="📄", layout="wide")

st.title("📄 AI Resume Scanner")
st.caption("Upload multiple resumes and a job description (JD). The app will parse, score, and rank candidates.")

# ------------- Utilities -------------

COMMON_SKILLS = {
    "communication","leadership","teamwork","problem solving","time management","project management",
    "c","c++","java","python","javascript","typescript","html","css","react","next.js","node.js","express",
    "mongodb","mysql","postgresql","sql","django","flask","spring","spring boot","graphql","rest","api",
    "git","github","docker","kubernetes","aws","gcp","azure","linux","bash",
    "pandas","numpy","scikit-learn","tensorflow","pytorch","nlp","computer vision","matplotlib","power bi","tableau",
    "android","kotlin","swift","flutter","react native","ci/cd","terraform","ansible",
}

def read_docx(file):
    """Safe .docx reader using python-docx"""
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file.read())
        tmp.flush()
        path = tmp.name
    doc = Document(path)
    os.unlink(path)
    return "\n".join([para.text for para in doc.paragraphs])

def read_file(file) -> str:
    """Extract text from PDF, DOCX, or TXT resume files"""
    name = file.name.lower()
    if name.endswith(".pdf"):
        if fitz is None:
            return ""
        try:
            file.seek(0)  # reset pointer
            with fitz.open(stream=file.read(), filetype="pdf") as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
            return text
        except Exception:
            return ""
    elif name.endswith(".docx"):
        try:
            return read_docx(file)
        except Exception:
            return ""
    elif name.endswith(".txt"):
        try:
            return file.read().decode("utf-8", errors="ignore")
        except Exception:
            return ""
    else:
        return ""

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{8,}\d")

def extract_contact(text: str):
    emails = EMAIL_RE.findall(text) if text else []
    phones = PHONE_RE.findall(text) if text else []
    return (emails[0] if emails else ""), (phones[0] if phones else "")

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()

def extract_skills(text: str):
    if not text:
        return set()
    low = normalize(text)
    found = set()
    for sk in COMMON_SKILLS:
        if sk in low:
            found.add(sk)
    return found

def jd_keywords(jd_text: str):
    if not jd_text:
        return set()
    low = normalize(jd_text)
    tokens = re.findall(r"[a-zA-Z][a-zA-Z+\-.#]*", low)
    toks = set(tokens)
    sks = {sk for sk in COMMON_SKILLS if sk in low}
    return toks.union(sks)

def similarity_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if _EMBEDDINGS_AVAILABLE and _model is not None:
        va = _model.encode([a], normalize_embeddings=True)
        vb = _model.encode([b], normalize_embeddings=True)
        sim = float(np.dot(va[0], vb[0]))
        return (sim + 1) / 2
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform([a, b])
    sim = cosine_similarity(X[0], X[1])[0][0]
    return float(sim)

def score_resume(resume_text: str, jd_text: str):
    r_skills = extract_skills(resume_text)
    j_skills = jd_keywords(jd_text)

    if len(j_skills) == 0:
        skills_overlap = 0.0
    else:
        overlap = len(r_skills.intersection(j_skills))
        skills_overlap = 60.0 * (overlap / max(1, len(j_skills)))

    sim = similarity_score(resume_text, jd_text)
    semantic = 40.0 * sim

    total = round(min(100.0, skills_overlap + semantic), 2)
    section = {
        "Skills Match (60)": round(skills_overlap, 2),
        "Semantic Fit (40)": round(semantic, 2),
    }
    return total, section, r_skills

def probable_name(text: str):
    if not text:
        return ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for l in lines[:10]:
        if 2 <= len(l.split()) <= 5 and not EMAIL_RE.search(l):
            if sum(w[0].isupper() for w in l.split() if w) >= 2:
                return l
    return lines[0] if lines else ""

# ------------- Sidebar -------------
with st.sidebar:
    st.subheader("Settings")
    st.write("**Scoring** = Skills overlap (60) + Semantic similarity (40)")
    st.write("Uses Sentence-BERT if available; otherwise TF-IDF.")
    st.write("Supported file types: PDF, DOCX, TXT.")
    st.link_button("Project Docs (README)", "https://")  # placeholder link

# ------------- Main UI -------------
col1, col2 = st.columns([1,1])
with col1:
    jd_text = st.text_area("Paste Job Description (JD)", height=240, placeholder="Paste JD here...")
with col2:
    uploads = st.file_uploader(
        "Upload Resumes (PDF/DOCX/TXT) — multiple allowed",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

analyze = st.button("Scan & Score Resumes", type="primary")

if analyze:
    if not jd_text:
        st.warning("Please paste a Job Description first.")
    elif not uploads:
        st.warning("Please upload at least one resume file.")
    else:
        rows = []
        details = {}

        for f in uploads:
            text = read_file(f)

            # 🔎 Debug: Show extracted text
            st.text_area(f"Extracted text from {f.name}", text[:1000], height=200)

            name = probable_name(text)
            email, phone = extract_contact(text)
            total, section, r_skills = score_resume(text, jd_text)
            rows.append({
                "File": f.name,
                "Candidate": name,
                "Email": email,
                "Phone": phone,
                "Score": total,
                "Skills Match (60)": section["Skills Match (60)"],
                "Semantic Fit (40)": section["Semantic Fit (40)"],
                "Skills Extracted": ", ".join(sorted(r_skills)) if r_skills else "",
            })
            details[f.name] = {
                "candidate": name,
                "email": email,
                "phone": phone,
                "skills": sorted(list(r_skills)),
                "section_scores": section,
            }

        df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
        st.success(f"Scanned {len(rows)} resume(s).")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Download CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results (CSV)", data=csv, file_name="resume_scores.csv", mime="text/csv")

        # Candidate drill-down
        st.subheader("Candidate Details")
        select_file = st.selectbox("Pick a resume to view details", [r["File"] for r in rows])
        if select_file:
            d = details[select_file]
            st.write(f"**Candidate**: {d['candidate']}")
            st.write(f"**Email**: {d['email']} | **Phone**: {d['phone']}")

            # Section score bar chart
            sec = d["section_scores"]
            chart_df = pd.DataFrame({
                "Section": list(sec.keys()),
                "Score": list(sec.values()),
            })
            st.bar_chart(chart_df.set_index("Section"))

            st.write("**Extracted Skills**:", ", ".join(d["skills"]) if d["skills"] else "—")

st.markdown("---")
st.caption("Tip: Improve results by adding precise skills and responsibilities in the JD.")

