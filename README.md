# 📄 AI Resume Scanner

A simple Streamlit app to parse PDF/DOCX/TXT resumes, extract contact info & skills, and score them against a Job Description (JD).

## 🚀 Features
- Upload multiple resumes
- Extract email/phone, heuristic candidate name
- Skill extraction (curated set + JD keywords)
- Scoring = Skills overlap (60) + Semantic similarity (40)
  - Uses Sentence-BERT (`all-MiniLM-L6-v2`) if available; falls back to TF‑IDF
- Ranked table + downloadable CSV
- Per-candidate section scores (bar chart)

## 🛠️ Installation

```bash
# 1) Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt
```

> If the sentence-transformers model is large for your environment, you can remove it from `requirements.txt`. The app will automatically fall back to TF‑IDF.

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

## 📦 Files
- `app.py` — Streamlit app
- `requirements.txt` — deps
- `README.md` — this guide

## 🔒 Notes
- This tool runs locally and does not send resume data to external services.
- Parsing quality can vary by resume format; PDFs exported as images may need OCR (not included by default).

## 🧭 Next Steps (Optional Enhancements)
- Add OCR for image-based PDFs (e.g., `pytesseract` + `pdf2image`).
- Add section-aware parsing (Education, Experience) using regex and heuristics.
- Add multilingual support and domain-specific skill dictionaries.
- Export per-candidate JSON reports.