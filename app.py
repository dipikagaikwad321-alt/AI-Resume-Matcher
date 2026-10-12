%pip install -q gradio pypdf sentence-transformers scikit-learn

import os
import re
import gradio as gr

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. LOAD NLP MODEL
# ============================================================

print("Loading NLP model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("NLP model loaded successfully!")


# ============================================================
# 2. EXTRACT TEXT FROM PDF
# ============================================================

def extract_pdf_text(pdf_file):

    reader = PdfReader(pdf_file)

    if reader.is_encrypted:
        raise ValueError(
            "Please upload an unlocked PDF resume."
        )

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text[:150000]


# ============================================================
# 3. CLEAN AND SPLIT TEXT
# ============================================================

def clean_text(text):

    text = text.replace("\u2022", "\n")
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def split_sentences(text):

    text = clean_text(text)

    # Split on bullets/new lines and sentence boundaries.
    chunks = re.split(
        r"\n+|(?<=[.!?])\s+",
        text
    )

    sentences = []

    for chunk in chunks:

        sentence = re.sub(
            r"^\s*(?:[-*•▪]|\d+[.)])\s*",
            "",
            chunk
        ).strip()

        # Ignore headings and very short fragments.
        if len(sentence) >= 20:
            sentences.append(sentence)

    # Remove duplicate sentences.
    return list(dict.fromkeys(sentences))


# ============================================================
# 4. AUTOMATICALLY EXTRACT REQUIREMENTS
# ============================================================

def extract_requirements(job_description):

    # No predefined skills dictionary is used.
    # Job description sentences become requirement candidates.

    requirements = split_sentences(job_description)

    return requirements[:40]


# ============================================================
# 5. CALCULATE NLP SIMILARITY USING TF-IDF
# ============================================================

def calculate_nlp_similarity(job_description, resume_text):

    try:

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        tfidf_matrix = vectorizer.fit_transform([
            job_description,
            resume_text
        ])

        similarity = cosine_similarity(
            tfidf_matrix[0:1],
            tfidf_matrix[1:2]
        )[0][0]

        return max(0, float(similarity)) * 100

    except ValueError:
        return 0.0


# ============================================================
# 6. COMPARE REQUIREMENTS WITH RESUME SENTENCES
# ============================================================

def compare_requirements(requirements, resume_sentences):

    if not requirements or not resume_sentences:
        return [], 0.0

    # Convert job requirements into meaning vectors.
    job_vectors = model.encode(
        requirements,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Convert resume sentences into meaning vectors.
    resume_vectors = model.encode(
        resume_sentences,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Compare every job requirement with resume sentences.
    similarity_matrix = job_vectors @ resume_vectors.T

    results = []

    matched_count = 0

    for index, requirement in enumerate(requirements):

        best_index = int(
            similarity_matrix[index].argmax()
        )

        score = float(
            similarity_matrix[index][best_index]
        )

        # Initial thresholds for testing, not validated standards.
        if score >= 0.55:

            status = "Potential Match"
            matched_count += 1

        elif score >= 0.40:

            status = "Needs Review"

        else:

            status = "No Close Evidence"

        results.append({
            "requirement": requirement,
            "score": max(0, min(score, 1)) * 100,
            "status": status,
            "evidence": resume_sentences[best_index]
        })

    requirement_match = (
        matched_count / len(requirements)
    ) * 100

    return results, requirement_match


# ============================================================
# 7. MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_resume(job_description, resume_file):

    if not job_description or not job_description.strip():

        return (
            "Please enter a job description.",
            "",
            ""
        )

    if resume_file is None:

        return (
            "Please upload a PDF resume.",
            "",
            ""
        )

    try:

        # Extract resume text.
        resume_text = extract_pdf_text(resume_file)

        if not resume_text.strip():

            return (
                "No readable text was found in the PDF. "
                "Scanned PDFs may require OCR.",
                "",
                ""
            )

        # Extract requirements from the job description.
        requirements = extract_requirements(
            job_description
        )

        # Extract sentences from resume.
        resume_sentences = split_sentences(
            resume_text
        )

        if not requirements or not resume_sentences:

            return (
                "Could not identify enough text for comparison.",
                "",
                resume_text
            )

        # Calculate traditional NLP similarity.
        nlp_score = calculate_nlp_similarity(
            job_description,
            resume_text
        )

        # Compare each requirement against resume evidence.
        comparison_results, skill_score = compare_requirements(
            requirements,
            resume_sentences
        )

        # ====================================================
        # 8. FINAL WEIGHTED SCORE
        # ====================================================

        # Skills/requirement evidence = 80%
        # NLP similarity = 20%

        final_score = (
            skill_score * 0.80
            +
            nlp_score * 0.20
        )

        # The label is based primarily on requirement coverage.
        if skill_score >= 75:

            match_level = "Strong Requirement Coverage"

        elif skill_score >= 50:

            match_level = "Moderate Requirement Coverage"

        else:

            match_level = "Low Requirement Coverage"

        # ====================================================
        # 9. FORMAT REQUIREMENT DETAILS
        # ====================================================

        details = """
## Requirement-by-Requirement Analysis

| Job Requirement | Match Score | Status | Resume Evidence |
|---|---:|---|---|
"""

        for item in comparison_results:

            requirement = item["requirement"].replace(
                "|", "\\|"
            )

            evidence = item["evidence"].replace(
                "|", "\\|"
            )

            details += (
                f"| {requirement} "
                f"| {item['score']:.2f}% "
                f"| {item['status']} "
                f"| {evidence} |\n"
            )

        details += """

### How to read the report

- **Potential Match:** The model found semantically similar
  resume evidence. This does not prove the candidate has the skill.
- **Needs Review:** The evidence may be relevant but needs
  human verification.
- **No Close Evidence:** The model did not find sufficiently
  similar text. The candidate may still possess the skill.
"""

        # ====================================================
        # 10. DISPLAY FINAL RESULT
        # ====================================================

        result = f"""
# Resume Analysis Report

## {match_level}

### Overall Match Score
# {final_score:.2f}%

### Requirement Match
**{skill_score:.2f}%**

### NLP Similarity (TF-IDF)
**{nlp_score:.2f}%**

---

### Scoring Formula

Overall Score = (Requirement Match × 0.80)
+ (NLP Similarity × 0.20)

### Important Notes

- No manually maintained skills dictionary is used.
- Requirements are automatically derived from job-description
  sentences, so some sentences may not be actual skills.
- The requirement score measures potential evidence coverage,
  not verified skill proficiency.
- The 80:20 weights and similarity thresholds are prototype
  settings and should be evaluated on reviewed examples.
- Review the resume evidence before making recruitment decisions.
"""

        return result, details, resume_text

    except Exception as error:

        return (
            f"Analysis error: {str(error)}",
            "",
            ""
        )


# ============================================================
# 11. GRADIO INTERFACE
# ============================================================

with gr.Blocks(title="AI Resume Matcher") as app:

    gr.Markdown(
        """
        # AI Resume Matcher

        Compare a job description with a PDF resume using
        semantic NLP matching and TF-IDF.

        This version does not require a predefined skills list.
        """
    )

    with gr.Row():

        with gr.Column():

            job_input = gr.Textbox(
                label="Job Description",
                placeholder="Paste the complete job description...",
                lines=12
            )

            resume_input = gr.File(
                label="Upload Resume PDF",
                file_types=[".pdf"],
                type="filepath"
            )

            analyze_button = gr.Button(
                "Analyze Resume",
                variant="primary"
            )

        with gr.Column():

            result_output = gr.Markdown()

    details_output = gr.Markdown()

    resume_text_output = gr.Textbox(
        label="Extracted Resume Text",
        lines=10
    )

    analyze_button.click(
        fn=analyze_resume,
        inputs=[job_input, resume_input],
        outputs=[
            result_output,
            details_output,
            resume_text_output
        ]
    )

app.launch()
