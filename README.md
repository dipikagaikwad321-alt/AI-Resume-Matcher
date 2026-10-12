# AI-Resume-Matcher
NLP-based resume and job description matching using TF-IDF and Sentence Transformers

## Project Overview

AI Resume Matcher is a Python-based application that compares a candidate's resume with a job description to estimate how well the resume matches the job requirements.

The application uses Natural Language Processing (NLP) techniques to analyze text, compare job requirements with resume content, calculate matching scores, and display relevant evidence.

## Features

- Upload a resume in PDF format.
- Enter a job description.
- Extract text from the uploaded resume.
- Analyze job requirements and resume content.
- Compare text using semantic similarity techniques.
- Calculate requirement matching and text similarity scores.
- Display matching results and supporting evidence through an interactive interface.

## Technologies Used

- Python
- Natural Language Processing (NLP)
- TF-IDF
- Cosine Similarity
- Sentence Transformers
- Scikit-learn
- Gradio
- PyPDF

## How to Run

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:

   ```bash
   python app.py
   ```

3. Open the local URL displayed in the terminal.
4. Enter a job description and upload a resume to view the matching results.

## Disclaimer

The matching scores are estimates and should not be treated as definitive hiring decisions.
