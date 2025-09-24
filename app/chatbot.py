import json
from pathlib import Path
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PyPDF2 import PdfReader
import pdfplumber
from PIL import Image
import pytesseract
import io
import re

from .config import GEMINI_API_KEY, FAISS_INDEX_PATH, METADATA_PATH

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAISS index and metadata
index = faiss.read_index(FAISS_INDEX_PATH)
with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)

def get_relevant_contexts(query, k=3):
    emb = embedder.encode(query)
    D, I = index.search(np.array([emb]).astype(np.float32), k=k)
    contexts = [metadata[i]['full_text'] for i in I[0] if i >= 0]
    return contexts

def get_disease_info(question):
    try:
        contexts = get_relevant_contexts(question)
        prompt = f"Question: {question}\nRelevant info: {contexts}\nAnswer based on info:"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def simplify_terms(term):
    try:
        prompt = f"Simplify the medical term '{term}' in simple language:"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def clean_response(response_text):
    """Post-process the response to ensure consistent formatting and capitalization."""
    # Fix specific acronyms to uppercase (e.g., Cbc -> CBC, Wbc -> WBC)
    acronyms = {
        r'\bCbc\b': 'CBC',
        r'\bWbc\b': 'WBC',
        r'\bHct\b': 'HCT',
        r'\bMcv\b': 'MCV',
        r'\bMch\b': 'MCH',
        r'\bMchc\b': 'MCHC',
        r'\bRbc\b': 'RBC'
    }
    for pattern, replacement in acronyms.items():
        response_text = re.sub(pattern, replacement, response_text, flags=re.IGNORECASE)
    
    # Capitalize test names (e.g., Hemoglobin, Leukocyte)
    response_text = re.sub(r'\b([A-Za-z]+)\b(?=:\s+\d|\(Missing\))', lambda m: m.group(1).title(), response_text)
    # Ensure double line breaks between sections
    response_text = re.sub(r'(\n#+\s)', r'\n\n\1', response_text)
    response_text = re.sub(r'(\n\*\*.*\*\*)', r'\n\n\1', response_text)
    # Fix bullet spacing
    response_text = re.sub(r'-\s*([^\s])', r'- \1', response_text)
    # Remove extra spaces or newlines
    response_text = re.sub(r'\n{3,}', r'\n\n', response_text)
    return response_text.strip()

def analyze_report(file_location):
    try:
        text = ''
        
        # Use pdfplumber for better text extraction
        with pdfplumber.open(file_location) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
                else:
                    print(f"Warning: No text extracted from page {page.page_number}. Attempting OCR...")
                    img = page.to_image(resolution=300)
                    img_byte_arr = io.BytesIO()
                    img.original.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    ocr_text = pytesseract.image_to_string(Image.open(img_byte_arr))
                    text += ocr_text + '\n'

        if not text.strip():
            return "Error: No text could be extracted from the PDF. It may be fully image-based or corrupted."

        # Strict prompt for structured, readable Markdown output
        prompt = f"""
Analyze this medical report and output in this exact structured Markdown format. Follow these rules strictly:
- Use bold headings (**Heading**) and subheadings (### Heading).
- Use bullets (-) with a space after each bullet.
- Capitalize acronyms correctly (e.g., CBC, WBC, HCT, MCV, MCH, MCHC, RBC).
- Capitalize test names consistently (e.g., Hemoglobin, Total Leukocyte Count).
- Add a blank line between sections for readability.
- Do not add extra text or deviate from this structure.
- If data is missing, explicitly state '(Missing)' for the value.
- Use 'None' for empty treatments or precautions if none are mentioned.
- Ensure all sections are included, even if empty.
- Use 'their' instead of 'his' for gender-neutral language in the closing statement.

### Medical Report Analysis: Complete Blood Count (CBC)

**Report Overview**  
[Brief summary of the report, its purpose, and limitations like missing data or need for physician consultation.]

#### **Patient Information**
- **Name**: [Name]  
- **Age**: [Age]  
- **Sex**: [Sex, e.g., Male, Female, M, F]  
- **Date of Report**: [Date, format as DD/MM/YYYY]  

#### **Test Results**
[Note about units/reference ranges if applicable, e.g., 'Units and reference ranges are not provided.']  

- **Hemoglobin**: [Value or (Missing)]  
- **Total Leukocyte Count (WBC)**: [Value or (Missing)]  
- **Differential Leukocyte Count**: [Value or (Missing)]  
- **Platelet Count**: [Value or (Missing)]  
- **Total RBC Count**: [Value or (Missing)]  
- **Hematocrit (HCT)**: [Value or (Missing)]  
- **Mean Corpuscular Volume (MCV)**: [Value or (Missing)]  
- **Mean Cell Hemoglobin (MCH)**: [Value or (Missing)]  
- **Mean Cell Hemoglobin Concentration (MCHC)**: [Value or (Missing)]  

#### **Clinical Notes from Report**
- [Note 1 or None if no notes]  
- [Note 2 or None if no notes]  

#### **Potential Diseases/Conditions**
[Speculative conditions based on data; emphasize no diagnosis and need for doctor.]

#### **Treatments Suggested**
[None or list treatments if mentioned.]

#### **Precautions and Recommendations**
[Precautions from report; include general advice to consult a doctor.]

This analysis is for informational purposes only and is not a substitute for professional medical advice. [Patient Name] should follow up with their healthcare provider promptly.

Report text: {text[:2000]}
"""
        response = model.generate_content(prompt)
        cleaned_response = clean_response(response.text)
        return cleaned_response
    except Exception as e:
        return f"Error analyzing report: {str(e)}. Extracted text: '{text[:500]}...'"