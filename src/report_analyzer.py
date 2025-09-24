import re
import io
import pdfplumber
from PIL import Image
import pytesseract
from src.chatbot_service import model
from src.rag import store_report_analysis
from src.logger import setup_logger

logger = setup_logger("report_analyzer")

def clean_response(response_text: str) -> str:
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
    
    response_text = re.sub(r'\b([A-Za-z]+)\b(?=:\s+\d|\(Missing\))', lambda m: m.group(1).title(), response_text)
    response_text = re.sub(r'(\n#+\s)', r'\n\n\1', response_text)
    response_text = re.sub(r'(\n\*\*.*\*\*)', r'\n\n\1', response_text)
    response_text = re.sub(r'-\s*([^\s])', r'- \1', response_text)
    response_text = re.sub(r'\n{3,}', r'\n\n', response_text)
    return response_text.strip()

def analyze_report(file_content: bytes) -> str:
    try:
        text = ''
        with io.BytesIO(file_content) as file_io:
            with pdfplumber.open(file_io) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
                    else:
                        logger.warning(f"No text extracted from page {page.page_number}. Attempting OCR...")
                        img = page.to_image(resolution=300)
                        img_byte_arr = io.BytesIO()
                        img.original.save(img_byte_arr, format='PNG')
                        img_byte_arr.seek(0)
                        ocr_text = pytesseract.image_to_string(Image.open(img_byte_arr))
                        text += ocr_text + '\n'

        if not text.strip():
            return "Error: No text could be extracted from the PDF. It may be fully image-based or corrupted."

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
        store_report_analysis(text, cleaned_response)
        return cleaned_response
    except Exception as e:
        logger.error(f"Error analyzing report: {str(e)}")
        return f"Error analyzing report: {str(e)}. Extracted text: '{text[:500]}...'"