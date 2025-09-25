# Handles medical report analysis from PDFs/images and AI post-processing
# Updated: Integrated with AI report generation from call transcripts

import pdfplumber
import pytesseract
from PIL import Image
import io
import json
from datetime import datetime
from src.logger import setup_logger
from src.chatbot_service import chatbot
from src.rag import embed_text, upsert_to_pinecone, store_ai_report

logger = setup_logger("report_analyzer")

def analyze_report(file_content):
    """Analyze uploaded report (PDF/image)"""
    try:
        # Extract text from file
        text = extract_text_from_file(file_content)
        logger.info(f"Extracted text length: {len(text)} characters")
        
        if not text.strip():
            return "Unable to extract readable text from the report. Please ensure the file is a clear PDF or image."
        
        # Generate comprehensive analysis
        analysis = perform_comprehensive_analysis(text)
        
        # Store in Pinecone
        store_report_in_pinecone(text, analysis)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing report: {e}")
        return f"Error analyzing report: {e}. Please consult a healthcare professional."

def perform_comprehensive_analysis(text, transcript: Optional[str] = None, prescription: Optional[Dict] = None):
    """Perform detailed analysis of medical report or call transcript"""
    
    comprehensive_prompt = f"""
    You are a senior medical doctor and pathologist with 20+ years of experience. Analyze this medical report with extreme attention to detail. Provide a comprehensive analysis that covers everything a patient would want to know.

    MEDICAL REPORT TEXT:
    {text}

    Please provide a DETAILED analysis in the following structured format:

    ## 📋 REPORT OVERVIEW
    - Patient Details: [Extract name, age, gender, date]
    - Report Type: [Blood test, Urine analysis, Imaging, etc.]
    - Laboratory/Hospital: [Where test was conducted]
    - Report Date: [When conducted]
    - Referring Doctor: [If mentioned]

    ## 🔬 DETAILED TEST RESULTS ANALYSIS
    For each test parameter, provide:
    - Test Name & Purpose: [What does this test measure and why is it important]
    - Your Result: [Actual value found]
    - Normal Range: [Reference values]
    - Status: [Normal/Abnormal/Borderline]
    - Clinical Significance: [What this means for your health]

    ## 🚨 CRITICAL FINDINGS & RED FLAGS
    - Immediate Concerns: [Any urgent abnormalities]
    - Values Outside Normal Range: [All abnormal results explained]
    - Trends: [Compare with previous results if dates suggest follow-up]

    ## 🎯 POSSIBLE CONDITIONS & IMPLICATIONS
    ### Short-term Health Implications:
    - What these results suggest about current health status
    - Any immediate symptoms or conditions indicated

    ### Long-term Health Implications:
    - Risk factors identified
    - Potential future health concerns
    - Preventive measures needed

    ## 💊 DOCTOR'S RECOMMENDATIONS & NEXT STEPS
    ### Immediate Actions Required:
    - Urgent medical attention needed (Yes/No and why)
    - Medications that might be prescribed
    - Lifestyle changes recommended

    ### Follow-up Care:
    - Additional tests that may be ordered
    - Specialist referrals likely needed
    - Monitoring schedule recommendations

    ## 🏠 HOME CARE & LIFESTYLE MODIFICATIONS
    ### Dietary Recommendations:
    - Foods to include/avoid based on results
    - Nutritional supplements that might help

    ### Exercise & Activity:
    - Safe activity levels
    - Exercises that could help improve results

    ### Daily Life Management:
    - Practical tips for managing identified conditions
    - Warning signs to watch for

    ## 📚 DETAILED MEDICAL EXPLANATIONS
    ### Technical Terms Simplified:
    [Explain all medical jargon in simple terms]

    ### Why These Tests Were Ordered:
    [Explain the medical reasoning behind each test]

    ### How Results Interconnect:
    [Explain how different test results relate to each other]

    ## 🔍 WHAT PATIENTS TYPICALLY GOOGLE
    ### Common Concerns About These Results:
    - "What does elevated [parameter] mean?"
    - "Should I be worried about [abnormal value]?"
    - "What causes [specific finding]?"

    ### Evidence-Based Answers:
    [Provide scientific, reassuring, but honest explanations]

    ## ⚖️ RISK ASSESSMENT
    ### Overall Health Risk Level: [Low/Moderate/High]
    ### Specific Risk Factors Identified:
    ### Protective Factors Present:

    ## 🎯 PERSONALIZED HEALTH PLAN
    ### 30-Day Action Plan:
    ### 90-Day Goals:
    ### Annual Monitoring Schedule:

    ## ⚠️ IMPORTANT DISCLAIMERS
    - This analysis is for educational purposes
    - Always consult your healthcare provider
    - Don't make treatment decisions based solely on this analysis
    - Seek immediate medical attention if you have concerning symptoms

    REMEMBER: Be thorough, compassionate, and provide hope where appropriate while being honest about concerns. Think like a caring doctor explaining to their own family member.
    """
    
    if transcript:
        comprehensive_prompt += f"\n\nIncorporate this call transcript: {transcript}"
    if prescription:
        comprehensive_prompt += f"\n\nInclude this prescription: {json.dumps(prescription)}"
    
    try:
        response = chatbot.model.generate_content(comprehensive_prompt)
        logger.info("Successfully generated comprehensive report analysis")
        return response.text
    except Exception as e:
        logger.error(f"Error generating comprehensive analysis: {e}")
        return f"Error analyzing report: {e}"

def store_report_in_pinecone(report_text, analysis):
    """Store report and analysis in Pinecone for RAG retrieval"""
    try:
        # Create embeddings
        report_embedding = embed_text(report_text)
        analysis_embedding = embed_text(analysis)
        
        # Generate unique IDs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_id = f"report_{timestamp}_{hash(report_text) % 10000}"
        analysis_id = f"analysis_{timestamp}_{hash(analysis) % 10000}"
        
        # Prepare metadata
        report_metadata = {
            'type': 'medical_report',
            'content': 'original_report',
            'full_text': report_text,
            'timestamp': timestamp,
            'source': 'uploaded_document'
        }
        
        analysis_metadata = {
            'type': 'medical_analysis',
            'content': 'ai_analysis',
            'full_text': analysis,
            'timestamp': timestamp,
            'source': 'ai_generated'
        }
        
        # Store in Pinecone
        vectors = [report_embedding, analysis_embedding]
        ids = [report_id, analysis_id]
        metadata_list = [report_metadata, analysis_metadata]
        
        upsert_to_pinecone(vectors, ids, metadata_list)
        logger.info(f"Successfully stored report and analysis in Pinecone with IDs: {report_id}, {analysis_id}")
        
    except Exception as e:
        logger.error(f"Error storing report in Pinecone: {e}")

def extract_text_from_file(file_content):
    """Extract text from PDF or image file"""
    try:
        text = ""
        
        # Try PDF extraction
        try:
            with io.BytesIO(file_content) as pdf_file:
                with pdfplumber.open(pdf_file) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n"
                            text += page_text + "\n"
                            
            if text.strip():
                logger.info(f"Successfully extracted text from PDF: {len(text)} characters")
                return text.strip()
                
        except Exception as pdf_error:
            logger.warning(f"PDF extraction failed: {pdf_error}")
        
        # Try OCR if PDF fails
        try:
            image = Image.open(io.BytesIO(file_content))
            image = image.convert('RGB')
            text = pytesseract.image_to_string(image, config='--psm 6')
            logger.info(f"Successfully extracted text using OCR: {len(text)} characters")
            return text.strip()
            
        except Exception as ocr_error:
            logger.error(f"OCR extraction failed: {ocr_error}")
        
        return ""
        
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return ""