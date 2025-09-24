import google.generativeai as genai
from src.config import GEMINI_API_KEY
from src.rag import get_relevant_contexts, store_interaction
from src.logger import setup_logger

logger = setup_logger("chatbot_service")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def get_disease_info(question: str) -> str:
    try:
        contexts = get_relevant_contexts(question)
        prompt = f"Question: {question}\nRelevant info: {contexts}\nAnswer based on info:"
        response = model.generate_content(prompt)
        store_interaction(question, response.text)
        return response.text
    except Exception as e:
        logger.error(f"Error in get_disease_info: {str(e)}")
        return f"Error: {str(e)}"

def simplify_terms(term: str) -> str:
    try:
        prompt = f"Simplify the medical term '{term}' in simple language:"
        response = model.generate_content(prompt)
        store_interaction(term, response.text, type='simplify')
        return response.text
    except Exception as e:
        logger.error(f"Error in simplify_terms: {str(e)}")
        return f"Error: {str(e)}"