# Handles medical chatbot functionality using Gemini LLM
# Updated: Added support for AI reports with transcript and prescription

import google.generativeai as genai
from typing import Optional, Dict, Any
from src.config import GEMINI_API_KEY
from src.logger import setup_logger
from src.rag import get_relevant_contexts

# Configure Gemini with API key
genai.configure(api_key=GEMINI_API_KEY)

logger = setup_logger("chatbot")

class MedicalChatbot:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
    def analyze_prompt(self, user_input: str) -> Dict[str, Any]:
        """Analyze user prompt to determine intent and required actions"""
        analysis_prompt = f"""
        Analyze this user message and determine:
        1. Is this a medical question/concern? (yes/no)
        2. Does it need additional medical knowledge from database? (yes/no)
        3. What type of response is needed? (diagnosis, general_info, symptom_check, emergency, casual)
        4. Urgency level (low, medium, high, emergency)
        
        User message: "{user_input}"
        
        Respond in this exact format:
        Medical: yes/no
        Needs_database: yes/no
        Response_type: [type]
        Urgency: [level]
        Intent: [brief description of what user wants]
        """
        
        try:
            response = self.model.generate_content(analysis_prompt)
            analysis = self._parse_analysis(response.text)
            logger.info(f"Prompt analysis: {analysis}")
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing prompt: {e}")
            return {
                'medical': True,
                'needs_database': True,
                'response_type': 'general_info',
                'urgency': 'medium',
                'intent': 'Medical inquiry'
            }
    
    def _parse_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """Parse the analysis response into structured data"""
        lines = analysis_text.strip().split('\n')
        result = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace('_', '_')
                value = value.strip()
                
                if key == 'medical':
                    result['medical'] = value.lower() == 'yes'
                elif key == 'needs_database':
                    result['needs_database'] = value.lower() == 'yes'
                elif key == 'response_type':
                    result['response_type'] = value
                elif key == 'urgency':
                    result['urgency'] = value
                elif key == 'intent':
                    result['intent'] = value
                    
        return result
    
    def generate_response(self, user_input: str, context: Optional[str] = None, transcript: Optional[str] = None, prescription: Optional[Dict] = None) -> str:
        """Generate response with or without RAG context, transcript, or prescription for AI reports"""
        
        # Analyze prompt to determine intent
        analysis = self.analyze_prompt(user_input)
        
        # Get context from RAG if needed
        rag_context = ""
        if analysis.get('needs_database', False):
            try:
                contexts = get_relevant_contexts(user_input, k=3)
                if contexts:
                    rag_context = "\n".join(contexts[:2])  # Limit to top 2 contexts
                    logger.info(f"Retrieved {len(contexts)} contexts from RAG")
                else:
                    logger.info("No relevant contexts found in RAG")
            except Exception as e:
                logger.error(f"Error getting RAG context: {e}")
        
        # Handle emergency cases
        if analysis.get('urgency') == 'emergency':
            return self._handle_emergency(user_input)
        
        # Build response prompt with optional transcript/prescription
        response_prompt = self._build_response_prompt(
            user_input, analysis, rag_context, transcript, prescription
        )
        
        try:
            response = self.model.generate_content(response_prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please consult a healthcare professional for medical advice."
    
    def _handle_emergency(self, user_input: str) -> str:
        """Handle emergency situations with immediate guidance"""
        return """🚨 **MEDICAL EMERGENCY DETECTED** 🚨

If you are experiencing a medical emergency, please:
1. Call emergency services immediately (911 in US, your local emergency number)
2. Go to the nearest emergency room
3. Contact your doctor immediately

I'm an AI assistant and cannot provide emergency medical care. Please seek immediate professional medical attention."""
    
    def _build_response_prompt(self, user_input: str, analysis: Dict, context: str, transcript: Optional[str] = None, prescription: Optional[Dict] = None) -> str:
        """Build the response generation prompt, including transcript/prescription for AI reports"""
        
        base_prompt = f"""You are a helpful medical AI assistant. 

User question: "{user_input}"

Guidelines:
- Provide helpful, accurate medical information
- Always recommend consulting healthcare professionals for serious concerns
- Be empathetic and supportive
- Keep responses concise but informative
"""
        
        if context:
            base_prompt += f"""

Relevant medical information from database:
{context}

Use this information to provide more specific and accurate guidance.
"""
        
        if transcript:
            base_prompt += f"""

Call Transcript:
{transcript}

Generate a structured report based on this transcript.
"""
        
        if prescription:
            base_prompt += f"""

Include this prescription in the report:
Medication: {prescription.get('medication', '')}
Dosage: {prescription.get('dosage', '')}
Instructions: {prescription.get('instructions', '')}
"""
        
        if analysis.get('response_type') == 'symptom_check':
            base_prompt += """
Focus on:
- Possible causes of the symptoms
- When to seek medical attention  
- Basic care recommendations
- Warning signs to watch for
"""
        
        return base_prompt

# Global chatbot instance
chatbot = MedicalChatbot()

# Backward compatibility functions
def get_disease_info(query: str) -> str:
    """Get disease information using chatbot"""
    return chatbot.generate_response(query)

def simplify_terms(text: str) -> str:
    """Simplify medical terms for general understanding"""
    prompt = f"Simplify this medical text for general understanding: {text}"
    try:
        response = chatbot.model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error simplifying terms: {e}")
        return text