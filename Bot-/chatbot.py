from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import logging
import re
from typing import List, Dict, Tuple, Optional
import torch
import numpy as np
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Initialize the medical models globally
logger.info("Loading medical models...")
try:
    # Main classification model
    medical_model = pipeline(
        "zero-shot-classification",
        model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Severity analysis model
    severity_model = pipeline(
        "text-classification",
        model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        device=0 if torch.cuda.is_available() else -1
    )
    
    logger.info("Medical models loaded successfully")
except Exception as e:
    logger.error(f"Error loading medical models: {e}")
    raise

# Symptom patterns for better matching
SYMPTOM_PATTERNS = {
    "duration": r"(?:for|since|last|past)\s+(\d+)\s+(?:day|days|week|weeks|hour|hours|month|months)",
    "frequency": r"(?:every|each|per)\s+(\d+)\s+(?:hour|hours|day|days|week|weeks)",
    "intensity": r"(mild|moderate|severe|extreme|slight|intense)\s+(?:pain|discomfort|symptoms?)",
    "progression": r"(getting|becoming|grown|developed)\s+(worse|better|severe|intense)",
    "time_of_day": r"(morning|evening|night|afternoon|daily|constantly|intermittently)",
}

# Enhanced medical conditions with more detailed symptoms and relationships
MEDICAL_CONDITIONS = {
    "Common Cold": {
        "symptoms": [
            "runny nose", "sneezing", "congestion", "mild fever", "sore throat", 
            "cough", "fatigue", "body aches", "mild headache"
        ],
        "severity": "mild",
        "description": "A viral infection of the upper respiratory tract",
        "typical_duration": "7-10 days"
    },
    "Flu (Influenza)": {
        "symptoms": [
            "high fever", "severe body aches", "extreme fatigue", "headache", 
            "dry cough", "congestion", "chills", "sweats", "weakness"
        ],
        "severity": "moderate",
        "description": "A viral infection that attacks your respiratory system",
        "typical_duration": "1-2 weeks"
    },
    "COVID-19": {
        "symptoms": [
            "fever", "dry cough", "fatigue", "loss of taste", "loss of smell",
            "difficulty breathing", "body aches", "headache", "sore throat",
            "congestion", "nausea", "diarrhea"
        ],
        "severity": "moderate to severe",
        "description": "A viral infection caused by the SARS-CoV-2 virus",
        "typical_duration": "2-6 weeks"
    },
    "Migraine": {
        "symptoms": [
            "severe headache", "throbbing pain", "sensitivity to light",
            "sensitivity to sound", "nausea", "vomiting", "visual aura",
            "dizziness", "fatigue"
        ],
        "severity": "moderate to severe",
        "description": "A neurological condition causing severe headaches",
        "typical_duration": "4-72 hours"
    },
    "Acute Bronchitis": {
        "symptoms": [
            "persistent cough", "chest congestion", "wheezing", "shortness of breath",
            "low fever", "chest discomfort", "fatigue", "mucus production"
        ],
        "severity": "moderate",
        "description": "Inflammation of the bronchial tubes",
        "typical_duration": "2-3 weeks"
    },
    "Gastroenteritis": {
        "symptoms": [
            "nausea", "vomiting", "diarrhea", "stomach cramps", "mild fever",
            "headache", "muscle aches", "loss of appetite", "dehydration"
        ],
        "severity": "mild to moderate",
        "description": "Inflammation of the stomach and intestines",
        "typical_duration": "1-3 days"
    },
    "Allergic Reaction": {
        "symptoms": [
            "rash", "hives", "itching", "swelling", "difficulty breathing",
            "runny nose", "watery eyes", "sneezing", "coughing"
        ],
        "severity": "mild to severe",
        "description": "Immune system response to allergens",
        "typical_duration": "hours to days"
    },
    "Anxiety Attack": {
        "symptoms": [
            "rapid heartbeat", "sweating", "trembling", "shortness of breath",
            "chest pain", "dizziness", "fear", "numbness", "tingling"
        ],
        "severity": "moderate",
        "description": "Intense period of anxiety with physical symptoms",
        "typical_duration": "minutes to hours"
    },
    "Asthma Attack": {
        "symptoms": [
            "wheezing", "severe shortness of breath", "chest tightness",
            "coughing", "rapid breathing", "anxiety", "difficulty speaking"
        ],
        "severity": "moderate to severe",
        "description": "Narrowing and inflammation of airways",
        "typical_duration": "minutes to hours"
    },
    "Acid Reflux": {
        "symptoms": [
            "heartburn", "chest pain", "difficulty swallowing", "regurgitation",
            "sour taste", "throat irritation", "coughing", "hoarseness"
        ],
        "severity": "mild to moderate",
        "description": "Backward flow of stomach acid into the esophagus",
        "typical_duration": "minutes to hours"
    }
}

# Add symptom relationships for better diagnosis
SYMPTOM_RELATIONSHIPS = {
    "fever": ["headache", "body aches", "fatigue", "chills"],
    "cough": ["chest pain", "shortness of breath", "wheezing", "congestion"],
    "headache": ["nausea", "sensitivity to light", "dizziness", "neck pain"],
    "nausea": ["vomiting", "dizziness", "stomach pain", "loss of appetite"],
    "chest pain": ["shortness of breath", "anxiety", "sweating", "arm pain"],
}

EMERGENCY_SYMPTOMS = {
    "severe chest pain": "Could indicate heart attack",
    "difficulty breathing": "Could indicate severe allergic reaction or respiratory distress",
    "severe bleeding": "Requires immediate medical attention",
    "loss of consciousness": "Could indicate various serious conditions",
    "stroke symptoms": "Including sudden numbness, confusion, difficulty speaking",
    "severe head injury": "Could lead to brain damage if untreated",
    "severe abdominal pain": "Could indicate appendicitis or other serious conditions",
    "suicidal thoughts": "Requires immediate mental health intervention",
    "severe allergic reaction": "Including swelling of face/throat, difficulty breathing"
}

SEVERITY_INDICATORS = {
    "mild": ["mild", "slight", "minor", "light"],
    "moderate": ["moderate", "medium", "significant"],
    "severe": ["severe", "intense", "extreme", "very", "serious"]
}

MEDICAL_DISCLAIMER = """
🏥 IMPORTANT MEDICAL DISCLAIMER:
This is not a substitute for professional medical advice. 
These are potential conditions based on AI analysis of your symptoms, but only a qualified healthcare provider can give you a proper diagnosis.
If symptoms persist or worsen, please consult with a healthcare professional immediately.

Remember:
• This is an AI-powered analysis
• It's not a definitive diagnosis
• Always seek professional medical advice
• Call emergency services for severe symptoms
"""

# Add a new dictionary for follow-up questions based on symptoms
FOLLOW_UP_QUESTIONS = {
    "headache": [
        "How long have you had this headache?",
        "On a scale of 1-10, how severe is the pain?",
        "Is it constant or does it come and go?",
        "Does anything make it better or worse?",
        "Do you have any other symptoms like nausea or sensitivity to light?"
    ],
    "cough": [
        "How long have you been coughing?",
        "Is it a dry cough or are you producing mucus?",
        "How frequent is the coughing?",
        "Do you have any other symptoms like fever or chest pain?",
        "Is it worse at certain times of day?"
    ],
    "fever": [
        "Do you know your current temperature?",
        "How long have you had the fever?",
        "Have you taken any medication for it?",
        "Do you have any other symptoms?",
        "Does the fever come and go or is it constant?"
    ],
    "pain": [
        "Where exactly is the pain located?",
        "How long have you had this pain?",
        "On a scale of 1-10, how severe is it?",
        "Is it constant or does it come and go?",
        "Does anything make it better or worse?"
    ],
    "breathing": [
        "How long have you had breathing difficulties?",
        "Does it happen at rest or with activity?",
        "Do you have chest pain or tightness?",
        "Is it getting worse over time?",
        "Do you have any other symptoms?"
    ],
    "rash": [
        "Where is the rash located?",
        "How long have you had it?",
        "Is it itchy or painful?",
        "Have you been exposed to anything new?",
        "Does anything make it better or worse?"
    ],
    "stomach": [
        "What type of stomach discomfort are you experiencing?",
        "How long has this been going on?",
        "Have you noticed any patterns with eating?",
        "Do you have any other symptoms like nausea or vomiting?",
        "On a scale of 1-10, how severe is the discomfort?"
    ],
    "fatigue": [
        "How long have you been feeling tired?",
        "Is it constant or worse at certain times?",
        "Are you getting enough sleep?",
        "Do you have any other symptoms?",
        "Has anything changed in your routine recently?"
    ]
}

def get_follow_up_questions(symptoms: str) -> List[str]:
    """Get relevant follow-up questions based on symptoms"""
    questions = []
    symptoms_lower = symptoms.lower()
    
    # Check each symptom category and add relevant questions
    for symptom, follow_ups in FOLLOW_UP_QUESTIONS.items():
        if symptom in symptoms_lower:
            questions.extend(follow_ups)
    
    # If no specific questions found, add general questions
    if not questions:
        questions = [
            "How long have you had these symptoms?",
            "On a scale of 1-10, how severe are your symptoms?",
            "Do the symptoms come and go or are they constant?",
            "Does anything make them better or worse?",
            "Do you have any other symptoms?"
        ]
    
    # Return top 3 most relevant questions
    return questions[:3]

def check_emergency(text: str) -> Tuple[bool, str]:
    """Check if the symptoms described are emergency conditions"""
    text = text.lower()
    for symptom, explanation in EMERGENCY_SYMPTOMS.items():
        if symptom in text:
            return True, explanation
    return False, ""

def analyze_severity(text: str) -> float:
    """Analyze symptom severity using the medical model"""
    try:
        # Use the severity model to classify the text
        result = severity_model(text)
        # Convert the output score to a severity value between 0 and 1
        severity_score = result[0]['score']
        return severity_score
    except Exception as e:
        logger.error(f"Error in severity analysis: {e}")
        return 0.5

class ConversationState:
    def __init__(self):
        self.current_symptoms = {}  # Store symptom information
        self.follow_up_questions = []  # Queue of follow-up questions
        self.current_question_index = 0  # Track which question we're on
        self.initial_complaint = ""  # Store initial symptom description

    def start_new_conversation(self, initial_input: str):
        """Start a new conversation with initial symptoms"""
        self.current_symptoms = {}
        self.follow_up_questions = []
        self.current_question_index = 0
        self.initial_complaint = initial_input

    def add_answer(self, answer: str):
        """Add answer to current symptoms"""
        if self.follow_up_questions:
            current_question = self.follow_up_questions[self.current_question_index - 1]
            self.current_symptoms[current_question] = answer

    def get_next_question(self) -> Optional[str]:
        """Get next follow-up question if available"""
        if self.current_question_index < len(self.follow_up_questions):
            question = self.follow_up_questions[self.current_question_index]
            self.current_question_index += 1
            return question
        return None

    def has_enough_info(self) -> bool:
        """Check if we have gathered enough information"""
        return self.current_question_index >= len(self.follow_up_questions)

    def get_full_description(self) -> str:
        """Combine all gathered information into a detailed description"""
        description = self.initial_complaint + ". "
        for question, answer in self.current_symptoms.items():
            # Extract the key information from the answer
            description += f"{answer}. "
        return description

# Create a global conversation state manager
conversation_state = ConversationState()

def analyze_symptoms(user_input: str) -> str:
    """Analyze symptoms using the medical AI models"""
    global conversation_state

    # Check for emergency conditions first
    is_emergency, emergency_explanation = check_emergency(user_input)
    if is_emergency:
        conversation_state = ConversationState()  # Reset state
        return f"🚨 EMERGENCY: {emergency_explanation}\nPlease seek immediate medical attention or call emergency services."

    try:
        # If this is a new conversation (no follow-up questions pending)
        if not conversation_state.follow_up_questions:
            # Check if the description is too vague
            word_count = len(user_input.split())
            has_duration = any(word in user_input.lower() for word in ['day', 'days', 'week', 'weeks', 'hour', 'hours', 'month', 'months'])
            has_severity = any(word in user_input.lower() for word in ['mild', 'moderate', 'severe', 'slight', 'intense', 'bad'])
            
            if word_count < 10 or not (has_duration or has_severity):
                # Start new conversation with follow-up questions
                conversation_state.start_new_conversation(user_input)
                conversation_state.follow_up_questions = get_follow_up_questions(user_input)
                
                # Return first question
                first_question = conversation_state.get_next_question()
                return f"To better understand your symptoms, please answer this question:\n\n{first_question}"
        else:
            # Process answer to previous question
            conversation_state.add_answer(user_input)
            
            # Check if we have more questions
            next_question = conversation_state.get_next_question()
            if next_question:
                return f"Thank you. Please answer this follow-up question:\n\n{next_question}"
            
            # If we have all answers, use the complete description for analysis
            user_input = conversation_state.get_full_description()

        # Get severity score
        severity_score = analyze_severity(user_input)
        logger.info(f"Severity score: {severity_score}")

        # Use medical model for condition classification
        result = medical_model(
            user_input,
            candidate_labels=list(MEDICAL_CONDITIONS.keys()),
            multi_label=True,
            hypothesis_template="The patient has {}"
        )
        logger.info(f"Model results: {result}")

        # Process results with higher confidence threshold
        enhanced_results = []
        for label, score in zip(result["labels"], result["scores"]):
            if score > 0.35:
                condition_info = MEDICAL_CONDITIONS[label]
                matching_symptoms = [s for s in condition_info["symptoms"] if s.lower() in user_input.lower()]
                
                if matching_symptoms:
                    symptom_score = len(matching_symptoms) / len(condition_info["symptoms"])
                    adjusted_score = (score + symptom_score) / 2
                    
                    if severity_score > 0.7 and "severe" in condition_info["severity"]:
                        adjusted_score *= 1.2
                    elif severity_score < 0.3 and "mild" in condition_info["severity"]:
                        adjusted_score *= 1.2

                    enhanced_results.append((label, adjusted_score, condition_info, matching_symptoms))

        if enhanced_results:
            # Reset conversation state after successful analysis
            conversation_state = ConversationState()
            
            enhanced_results.sort(key=lambda x: x[1], reverse=True)
            response = "Based on AI analysis of your symptoms:\n\n"
            severity_text = "severe" if severity_score > 0.7 else "moderate" if severity_score > 0.3 else "mild"
            response += f"Severity Assessment: {severity_text.title()}\n\n"
            
            for condition, confidence, info, matching_symptoms in enhanced_results[:3]:
                response += f"🔍 {condition} (Confidence: {confidence:.1%})\n"
                response += f"• Matching symptoms: {', '.join(matching_symptoms)}\n"
                response += f"• Other potential symptoms: {', '.join(s for s in info['symptoms'] if s not in matching_symptoms)}\n"
                response += f"• Severity level: {info['severity']}\n"
                response += f"• Description: {info['description']}\n"
                response += f"• Typical duration: {info['typical_duration']}\n\n"
            
            response += MEDICAL_DISCLAIMER
            return response
        else:
            # If no conditions found but we have complete information
            if conversation_state.has_enough_info():
                conversation_state = ConversationState()  # Reset state
                return ("Based on the information provided, I cannot confidently identify any specific conditions. "
                       "Please consult with a healthcare professional for a proper diagnosis.")
            
            # If we still need more information
            next_question = conversation_state.get_next_question()
            if next_question:
                return f"Please answer this follow-up question:\n\n{next_question}"
            
            # Fallback response
            conversation_state = ConversationState()  # Reset state
            return "I need more specific information about your symptoms. Could you please describe them in more detail?"

    except Exception as e:
        logger.error(f"Error in medical analysis: {e}")
        conversation_state = ConversationState()  # Reset state on error
        return "I apologize, but I'm having trouble analyzing your symptoms. Please try describing them differently or consult with a healthcare professional."

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == "OPTIONS":
        logger.info("Received OPTIONS request")
        return make_response(jsonify({"status": "ok"}), 200)

    try:
        logger.info("Received chat request")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Request data: {request.get_data()}")
        
        data = request.json
        if not data:
            logger.error("No JSON data received")
            return jsonify({'error': 'No data received'}), 400
        
        user_input = data.get('message', '')
        if not user_input:
            logger.error("No message in request")
            return jsonify({'error': 'No message provided'}), 400

        logger.info(f"Processing message: {user_input}")
        response = analyze_symptoms(user_input)
        logger.info(f"Generated response: {response}")
        
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(debug=True, port=5001, host='0.0.0.0')