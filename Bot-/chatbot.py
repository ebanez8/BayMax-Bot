from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import logging
import re
from typing import List, Dict, Tuple
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

# Load multiple medical models for better accuracy
try:
    logger.info("Loading medical models...")
    
    # Main medical model for general classification
    medical_model = pipeline(
        "zero-shot-classification",
        model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    )
    
    # Specialized model for symptom severity
    severity_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    severity_model = AutoModelForSequenceClassification.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    
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

def extract_severity(text: str) -> str:
    """Determine the severity level from the text"""
    text = text.lower()
    for level, indicators in SEVERITY_INDICATORS.items():
        if any(indicator in text for indicator in indicators):
            return level
    return "unknown"

def check_emergency(text: str) -> Tuple[bool, str]:
    """Check if the symptoms described are emergency conditions"""
    text = text.lower()
    for symptom, explanation in EMERGENCY_SYMPTOMS.items():
        if symptom in text:
            return True, explanation
    return False, ""

def count_matching_symptoms(user_input: str, condition_symptoms: List[str]) -> int:
    """Count how many symptoms of a condition match the user's input"""
    user_input = user_input.lower()
    return sum(1 for symptom in condition_symptoms if symptom in user_input)

def extract_symptom_patterns(text: str) -> Dict[str, List[str]]:
    """Extract detailed symptom patterns from text"""
    patterns = defaultdict(list)
    for pattern_type, regex in SYMPTOM_PATTERNS.items():
        matches = re.finditer(regex, text.lower())
        patterns[pattern_type].extend(match.group(1) for match in matches)
    return patterns

def calculate_symptom_relationships(symptoms: List[str]) -> float:
    """Calculate relationship score between symptoms"""
    relationship_score = 0
    for symptom in symptoms:
        if symptom in SYMPTOM_RELATIONSHIPS:
            related = SYMPTOM_RELATIONSHIPS[symptom]
            relationship_score += sum(1 for s in symptoms if s in related)
    return relationship_score / (len(symptoms) or 1)

def analyze_severity(text: str) -> float:
    """Analyze symptom severity using the specialized model"""
    try:
        inputs = severity_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = severity_model(**inputs)
            severity_score = torch.sigmoid(outputs.logits).item()
        return severity_score
    except Exception as e:
        logger.error(f"Error in severity analysis: {e}")
        return 0.5

def analyze_symptoms(user_input: str) -> str:
    """Enhanced symptom analysis using multiple models and pattern matching"""
    # Check for emergency conditions first
    is_emergency, emergency_explanation = check_emergency(user_input)
    if is_emergency:
        return f"🚨 EMERGENCY: {emergency_explanation}\nPlease seek immediate medical attention or call emergency services."

    try:
        # Extract symptom patterns
        patterns = extract_symptom_patterns(user_input)
        
        # Get severity score
        severity_score = analyze_severity(user_input)
        
        # Use main model for initial classification
        result = medical_model(
            user_input,
            candidate_labels=list(MEDICAL_CONDITIONS.keys()),
            multi_label=True
        )

        # Enhanced analysis with multiple factors
        enhanced_results = []
        for label, score in zip(result["labels"], result["scores"]):
            condition_info = MEDICAL_CONDITIONS[label]
            
            # Count matching symptoms
            matching_symptoms = [s for s in condition_info["symptoms"] if s in user_input.lower()]
            symptom_match_score = len(matching_symptoms) / len(condition_info["symptoms"])
            
            # Calculate relationship score
            relationship_score = calculate_symptom_relationships(matching_symptoms)
            
            # Calculate pattern match score
            pattern_score = sum(len(patterns[p]) > 0 for p in patterns) / len(SYMPTOM_PATTERNS)
            
            # Combine scores with weights
            adjusted_score = (
                score * 0.3 +                  # Base model score
                symptom_match_score * 0.3 +    # Symptom matching
                relationship_score * 0.2 +     # Symptom relationships
                pattern_score * 0.2            # Pattern matching
            )
            
            # Adjust for severity
            if severity_score > 0.7 and "severe" in condition_info["severity"]:
                adjusted_score *= 1.2
            elif severity_score < 0.3 and "mild" in condition_info["severity"]:
                adjusted_score *= 1.2

            if adjusted_score > 0.4:  # Higher threshold for more accuracy
                enhanced_results.append((label, adjusted_score, condition_info))

        if enhanced_results:
            # Sort by adjusted score
            enhanced_results.sort(key=lambda x: x[1], reverse=True)
            
            response = "Based on comprehensive AI analysis of your symptoms:\n\n"
            
            # Add pattern information if available
            if any(patterns.values()):
                response += "📊 Symptom Analysis:\n"
                if patterns["duration"]:
                    response += f"• Duration: {', '.join(patterns['duration'])}\n"
                if patterns["frequency"]:
                    response += f"• Frequency: {', '.join(patterns['frequency'])}\n"
                if patterns["intensity"]:
                    response += f"• Intensity: {', '.join(patterns['intensity'])}\n"
                if patterns["progression"]:
                    response += f"• Progression: {', '.join(patterns['progression'])}\n"
                if patterns["time_of_day"]:
                    response += f"• Time pattern: {', '.join(patterns['time_of_day'])}\n"
                response += "\n"
            
            # Add top conditions
            for condition, confidence, info in enhanced_results[:3]:
                response += f"🔍 {condition} (Confidence: {confidence:.1%})\n"
                response += f"• Matching symptoms: {', '.join(s for s in info['symptoms'] if s in user_input.lower())}\n"
                response += f"• Other common symptoms: {', '.join(s for s in info['symptoms'] if s not in user_input.lower())}\n"
                response += f"• Severity level: {info['severity']}\n"
                response += f"• Description: {info['description']}\n"
                response += f"• Typical duration: {info['typical_duration']}\n\n"
            
            response += MEDICAL_DISCLAIMER
            return response
        else:
            return ("I couldn't identify any specific medical conditions with high confidence based on the symptoms you described. "
                   "Please provide more details about your symptoms, such as:\n"
                   "• How long you've had them\n"
                   "• Their severity\n"
                   "• Any patterns or triggers\n"
                   "• Related symptoms\n\n"
                   "This will help me provide a more accurate analysis.")

    except Exception as e:
        logger.error(f"Error in medical analysis: {e}")
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