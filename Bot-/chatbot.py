from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model_name = "facebook/blenderbot-400M-distill"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
conversations={}

def generate_response(user_input, conversation_id):
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(inputs.input_ids,max_length=100,num_return_sequences=1,temperature=0.7,pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if any(word in user_input.lower() for word in ['suicide', 'kill', 'die']):
        response = "EMERGENCY: Please call 911 immediately for professional help. " + response
    elif any(word in user_input.lower() for word in ['sad', 'depressed', 'anxiety']):
        response = "I hear you're struggling. " + response
    
    return response

@app.route('/chat', methods=['POST'])
def chat():
    data=request.json