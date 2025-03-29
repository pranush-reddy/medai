import torch
import re , os
from transformers import BertTokenizer, BertForSequenceClassification
from flask import Flask,render_template, request, jsonify
import logging
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
url=os.getenv("url")

app.config['url'] = url

# Expanded medical reports (diverse samples for each disease)
try:
    client = MongoClient(url)
    db = client.get_database()
    collection = db.medai
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")

# Enhanced disease labels with detailed criteria
disease_labels = {
    0: {
        "name": "No Disease",
        "effects": "No significant health concerns detected",
        "keywords": ["normal", "healthy", "no complaints", "routine checkup"]
    },
    1: {
        "name": "Diabetes",
        "effects": "Risk of nerve damage, kidney failure, vision problems",
        "strict_keywords": [
            r"diabet(es|ic)", r"glucose\s*[>]\s*126", r"hba1c\s*[>]\s*6\.5",
            "polyuria", "polydipsia", "insulin", "metformin"
        ]
    },
    2: {
        "name": "Heart Disease",
        "effects": "Possible heart attack, chest pain, high cholesterol risks",
        "strict_keywords": [
            "chest pain", "angina", "palpitations", "ecg abnormal",
            "troponin", "st elevation", "heart attack", "cardiovascular"
        ]
    },
    3: {
        "name": "Asthma",
        "effects": "Breathing difficulties, potential lung infections",
        "strict_keywords": [
            "wheez(ing|es)", "shortness of breath", "inhaler",
            "bronchospasm", "peak flow", "respiratory distress"
        ]
    },
    4: {
        "name": "Kidney Disease",
        "effects": "May require dialysis, risk of high blood pressure",
        "strict_keywords": [
            r"creatinine\s*[>]\s*1\.5", "dialysis", "proteinuria",
            "hematuria", r"gfr\s*[<]\s*60", "renal failure"
        ]
    },
    5: {
        "name": "Other Disease",
        "effects": "Specific condition requiring medical attention",
        "strict_keywords": [
            "neurological", "mri scan", "thyroid disorder",
            "unexplained weight loss", "night sweats", "chronic headaches"
        ]
    }
}

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = BertForSequenceClassification.from_pretrained('D:/java/nlp/ad/advMAI/fine_tuned_biobert')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def contains_disease_evidence(text, disease_id):
    """Check for disease-specific evidence with context"""
    text_lower = text.lower()
    keywords = disease_labels[disease_id].get("strict_keywords", [])
    
    for pattern in keywords:
        if re.search(pattern, text_lower):
            # Check for negation patterns
            if not re.search(r"(no\s|not\s|negative\sfor\s)" + pattern.split()[0], text_lower):
                return True
    return False

def predict_disease(text):
    """Enhanced prediction with multi-disease detection"""
    try:
        if not text.strip():
            return {
                **disease_labels[0],
                "confidence": 0.0,
                "reason": "Empty input"
            }

        # First check for explicit no disease indicators
        healthy_phrases = [
            "no significant", "normal results", "healthy",
            "no complaints", "no evidence of", "no findings of",
            "unremarkable", "within normal limits", "routine checkup"
        ]
        
        if any(phrase in text.lower() for phrase in healthy_phrases):
            return {
                **disease_labels[0],
                "confidence": 0.95,
                "reason": "Explicit healthy indicators"
            }

        # Check for specific disease evidence before model prediction
        detected_diseases = []
        for disease_id in [1, 2, 3, 4, 5]:  # Skip "No Disease"
            if contains_disease_evidence(text, disease_id):
                detected_diseases.append(disease_id)
        
        # If only one disease detected with high confidence
        if len(detected_diseases) == 1:
            return {
                **disease_labels[detected_diseases[0]],
                "confidence": 0.9,
                "reason": "Clear disease evidence found"
            }

        # Tokenize and get model predictions
        encoding = tokenizer(text, return_tensors="pt", 
                           truncation=True, padding=True, max_length=512)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        
        # Get probabilities and prediction
        probabilities = torch.softmax(outputs.logits, dim=1)[0]
        prediction = torch.argmax(probabilities).item()
        confidence = probabilities[prediction].item()
        
        # For disease predictions, verify with keywords
        if prediction != 0:
            if not contains_disease_evidence(text, prediction):
                logger.info(f"Overriding prediction {prediction} - no evidence found")
                return {
                    **disease_labels[0],
                    "confidence": 1 - confidence,
                    "reason": "No disease-specific evidence detected"
                }
        
        return {
            **disease_labels[prediction],
            "confidence": confidence,
            "reason": "Model prediction with evidence verification"
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {
            **disease_labels[0],
            "confidence": 0.0,
            "reason": "Error in processing"
        }
@app.route("/",methods=['GET'])
def index():
    return render_template("index.html")

@app.route("/feedback", methods=["GET"])
def ret():
    return render_template("feedback.html")

@app.route('/data', methods=['POST'])
def save_feedback():
    try:
        # Get form data
        name = request.form.get('name')
        email = request.form.get('email')
        rating = request.form.get('rating')
        description = request.form.get('description')

        if not name or not email or not description:
            return jsonify({"error": "Missing required fields"}), 400

        # Create a document to insert
        feedback = {
            "name": name,
            "email": email,
            "rating": int(rating) if rating else None,  # Convert rating to integer
            "description": description
        }

        # Insert into MongoDB
        result = collection.insert_one(feedback)

        return jsonify({
            "message": "Feedback saved successfully",
            "id": str(result.inserted_id)
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        
        if not text:
            return jsonify({
                "error": "No text provided",
                "predicted_disease": "Cannot determine",
                "confidence": 0.0
            }), 400
            
        result = predict_disease(text)
        return jsonify({
            "predicted_disease": result["name"],
            "possible_effects": result["effects"],
            "confidence": result["confidence"],
            "reason": result["reason"]
        })
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "predicted_disease": "Cannot determine",
            "confidence": 0.0
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)