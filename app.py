import os
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import PyPDF2
import re
import io  # Import io for handling in-memory files
from dotenv import load_dotenv
from pymongo import MongoClient
load_dotenv()

app = Flask(__name__)
# Accessing the variables
url=os.getenv("url")

app.config['url'] = url

# Expanded medical reports (diverse samples for each disease)
try:
    client = MongoClient(url)
    db = client.get_database()
    collection = db.medai
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
# Expanded medical reports (diverse samples for each disease)
sample_reports = [
    # Cardiovascular Disease
    "The patient experiences chest pain, shortness of breath, and fatigue, diagnosed with heart disease.",
    "The patient has a history of hypertension, palpitations, and recent chest pain suggesting cardiovascular disease.",
    "Patient presents with irregular heartbeat and fatigue, recently diagnosed with coronary artery disease.",
    "A patient reports dizziness, heart palpitations, and pain in the chest, indicating potential heart failure.",
    
    # Diabetes
    "The patient frequently urinates and experiences excessive thirst and hunger, symptoms of diabetes.",
    "High blood sugar levels, fatigue, and frequent urination indicate that the patient may have diabetes.",
    "Patient has a family history of diabetes and reports excessive thirst and hunger.",
    "The patient has blurry vision, excessive thirst, and frequent urination, which are classic signs of diabetes.",
    
    # Respiratory Diseases
    "The patient presents with wheezing, shortness of breath, and persistent coughing, suggesting asthma.",
    "Patient with difficulty breathing, cough, and wheezing likely has an allergic reaction or asthma.",
    "Shortness of breath, coughing, and a tight chest are symptoms of an asthma attack.",
    "Patient exhibits shortness of breath and fever, signs of a possible respiratory infection.",
    
    # Infectious Diseases
    "The patient shows fever, cough, and sore throat, which are typical symptoms of the flu.",
    "The patient has a persistent cough, fever, and a sore throat, indicative of a viral infection.",
    "Patient presents with fatigue, body aches, and a high fever, likely due to an influenza virus.",
    "Fever, chills, and a dry cough in the patient point towards a respiratory infection.",
    
    # Cancer
    "A CT scan reveals a mass in the lung, diagnosed as lung cancer.",
    "The patient has persistent cough, chest pain, and difficulty breathing, diagnosed with lung cancer.",
    "The patient presents with unexplained weight loss and fatigue, symptoms suggestive of lung cancer.",
    "Based on biopsy results, the patient is diagnosed with stage 3 lung cancer.",
    
]

# Corresponding labels for the expanded reports
sample_labels = [
    # Cardiovascular Disease
    "Cardiovascular Disease", 
    "Cardiovascular Disease", 
    "Cardiovascular Disease", 
    "Cardiovascular Disease",

    # Diabetes
    "Diabetes", 
    "Diabetes", 
    "Diabetes", 
    "Diabetes",

    # Respiratory Diseases
    "Respiratory Diseases", 
    "Respiratory Diseases", 
    "Respiratory Diseases", 
    "Respiratory Diseases",

    # Infectious Diseases
    "Infectious Diseases", 
    "Infectious Diseases", 
    "Infectious Diseases", 
    "Infectious Diseases",

    # Cancer
    "Cancer", 
    "Cancer", 
    "Cancer", 
    "Cancer",

]



# (Your sample_reports and sample_labels remain the same as before)

# TF-IDF Vectorizer and Naive Bayes model setup remain unchanged
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(sample_reports)
y = sample_labels

classifier = MultinomialNB()
classifier.fit(X, y)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to process uploaded file and extract text from PDF in memory
def extract_text_from_file(file):
    if file and allowed_file(file.filename):
        try:
            # Use io.BytesIO to handle the file in memory
            file_stream = io.BytesIO(file.read())
            pdf_reader = PyPDF2.PdfReader(file_stream)
            
            # Check if the PDF is encrypted
            if pdf_reader.is_encrypted:
                try:
                    pdf_reader.decrypt('')
                except Exception as e:
                    return f"Error: The PDF is encrypted and cannot be read. {e}"
            
            text = ""
            for page in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page].extract_text()
            return text
        except Exception as e:
            return f"Error extracting text from PDF: {e}"
    else:
        return "File type not supported"

def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text.lower())  # Remove any special characters
    text = text.strip()  # Remove leading/trailing spaces
    return text

@app.route("/feedback", methods=["GET"])
def ret():
    return render_template("feedback.html")
@app.route('/data', methods=['POST'])
def data():
    try:
        # Get the form data
        data_received = request.form.get('data')
        
        # If no data received, return a bad request response
        if not data_received:
            return jsonify({"error": "No data received"}), 400
        
        return jsonify({"message": "Data received successfully", "data": data_received}), 200
    
    except Exception as e:
        # Handle any exceptions that occur
        return jsonify({"error": str(e)}), 500  # Internal Server Error

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_disease = None
    seriousness = None
    if request.method == "POST":
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Process the file directly in memory (no need to save to uploads)
        text = extract_text_from_file(file)  # Pass the actual file object here
        if "Error" in text:
            return jsonify({"error": text}), 400

        text = preprocess_text(text)  # Preprocess the text
        
        # Use the disease prediction model (classification)
        vectorized_text = vectorizer.transform([text])  # Transform extracted text into features
        predicted_disease = classifier.predict(vectorized_text)[0]
        
        # Get the prediction probabilities
        prediction_probabilities = classifier.predict_proba(vectorized_text)[0]
        mx = max(prediction_probabilities)
        
        # Check if the prediction confidence is below a threshold (e.g., 0.5)
        if mx > 0.5:
            predicted_disease = "Cannot determine"
        elif mx >= 0.4:
            seriousness = "high"
        elif mx >= 0.3:
            seriousness = "moderate"
        else:
            seriousness = "low"

    print(predicted_disease)
    return render_template("index.html", predicted_disease=predicted_disease, seriousness=seriousness)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
  # Use a different port

    app.run(debug=False)
