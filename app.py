from flask import Flask, request, render_template
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

app = Flask(__name__)

# Ensure dependencies are available at runtime
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Define preprocessing identically to training
lemmatizer = WordNetLemmatizer()
standard_stopwords = set(stopwords.words('english'))
negation_words = {'not', 'no', 'nor', 'neither', 'never', 'cannot', "isn't", "wasn't", "shouldn't", "wouldn't", "couldn't", "won't"}
custom_stopwords = standard_stopwords - negation_words

def preprocess_text(text):
    text = str(text)
    text = re.sub(r'&#039;', "'", text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text).lower()
    words = text.split()
    return " ".join([lemmatizer.lemmatize(w) for w in words if w not in custom_stopwords])

# Load Model Artifacts globally
model_path = 'model/classifier.pkl'
vectorizer_path = 'model/tfidf_vectorizer.pkl'

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
else:
    model, vectorizer = None, None

@app.route('/')
def home():
    if not model:
        return "Model not found. Please run train.py first.", 500
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    drug_name = request.form.get('drug_name', '')
    review_text = request.form.get('review_text', '')
    
    # Replicate combined feature
    combined_input = f"{drug_name} {review_text}"
    cleaned_input = preprocess_text(combined_input)
    
    transformed_input = vectorizer.transform([cleaned_input])
    prediction = model.predict(transformed_input)[0]
    
    return render_template('index.html', prediction=prediction, drug_name=drug_name, review_text=review_text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)