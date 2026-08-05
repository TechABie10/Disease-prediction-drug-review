import pandas as pd
import re
import nltk
import joblib
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Initialize NLTK
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
standard_stopwords = set(stopwords.words('english'))
# Preserve medical negations
negation_words = {'not', 'no', 'nor', 'neither', 'never', 'cannot', "isn't", "wasn't", "shouldn't", "wouldn't", "couldn't", "won't"}
custom_stopwords = standard_stopwords - negation_words

def preprocess_text(text):
    text = str(text)
    text = re.sub(r'&#039;', "'", text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text).lower()
    words = text.split()
    return " ".join([lemmatizer.lemmatize(w) for w in words if w not in custom_stopwords])

def build_pipeline():
    print("1. Loading training dataset...")
    df = pd.read_csv('drugsComTrain_raw.csv')
    df = df.dropna(subset=['condition', 'review'])

    print("2. Filtering top 5 conditions...")
    top_5 = df['condition'].value_counts().nlargest(5).index.tolist()
    df_filtered = df[df['condition'].isin(top_5)].copy()

    print("3. Feature Engineering & Preprocessing...")
    df_filtered['combined_text'] = df_filtered['drugName'] + " " + df_filtered['review']
    df_filtered['cleaned_text'] = df_filtered['combined_text'].apply(preprocess_text)

    # 90/10 Split
    X_train, X_test, y_train, y_test = train_test_split(
        df_filtered['cleaned_text'], df_filtered['condition'], 
        test_size=0.10, random_state=42, stratify=df_filtered['condition']
    )

    print("4. Vectorizing Text (Unrestricted Trigrams)...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=None, sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("5. Training Passive Aggressive Classifier...")
    model = PassiveAggressiveClassifier(C=0.5, loss='squared_hinge', max_iter=311, random_state=42)
    model.fit(X_train_tfidf, y_train)

    print(f"Validation Accuracy: {accuracy_score(y_test, model.predict(X_test_tfidf)) * 100:.2f}%")

    print("6. Exporting artifacts to /model...")
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/classifier.pkl')
    joblib.dump(vectorizer, 'model/tfidf_vectorizer.pkl')
    print("Pipeline build complete!")

if __name__ == '__main__':
    build_pipeline()