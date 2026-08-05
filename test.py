import pandas as pd
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, classification_report

def run_tests():
    print("Loading test data...")
    try:
        df_test = pd.read_csv('drugsComTest_raw.csv')
    except FileNotFoundError:
        print("Please place 'drugsComTest_raw.csv' in the root directory.")
        return

    model = joblib.load('model/classifier.pkl')
    vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

    df_test = df_test.dropna(subset=['condition', 'review'])
    top_5 = ['Birth Control', 'Depression', 'Pain', 'Anxiety', 'Acne']
    df_test = df_test[df_test['condition'].isin(top_5)].copy()

    lemmatizer = WordNetLemmatizer()
    custom_stopwords = set(stopwords.words('english')) - {'not', 'no', 'nor', 'neither', 'never', 'cannot', "isn't", "wasn't", "shouldn't", "wouldn't", "couldn't", "won't"}

    def clean(text):
        text = re.sub(r'&#039;', "'", str(text))
        text = re.sub(r'[^a-zA-Z\s]', ' ', text).lower()
        return " ".join([lemmatizer.lemmatize(w) for w in text.split() if w not in custom_stopwords])

    print("Testing Holdout Accuracy...")
    df_test['combined'] = df_test['drugName'] + " " + df_test['review']
    X_test_tfidf = vectorizer.transform(df_test['combined'].apply(clean))
    
    y_pred = model.predict(X_test_tfidf)
    print(f"\n--- HOLDOUT ACCURACY: {accuracy_score(df_test['condition'], y_pred) * 100:.2f}% ---\n")
    print(classification_report(df_test['condition'], y_pred))

if __name__ == '__main__':
    run_tests()