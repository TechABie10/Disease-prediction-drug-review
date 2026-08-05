# Disease Prediction Using Drug Reviews

This project utilizes Natural Language Processing (NLP) to predict patient medical conditions across 5 distinct classes based on over 50,000 drug reviews[cite: 1].

## Architecture
- **Feature Engineering:** Concatenation of `drugName` and `review` text for high-confidence signaling.
- **Preprocessing:** Custom stopword filtration preserving critical linguistic negations (e.g., "not", "no").
- **Vectorization:** TF-IDF extraction utilizing unrestricted trigrams (1, 3) resulting in ~1.6M features.
- **Modeling:** Highly optimized `PassiveAggressiveClassifier` achieving >98% accuracy.
- **Web App:** Deployed as a RESTful web interface using Flask.

## Quick Start
1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
