import pandas as pd
import numpy as np
import joblib
import re
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer

DATA_FILEPATH = 'emails.csv'
TEXT_COLUMN = 'text'
LABEL_COLUMN = 'spam'
SCAM_LABEL_VALUE = 1

MODEL_ARTIFACTS_PATH = 'scam_model_artifacts.joblib'

MODEL_PATH = 'scam_detector_model.joblib'
VECTORIZER_PATH = 'tfidf_vectorizer.joblib'
SCALER_PATH = 'metadata_scaler.joblib' 
METADATA_FEATURES = ['percent_caps', 'percent_punct', 'link_count', 'word_count']

SBERT_MODEL_NAME = "all-MiniLM-L6-v2"  # good, small, fast SBERT model
SBERT_SCALER_PATH = "sbert_scaler.joblib"

def compute_metadata_features(texts):
    features = []
    for text in texts:
        if not isinstance(text, str):
            text = "" # handle potential np.nan
            
        total_len = len(text)
        if total_len == 0:
            features.append([0, 0, 0, 0])
            continue
            
        # percent of caps
        caps_count = sum(1 for c in text if c.isupper())
        percent_caps = (caps_count / total_len) * 100
        
        # percent of punctuation
        punct_count = sum(1 for c in text if c in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        percent_punct = (punct_count / total_len) * 100
        
        # link count
        link_count = len(re.findall(r'http[s]?://|www\.', text))
        
        # word count
        word_count = len(text.split())
        
        features.append([percent_caps, percent_punct, link_count, word_count])
        
    return np.array(features)

# core functions

def load_and_prepare_data():
    try:
        df = pd.read_csv(DATA_FILEPATH)
    except FileNotFoundError:
        print(f"Error: The file '{DATA_FILEPATH}' was not found.")
        return None, None
    
    if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
        print(f"Error: The CSV must contain '{TEXT_COLUMN}' and '{LABEL_COLUMN}' columns.")
        return None, None
        
    print("Successfully loaded data.")
    
    df['is_scam'] = np.where(df[LABEL_COLUMN] == SCAM_LABEL_VALUE, 1, 0)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna('')
    
    X = df[TEXT_COLUMN]
    y = df['is_scam']
    
    return X, y

def train_model():
    X, y = load_and_prepare_data()
    if X is None:
        return

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nData split: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # vectorize text
    print("Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # calculate metadata features
    print("Calculating metadata features...")
    X_train_meta = compute_metadata_features(X_train)
    X_test_meta = compute_metadata_features(X_test)
    
    # scale metadata
    print("Scaling metadata...")
    scaler = StandardScaler()
    X_train_meta_scaled = scaler.fit_transform(X_train_meta)
    X_test_meta_scaled = scaler.transform(X_test_meta)

    # SBERT embeddings
    print("Loading SBERT model and encoding emails...")
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME)

    # SBERT returns dense numpy arrays
    X_train_sbert = sbert_model.encode(X_train.tolist(), show_progress_bar=True)
    X_test_sbert = sbert_model.encode(X_test.tolist(), show_progress_bar=True)

    print("Scaling SBERT embeddings...")
    sbert_scaler = StandardScaler()
    X_train_sbert_scaled = sbert_scaler.fit_transform(X_train_sbert)
    X_test_sbert_scaled = sbert_scaler.transform(X_test_sbert)

    # train TF-IDF model (Keywords)
    print("Training TF-IDF Model (Lexical)...")
    X_train_lexical = hstack([X_train_tfidf, X_train_meta_scaled]).tocsr()
    model_tfidf = LogisticRegression(solver='liblinear', random_state=42)
    model_tfidf.fit(X_train_lexical, y_train)

    # train SBERT model (Context)
    print("Training SBERT Model (Semantic)...")
    model_sbert = LogisticRegression(solver='liblinear', random_state=42)
    model_sbert.fit(X_train_sbert_scaled, y_train)

    # evaluate the model
    print("\n--- Evaluating Ensemble Accuracy ---")
    
    # prepare test data
    X_test_lexical = hstack([X_test_tfidf, X_test_meta_scaled]).tocsr()
    
    # get probabilities
    pred_tfidf_prob = model_tfidf.predict_proba(X_test_lexical)[:, 1]
    pred_sbert_prob = model_sbert.predict_proba(X_test_sbert_scaled)[:, 1]

    # combine (average)
    final_prob = (pred_tfidf_prob + pred_sbert_prob) / 2
    final_pred = (final_prob > 0.5).astype(int)

    accuracy = accuracy_score(y_test, final_pred)
    print(f"Combined Model Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, final_pred, target_names=['Not Scam', 'Scam']))

    # save model components
    artifacts = {
        "model_tfidf": model_tfidf,
        "model_sbert": model_sbert,
        "vectorizer": vectorizer,
        "scaler": scaler,
        "sbert_scaler": sbert_scaler
    }
    joblib.dump(artifacts, MODEL_ARTIFACTS_PATH)
    print(f"\n Training Complete. Saved to {MODEL_ARTIFACTS_PATH}")

# Extract suspicious keywords (TF-IDF only)
# We use TF-IDF for this because SBERT can't easily show us which individual word triggered the alarm.
def get_scam_indicators(email_text, model, vectorizer):
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    
    # get only coefficients for tf-idf features
    tfidf_coeffs = coefficients[:len(feature_names)]
    word_scam_scores = dict(zip(feature_names, tfidf_coeffs))

    tokens = vectorizer.build_analyzer()(email_text.lower())
    
    found_words = {}
    for token in set(tokens):
        if token in word_scam_scores:
            if word_scam_scores[token] > 0:
                found_words[token] = word_scam_scores[token]
    
    sorted_scam_words = sorted(found_words.items(), key=lambda item: item[1], reverse=True)
    return sorted_scam_words

# Main Prediction Logic: Hybrid Model (TF-IDF + SBERT) / 2
def predict_email(email_text, artifacts=None):
    # load artifacts (fixes speed issues)
    if artifacts is None:
        try:
            artifacts = joblib.load(MODEL_ARTIFACTS_PATH)
            artifacts['sbert_model_obj'] = SentenceTransformer(SBERT_MODEL_NAME)
        except FileNotFoundError:
            print("Error: Model files not found. Run train_model() first.")
            return None
    
    if not isinstance(email_text, str):
        email_text = ""

    # unpack models
    model_tfidf = artifacts['model_tfidf']
    model_sbert = artifacts['model_sbert']
    vectorizer = artifacts['vectorizer']
    scaler = artifacts['scaler']
    sbert_scaler = artifacts['sbert_scaler']
    
    # handle SBERT loading difference (API vs script)
    if 'sbert_model_obj' in artifacts:
        sbert_model_obj = artifacts['sbert_model_obj']
    elif 'sbert_model' in artifacts:
        sbert_model_obj = artifacts['sbert_model']
    else:
        # fallback
        sbert_model_obj = SentenceTransformer(SBERT_MODEL_NAME)

    # TF-IDF prediction (Keywords)
    text_tfidf = vectorizer.transform([email_text])
    text_meta = compute_metadata_features([email_text])
    text_meta_scaled = scaler.transform(text_meta)
    
    text_combined_lexical = hstack([text_tfidf, text_meta_scaled]).tocsr()
    tfidf_prob = model_tfidf.predict_proba(text_combined_lexical)[0][1]

    # SBERT prediction (Context)
    text_sbert = sbert_model_obj.encode([email_text])
    text_sbert_scaled = sbert_scaler.transform(text_sbert)
    
    sbert_prob = model_sbert.predict_proba(text_sbert_scaled)[0][1]

    # combine scores
    final_prob = (tfidf_prob + sbert_prob) / 2
    is_scam = final_prob > 0.50
    
    # get flagged words (from TF-IDF model)
    flagged_words = []
    if tfidf_prob > 0.4:
        flagged_words = [word for word, score in get_scam_indicators(email_text, model_tfidf, vectorizer)[:5]]

    result = {
        "is_scam": bool(is_scam),
        "prediction": "Scam" if is_scam else "Not Scam",
        "confidence_percent": f"{final_prob * 100:.2f}%",
        "breakdown": {
            "tfidf_score": f"{tfidf_prob * 100:.1f}%",
            "sbert_score": f"{sbert_prob * 100:.1f}%"
        },
        "flagged_words": flagged_words
    }
    
    return result

# main execution

if __name__ == "__main__":
    
    print("--- Starting Model Training ---")
    train_model()
    
    print("\n--- Starting Model Prediction ---")

    ham_email = """
    Hi team,
    Just a reminder that our quarterly review meeting is scheduled
    for this Friday at 10:00 AM.
    See you there,
    Jane
    """
    
    scam_email = """
    URGENT ACTION REQUIRED!
    Click http://bit.ly/secure-your-account-now to verify.
    """

    print("\n--- Testing 'Not Scam' Email ---")
    ham_prediction = predict_email(ham_email)
    
    if ham_prediction:
        print(f"Prediction: {ham_prediction['prediction']}")
        print(f"Confidence: {ham_prediction['confidence_percent']}")
        print(f"Breakdown: {ham_prediction['breakdown']}")
        print(f"Flagged Words: {ham_prediction['flagged_words']}")

    print("\n--- Testing 'Scam' Email ---")
    scam_prediction = predict_email(scam_email)
    
    if scam_prediction:
        print(f"Prediction: {scam_prediction['prediction']}")
        print(f"Confidence: {scam_prediction['confidence_percent']}")
        print(f"Breakdown: {scam_prediction['breakdown']}")
        print(f"Flagged Words: {scam_prediction['flagged_words']}")