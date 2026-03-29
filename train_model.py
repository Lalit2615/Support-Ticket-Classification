import pandas as pd
import numpy as np
import re
import joblib
import os
import random

# NLP & Machine Learning Libraries
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# Download required NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --- CONFIGURATION ---
DATA_FILE = "tickets.csv"

# --- 1. SYNTHETIC DATA GENERATOR ---
def generate_synthetic_data(num_samples=1500):
    """
    Generates a synthetic IT support dataset if a real one isn't found.
    """
    print("Generating synthetic ticket dataset...")
    
    categories = ['Hardware', 'Access', 'HR Support', 'Storage', 'Miscellaneous']
    priorities = ['High', 'Medium', 'Low']
    
    # Keyword pools for realistic text generation
    kw_hardware = ["monitor", "mouse", "broken", "keyboard", "laptop", "screen", "power", "battery", "printer", "paper"]
    kw_access = ["password", "login", "locked out", "account", "credentials", "access denied", "vpn", "reset", "authentication"]
    kw_hr = ["payroll", "benefits", "vacation", "pto", "w2", "timesheet", "onboarding", "leave", "insurance"]
    kw_storage = ["disk space", "quota", "full", "drive", "cloud storage", "onedrive", "backup", "missing files", "database"]
    kw_misc = ["question", "feedback", "slow", "update", "error", "crashing", "software", "meeting", "zoom", "help"]

    # Urgency markers for assigning priority
    urgent_words = ["urgent", "critical", "immediately", "asap", "emergency", "broken", "blocked", "all users"]
    low_urgency_words = ["wondering", "whenever", "no rush", "update", "trivial", "question", "suggestion"]

    data = []
    for _ in range(num_samples):
        cat = random.choice(categories)
        if cat == 'Hardware': pool = kw_hardware
        elif cat == 'Access': pool = kw_access
        elif cat == 'HR Support': pool = kw_hr
        elif cat == 'Storage': pool = kw_storage
        else: pool = kw_misc

        # Construct a ticket text
        words_num = random.randint(5, 15)
        text_words = random.choices(pool, k=words_num)
        
        # Decide Priority based on urgency injection
        pri_roll = random.random()
        if pri_roll < 0.2: # 20% High priority
            text_words.extend(random.choices(urgent_words, k=2))
            priority = 'High'
        elif pri_roll > 0.6: # 40% Low priority
            text_words.extend(random.choices(low_urgency_words, k=2))
            priority = 'Low'
        else:
            priority = 'Medium'
            
        # Shuffle words into a sentence
        random.shuffle(text_words)
        text = " ".join(text_words).capitalize() + "."
        
        data.append({"Document": text, "Topic_group": cat, "Priority": priority})
        
    df = pd.DataFrame(data)
    df.to_csv(DATA_FILE, index=False)
    print(f"Dataset generated and saved to {DATA_FILE}\n")
    return df

# --- 2. TEXT CLEANING ---
def clean_text(text):
    """
    Lowercases, removes punctuation, and removes English stopwords.
    """
    if pd.isna(text):
        return ""
    
    # Lowercase
    text = str(text).lower()
    # Remove punctuation & numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    
    return " ".join(tokens)


# --- 3. MAIN ML PIPELINE ---
def run_pipeline():
    print("===== Support Ticket Classification System Pipeline =====\n")
    
    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        print(f"'{DATA_FILE}' not found.")
        df = generate_synthetic_data(2000)
    else:
        df = pd.read_csv(DATA_FILE)
        print(f"Loaded existing data from {DATA_FILE}")

    # Ensure required columns exist
    if 'Priority' not in df.columns:
        print("Dataset missing 'Priority' column. Simulating priority logic based on keywords...")
        # Since Kaggle dataset might not have Priority, let's inject a very basic rule just for modelling purposes.
        df['Priority'] = np.where(df['Document'].str.contains('urgent|critical|immediately|down|broken', case=False, na=False), 'High', 
                                  np.where(df['Document'].str.contains('slow|update|issue|error', case=False, na=False), 'Medium', 'Low'))

    print("\nTotal Records:", len(df))
    print("Target 1 (Categories):\n", df['Topic_group'].value_counts().to_dict())
    print("Target 2 (Priorities):\n", df['Priority'].value_counts().to_dict())

    # 2. Preprocessing
    print("\n--- Cleaning Text ---")
    df['cleaned_text'] = df['Document'].apply(clean_text)
    
    # 3. Handling features (TF-IDF Vectorization)
    print("\n--- Extracting Features (TF-IDF) ---")
    tfidf = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.8)
    X = tfidf.fit_transform(df['cleaned_text'])
    
    # Variables for multiple models
    y_category = df['Topic_group']
    y_priority = df['Priority']
    
    # Split Data (80% Train, 20% Test)
    X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test = train_test_split(
        X, y_category, y_priority, test_size=0.2, random_state=42
    )

    # 4. Train Models
    print("\n--- Training Models ---")
    # Category classifier: Logistic Regression (Fast and efficient for text)
    clf_category = LogisticRegression(random_state=42, max_iter=200)
    print("Training Category Predictor (Logistic Regression)...")
    clf_category.fit(X_train, y_cat_train)
    
    # Priority classifier: Random Forest Model (to demonstrate tree-based models)
    clf_priority = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Training Priority Predictor (Random Forest)...")
    clf_priority.fit(X_train, y_pri_train)

    # 5. Evaluation
    print("\n=== EVALUATION METRICS ===")
    
    # Evaluate Category
    print("\n--- CATEGORY PREDICTION ---")
    cat_preds = clf_category.predict(X_test)
    print("Accuracy:", round(accuracy_score(y_cat_test, cat_preds), 4))
    print("\nClassification Report:\n", classification_report(y_cat_test, cat_preds))
    print("Confusion Matrix:\n", confusion_matrix(y_cat_test, cat_preds))

    # Evaluate Priority
    print("\n--- PRIORITY PREDICTION ---")
    pri_preds = clf_priority.predict(X_test)
    print("Accuracy:", round(accuracy_score(y_pri_test, pri_preds), 4))
    print("\nClassification Report:\n", classification_report(y_pri_test, pri_preds))
    print("Confusion Matrix:\n", confusion_matrix(y_pri_test, pri_preds))

    # 6. Save Artifacts for deployment
    os.makedirs('models', exist_ok=True)
    joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
    joblib.dump(clf_category, 'models/category_model.pkl')
    joblib.dump(clf_priority, 'models/priority_model.pkl')
    
    print("\n✅ System trained successfully! Models and Vectorizer saved in the 'models/' directory.")


if __name__ == "__main__":
    run_pipeline()
