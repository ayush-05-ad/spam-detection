import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import boto3
import os
import nltk
from dotenv import load_dotenv

load_dotenv()
nltk.download('stopwords', quiet=True)

# ── Step 1: Load Data ────────────────────
df = pd.read_csv("notebooks/spamham.csv")
df = df.rename(columns={'Label': 'label', 'Message': 'message'})
df = df.dropna(subset=['message'])
print(f"✅ Data loaded — {len(df)} rows")

# ── Step 2: Preprocess ───────────────────
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
corpus = []

for msg in df['message']:
    review = re.sub('[^a-zA-Z]', ' ', str(msg)).lower().split()
    review = [ps.stem(w) for w in review if w not in stop_words]
    corpus.append(' '.join(review))
print("✅ Preprocessing done!")

# ── Step 3: TF-IDF ───────────────────────
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(corpus).toarray()
y = pd.get_dummies(df['label'], drop_first=True).values.ravel()
print(f"✅ TF-IDF done — shape: {X.shape}")

# ── Step 4: Train SVM ────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = SVC(kernel='linear')
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Model trained — Accuracy: {acc * 100:.2f}%")

# ── Step 5: Save locally ─────────────────
pickle.dump(model, open("notebooks/spam_detector_model.pkl", "wb"))
pickle.dump(tfidf,  open("notebooks/tfidf_vectorizer.pkl", "wb"))
print("✅ Models saved locally!")

# ── Step 6: Upload to S3 ─────────────────
s3 = boto3.client(
    "s3",
    aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name           = os.getenv("AWS_REGION")
)

BUCKET = os.getenv("S3_BUCKET_NAME", "spam-detection-model-ayush")
print(f"Using bucket: {BUCKET}")

s3.upload_file("notebooks/spam_detector_model.pkl", BUCKET, "models/spam_detector_model.pkl")
s3.upload_file("notebooks/tfidf_vectorizer.pkl",    BUCKET, "models/tfidf_vectorizer.pkl")
print(f"✅ Models uploaded to S3: {BUCKET}")

print("\n🎉 Training and export complete!")