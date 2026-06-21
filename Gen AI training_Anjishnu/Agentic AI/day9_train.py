# train.py (UPLOADS TO GCS)
import argparse
import os
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from google.cloud import storage

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=0.001)
args = parser.parse_args()

print(f"🚀 Training iris classifier...")

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Train
model = LogisticRegression(max_iter=args.epochs, C=1/args.lr, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"✅ Accuracy: {accuracy:.4f}")

# Save to local file first
local_model_path = "/tmp/model.pkl"
with open(local_model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"💾 Local model saved to: {local_model_path}")

# Upload to GCS
bucket_name = "ur-training-bucket"
gcs_model_path = "trained_models/model.pkl"

storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(gcs_model_path)
blob.upload_from_filename(local_model_path)

print(f"☁️  Uploaded to: gs://{bucket_name}/{gcs_model_path}")
print("✅ Training complete!")
