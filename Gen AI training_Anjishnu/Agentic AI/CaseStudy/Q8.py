# ── CASE STUDY 8 FINAL WORKING CODE ──────────────────────────────────────────────
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
from google.cloud.aiplatform.matching_engine import matching_engine_index_config
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import MatchingEngineIndexEndpoint
from google.cloud.aiplatform.matching_engine import MatchingEngineIndex, MatchingEngineIndexDatapoint
import vertexai
from google.cloud import storage
from dotenv import load_dotenv
import os
import json

# --- CONFIGURATION ---
load_dotenv()

PROJECT_ID = "wpuser22"               # Replace with your GCP project ID
LOCATION = "us-central1"
BUCKET =  "your-gcs-bucket"                     # Replace with your GCP BUCKET
GCS_INDEX_DIR = f"{BUCKET}/banking_faq_index/"   # Used to store embeddings

vertexai.init(project=PROJECT_ID, location=LOCATION)
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# --- Step 1: Banking FAQ corpus ---
banking_faqs = [
    {"id": "FAQ001", "question": "How do I reset my internet banking password?"},
    {"id": "FAQ002", "question": "What documents are required to open a savings account?"},
    {"id": "FAQ003", "question": "How can I dispute an unauthorized credit card transaction?"},
    {"id": "FAQ004", "question": "What is the daily ATM withdrawal limit for my account?"},
    {"id": "FAQ005", "question": "How do I apply for a home loan pre-approval?"},
    {"id": "FAQ006", "question": "How can I block my debit card if it is lost or stolen?"},
    {"id": "FAQ007", "question": "What are the NEFT transfer timings and charges?"},
    {"id": "FAQ008", "question": "How do I update my mobile number linked to my bank account?"},
    {"id": "FAQ009", "question": "What is the process for closing a fixed deposit?"},
    {"id": "FAQ010", "question": "How do I set up auto-debit for my loan EMI payments?"}
]

# --- Step 2 — Generate embeddings ---
embed_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

faq_embeddings = []
for faq in banking_faqs:
    emb = embed_model.get_embeddings([faq["question"]])[0].values  # ✅ FIXED
    faq_embeddings.append({**faq, "embedding": emb})

print(f"Generated {len(faq_embeddings)} FAQ embeddings. Dimension: {len(faq_embeddings[0]['embedding'])}")

# Store embeddings into GCS (Matching Engine expects files in GCS)
local_file = "faq_vectors.jsonl"
with open(local_file, "w") as f:
    for faq in faq_embeddings:
        row = {
            "id": faq["id"],
            "embedding": faq["embedding"]
        }
        f.write(json.dumps(row) + "\n")

# Upload to GCS


client = storage.Client()
bucket = client.bucket("your-gcs-bucket")
blob = bucket.blob("banking_faq_index/faq_vectors.jsonl")
blob.upload_from_filename("faq_vectors.jsonl")

print("Uploaded to GCS successfully.")

# --- Step 3 — Create Matching Engine Index ---
tree_ah_config = matching_engine_index_config.TreeAhConfig(
    leaf_node_embedding_count=500,
    leaf_nodes_to_search_percent=7
)

faq_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="banking-faq-index",
    contents_delta_uri=GCS_INDEX_DIR,
    dimensions=768,
    distance_measure_type="COSINE_DISTANCE",
    index_update_method="STREAM_UPDATE",     # Real-time updates
    tree_ah_config=tree_ah_config
)

print("Index created:", faq_index.resource_name)

# --- Step 4 — Create Matching Engine Index Endpoint ---
faq_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="banking-faq-endpoint",
    public_endpoint_enabled=True
)

print("Endpoint created:", faq_endpoint.resource_name)

# --- Step 5 — Deploy index to endpoint ---
faq_endpoint.deploy_index(
    index=faq_index,
    deployed_index_id="banking_faq_deployed"
)

print("Index deployed to endpoint.")

# --- Step 6 — Upsert FAQ embeddings ---

datapoints = []
for faq in faq_embeddings:
    dp = MatchingEngineIndexDatapoint(
        datapoint_id=faq["id"],
        feature_vector=faq["embedding"]
    )
    datapoints.append(dp)

faq_index.upsert_datapoints(datapoints=datapoints)
print("✅ FAQs upserted into Matching Engine.")

# --- Step 7 — Test Semantic Search ---

id_to_faq = {f["id"]: f["question"] for f in banking_faqs}

test_queries = [
    "I forgot my online banking login credentials",      # → FAQ001 
    "My ATM card was misplaced, I need to stop it",       # → FAQ006
    "I want to pay off my fixed deposit early",           # → FAQ009
]

for query in test_queries:
    print(f"\n🔍 Query: {query}")

    query_emb = embed_model.get_embeddings([query])[0].values

    neighbors = faq_endpoint.find_neighbors(
        deployed_index_id="banking_faq_deployed",
        queries=[query_emb],
        num_neighbors=3
    )

    for rank, nb in enumerate(neighbors[0], 1):
        faq_text = id_to_faq.get(nb.id, "Unknown")
        print(f"  #{rank} [{nb.id}] {faq_text}  (distance={nb.distance:.4f})")