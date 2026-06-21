# %% [markdown]
# # 🚀 Vertex AI & Google GenAI — Complete Reference Guide
# 
# **Topics Covered in this Notebook:**
# 
# | Sl No | Topic | Sub-topic |
# |-------|-------|-----------|
# | 1 | Google AI Platform Overview | Vertex AI landscape and GenAI positioning |
# | 2 | Vertex AI Fundamentals | Projects, regions, endpoints, deployment concepts |
# | 3 | Gemini Models on Vertex AI | Gemini models overview, embeddings, quotas, safety |
# | 4 | Vertex AI Studio | Prompt management, experimentation, evaluations |
# | 5 | Gemini ML Overview | Workbench, training concepts, pipelines (high level) |
# | 6 | Model Lifecycle | Model registry, versioning, endpoints |
# | 7 | Security & Networking | IAM, service accounts, VPC Service Controls |
# | 8 | Operations Management | Quotas, logging, monitoring, cost awareness |
# | 9 | Governance & Responsible AI | Responsible AI principles, policy and compliance |
# | 10 | Capstone Project | Deploy and govern a GenAI solution using Vertex AI |
# 
# ---
# 
# > 💡 **Prerequisites:**  
# > - A Google Cloud Project with billing enabled  
# > - `GOOGLE_API_KEY` **or** a Service Account JSON key (for direct Gemini API access)  
# > - Vertex AI API enabled in your GCP project  
# > - Python packages: `google-genai`, `google-cloud-aiplatform`, `google-auth`, `python-dotenv`

# %%
# ── Environment Setup ────────────────────────────────────────────────────────
# Run this cell FIRST before any other section

# Install required packages (uncomment if needed)
# !pip install google-genai google-cloud-aiplatform google-auth python-dotenv

import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Option A — Gemini API key (AI Studio / direct access)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")

# Option B — Vertex AI (GCP project settings)
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "wpuser01")
LOCATION   = os.getenv("GCP_REGION",     "us-central1")
SA_KEY     = os.getenv("SA_KEY_PATH",    "wpuser01-88df631dc4fd.json")

print("✅ Environment loaded")
print(f"   Project : {PROJECT_ID}")
print(f"   Region  : {LOCATION}")

# %% [markdown]
# ---
# # Section 1 — Google AI Platform Overview
# 
# ## 1.1 Google's AI / ML Product Landscape
# 
# Google Cloud offers a layered stack for AI and machine learning:
# 
# ```
# ┌──────────────────────────────────────────────────────────────┐
# │  AI Applications (Workspace AI, Contact Center AI, etc.)     │
# ├──────────────────────────────────────────────────────────────┤
# │  Vertex AI  ←  Unified ML Platform (this course focus)       │
# │    ├── Vertex AI Studio  (prompt, evaluate, tune)            │
# │    ├── Model Garden      (foundation models)                 │
# │    ├── Pipelines         (MLOps orchestration)               │
# │    └── Matching Engine   (vector search)                     │
# ├──────────────────────────────────────────────────────────────┤
# │  Google AI APIs  (Gemini API, Vision, Speech, NL, etc.)      │
# ├──────────────────────────────────────────────────────────────┤
# │  Google Cloud Infrastructure  (Compute, Storage, Network)    │
# └──────────────────────────────────────────────────────────────┘
# ```
# 
# ## 1.2 Two Ways to Access Gemini
# 
# | Access Path | SDK / Tool | Best For |
# |-------------|-----------|----------|
# | **Gemini API (AI Studio)** | `google-genai` (`genai.Client(api_key=…)`) | Prototyping, personal projects, low-cost testing |
# | **Vertex AI** | `google-genai` (`genai.Client(vertexai=True, …)`) OR `vertexai` SDK | Enterprise, governance, VPC, MLOps, fine-tuning |
# 
# ## 1.3 GenAI Positioning in Vertex AI
# 
# ```
# Vertex AI
#   └── Model Garden
#         ├── Google Foundation Models
#         │     ├── Gemini 2.5 Flash / Pro  ← multimodal LLM
#         │     ├── Imagen 3                ← image generation
#         │     ├── Chirp                   ← speech
#         │     └── Codey                   ← code generation
#         └── Partner / Open-Source Models (Llama, Mistral, etc.)
# ```
# 
# Key terms:
# - **Foundation Model** — Large pre-trained model available via API  
# - **Tuning** — Supervised fine-tuning or RLHF on your data  
# - **Grounding** — Connecting model to live data (Search, private data via RAG)  
# - **Model Garden** — Catalogue to discover, test, and deploy foundation models

# %%
# Section 1 — Demo: List Available Gemini Models via Vertex AI
# Connects to Vertex AI and lists the Gemini models available in your project.

from google.genai import Client

client_vertex = Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
)

# List all models available in your project/region
models = client_vertex.models.list()
print("Available Models in Vertex AI:")
print("-" * 50)
for m in models:
    if "gemini" in m.name.lower():
        print(f"  • {m.name}")

# %% [markdown]
# ---
# # Section 2 — Vertex AI Fundamentals
# 
# ## 2.1 Core Concepts
# 
# | Concept | Description |
# |---------|-------------|
# | **Project** | GCP organisational unit; all resources (models, datasets, endpoints) live inside a project |
# | **Region** | Physical data-centre location (e.g., `us-central1`, `europe-west4`). Affects latency, data residency, and quota. |
# | **Endpoint** | A managed REST service that hosts a deployed model and accepts prediction requests |
# | **Model** | A trained artifact stored in Vertex AI Model Registry |
# | **Deployment** | The act of attaching a Model to an Endpoint with compute resources |
# 
# ## 2.2 Regions at a Glance
# 
# ```
# Americas         : us-central1, us-east1, us-east4, us-west1, northamerica-northeast1
# Europe           : europe-west1, europe-west2, europe-west4, europe-north1
# Asia-Pacific     : asia-east1, asia-northeast1, asia-southeast1, australia-southeast1
# ```
# 
# > ⚠️ **Not all models are available in every region.** Gemini 2.5 Flash/Pro are available in select regions — always check the [Vertex AI locations page](https://cloud.google.com/vertex-ai/docs/general/locations).
# 
# ## 2.3 Initialising the Vertex AI SDK
# 
# ```python
# import vertexai
# vertexai.init(project="my-project", location="us-central1")
# ```
# 
# After `init()`, all subsequent SDK calls automatically use these defaults.
# 
# ## 2.4 Prediction vs Batch Prediction
# 
# | Mode | When to Use | API |
# |------|-------------|-----|
# | **Online Prediction** | Real-time, low-latency, one request at a time | `endpoint.predict()` |
# | **Batch Prediction** | Offline, large volumes, cost-efficient | `aiplatform.BatchPredictionJob` |

# %%
# Section 2 — Demo: Initialise Vertex AI SDK + List Endpoints
import vertexai
from google.cloud import aiplatform

# Initialise SDK with project and region
vertexai.init(project=PROJECT_ID, location=LOCATION)
aiplatform.init(project=PROJECT_ID, location=LOCATION)

print(f"✅ Vertex AI SDK initialised")
print(f"   Project  : {PROJECT_ID}")
print(f"   Location : {LOCATION}")

# List existing endpoints in this project/region
print("\n📋 Existing Endpoints:")
endpoints = aiplatform.Endpoint.list()
if endpoints:
    for ep in endpoints:
        print(f"  • {ep.display_name}  [{ep.resource_name}]")
else:
    print("  (no endpoints found — create one in Section 6)")

# %% [markdown]
# Use vertexai for model inference and GenAI tasks
# Use aiplatform for MLOps and infrastructure tasks (creating endpoints, uploading models, etc.)

# %% [markdown]
# ---
# # Section 3 — Gemini Models on Vertex AI
# 
# ## 3.1 Gemini Model Family (as of 2025)
# 
# | Model | Best Use | Context Window | Modalities |
# |-------|----------|---------------|------------|
# | `gemini-2.5-pro` | Complex reasoning, long context | 1M tokens | Text, image, video, audio, code |
# | `gemini-2.5-flash` | Speed + cost balance | 1M tokens | Text, image, video, audio, code |
# | `gemini-2.0-flash` | Production workloads | 1M tokens | Text, image, audio, code |
# | `gemini-1.5-pro` | Legacy, stable | 2M tokens | Text, image, video, audio, code |
# | `gemini-1.5-flash` | Fast & cheap | 1M tokens | Text, image, video, audio, code |
# 
# > 💡 On **Vertex AI**, model names are accessed as:  
# > `publishers/google/models/gemini-2.5-flash`  
# > or simply `gemini-2.5-flash` when using the `google-genai` SDK with `vertexai=True`.
# 
# ## 3.2 Embeddings
# 
# | Model | Dimensions | Best For |
# |-------|-----------|---------|
# | `text-embedding-004` | 768 | General-purpose text embeddings |
# | `text-multilingual-embedding-002` | 768 | Multilingual semantic search |
# | `multimodalembedding@001` | 1408 | Text + image embeddings |
# 
# ## 3.3 Quotas
# 
# Quotas in Vertex AI are enforced per **project, region, and model**:
# 
# - **Requests per minute (RPM)** — number of API calls
# - **Tokens per minute (TPM)** — total input+output tokens per minute
# - **Requests per day (RPD)** — daily cap
# 
# View quotas: **Google Cloud Console → IAM & Admin → Quotas** → filter by `Vertex AI`
# 
# To request a quota increase: Cloud Console → Quotas → tick the quota row → Edit Quotas.
# 
# ## 3.4 Safety Settings
# 
# Gemini models enforce four harm categories by default:
# 
# | Category | `HarmCategory` Enum |
# |----------|---------------------|
# | Hate speech | `HARM_CATEGORY_HATE_SPEECH` |
# | Dangerous content | `HARM_CATEGORY_DANGEROUS_CONTENT` |
# | Sexually explicit | `HARM_CATEGORY_SEXUALLY_EXPLICIT` |
# | Harassment | `HARM_CATEGORY_HARASSMENT` |
# 
# Threshold options: `BLOCK_NONE` · `BLOCK_LOW_AND_ABOVE` · `BLOCK_MEDIUM_AND_ABOVE` · `BLOCK_HIGH_AND_ABOVE` · `BLOCK_ONLY_HIGH`

# %%
# Section 3 — Demo A: Generate content with custom safety settings on Vertex AI
from google.genai import Client
from google.genai import types

client_v = Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
MODEL = "gemini-2.5-flash"

response = client_v.models.generate_content(
    model=MODEL,
    contents="Explain the concept of zero-shot prompting in 3 sentences.",
    config=types.GenerateContentConfig(
        temperature=0.4,
        max_output_tokens=200,
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            ),
        ],
    ),
)

print("Response:", response.text)

# Check safety ratings
if response.candidates:
    print("\n🛡️  Safety Ratings:")
    for rating in response.candidates[0].safety_ratings:
        print(f"   {rating.category.name}: {rating.probability.name}")

# %%
# Section 3 — Demo B: Generate text embeddings with Vertex AI
# text-embedding-004 produces 768-dimensional vectors

from vertexai.language_models import TextEmbeddingModel

embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

sentences = [
    "What is the interest rate on a home loan?",
    "How do I apply for a mortgage?",
    "What is the weather today?",
]

embeddings = embedding_model.get_embeddings(sentences)

print("✅ Embeddings generated")
print(f"   Model    : text-embedding-004")
print(f"   Sentences: {len(sentences)}")
print(f"   Dimensions: {len(embeddings[0].values)}")
print()
for i, (sent, emb) in enumerate(zip(sentences, embeddings)):
    preview = emb.values[:5]
    print(f"  [{i}] \"{sent[:55]}\"")
    print(f"       Vector preview: {[round(v, 4) for v in preview]} ...")

# %% [markdown]
# ---
# # Section 4 — Vertex AI Studio
# 
# ## 4.1 What is Vertex AI Studio?
# 
# **Vertex AI Studio** is the no-code/low-code UI inside Google Cloud Console for exploring and experimenting with foundation models.
# 
# **URL:** `console.cloud.google.com` → Vertex AI → Vertex AI Studio
# 
# ## 4.2 Key Areas
# 
# | Tab | Purpose |
# |-----|---------|
# | **Overview** | Get started, links to models |
# | **Freeform** | Single-turn prompt playground (text, multimodal) |
# | **Chat** | Multi-turn conversation testing |
# | **Batch** | Run a large prompt dataset through a model offline |
# | **Grounding** | Connect Gemini to Google Search or your own data |
# | **Tuning** | Supervised fine-tuning (SFT) and RLHF |
# | **Evaluation** | Score model responses against reference answers |
# 
# ## 4.3 Prompt Management Best Practices
# 
# 1. **Version your prompts** — save named prompt versions in Vertex AI Studio before tuning.  
# 2. **Use system instructions** — set persona and constraints separately from user content.  
# 3. **Temperature for creativity:**  
#    - `0.0 – 0.3` → factual, deterministic  
#    - `0.4 – 0.7` → balanced  
#    - `0.8 – 1.0+` → creative, diverse  
# 4. **Top-P sampling** — controls nucleus sampling; `0.9` is a common default.  
# 5. **Max output tokens** — always set an upper bound to control cost.
# 
# ## 4.4 Evaluation Metrics Available in Studio
# 
# | Metric | What it Measures |
# |--------|-----------------|
# | **ROUGE-L** | Overlap of longest common subsequence with reference |
# | **BLEU** | N-gram precision vs. reference text |
# | **Coherence** | Logical flow of the response (model-based) |
# | **Fluency** | Grammatical quality (model-based) |
# | **Groundedness** | Facts anchored to provided context |
# | **Safety** | Proportion of responses flagged by safety classifiers |

# %%
# Section 4 — Demo A: Prompt versioning pattern using Python dict
# (Mirrors the "save prompt versions" workflow from Vertex AI Studio)

PROMPT_LIBRARY = {
    "v1_basic": {
        "system": "You are a helpful banking assistant.",
        "user_template": "What is {topic}?",
        "temperature": 0.3,
        "max_output_tokens": 1500,
    },
    "v2_with_format": {
        "system": (
            "You are a concise banking assistant. "
            "Always respond in exactly 3 bullet points."
        ),
        "user_template": "Explain {topic} in 3 bullet points.",
        "temperature": 0.2,
        "max_output_tokens": 1500,
    },
}

def run_prompt_version(version: str, topic: str):
    from google.genai import Client, types

    p = PROMPT_LIBRARY[version]
    client = Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Content(role="user", parts=[types.Part(text=p["user_template"].format(topic=topic))]),
        ],
        config=types.GenerateContentConfig(
            system_instruction=p["system"],
            temperature=p["temperature"],
            max_output_tokens=p["max_output_tokens"],
        ),
    )
    
    # Debug the response
    print(f"Finish reason: {response.candidates[0].finish_reason}")  # ← Check this
    print(f"Stop reason: {response.candidates[0].finish_message}")
    return response.text


# Test both versions side by side
topic = "fixed deposit"
for ver in PROMPT_LIBRARY:
    print(f"─── {ver} ───")
    result = run_prompt_version(ver, topic)
    print(f"Response length: {len(result)} chars")
    print(f"Token estimate: ~{len(result.split()) // 0.75}")  # rough estimate
    print(result)
    print()


# %%
# Section 4 — Demo B: Model evaluation using Vertex AI Rapid Evaluation SDK
# Computes BLEU and ROUGE-L scores programmatically

from vertexai.evaluation import EvalTask, MetricPromptTemplateExamples
from vertexai.generative_models import GenerativeModel
import pandas as pd

# Sample evaluation dataset
eval_dataset = pd.DataFrame({
    "prompt": [
        "What is a savings account?",
        "Define compound interest in one sentence.",
    ],
    "reference": [
        "A savings account is a deposit account that earns interest over time.",
        "Compound interest is interest calculated on both the initial principal and accumulated past interest.",
    ],
})

# Define the model to evaluate
model_to_eval = GenerativeModel("gemini-2.5-flash")

# Create the evaluation task
eval_task = EvalTask(
    dataset=eval_dataset,
    metrics=["bleu", "rouge_l_sum"],
)

print("📊 Running evaluation — this may take 30–60 seconds...")
eval_result = eval_task.evaluate(model=model_to_eval, prompt_template="{prompt}")

print("\nEvaluation Summary:")
print(eval_result.summary_metrics)
print("\nPer-row Results:")
print("Available columns:", eval_result.metrics_table.columns.tolist())
print(eval_result.metrics_table.head())

print(eval_result.metrics_table[["prompt", "bleu/score", "rouge_l_sum/score"]].to_string(index=False))



# %% [markdown]
# ---
# # Section 5 — Gemini ML Overview (Workbench, Training & Pipelines)
# 
# ## 5.1 Vertex AI Workbench
# 
# **Vertex AI Workbench** is a managed Jupyter notebook environment on GCP.
# 
# | Type | Description |
# |------|-------------|
# | **Managed Notebooks** | Fully managed by Google — auto-start/stop, pre-installed frameworks |
# | **User-managed Notebooks** | More control — choose VM type, GPU, custom Docker image |
# 
# Key features:
# - Pre-installed: TensorFlow, PyTorch, scikit-learn, XGBoost, Spark
# - Direct GCS/BigQuery integration
# - One-click kernel restart, scheduled runs
# - Integrates with Vertex AI Training and Pipelines
# 
# ## 5.2 Training Concepts on Vertex AI
# 
# ```
# Training Job Types
# ├── AutoML              ← No-code: provide data, get a model
# ├── Custom Training     ← Bring your own training script
# │     ├── Pre-built Container  (TF, PyTorch, scikit-learn)
# │     └── Custom Container     (any framework via Docker)
# └── Hyperparameter Tuning (Vizier) ← Automated HPT
# ```
# 
# ### Training Script Pattern
# ```python
# # train.py — runs inside a Vertex AI Training container
# import argparse, os
# from google.cloud import storage
# 
# parser = argparse.ArgumentParser()
# parser.add_argument("--epochs",    type=int,   default=10)
# parser.add_argument("--lr",        type=float, default=0.001)
# parser.add_argument("--model-dir", type=str,   default=os.environ.get("AIP_MODEL_DIR"))
# args = parser.parse_args()
# 
# # ... training logic ...
# 
# # Save model artefacts to GCS (AIP_MODEL_DIR is set by Vertex AI)
# model.save(args.model_dir)
# ```
# 
# ## 5.3 Vertex AI Pipelines (High Level)
# 
# Vertex AI Pipelines is based on **Kubeflow Pipelines (KFP) v2**, which compiles Python functions into a DAG of containerised steps.
# 
# ```
# Pipeline  →  Steps (Components)
#                 ├── data_ingestion
#                 ├── preprocessing
#                 ├── training
#                 ├── evaluation
#                 └── conditional_deploy (if accuracy > threshold)
# ```
# 
# ### Component Definition Pattern
# ```python
# from kfp.v2 import dsl
# from kfp.v2.dsl import component, Output, Model
# 
# @component(base_image="python:3.11", packages_to_install=["scikit-learn"])
# def train_model(learning_rate: float, model_output: Output[Model]):
#     from sklearn.linear_model import LogisticRegression
#     import pickle, os
#     clf = LogisticRegression(C=1/learning_rate)
#     # ... fit ...
#     with open(model_output.path, "wb") as f:
#         pickle.dump(clf, f)
# ```

# %%
import os

# Check bucket name
BUCKET_NAME = "ur-training-bucket"

# List contents
print("📋 Checking bucket contents...")
os.system(f"gsutil ls gs://{BUCKET_NAME}/")

# List train.py specifically
print("\n🔍 Looking for train.py...")
result = os.system(f"gsutil ls gs://{BUCKET_NAME}/train.py")

if result == 0:
    print("✅ train.py found!")
else:
    print("❌ train.py NOT found - need to upload it")
    print(f"\nTo upload, run:\n   gsutil cp train.py gs://{BUCKET_NAME}/")


# %%
from google.cloud import aiplatform

PROJECT_ID = "wpuser01"
LOCATION = "us-central1"
BUCKET_NAME = "ur-training-bucket"

aiplatform.init(
    project=PROJECT_ID, 
    location=LOCATION,
    staging_bucket=f"gs://{BUCKET_NAME}"
)

job = aiplatform.CustomTrainingJob(
    display_name="iris-training-job",
    script_path="train.py",  # ← LOCAL file, not gs://
    container_uri="us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-12.py310:latest",
    requirements=["scikit-learn", "google-cloud-storage"],
)

print("📤 Uploading training script...")

model = job.run(
    args=["--epochs=10", "--lr=0.001"],
    replica_count=1,
    machine_type="n1-standard-4",
    base_output_dir=f"gs://{BUCKET_NAME}/models",
)

print("✅ Training submitted!")
print(f"Job: {job.resource_name}")


# %%
import os

print("📥 Downloading trained model...")
os.system("gsutil cp gs://ur-training-bucket/trained_models/model.pkl ./trained_model.pkl")

if os.path.exists('trained_model.pkl'):
    print("✅ Model downloaded successfully!")
else:
    print("❌ Download failed")


# %%
import pickle
from sklearn.datasets import load_iris

print("🧪 Loading model...")
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("✅ Model loaded!")

# Test it
iris = load_iris()
predictions = model.predict(iris.data[:10])

print("\n🎯 Predictions on first 10 flowers:")
for i, pred in enumerate(predictions):
    flower_name = iris.target_names[pred]
    print(f"  Flower {i}: {flower_name}")

# Check accuracy
accuracy = model.score(iris.data, iris.target)
print(f"\n📊 Overall Accuracy: {accuracy:.2%}")


# %%
import pickle
from sklearn.datasets import load_iris

# Load model
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("🌸 Making predictions on new flower data...\n")

# New flower measurements (you can change these values)
# Format: [sepal_length, sepal_width, petal_length, petal_width]

new_flowers = [
    [5.1, 3.5, 1.4, 0.2],   # Likely setosa
    [7.0, 3.2, 4.7, 1.4],   # Likely versicolor
    [6.3, 3.3, 6.0, 2.5],   # Likely virginica
]

iris = load_iris()

print("🎯 Predictions:")
print("─" * 60)

for i, flower in enumerate(new_flowers):
    prediction = model.predict([flower])[0]
    flower_name = iris.target_names[prediction]
    
    print(f"\nFlower {i+1}:")
    print(f"  Measurements: {flower}")
    print(f"  Predicted type: {flower_name}")
    print(f"  Confidence: {model.predict_proba([flower]).max():.2%}")


# %%
from google.cloud import aiplatform

aiplatform.init(project="wpuser01", location="us-central1")

print("📤 Uploading model to Vertex AI...")

model = aiplatform.Model.upload(
    display_name="iris-classifier-v1",
    artifact_uri="gs://ur-training-bucket/trained_models/",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"
)

print(f"✅ Model uploaded!")
print(f"   Model ID: {model.resource_name}")
print(f"   Display name: iris-classifier-v1")


# %%
print("🔗 Creating endpoint...")

endpoint = aiplatform.Endpoint.create(
    display_name="iris-endpoint",
    project="wpuser01",
    location="us-central1"
)

print(f"✅ Endpoint created!")
print(f"   Endpoint ID: {endpoint.resource_name}")


# %%
from google.cloud import aiplatform

aiplatform.init(project="wpuser01", location="us-central1")

# Get the endpoint you just created
endpoint = aiplatform.Endpoint('projects/2916254904/locations/us-central1/endpoints/3111882339152560128')

# Get the model you uploaded
model = aiplatform.Model.list(
    filter='display_name="iris-classifier-v1"'
)[0]

print("🚀 Deploying model to endpoint...")
print(f"   Endpoint: {endpoint.display_name}")
print(f"   Model: {model.display_name}")

endpoint.deploy(
    model=model,
    deployed_model_display_name="iris-classifier-v1",
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=1
)

print("✅ Model deployed!")
print("   Waiting for deployment to complete (5-10 minutes)...")


# %% [markdown]
# ---
# # Section 8 — Operations Management
# 
# ## 8.1 Quotas
# 
# Vertex AI enforces quotas per **project × region × resource type**:
# 
# | Quota Type | Typical Default | How to Find |
# |-----------|----------------|-------------|
# | Online prediction requests/min per model | 600 | Cloud Console → Quotas |
# | Gemini tokens per minute | 1,000,000 | Cloud Console → Quotas → filter "generativelanguage" |
# | Training hours per day | 38 GPU-hours | Cloud Console → Quotas |
# | Custom jobs running concurrently | 8 | Cloud Console → Quotas |
# 
# **To request more quota:**  
# Cloud Console → API & Services → Vertex AI API → Quotas → tick row → Edit Quotas
# 
# ## 8.2 Cloud Logging for Vertex AI
# 
# All Vertex AI API calls can be captured in **Cloud Logging**:
# 
# ```
# Log types:
#   • Admin Activity  — changes to config (always enabled, cannot disable)
#   • Data Access     — read/write to model data (off by default — enable for compliance)
#   • System Events   — Vertex infrastructure events
# ```
# 
# To enable Data Access logs:
# ```bash
# gcloud projects add-iam-policy-binding PROJECT_ID \
#   --member="allUsers" --role="roles/logging.viewer"   # audit only example
# ```
# 
# Or via Cloud Console → IAM & Admin → Audit Logs → Vertex AI API → tick "Data Read" + "Data Write"
# 
# ## 8.3 Cloud Monitoring
# 
# Key metrics to monitor for a deployed endpoint:
# 
# | Metric | Significance |
# |--------|-------------|
# | `aiplatform.googleapis.com/prediction/online/request_count` | Traffic volume |
# | `aiplatform.googleapis.com/prediction/online/error_count` | Error rate |
# | `aiplatform.googleapis.com/prediction/online/latencies` | P50/P95/P99 latency |
# | `aiplatform.googleapis.com/prediction/online/accelerator/duty_cycle` | GPU utilisation |
# 
# ## 8.4 Cost Awareness
# 
# Cost drivers for Vertex AI:
# 
# | Resource | Billed On |
# |----------|----------|
# | **Online Prediction** | Node uptime (min 1 hour) + per-request for serverless |
# | **Training** | Machine-hours (CPU/GPU) |
# | **Gemini API calls** | Input tokens + output tokens |
# | **Storage (GCS)** | GB stored per month |
# | **Matching Engine / Vector Search** | Index storage + queries |
# 
# **Cost reduction tips:**
# - Use `gemini-2.5-flash` instead of `gemini-2.5-pro` for high-volume tasks (≈10× cheaper per token)
# - Set `max_output_tokens` to avoid runaway responses
# - Enable endpoint auto-scaling with `min_replica_count=0` for intermittent workloads
# - Use batch prediction instead of online prediction for offline use-cases

# %% [markdown]
# ---
# # Section 9 — Governance & Responsible AI
# 
# ## 9.1 Google's Responsible AI Principles
# 
# Google publishes seven AI principles that guide all AI products, including Vertex AI and Gemini:
# 
# | # | Principle | Meaning |
# |---|-----------|---------|
# | 1 | **Be socially beneficial** | Net benefit to society and users |
# | 2 | **Avoid creating or reinforcing unfair bias** | Fairness across demographics |
# | 3 | **Be built and tested for safety** | Rigorous testing before deployment |
# | 4 | **Be accountable to people** | Human oversight mechanisms |
# | 5 | **Incorporate privacy design principles** | Data minimisation, user control |
# | 6 | **Uphold high standards of scientific excellence** | Evidence-based approach |
# | 7 | **Be made available for uses that accord with these principles** | Access restrictions for harmful use |
# 
# ## 9.2 Model Cards and Model Documentation
# 
# - **Model Cards** — standardised documents that describe a model's intended use, training data, performance metrics, and known limitations.
# - Vertex AI Model Registry stores model metadata alongside artefacts.
# - Always document: intended use, out-of-scope use, evaluation datasets, bias assessment.
# 
# ## 9.3 Compliance & Data Governance
# 
# | Framework | What it Covers |
# |-----------|---------------|
# | **GDPR** | EU personal data — right to erasure, data minimisation |
# | **CCPA** | California consumer privacy rights |
# | **HIPAA** | Health data — requires BAA with Google |
# | **SOC 2** | Google Cloud holds SOC 2 Type II certification |
# | **ISO 27001** | Information security management |
# 
# **Data residency:** Use regional endpoints and enable **CMEK** (Customer-Managed Encryption Keys) in Cloud KMS to maintain key ownership.
# 
# ## 9.4 Bias Detection & Fairness
# 
# Vertex AI provides **Model Evaluation** and **What-If Tool** for bias analysis:
# 
# 1. **Slice your evaluation dataset** by sensitive attributes (age group, gender, etc.)
# 2. **Compare metrics** (accuracy, false positive rate) across slices
# 3. **Document disparities** in the model card
# 4. **Mitigate** via re-sampling, re-weighting, or adversarial debiasing
# 
# ## 9.5 Explainability (Vertex Explainable AI)
# 
# Vertex AI supports feature attribution methods for structured models:
# 
# | Method | Works With |
# |--------|-----------|
# | **Integrated Gradients** | Neural networks |
# | **XRAI** | Image models |
# | **Sampled Shapley** | Any tabular model |
# 
# Enable by setting `explanation_parameters` in your Model upload.

# %% [markdown]
# ---
# # Section 10 — Capstone Project: Deploy & Govern a GenAI Solution on Vertex AI
# 
# ## 🎯 Objective
# Build and deploy a **Responsible GenAI Loan Advisory Chatbot** on Vertex AI that:
# 1. Answers customer queries about loan products using Gemini
# 2. Enforces safety guardrails and blocks harmful content
# 3. Logs every interaction to Cloud Logging (audit trail)
# 4. Estimates and prints per-call token cost
# 5. Uses a Service Account for secure authentication (no API key in code)
# 6. Rejects biased or inappropriate outputs post-generation
# 
# ## 📐 Architecture
# 
# ```
# Customer Query
#      │
#      ▼
# ┌─────────────────────────────────────────────────────────┐
# │  Loan Advisory Bot (Python)                             │
# │                                                         │
# │  1. Authenticate (Service Account → Vertex AI)          │
# │  2. Count tokens → estimate cost                        │
# │  3. call Gemini 2.5 Flash via Vertex AI                 │
# │     └── safety settings enabled                         │
# │  4. Post-gen safety audit (raise if harmful)            │
# │  5. Log interaction to Cloud Logging                    │
# │  6. Return response to user                             │
# └─────────────────────────────────────────────────────────┘
#      │
#      ▼
# Cloud Logging → Cloud Monitoring → Alerts (if error rate > threshold)
# ```
# 
# ## ✅ Governance Checklist
# 
# - [ ] Service Account with `roles/aiplatform.user` only (least privilege)
# - [ ] SA key stored in Secret Manager, not in code or git
# - [ ] Safety settings: `BLOCK_LOW_AND_ABOVE` for all harm categories
# - [ ] Post-generation safety audit implemented
# - [ ] All calls logged with user session ID (anonymised)
# - [ ] Token cost tracked per session
# - [ ] Model version pinned (not `latest`) for reproducibility
# - [ ] Response grounded in product catalogue (RAG) — no hallucination
# - [ ] Bias evaluation run quarterly on sample outputs
# - [ ] Model card maintained and shared with compliance team

# %% [markdown]
# # ────────────────────────────────────────────────────────────────────────────
# # Capstone — Step 1: Configuration & Authentication
# # ────────────────────────────────────────────────────────────────────────────
# 
# import os, json, uuid, datetime
# from dotenv import load_dotenv
# from google.genai import Client
# from google.genai import types
# from google.oauth2.service_account import Credentials
# from google.cloud import logging as gcp_logging
# 
# load_dotenv(override=True)
# 
# # ── Config (load from .env in production) ───────────────────────────────────
# CAP_PROJECT_ID = os.getenv("GCP_PROJECT_ID",  "your-gcp-project-id")
# CAP_LOCATION   = os.getenv("GCP_REGION",      "us-central1")
# CAP_SA_KEY     = os.getenv("SA_KEY_PATH",     "path/to/service-account.json")
# CAP_MODEL      = "gemini-2.5-flash"          # pinned — do NOT use "latest"
# 
# PRICE_IN_PER_1K  = 0.000075   # USD per 1 000 input tokens
# PRICE_OUT_PER_1K = 0.00030    # USD per 1 000 output tokens
# 
# # ── Authentication ───────────────────────────────────────────────────────────
# credentials = Credentials.from_service_account_file(
#     CAP_SA_KEY,
#     scopes=["https://www.googleapis.com/auth/cloud-platform"],
# )
# 
# genai_client = Client(
#     vertexai=True,
#     project=CAP_PROJECT_ID,
#     location=CAP_LOCATION,
#     credentials=credentials,
# )
# 
# # ── Cloud Logging client ─────────────────────────────────────────────────────
# log_client = gcp_logging.Client(project=CAP_PROJECT_ID, credentials=credentials)
# logger     = log_client.logger("loan-advisory-bot")
# 
# print("✅ Capstone — authentication configured")
# print(f"   Model   : {CAP_MODEL}")
# print(f"   Project : {CAP_PROJECT_ID}")
# print(f"   Region  : {CAP_LOCATION}")

# %% [markdown]
# # ────────────────────────────────────────────────────────────────────────────
# # Capstone — Step 2: Product Knowledge Base (simulated RAG context)
# # In production, retrieve this from a vector database (Vertex AI Vector Search)
# # ────────────────────────────────────────────────────────────────────────────
# 
# PRODUCT_CATALOGUE = """
# ## Bank Loan Products
# 
# ### Home Loan
# - Interest rates: 8.5% – 10.2% p.a. (floating) | 9.0% – 10.8% (fixed)
# - Tenure: Up to 30 years
# - LTV ratio: Up to 80% of property value
# - Processing fee: 0.5% of loan amount (min ₹5,000)
# - Eligible income: ≥ ₹25,000/month
# 
# ### Personal Loan
# - Interest rates: 11% – 24% p.a.
# - Tenure: 1 – 5 years
# - Loan amount: ₹50,000 – ₹40,00,000
# - No collateral required
# - Eligible credit score: ≥ 700
# 
# ### Car Loan
# - Interest rates: 8.7% – 12.5% p.a.
# - Tenure: Up to 7 years
# - LTV ratio: Up to 85% of vehicle invoice value
# - Processing fee: ₹2,500 – ₹5,000
# 
# ### Education Loan
# - Interest rates: 9.5% – 12% p.a.
# - Loan amount: Up to ₹50,00,000 (abroad) | ₹20,00,000 (India)
# - Moratorium: Course period + 1 year
# - Collateral required for loans > ₹7.5 lakh
# """
# 
# SYSTEM_INSTRUCTION = f"""You are a professional, friendly loan advisory chatbot for SecureBank.
# Use ONLY the information in the product catalogue below to answer questions.
# If the answer is not in the catalogue, say: "I don't have that information — please visit a branch."
# Never fabricate interest rates, fees, or eligibility criteria.
# Always recommend the customer consult a branch manager for personalised advice.
# 
# {PRODUCT_CATALOGUE}
# """
# 
# print("✅ Product catalogue loaded")
# print(f"   Catalogue size: {len(PRODUCT_CATALOGUE)} characters")

# %% [markdown]
# # ────────────────────────────────────────────────────────────────────────────
# # Capstone — Step 3: Core Bot Function (safety + logging + cost tracking)
# # ────────────────────────────────────────────────────────────────────────────
# 
# SAFE_PROBS = {"NEGLIGIBLE", "LOW"}
# 
# def loan_bot(user_query: str, session_id: str | None = None) -> dict:
#     """
#     Sends user_query to Gemini on Vertex AI.
#     Returns dict with: response, input_tokens, output_tokens, cost_usd.
#     Logs the interaction to Cloud Logging.
#     Raises ValueError if safety threshold is exceeded.
#     """
#     session_id = session_id or str(uuid.uuid4())[:8]
#     start_ts   = datetime.datetime.utcnow()
# 
#     # ── 1. Count input tokens (cost estimate before calling) ────────────────
#     token_check = genai_client.models.count_tokens(
#         model=CAP_MODEL,
#         contents=user_query,
#     )
#     input_tokens = token_check.total_tokens
# 
#     # ── 2. Generate response ────────────────────────────────────────────────
#     response = genai_client.models.generate_content(
#         model=CAP_MODEL,
#         contents=user_query,
#         config=types.GenerateContentConfig(
#             system_instruction=SYSTEM_INSTRUCTION,
#             temperature=0.2,
#             max_output_tokens=400,
#             safety_settings=[
#                 types.SafetySetting(
#                     category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
#                     threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
#                 ),
#                 types.SafetySetting(
#                     category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
#                     threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
#                 ),
#                 types.SafetySetting(
#                     category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
#                     threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
#                 ),
#                 types.SafetySetting(
#                     category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
#                     threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
#                 ),
#             ],
#         ),
#     )
# 
#     # ── 3. Post-generation safety audit ─────────────────────────────────────
#     if response.candidates:
#         for rating in response.candidates[0].safety_ratings:
#             if rating.probability.name not in SAFE_PROBS:
#                 raise ValueError(
#                     f"⛔ Response blocked [{session_id}]: "
#                     f"{rating.category.name} = {rating.probability.name}"
#                 )
# 
#     answer       = response.text
#     output_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
#     cost_usd      = (input_tokens / 1000 * PRICE_IN_PER_1K) + \
#                     (output_tokens / 1000 * PRICE_OUT_PER_1K)
#     duration_ms   = int((datetime.datetime.utcnow() - start_ts).total_seconds() * 1000)
# 
#     # ── 4. Log to Cloud Logging ──────────────────────────────────────────────
#     logger.log_struct({
#         "event":         "loan_bot_call",
#         "session_id":    session_id,
#         "model":         CAP_MODEL,
#         "input_tokens":  input_tokens,
#         "output_tokens": output_tokens,
#         "cost_usd":      round(cost_usd, 8),
#         "duration_ms":   duration_ms,
#         "timestamp":     start_ts.isoformat() + "Z",
#     }, severity="INFO")
# 
#     return {
#         "response":      answer,
#         "input_tokens":  input_tokens,
#         "output_tokens": output_tokens,
#         "cost_usd":      cost_usd,
#         "duration_ms":   duration_ms,
#     }
# 
# print("✅ loan_bot() function defined")


