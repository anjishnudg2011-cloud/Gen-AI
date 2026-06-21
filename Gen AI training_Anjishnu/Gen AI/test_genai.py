from google import genai
import os
from dotenv import load_dotenv
from IPython.display import display, Markdown

load_dotenv(override=True)

# Configure API key (replace with your actual key or use environment variable)
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
# Initialize the client
client = genai.Client(api_key=GOOGLE_API_KEY)

# Set the model to use
MODEL_ID = "gemini-2.5-flash" 

prompt = """OBJECTIVE: Make legal content accessible to non-technical staff.
TASK: Below is a paragraph from a banking compliance document. You have to rewrite it in plain English suitable for branch staff training. Use very simple words for the non-technical staff to understand very clearly.
Banking Compliance Document:
"As per the bank’s Customer Due Diligence (CDD) Policy, all branches must verify the identity and address of every new customer before opening an account. This includes collecting valid KYC documents, screening the customer against regulatory watchlists, and ensuring that the customer profile is consistent with their stated source of funds. Branch staff must also report any suspicious or unusual customer behavior to the Compliance Department without informing the customer, as per the internal AML reporting guidelines."
CONTEXT: This will be
AUDIENCE: Non-technical staff
"""
response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt
)

display(Markdown("### Prompt Response:"))
display(Markdown(response.text))