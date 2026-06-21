# ── CASE STUDY 7 STARTER CODE ──────────────────────────────────────────────
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL = "gemini-2.5-flash"

# --- Step 1: System instruction for credit analyst persona --
CREDIT_ANALYST_INSTRUCTION = """
You are a certified credit risk analyst at RiskGuard Bank with 15 years of experience.
Analyze the given applicant profile and provide:
1. Risk Rating: VERY LOW / LOW / MEDIUM / HIGH / VERY HIGH
2. Key Risk Factors (if any)
3. Mitigating Factors (if any)
4. Recommended Decision: APPROVE / APPROVE WITH CONDITIONS / REJECT
5. Conditions (if applicable)
Always base your assessment on financial data only. Be objective and concise.
"""

# --- Step 2: Safety settings --
safety_settings = [
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_LOW_AND_ABOVE"
    ),
    # Added harassment blocking
    types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="BLOCK_MEDIUM_AND_ABOVE"
    )
]

# --- Step 3: Test applicant profiles --
applicant_profiles = {
    "LOW_RISK": """
    Applicant: Meera Nair, Age 38
    Employment: Senior Software Engineer, TCS (11 years)
    Annual Income: Rs 24,00,000 (24 LPA)
    Credit Score: 812
    Existing Loans: Home loan EMI Rs 35,000/month (3 years remaining)
    Savings: Rs 18,00,000 in FDs
    Requested: Personal loan of Rs 5,00,000 for home renovation
    """,
    "HIGH_RISK": """
    Applicant: Vijay Kumar, Age 27
    Employment: Freelancer (6 months tenure, irregular income)
    Annual Income: Rs 2,40,000 (2.4 LPA) — estimated
    Credit Score: 542
    Existing Loans: 2 personal loans (Rs 8,000 EMI each), 1 credit card (maxed a
    Savings: Rs 12,000
    Requested: Personal loan of Rs 3,00,000 for business expansion
    Payment History: 3 missed EMI payments in the last 12 months
    """
}

# --- Step 4: Run analysis for both profiles --
for risk_label, profile in applicant_profiles.items():
    print(f"\n{'='*65}")
    print(f"CREDIT RISK ANALYSIS — {risk_label} APPLICANT")
    print('='*65)

    # Detailed analysis config (with thinking)
    detailed_config = types.GenerateContentConfig(
        system_instruction=CREDIT_ANALYST_INSTRUCTION,
        temperature=0.1,
        max_output_tokens=800,
        safety_settings=safety_settings,
        thinking_config=types.ThinkingConfig(thinking_budget=1024)
    )

    # Count tokens before API call
    token_count = client.models.count_tokens(
        model=MODEL,
        contents=[CREDIT_ANALYST_INSTRUCTION, profile]
    )
    print(f"Estimated prompt tokens: {token_count.total_tokens}")

    # Streaming mode with thinking budget
    print("\n📊 Detailed Analysis (with deep reasoning):\n")
    response_stream = client.models.generate_content_stream(
        model=MODEL,
        contents=[profile],
        config=detailed_config
    )

    # Print streaming chunks
    for chunk in response_stream:
        if chunk.text:
            print(chunk.text, end="", flush=True)

    print("\n\n" + "-"*70)

    # Fast screening config (no thinking)
    fast_config = types.GenerateContentConfig(
        system_instruction=CREDIT_ANALYST_INSTRUCTION,
        temperature=0,
        max_output_tokens=120,
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )

    # Fast response — non-streaming
    fast_response = client.models.generate_content(
        model=MODEL,
        contents=[profile],
        config=fast_config
    )

    print("\n⚡ Fast Screening Result:")
    print(fast_response.text[:250])

    print("\n🔢 Token Usage:")
    print(f"Prompt Tokens: {fast_response.usage_metadata.prompt_token_count}")
    print(f"Output Tokens: {fast_response.usage_metadata.candidates_token_count}")
