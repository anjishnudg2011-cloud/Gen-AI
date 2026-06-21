import requests
import os
from dotenv import load_dotenv
 
load_dotenv(override=True)
 
gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key is None:
    raise ValueError("GEMINI_API_KEY not found in .env file")
 
url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent"
 
headers = {
    "Content-Type": "application/json",
    "x-goog-api-key": gemini_api_key
}
 
def get_ai_response(user_message, max_tokens=200, temperature=0.7):
    """Simple function to get AI response from Gemini API"""
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": user_message
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
    }
    response = requests.post(url, json=payload, headers=headers)
    response_data = response.json()
    # Extract the text content
    if 'candidates' in response_data:
        return response_data['candidates'][0]['content']['parts'][0]['text'].strip()
    else:
        return "Error: No response generated"
 
print("=== Demo 1: Interactive Q&A (Simple) ===")
 
answer = get_ai_response("What is the capital of France?")
print(f'Q: What is the capital of France?\nA: {answer}\n')
 
answer = get_ai_response("Who is the president of India?")
print(f'Q: Who is the president of India?\nA: {answer}\n')