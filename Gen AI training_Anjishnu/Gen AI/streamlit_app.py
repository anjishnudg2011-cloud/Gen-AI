import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ── Load environment variables from .env ──────────────────────────────────────
load_dotenv(override=True)
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gemini Chatbot",
    page_icon="✨",
    layout="centered",
)
 
# ── Minimal custom CSS ────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        #MainMenu, footer, header {visibility: hidden;}
        .app-header {
            text-align: center;
            padding: 1.2rem 0 0.4rem;
            font-size: 1.7rem;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        .app-subheader {
            text-align: center;
            color: #888;
            font-size: 0.9rem;
            margin-bottom: 1.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
 
# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="app-header">✨ Gemini Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subheader">Powered by Google Gemini · Conversation history enabled</div>',
    unsafe_allow_html=True,
)
 
# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
 
    if GOOGLE_API_KEY:
        st.success("API key loaded from .env", icon="🔑")
    else:
        st.error("GEMINI_API_KEY not found in .env")
 
    model_name = st.selectbox(
        "Model",
        ["gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-1.5-pro", "gemini-1.5-flash"],
        index=0,
    )
 
    temperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.05)
 
    system_prompt = st.text_area(
        "System prompt (optional)",
        placeholder="You are a helpful assistant…",
        height=100,
    )
 
    st.divider()
 
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
 
    st.caption("Conversation history is kept in-session.")
 
# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
 
# ── Render existing messages ──────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
 
# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Message Gemini…")
 
if user_input:
    if not GOOGLE_API_KEY:
        st.warning("GEMINI_API_KEY not found. Please add it to your .env file and restart the app.")
        st.stop()
 
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
 
    # Build history (all turns except the latest user message)
    history: list[types.Content] = []
    for msg in st.session_state.messages[:-1]:
        role = "model" if msg["role"] == "assistant" else "user"
        history.append(
            types.Content(role=role, parts=[types.Part(text=msg["content"])])
        )
 
    # Call Gemini and stream response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
 
        try:
            client = genai.Client(api_key=GOOGLE_API_KEY)
 
            config_kwargs: dict = {"temperature": temperature}
            if system_prompt.strip():
                config_kwargs["system_instruction"] = system_prompt.strip()
 
            # Create chat with history and stream
            chat = client.chats.create(
                model=model_name,
                history=history,
                config=types.GenerateContentConfig(**config_kwargs),
            )
 
            for chunk in chat.send_message_stream(user_input):
                if chunk.text:
                    full_response += chunk.text
                    response_placeholder.markdown(full_response + "▌")
 
            response_placeholder.markdown(full_response)
 
        except Exception as exc:
            error_msg = str(exc)
            if "API_KEY_INVALID" in error_msg or "invalid" in error_msg.lower():
                full_response = "❌ **Invalid API key.** Please check your .env file."
            elif "quota" in error_msg.lower():
                full_response = "❌ **Quota exceeded.** Please check your Google AI Studio quota."
            else:
                full_response = f"❌ **Error:** {error_msg}"
            response_placeholder.markdown(full_response)
 
    st.session_state.messages.append({"role": "assistant", "content": full_response})
 