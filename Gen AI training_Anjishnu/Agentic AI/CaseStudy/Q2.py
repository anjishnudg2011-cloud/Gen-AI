# ── CASE STUDY 2 STARTER CODE ──────────────────────────────────────────────
from langchain_classic.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
)
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# --- Banking chatbot prompt --
banking_prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
You are a helpful banking assistant for CitizenBank.
Always address the customer by name once you know it.
Be professional, concise, and empathetic.
Previous conversation:
{history}
Customer: {input}
CitizenBank Assistant:"""
)

# --- Step 1: Buffer Memory Chain (first 2 turns — KYC capture) --
buffer_memory = ConversationBufferMemory()
# YOUR CODE HERE — create LLMChain with buffer_memory

buffer_chain = LLMChain(llm=llm, prompt=banking_prompt, memory=buffer_memory)
turn1 = buffer_chain.predict(input="Hi, my name is Rajesh and I have a savings account with you.")
print("Turn 1:", turn1)
turn2 = buffer_chain.predict(input="Can you tell me my current account balance?")
print("Turn 2:", turn2)

# --- Step 2: Window Memory Chain (turns 3-5 — active support) --
window_memory = ConversationBufferWindowMemory(k=3)
# YOUR CODE HERE — create LLMChain with window_memory
window_chain = LLMChain(llm=llm, prompt=banking_prompt, memory=window_memory)
turn3 = window_chain.predict(input="I see a transaction of Rs 5000 on March 10 to a merchant I don't recognize. Can you help?")
print("Turn 3:", turn3)
turn4 = window_chain.predict(input="Also, I'd like to temporarily block my debit card for security reasons.")
print("Turn 4:", turn4)
turn5 = window_chain.predict(input="Thank you. How long does a dispute resolution take?")
print("Turn 5:", turn5)

# --- Step 3: Summary Memory (end of session — handoff note) --
summary_memory = ConversationSummaryMemory(llm=llm)
# YOUR CODE HERE — create LLMChain with summary_memory
summary_chain = LLMChain(llm=llm, prompt=banking_prompt, memory=summary_memory)

# Feed the key events into summary memory
summary_chain.predict(input="Summarize this session: Rajesh with savings account, asked for balance, reported a suspicious transaction, requested card block, and inquired about dispute resolution time.")

print("\n--- HANDOFF SUMMARY FOR HUMAN AGENT ---")
# YOUR CODE HERE — print the summary memory buffer
print(summary_memory.load_memory_variables({})["history"])