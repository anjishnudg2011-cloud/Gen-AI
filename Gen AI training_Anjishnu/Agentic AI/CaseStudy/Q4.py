# ── CASE STUDY 4 STARTER CODE ──────────────────────────────────────────────
import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# --- Step 1: Define banking tools --
@tool
def run_kyc_check(account_id: str) -> str:
    """Performs KYC verification on a bank account. Returns KYC status."""
    kyc_data = {
        "ACC-001": "KYC VERIFIED — PAN, Aadhaar, Address confirmed",
        "ACC-002": "KYC PENDING — Missing address proof",
        "ACC-003": "KYC FAILED — Mismatch in PAN details",
    }
    return kyc_data.get(account_id, "KYC UNKNOWN — Account not found")

@tool
def check_beneficiary_watchlist(beneficiary_id: str) -> str:
    """Checks if a beneficiary account is on the RBI fraud watchlist."""
    watchlist = ["BEN-999", "BEN-777"]  # blacklisted
    if beneficiary_id in watchlist:
        return f"WATCHLIST HIT — {beneficiary_id} is flagged by RBI Fraud Registry"
    return f"CLEAR — {beneficiary_id} is not on any watchlist"

@tool
def get_transfer_history(account_id: str) -> str:
    """Returns the count of large transfers (>Rs 10L) made in the last 30 days."""
    history = {"ACC-001": 0, "ACC-002": 3, "ACC-003": 8}
    count = history.get(account_id, 0)
    risk = "HIGH" if count > 5 else "MEDIUM" if count > 2 else "LOW"
    return f"{count} large transfers in 30 days — Velocity Risk: {risk}"

tools = [run_kyc_check, check_beneficiary_watchlist, get_transfer_history]
llm_with_tools = llm.bind_tools(tools)

# --- Step 2: Define TransferState with message accumulator --
class TransferState(TypedDict):
    messages : Annotated[list[BaseMessage], operator.add]
    transfer_amount : float
    account_id : str
    beneficiary_id :  str
    transfer_status : str   # 'PENDING', 'APPROVED', 'REJECTED'
    officer_note : str

# --- Step 3: Define nodes --
def screening_agent(state: TransferState) -> dict:
    """Agent node — calls KYC, watchlist, and history tools."""
    # YOUR CODE HERE
    account = state["account_id"]
    bene = state["beneficiary_id"]

    user_prompt = (
        f"Screen this transfer:\n"
        f"- Account: {account}\n"
        f"- Beneficiary: {bene}\n"
        f"- Amount: Rs {state['transfer_amount']}\n\n"
        f"Run KYC, watchlist, and transfer history checks."
    )

    ai_msg = llm_with_tools.invoke([HumanMessage(content=user_prompt)])
    return {"messages": [ai_msg]}

def tools_executor(state: TransferState) -> dict:
    """Tools node — executes all pending tool calls from the agent."""
    # YOUR CODE HERE
    last_msg = state["messages"][-1]

    new_messages = []

    for call in last_msg.tool_calls:
        tool_name = call["name"]
        tool_input = call["args"]

        tool_obj = next(t for t in tools if t.name == tool_name)
        tool_result = tool_obj.invoke(tool_input)

        new_messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=call["id"]
            )
        )

    # Send tool results back to model for interpretation
    followup = llm_with_tools.invoke(state["messages"] + new_messages)
    new_messages.append(followup)

    return {"messages": new_messages}
    
def execute_transfer(state: TransferState) -> dict:
    """Final node — executes the transfer (only reached after officer approval)."""
    amount = state["transfer_amount"]
    account = state["account_id"]
    beneficiary = state["beneficiary_id"]
    return {
    "transfer_status": "EXECUTED",
    "officer_note": f"Transfer of Rs {amount:,.0f} from {account} to {beneficiary} has been executed successfully."
}
def should_continue(state: TransferState) -> str:
    """Routing: if agent has tool calls → tools node, else → execute_transfer."""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "execute_transfer"

# --- Step 4: Build the graph --
builder = StateGraph(TransferState)
# YOUR CODE HERE — add nodes, edges, conditional edges
builder.add_node("agent", screening_agent)
builder.add_node("tools", tools_executor)
builder.add_node("execute_transfer", execute_transfer)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue,
                              {"tools": "tools", "execute": "execute_transfer"})
builder.add_edge("tools", "agent")
builder.add_edge("execute_transfer", END)
mem = MemorySaver()
transfer_graph = builder.compile(
    checkpointer=mem,
    interrupt_before=["execute_transfer"]   # HITL pause point
)

# --- Step 5: Initiate transfer request ---
transfer_request = {
    "messages": [
        HumanMessage(content="Please screen and process this wire transfer.")
    ],
    "transfer_amount": 1500000,  # Rs 15 lakhs
    "account_id": "ACC-001",
    "beneficiary_id": "BEN-202",
    "transfer_status": "PENDING",
    "officer_note": ""
}

cfg = {"configurable": {"thread_id": "transfer-T001"}}

paused_state = transfer_graph.invoke(transfer_request, config=cfg)

print("\n--- AGENT SCREENING COMPLETE — AWAITING OFFICER REVIEW ---")
print(paused_state["messages"][-1].content)

# --- Step 6: Officer Decision ---

officer_decision = input("\nOfficer Decision — Type APPROVE or REJECT: ").strip().upper()

if officer_decision == "APPROVE":
    final_state = transfer_graph.invoke(None, config=cfg)  # resume and complete
    print("\n✅ TRANSFER EXECUTED:")
    print(final_state["officer_note"])

else:
    transfer_graph.update_state(cfg, {
        "transfer_status": "REJECTED",
        "officer_note": "Transfer rejected by bank officer."
    })
    print("\n❌ TRANSFER REJECTED by bank officer.")