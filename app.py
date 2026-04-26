import os
import chromadb
import gradio as gr
import datetime
import pytz
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from typing import TypedDict
from langgraph.graph import StateGraph, END

# --- 1. CONFIGURATION ---
SOURCE_DATA_DIR = "./hr_data"
DRIVE_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-mpnet-base-v2"

os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1) 
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# --- 2. THE BUILDER ---
def build_knowledge_base():
    if not os.path.exists(SOURCE_DATA_DIR): return
    persistent_client = chromadb.PersistentClient(path=DRIVE_DB_PATH)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    
    for item_name in os.listdir(SOURCE_DATA_DIR):
        if item_name.startswith('.'): continue
        safe_name = item_name.split('.')[0] + "_collection"
        file_path = os.path.join(SOURCE_DATA_DIR, item_name)
        try:
            loader = DirectoryLoader(file_path, glob="**/*", loader_cls=UnstructuredFileLoader) if os.path.isdir(file_path) else UnstructuredFileLoader(file_path)
            docs = loader.load()
            if docs:
                splits = splitter.split_documents(docs)
                Chroma(client=persistent_client, collection_name=safe_name, embedding_function=embeddings).add_documents(splits)
        except Exception: pass

build_knowledge_base()

# --- 3. AGENT NODES ---
class AgentState(TypedDict):
    query: str
    response: str
    next_node: str
    debug_log: str

def router_node(state):
    user_input = state["query"].strip().lower()
    
    # Selection Mapping
    if user_input in ["1", "one"]: return {"next_node": "recruitment_agent", "query": "Summary of all candidates", "debug_log": "🚦 Router: Option 1"}
    if user_input in ["2", "two"]: return {"next_node": "tech_agent", "query": "Technical interview benchmarks", "debug_log": "🚦 Router: Option 2"}
    if user_input in ["3", "three"]: return {"next_node": "hr_policy_agent", "query": "Policy and leave overview", "debug_log": "🚦 Router: Option 3"}

    # Keyword Mapping
    rec_kw = ["candidate", "cv", "resume", "who are", "quek", "tan", "linda", "john", "robert", "sarah", "david", "chen", "role", "position", "hire", "job", "hr"]
    pol_kw = ["leave", "medical", "allowance", "claim", "handbook", "hybrid", "remote", "annual", "policy", "benefit", "conduct"]
    tec_kw = ["interview questions", "benchmark", "technical", "coding test", "assessment"]
    
    if any(word in user_input for word in pol_kw): return {"next_node": "hr_policy_agent", "debug_log": "🚦 Router: Policy"}
    if any(word in user_input for word in tec_kw): return {"next_node": "tech_agent", "debug_log": "🚦 Router: Tech"}
    if any(word in user_input for word in rec_kw): return {"next_node": "recruitment_agent", "debug_log": "🚦 Router: Recruitment"}
    
    return {"next_node": "general_agent", "debug_log": "💬 Router: General"}

def run_grounded_agent(state, agent_name, collection_name, persona, specific_instructions=""):
    query = state["query"]
    try:
        db = Chroma(client=chromadb.PersistentClient(path=DRIVE_DB_PATH), collection_name=collection_name, embedding_function=embeddings)
        docs = db.similarity_search(query, k=6)
        context = "\n\n".join([d.page_content for d in docs])
    except: context = ""
    
    system_prompt = (
        f"You are the {persona}. Answer clearly based ONLY on the documents.\n\n"
        "### FORMATTING:\n"
        "- Use **Bold Headers** and Markdown TABLES.\n\n"
        "### CORE RULES:\n"
        f"{specific_instructions}\n"
        "- NO re-introductions.\n"
        "- Never mention file names.\n"
        "- End with a single professional follow-up question."
        f"\n\nDOCS:\n{context}"
    )
    res = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
    return {"response": res.content, "debug_log": f"✅ {agent_name}: Data processed."}

def recruitment_node(state): 
    # ADDED INSTRUCTION: Distinguish between Experience and Aspiration
    instr = "- IMPORTANT: Distinguish clearly between a candidate's 'Core Experience' and their 'Career Aspirations/Pivots'. If a candidate is seeking a role outside their main history, mention it as a 'Pivot Opportunity'."
    return run_grounded_agent(state, "Recruitment Specialist", "1_collection", "Senior Recruitment Expert", instr)

def tech_node(state): return run_grounded_agent(state, "Technical Specialist", "2_collection", "Technical Interview Consultant")
def policy_node(state): return run_grounded_agent(state, "Policy Specialist", "3_collection", "HR Policy Expert")

def general_node(state):
    system_prompt = (
        "You are the Little Lotus HR AI Assistant. Be professional and helpful.\n"
        "If unsure, offer the 3 main areas of expertise (1. Recruitment, 2. Technical, 3. Policy) with their numbers."
    )
    res = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=state["query"])])
    return {"response": res.content, "debug_log": "💬 General Chat."}

# --- 4. THE GRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("recruitment_agent", recruitment_node)
workflow.add_node("tech_agent", tech_node)
workflow.add_node("hr_policy_agent", policy_node)
workflow.add_node("general_agent", general_node)
workflow.set_entry_point("router")
workflow.add_conditional_edges("router", lambda state: state["next_node"])
workflow.add_edge("recruitment_agent", END)
workflow.add_edge("tech_agent", END)
workflow.add_edge("hr_policy_agent", END)
workflow.add_edge("general_agent", END)
app = workflow.compile()

# --- 5. THE UI ---
def get_initial_message():
    sg_tz = pytz.timezone('Asia/Singapore')
    hour = datetime.datetime.now(sg_tz).hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
    return f"""{greeting}. I am the **Little Lotus HR AI Assistant**.

Please select an area of interest (type the **number** or keyword):
1. 👥 **Candidate Recruitment**
2. 💻 **Technical Interviewing**
3. 📋 **Company Policies**"""

def chat_interface(message, history):
    result = app.invoke({"query": message})
    return f"{result['response']}\n\n---\n<details><summary>🧠 Thinking Process</summary>{result['debug_log']}</details>"

demo = gr.ChatInterface(
    fn=chat_interface, 
    title="Little Lotus HR AI Assistant", 
    chatbot=gr.Chatbot(value=[{"role": "assistant", "content": get_initial_message()}])
)
demo.launch()
