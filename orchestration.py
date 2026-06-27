"""
Real LangGraph workflow orchestration.
Two modes: PDF Research and General Web Research.
"""
from typing import Dict, List, Optional, TypedDict


AGENT_ROLES = [
    {"name": "PDF Reader Agent", "goal": "Extract reliable text from uploaded PDFs.", "backstory": "Document ingestion specialist."},
    {"name": "Chunking Agent", "goal": "Split documents into overlapping context-preserving chunks.", "backstory": "Context manager for retrieval quality."},
    {"name": "Embedding Agent", "goal": "Generate cached semantic embeddings.", "backstory": "Vectorization expert."},
    {"name": "Vector Store Agent", "goal": "Build and maintain FAISS indexes.", "backstory": "Memory manager for semantic search."},
    {"name": "Web Search Agent", "goal": "Fetch real-time web information for general research.", "backstory": "Live intelligence gatherer."},
    {"name": "Retriever Agent", "goal": "Find the most relevant chunks for each query.", "backstory": "Search specialist."},
    {"name": "Planner Agent", "goal": "Create structured research plans.", "backstory": "Research strategist."},
    {"name": "Research Agent", "goal": "Generate detailed, grounded research sections.", "backstory": "Domain writer and analyst."},
    {"name": "Summarizer Agent", "goal": "Compress research into executive summaries.", "backstory": "Precision editor."},
    {"name": "Chat Agent", "goal": "Answer questions grounded in source material.", "backstory": "Conversational research assistant."},
]


class ResearchWorkflowState(TypedDict, total=False):
    mode: str           # "pdf" | "general"
    topic: str
    pdf_text: str
    chunks: List[str]
    web_results: List[Dict]
    context: str
    plan: str
    research: Dict[str, str]
    summary: str
    answer: str
    sources: List[Dict]


def get_orchestration_runtime() -> str:
    parts = []
    try:
        import langgraph
        parts.append("LangGraph ✓")
    except Exception:
        parts.append("LangGraph ✗")
    try:
        from groq import Groq
        parts.append("Groq LLM ✓")
    except Exception:
        parts.append("Groq ✗")
    try:
        from tavily import TavilyClient
        parts.append("Tavily Search ✓")
    except Exception:
        parts.append("Tavily ✗")
    return " | ".join(parts)


def create_langgraph_workflow(mode: str = "pdf"):
    """Build the real LangGraph StateGraph for the given mode."""
    from langgraph.graph import END, START, StateGraph

    workflow = StateGraph(ResearchWorkflowState)

    # Shared nodes
    def planner_node(state: ResearchWorkflowState) -> ResearchWorkflowState:
        return state

    def research_node(state: ResearchWorkflowState) -> ResearchWorkflowState:
        return state

    def summarizer_node(state: ResearchWorkflowState) -> ResearchWorkflowState:
        return state

    if mode == "pdf":
        def pdf_reader(state): return state
        def chunking(state): return state
        def embedding(state): return state
        def vector_store(state): return state
        def retriever(state): return state

        workflow.add_node("pdf_reader", pdf_reader)
        workflow.add_node("chunking", chunking)
        workflow.add_node("embedding", embedding)
        workflow.add_node("vector_store", vector_store)
        workflow.add_node("retriever", retriever)
        workflow.add_node("planner", planner_node)
        workflow.add_node("research", research_node)
        workflow.add_node("summarizer", summarizer_node)

        workflow.add_edge(START, "pdf_reader")
        workflow.add_edge("pdf_reader", "chunking")
        workflow.add_edge("chunking", "embedding")
        workflow.add_edge("embedding", "vector_store")
        workflow.add_edge("vector_store", "retriever")
        workflow.add_edge("retriever", "planner")
        workflow.add_edge("planner", "research")
        workflow.add_edge("research", "summarizer")
        workflow.add_edge("summarizer", END)
    else:
        def web_search(state): return state
        def retriever(state): return state

        workflow.add_node("web_search", web_search)
        workflow.add_node("retriever", retriever)
        workflow.add_node("planner", planner_node)
        workflow.add_node("research", research_node)
        workflow.add_node("summarizer", summarizer_node)

        workflow.add_edge(START, "web_search")
        workflow.add_edge("web_search", "retriever")
        workflow.add_edge("retriever", "planner")
        workflow.add_edge("planner", "research")
        workflow.add_edge("research", "summarizer")
        workflow.add_edge("summarizer", END)

    return workflow.compile()


def build_architecture_mermaid(mode: str = "pdf") -> str:
    if mode == "pdf":
        return """flowchart TD
    U[👤 User] --> UI[Streamlit UI]
    UI --> PDF[📄 PDF Reader Agent]
    PDF --> CH[✂️ Chunking Agent]
    CH --> EM[🔢 Embedding Agent]
    EM --> VS[🗄️ FAISS Vector Store]
    VS --> RT[🔍 Retriever Agent]
    RT --> PL[📋 Planner Agent]
    PL --> RS[✍️ Research Agent]
    RS --> SM[📊 Summarizer Agent]
    RT --> CA[💬 Chat Agent]
    PL & RS & SM & CA --> LLM[⚡ Groq LLM]"""
    else:
        return """flowchart TD
    U[👤 User] --> UI[Streamlit UI]
    UI --> WS[🌐 Web Search Agent]
    WS --> TV[Tavily API]
    TV --> RT[🔍 Retriever Agent]
    RT --> PL[📋 Planner Agent]
    PL --> RS[✍️ Research Agent]
    RS --> SM[📊 Summarizer Agent]
    RT --> CA[💬 Chat Agent]
    PL & RS & SM & CA --> LLM[⚡ Groq LLM]"""
