# 🔬 ResearchMind — Multi-Agent Research Assistant

A production-ready multi-agent AI research assistant powered by **Groq LLM**, **Tavily web search**, and **FAISS RAG** — with two completely separate modes and a clean light-mode UI.

## ✨ Features

| Feature | Detail |
|---------|--------|
| 📄 **PDF Research Mode** | Upload any PDF → RAG-powered report + grounded Q&A chat |
| 🌐 **Web Research Mode** | Enter any topic → real-time web search → AI report with cited sources |
| 💬 **Grounded Chat** | Ask questions answered strictly from your source |
| 📚 **References Panel** | Full source list with relevance scores and clickable URLs |
| ⬇️ **PDF Export** | Download the full research report as a styled PDF |
| ⚡ **Groq LLM** | Fast inference — llama-3.3-70b (free tier available) |
| 🗄️ **FAISS Cache** | Embeddings cached locally — no reprocessing same PDFs |

## 🤖 Agent Pipeline

```
PDF Mode:   PDF Reader → Chunking → Embedding → FAISS → Retriever → Planner → Research → Summarizer
Web Mode:   Tavily Search → Chunking → Embedding → FAISS → Retriever → Planner → Research → Summarizer
Both modes: Retriever → Chat Agent (for Q&A)
```

## 🚀 Quick Start

```bash
# 1. Clone or unzip
cd ResearchMind

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add API keys
cp .env.example .env
# Edit .env and fill in GROQ_API_KEY and TAVILY_API_KEY

# 5. Run
streamlit run app.py
```

## 🔑 API Keys (both have free tiers)

| Service | Get key at | Free tier |
|---------|-----------|-----------|
| **Groq** | https://console.groq.com | 14,400 req/day |
| **Tavily** | https://app.tavily.com | 1,000 searches/month |

## ☁️ Deploy

### Streamlit Cloud (recommended, free)
1. Push to GitHub
2. Go to **share.streamlit.io** → New App
3. Set secrets: `GROQ_API_KEY` and `TAVILY_API_KEY`

### Render / Railway
- Build: `pip install -r requirements.txt`
- Start: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`
- Add env vars in dashboard

## 📁 Structure

```
ResearchMind/
├── app.py                    # UI — light mode, tabs, two modes
├── config.py                 # All settings via env vars
├── orchestration.py          # LangGraph workflow + agent metadata
├── agents/
│   ├── llm.py                # Groq client with retry + streaming
│   ├── pdf_reader.py         # PDF text extraction (pypdf)
│   ├── chunking.py           # Overlapping sentence-aware chunking
│   ├── embeddings.py         # Cached SentenceTransformers
│   ├── vector_store.py       # FAISS index with disk cache
│   ├── retriever.py          # Semantic top-k retrieval
│   ├── web_search.py         # Tavily real-time web search
│   ├── planner.py            # Research plan generator
│   ├── research.py           # 6-section report writer
│   ├── summarizer.py         # Executive summary + recommendations
│   └── chat.py               # Grounded Q&A agent
├── utils/
│   └── report.py             # Styled PDF report export
├── .streamlit/
│   └── config.toml           # Light theme config
├── .env.example              # API key template
├── requirements.txt
└── Procfile                  # Deployment config
```

## Author
**Anu** · https://github.com/anurajput06
