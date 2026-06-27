import streamlit as st

from agents.chat import ChatAgent
from agents.chunking import ChunkingAgent
from agents.embeddings import EmbeddingAgent
from agents.pdf_reader import PDFReaderAgent
from agents.planner import PlannerAgent
from agents.research import ResearchAgent
from agents.retriever import RetrieverAgent
from agents.summarizer import SummarizerAgent
from agents.vector_store import VectorStoreAgent
from agents.web_search import WebSearchAgent
from orchestration import AGENT_ROLES, get_orchestration_runtime
from utils.report import build_research_pdf

st.set_page_config(
    page_title="ResearchMind",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*{font-family:'Inter',-apple-system,sans-serif!important;box-sizing:border-box;}
:root{
  --bg:#F0F2FF;--white:#FFFFFF;--surface:#F8F9FF;
  --border:#E2E6FF;--border2:#CBD0F0;
  --text:#1a1a2e;--muted:#6B7098;
  --brand:#4361EE;--brand-light:#EEF1FF;--brand-mid:#C5CEFF;
  --green:#10B981;--green-bg:#ECFDF5;--green-border:#A7F3D0;
  --red:#EF4444;--red-bg:#FEF2F2;
  --purple:#7C3AED;--purple-bg:#F5F3FF;
}
.stApp{background:var(--bg)!important;}
[data-testid="stSidebar"]{display:none!important;}
.block-container{padding:0!important;max-width:100%!important;}
#MainMenu,footer,header{visibility:hidden;}
[data-testid="stToolbar"]{display:none!important;}
[data-testid="stDecoration"]{display:none!important;}

/* ── NAV ── */
.topnav{background:var(--white);border-bottom:1.5px solid var(--border);
  padding:0 28px;height:56px;display:flex;align-items:center;justify-content:space-between;
  position:sticky;top:0;z-index:999;box-shadow:0 1px 12px rgba(67,97,238,0.07);}
.nav-brand{display:flex;align-items:center;gap:12px;}
.nav-logo{width:34px;height:34px;background:linear-gradient(135deg,#4361EE,#7C3AED);
  border-radius:9px;display:flex;align-items:center;justify-content:center;
  font-size:13px;font-weight:900;color:#fff;letter-spacing:-0.5px;}
.nav-title{font-size:1rem;font-weight:800;color:var(--text);}
.nav-sub{font-size:0.67rem;color:var(--muted);}
.nav-info{font-size:0.7rem;color:var(--muted);font-weight:500;}
.nav-info b{color:var(--text);font-weight:700;}

/* modebar replaced by .modebar-wrap radio styles below */

/* ── CARDS ── */
.card{background:var(--white);border:1.5px solid var(--border);border-radius:12px;
  padding:16px 18px;margin-bottom:14px;}
.card-title{font-size:0.63rem;font-weight:800;color:var(--muted);
  text-transform:uppercase;letter-spacing:0.09em;margin-bottom:12px;}

/* ── FILE UPLOAD ── */
[data-testid="stFileUploader"] label{display:none!important;}
[data-testid="stFileUploader"] [data-testid="stWidgetLabel"]{display:none!important;}
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploadDropzone"]{
  background:var(--brand-light)!important;
  border:1.5px dashed var(--brand-mid)!important;
  border-radius:10px!important;}
[data-testid="stFileUploaderDropzone"]:hover,
[data-testid="stFileUploadDropzone"]:hover{border-color:var(--brand)!important;}
[data-testid="stFileUploadDropzone"] button,
[data-testid="stFileUploaderDropzone"] button{
  background:var(--brand)!important;color:#fff!important;
  border-radius:7px!important;border:none!important;font-weight:700!important;}

/* ── FILE BADGE ── */
.file-badge{background:var(--green-bg);border:1px solid var(--green-border);
  border-radius:8px;padding:8px 12px;margin-top:10px;display:flex;align-items:center;gap:8px;}
.file-badge-name{font-size:0.78rem;font-weight:700;color:var(--green);}
.file-badge-size{font-size:0.7rem;color:var(--muted);}

/* ── METRICS ── */
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:14px;}
.mcard{background:var(--white);border:1.5px solid var(--border);border-radius:10px;padding:12px 14px;}
.mcard-label{font-size:0.6rem;font-weight:700;color:var(--muted);text-transform:uppercase;
  letter-spacing:0.07em;margin-bottom:4px;}
.mcard-val{font-size:1.25rem;font-weight:800;color:var(--text);}

/* ── PIPELINE ── */
.pipeline{display:flex;align-items:center;flex-wrap:wrap;gap:3px;padding:4px 0;}
.pn{display:flex;align-items:center;gap:5px;padding:5px 10px;border-radius:20px;
  border:1px solid var(--border);background:var(--surface);
  font-size:0.68rem;font-weight:600;white-space:nowrap;color:var(--muted);}
.pn.done{border-color:var(--green-border);background:var(--green-bg);color:var(--green);}
.pn.run{border-color:var(--brand-mid);background:var(--brand-light);color:var(--brand);}
.pn.err{border-color:#FCA5A5;background:var(--red-bg);color:var(--red);}
.pdot{width:6px;height:6px;border-radius:50%;flex-shrink:0;}
.done .pdot{background:var(--green);}
.run .pdot{background:var(--brand);}
.err .pdot{background:var(--red);}
.pn.wait .pdot{background:var(--border2);}
.parr{color:var(--border2);font-size:0.7rem;padding:0 2px;}

/* ── REPORT SECTIONS ── */
.sc{background:var(--white);border:1.5px solid var(--border);border-radius:10px;
  padding:14px 16px;margin-bottom:10px;}
.sc-label{font-size:0.6rem;font-weight:800;color:var(--brand);text-transform:uppercase;
  letter-spacing:0.08em;padding:3px 8px;background:var(--brand-light);border-radius:20px;
  display:inline-block;margin-bottom:8px;}
.sc-text{font-size:0.84rem;color:var(--text);line-height:1.7;}

/* ── SUMMARY ── */
.sumbox{background:var(--green-bg);border:1.5px solid var(--green-border);
  border-radius:10px;padding:14px 16px;margin-bottom:12px;}
.sumbox-label{font-size:0.6rem;font-weight:800;color:var(--green);
  text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;}
.sumbox-text{font-size:0.84rem;color:var(--text);line-height:1.7;}

/* ── REFERENCES ── */
.ref-card{background:var(--white);border:1.5px solid var(--border);border-radius:10px;
  padding:12px 14px;margin-bottom:8px;display:flex;gap:12px;align-items:flex-start;}
.ref-num{width:26px;height:26px;border-radius:50%;background:var(--brand-light);
  color:var(--brand);font-size:0.7rem;font-weight:800;display:flex;align-items:center;
  justify-content:center;flex-shrink:0;margin-top:2px;}
.ref-title{font-size:0.82rem;font-weight:700;color:var(--text);margin-bottom:3px;}
.ref-url{font-size:0.7rem;color:var(--brand);margin-bottom:4px;word-break:break-all;}
.ref-snip{font-size:0.74rem;color:var(--muted);line-height:1.5;}
.ref-score{font-size:0.63rem;font-weight:700;padding:2px 7px;border-radius:20px;
  background:var(--brand-light);color:var(--brand);white-space:nowrap;}

/* ── CHAT ── */
.chat-outer{background:var(--white);border:1.5px solid var(--border);
  border-radius:12px;overflow:hidden;margin-bottom:12px;}
.chat-head{background:var(--brand-light);border-bottom:1px solid var(--brand-mid);padding:10px 14px;}
.chat-head-label{font-size:0.75rem;font-weight:700;color:var(--brand);}
.chat-body{padding:14px;display:flex;flex-direction:column;gap:12px;min-height:180px;}
.cmsg{display:flex;gap:10px;}
.cav{width:28px;height:28px;border-radius:8px;display:flex;align-items:center;
  justify-content:center;font-size:0.63rem;font-weight:800;flex-shrink:0;}
.cav.u{background:var(--brand-light);color:var(--brand);}
.cav.a{background:var(--purple-bg);color:var(--purple);}
.cbody{flex:1;}
.crole{font-size:0.6rem;font-weight:800;color:var(--muted);
  text-transform:uppercase;letter-spacing:0.05em;margin-bottom:3px;}
.ctext{font-size:0.82rem;color:var(--text);line-height:1.6;
  background:var(--surface);padding:8px 12px;border-radius:8px;}
.cmsg.u .ctext{background:var(--brand-light);}

/* ── EMPTY STATE ── */
.empty{text-align:center;padding:44px 20px;background:var(--white);
  border:1.5px dashed var(--border);border-radius:12px;margin:10px 0;}
.empty-icon{font-size:2.2rem;margin-bottom:10px;}
.empty-title{font-size:0.95rem;font-weight:700;color:var(--text);margin-bottom:6px;}
.empty-desc{font-size:0.8rem;color:var(--muted);}

/* ── AGENT CARDS ── */
.agent-row{display:flex;align-items:flex-start;gap:12px;
  background:var(--surface);border:1.5px solid var(--border);
  border-radius:9px;padding:11px 14px;margin-bottom:7px;}
.agent-icon{width:32px;height:32px;border-radius:8px;background:var(--brand-light);
  display:flex;align-items:center;justify-content:center;font-size:15px;flex-shrink:0;}
.agent-body{flex:1;}
.agent-name{font-size:0.82rem;font-weight:700;color:var(--text);margin-bottom:3px;}
.agent-goal{font-size:0.72rem;color:var(--muted);line-height:1.5;}

/* ── SOURCE ITEMS ── */
.src-item{padding:6px 0;border-bottom:1px solid var(--border);}
.src-item:last-child{border-bottom:none;}
.src-title{font-size:0.75rem;font-weight:600;color:var(--brand);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.src-url{font-size:0.63rem;color:var(--muted);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}

/* ── STREAMLIT OVERRIDES ── */
.stTextInput input,.stTextArea textarea{
  background:var(--surface)!important;color:var(--text)!important;
  border:1.5px solid var(--border)!important;border-radius:8px!important;font-size:0.85rem!important;}
.stTextInput input:focus,.stTextArea textarea:focus{
  border-color:var(--brand)!important;box-shadow:0 0 0 3px rgba(67,97,238,0.1)!important;}
[data-testid="stWidgetLabel"] p{
  color:var(--muted)!important;font-size:0.65rem!important;font-weight:700!important;
  text-transform:uppercase!important;letter-spacing:0.06em!important;}
.stButton>button{border-radius:8px!important;font-weight:700!important;
  font-size:0.82rem!important;height:38px!important;border:none!important;transition:all 0.18s!important;}
.stButton>button[kind="primary"]{background:var(--brand)!important;color:#fff!important;}
.stButton>button[kind="secondary"]{background:var(--white)!important;
  border:1.5px solid var(--border)!important;color:var(--text)!important;}
.stProgress>div>div{background:var(--brand)!important;border-radius:4px!important;}
div[data-testid="stTabs"] [data-baseweb="tab-list"]{
  background:transparent!important;border-bottom:1.5px solid var(--border)!important;
  padding:0 28px!important;gap:0!important;}
div[data-testid="stTabs"] [data-baseweb="tab"]{
  background:transparent!important;color:var(--muted)!important;border:none!important;
  font-weight:600!important;font-size:0.82rem!important;padding:9px 14px!important;
  border-bottom:2px solid transparent!important;border-radius:0!important;}
div[data-testid="stTabs"] [aria-selected="true"]{
  color:var(--brand)!important;border-bottom-color:var(--brand)!important;}
div[data-testid="stTabs"] [data-baseweb="tab-panel"]{padding:20px 28px!important;}

/* ── EXPANDER ── */
[data-testid="stExpander"]{border:none!important;background:transparent!important;}
[data-testid="stExpander"] details{
  border:1.5px solid var(--border)!important;border-radius:8px!important;
  background:var(--surface)!important;}
[data-testid="stExpander"] details summary{
  list-style:none!important;padding:9px 14px!important;
  font-weight:600!important;font-size:0.82rem!important;color:var(--text)!important;
  cursor:pointer!important;background:transparent!important;}
[data-testid="stExpander"] details summary::-webkit-details-marker{display:none!important;}
[data-testid="stExpander"] details summary::marker{content:""!important;}
[data-testid="stExpanderToggleIcon"]{display:none!important;}

/* ── MODE RADIO SWITCHER ── */
.modebar-wrap{
  background:var(--white);border-bottom:1.5px solid var(--border);
  padding:4px 28px;}
.modebar-wrap [data-testid="stRadio"]{display:flex!important;align-items:center!important;}
.modebar-wrap [data-baseweb="radio-group"]{display:flex!important;gap:0!important;flex-direction:row!important;}
.modebar-wrap [data-baseweb="radio"]{
  display:flex!important;align-items:center!important;
  padding:6px 18px!important;cursor:pointer!important;
  border-bottom:2.5px solid transparent!important;
  margin:0!important;}
.modebar-wrap [data-baseweb="radio"]:has(input:checked){
  border-bottom:2.5px solid var(--brand)!important;}
/* radio circle — hide the actual circle, show text only */
.modebar-wrap [data-baseweb="radio"] [data-testid="stMarkdownContainer"] p{
  font-size:0.84rem!important;font-weight:600!important;color:var(--muted)!important;}
.modebar-wrap [data-baseweb="radio"]:has(input:checked) [data-testid="stMarkdownContainer"] p{
  color:var(--brand)!important;}
/* hide the actual circle dot */
.modebar-wrap [data-baseweb="radio"] div:first-child{display:none!important;}
.modebar-wrap [data-testid="stWidgetLabel"]{display:none!important;}
</style>
""", unsafe_allow_html=True)


# ── STATE ──────────────────────────────────────────────────────────────────────
def _init():
    defaults = dict(
        mode="pdf", pdf_text="", chunks=[], vector_store=None,
        document_key=None, last_pdf_name="", topic="", top_k=5,
        plan="", research={}, summary="", chat_history=[],
        agent_status={}, last_context="", web_results=[],
        web_sources=[], general_chunks=[], general_vector_store=None,
    )
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

def _set(k, s):
    st.session_state.agent_status[k] = s

def _cls(s):
    s = s.lower()
    if "done" in s or "ready" in s: return "done"
    if "run" in s: return "run"
    if "error" in s: return "err"
    return "wait"


# ── AGENTS ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_agents():
    return dict(
        reader=PDFReaderAgent(), chunking=ChunkingAgent(),
        embedding=EmbeddingAgent(), vector_store=VectorStoreAgent(),
        retriever=RetrieverAgent(), planner=PlannerAgent(),
        research=ResearchAgent(), summarizer=SummarizerAgent(),
        chat=ChatAgent(), web_search=WebSearchAgent(),
    )


# ── PIPELINE STEPS ─────────────────────────────────────────────────────────────
def process_pdf(uploaded_file):
    ag = get_agents()
    try:
        prog = st.progress(0)
        _set("PDF Reader", "Running"); prog.progress(10, "Extracting text…")
        r = ag["reader"].extract(uploaded_file)
        st.session_state.pdf_text = r.text
        st.session_state.document_key = r.document_key
        st.session_state.last_pdf_name = uploaded_file.name
        _set("PDF Reader", f"Done ({r.page_count} pages)")

        _set("Chunking", "Running"); prog.progress(30, "Chunking…")
        chunks = ag["chunking"].chunk(r.text)
        st.session_state.chunks = chunks
        _set("Chunking", f"Done ({len(chunks)} chunks)")

        _set("Embedding", "Running"); prog.progress(55, "Generating embeddings…")
        embeddings = ag["embedding"].embed_documents([c.text for c in chunks])
        _set("Embedding", "Done")

        _set("Vector Store", "Running"); prog.progress(80, "Building FAISS index…")
        vs = ag["vector_store"].build_or_load(r.document_key, chunks, embeddings)
        st.session_state.vector_store = vs
        _set("Vector Store", f"Ready ({vs.index.ntotal} vectors)")

        prog.progress(100, "✅ Ready!")
        return True
    except Exception as e:
        st.error(f"PDF processing error: {e}")
        _set("PDF Reader", "Error")
        return False


def run_web_search(topic):
    ag = get_agents()
    _set("Web Search", "Running")
    r = ag["web_search"].search(topic, max_results=8)
    if not r.ok:
        _set("Web Search", "Error")
        st.error(f"Web search failed: {r.error}")
        return False
    st.session_state.web_results = r.results
    st.session_state.web_sources = [
        {"title": x.title, "url": x.url, "content": x.content, "score": x.score}
        for x in r.results
    ]
    st.session_state.last_context = r.combined_context
    _set("Web Search", f"Done ({len(r.results)} sources)")
    _set("Retriever", "Running")
    chunks = ag["chunking"].chunk(r.combined_context)
    st.session_state.general_chunks = chunks
    if chunks:
        import hashlib
        embs = ag["embedding"].embed_documents([c.text for c in chunks])
        key = hashlib.md5(topic.encode()).hexdigest() + "_web"
        vs = ag["vector_store"].build_or_load(key, chunks, embs)
        st.session_state.general_vector_store = vs
    _set("Retriever", "Done")
    return True


def retrieve_ctx(query, top_k, mode):
    ag = get_agents()
    vs = st.session_state.vector_store if mode == "pdf" else st.session_state.general_vector_store
    if not vs:
        return st.session_state.last_context[:6000]
    q_emb = ag["embedding"].embed_query(query)
    chunks = ag["retriever"].retrieve(vs, q_emb, top_k=top_k)
    ctx = "\n\n".join(c.text for c in chunks)
    st.session_state.last_context = ctx
    return ctx


def create_plan():
    ag = get_agents()
    topic = st.session_state.topic
    mode = st.session_state.mode
    if not topic:
        st.warning("Please enter a research topic first.")
        return False
    _set("Planner", "Running")
    ctx = retrieve_ctx(topic, st.session_state.top_k, mode)
    r = ag["planner"].create_plan(topic=topic, context=ctx, mode=mode)
    if r.ok:
        st.session_state.plan = r.text
        _set("Planner", "Done")
        return True
    _set("Planner", "Error")
    st.error(r.text)
    return False


def run_research():
    ag = get_agents()
    topic = st.session_state.topic
    mode = st.session_state.mode
    if not topic:
        st.warning("Please enter a research topic first.")
        return False
    ctx = retrieve_ctx(topic, st.session_state.top_k, mode)
    _set("Research", "Running")
    res = ag["research"].research(topic=topic, plan=st.session_state.plan, context=ctx, mode=mode)
    if not res.ok:
        _set("Research", "Error")
        st.error(res.error)
        return False
    st.session_state.research = res.sections
    _set("Research", "Done")
    _set("Summarizer", "Running")
    summ = ag["summarizer"].summarize(topic=topic, research=res.sections, context=ctx)
    if summ.ok:
        st.session_state.summary = summ.text
        _set("Summarizer", "Done")
    else:
        _set("Summarizer", "Error")
    return True


def run_full(uploaded_file=None):
    mode = st.session_state.mode
    if not st.session_state.topic:
        st.warning("Please enter a research topic first.")
        return
    with st.status("Running pipeline…", expanded=True) as s:
        if mode == "pdf":
            if not uploaded_file:
                st.warning("Please upload a PDF first.")
                return
            st.write("📄 Processing PDF…")
            if not process_pdf(uploaded_file):
                return
        else:
            st.write("🌐 Searching the web…")
            if not run_web_search(st.session_state.topic):
                return
        st.write("📋 Creating research plan…")
        create_plan()
        st.write("✍️ Generating research report…")
        run_research()
        s.update(label="✅ Pipeline complete!", state="complete")


# ── PIPELINE BAR ───────────────────────────────────────────────────────────────
def _pipeline_bar():
    mode = st.session_state.mode
    nodes = (
        ["PDF Reader","Chunking","Embedding","Vector Store","Retriever","Planner","Research","Summarizer"]
        if mode == "pdf" else
        ["Web Search","Retriever","Planner","Research","Summarizer"]
    )
    parts = []
    for i, n in enumerate(nodes):
        stat = st.session_state.agent_status.get(n, "Waiting")
        parts.append(f'<div class="pn {_cls(stat)}"><div class="pdot"></div>{n}</div>')
        if i < len(nodes) - 1:
            parts.append('<span class="parr">›</span>')
    st.markdown(f'<div class="pipeline">{"".join(parts)}</div>', unsafe_allow_html=True)


# ── NAV + MODE BAR ─────────────────────────────────────────────────────────────
def render_nav():
    mode = st.session_state.mode

    st.markdown("""
    <div class="topnav">
      <div class="nav-brand">
        <div class="nav-logo">RM</div>
        <div>
          <div class="nav-title">ResearchMind</div>
          <div class="nav-sub">Multi-Agent Research Assistant</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Mode switcher — real Streamlit radio rendered as custom toggle
    st.markdown('<div class="modebar-wrap">', unsafe_allow_html=True)
    col_mode, _ = st.columns([3, 7])
    with col_mode:
        picked = st.radio(
            "mode_sel",
            ["📄 PDF Research", "🌐 General Web Research"],
            index=0 if mode == "pdf" else 1,
            horizontal=True,
            label_visibility="collapsed",
            key="mode_radio_main",
        )
    st.markdown('</div>', unsafe_allow_html=True)

    new_mode = "pdf" if "PDF" in picked else "general"
    if new_mode != st.session_state.mode:
        st.session_state.mode = new_mode
        st.session_state.agent_status = {}
        st.rerun()


# ── WORKSPACE: PDF ─────────────────────────────────────────────────────────────
def tab_workspace_pdf():
    left, right = st.columns([1, 1.85], gap="large")

    with left:
        st.markdown('<div class="card"><div class="card-title">📄 Source document</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            label_visibility="hidden",
            key="pdf_up",
        )
        if uploaded:
            st.markdown(f"""
            <div class="file-badge">
              <span style="font-size:1.2rem">📎</span>
              <div>
                <div class="file-badge-name">{uploaded.name}</div>
                <div class="file-badge-size">{uploaded.size/1024:.1f} KB</div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">⚙️ Settings</div>', unsafe_allow_html=True)
        topic = st.text_input("Research topic", value=st.session_state.topic,
                              placeholder="e.g. Machine Learning in Healthcare", key="t_pdf")
        if topic != st.session_state.topic:
            st.session_state.topic = topic
        top_k = st.slider("Retrieval depth (chunks)", 2, 10, st.session_state.top_k, key="k_pdf")
        st.session_state.top_k = top_k
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">🎯 Actions</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📄 Process PDF", use_container_width=True, key="btn_pp"):
                if not uploaded: st.warning("Upload a PDF first.")
                else:
                    with st.spinner(): process_pdf(uploaded)
                    st.rerun()
        with c2:
            if st.button("📋 Create Plan", use_container_width=True, key="btn_mp"):
                with st.spinner(): create_plan()
                st.rerun()
        if st.button("✍️ Generate Report", use_container_width=True, key="btn_gr"):
            with st.spinner(): run_research()
            st.rerun()
        if st.button("🚀 Run Full Pipeline", use_container_width=True, type="primary", key="btn_fp"):
            run_full(uploaded_file=uploaded)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        vc = st.session_state.vector_store.index.ntotal if st.session_state.vector_store else 0
        fname = st.session_state.last_pdf_name or "—"
        st.markdown(f"""
        <div class="metrics">
          <div class="mcard"><div class="mcard-label">Document</div>
            <div class="mcard-val" style="font-size:0.8rem;line-height:1.3">{fname[:20]}</div></div>
          <div class="mcard"><div class="mcard-label">Characters</div>
            <div class="mcard-val">{len(st.session_state.pdf_text):,}</div></div>
          <div class="mcard"><div class="mcard-label">Chunks</div>
            <div class="mcard-val">{len(st.session_state.chunks):,}</div></div>
          <div class="mcard"><div class="mcard-label">Vectors</div>
            <div class="mcard-val">{vc:,}</div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">🔄 Agent pipeline</div>', unsafe_allow_html=True)
        _pipeline_bar()
        st.markdown('</div>', unsafe_allow_html=True)
        _render_report("pdf")


# ── WORKSPACE: GENERAL ─────────────────────────────────────────────────────────
def tab_workspace_general():
    left, right = st.columns([1, 1.85], gap="large")

    with left:
        st.markdown('<div class="card"><div class="card-title">🌐 Research topic</div>', unsafe_allow_html=True)
        topic = st.text_input("What do you want to research?", value=st.session_state.topic,
                              placeholder="e.g. Quantum Computing in Cryptography",
                              label_visibility="collapsed", key="t_gen")
        if topic != st.session_state.topic:
            st.session_state.topic = topic
        top_k = st.slider("Context depth (chunks)", 2, 10, st.session_state.top_k, key="k_gen")
        st.session_state.top_k = top_k
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">🎯 Actions</div>', unsafe_allow_html=True)
        if st.button("🌐 Search Web", use_container_width=True, key="btn_sw"):
            if not st.session_state.topic: st.warning("Enter a topic first.")
            else:
                with st.spinner("Searching…"): run_web_search(st.session_state.topic)
                st.rerun()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📋 Plan", use_container_width=True, key="btn_pg"):
                with st.spinner(): create_plan()
                st.rerun()
        with c2:
            if st.button("✍️ Report", use_container_width=True, key="btn_rg"):
                with st.spinner(): run_research()
                st.rerun()
        if st.button("🚀 One-Click Research", use_container_width=True, type="primary", key="btn_ocr"):
            run_full(); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.web_sources:
            st.markdown('<div class="card"><div class="card-title">🔗 Sources retrieved</div>', unsafe_allow_html=True)
            for s in st.session_state.web_sources[:6]:
                t = s['title'][:52] + "…" if len(s['title']) > 52 else s['title']
                u = s['url'][:48] + "…" if len(s['url']) > 48 else s['url']
                st.markdown(f'<div class="src-item"><div class="src-title">{t}</div>'
                            f'<div class="src-url">{u}</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with right:
        ns = len(st.session_state.web_results)
        nc = len(st.session_state.general_chunks)
        vs = st.session_state.general_vector_store
        nv = vs.index.ntotal if vs else 0
        st.markdown(f"""
        <div class="metrics">
          <div class="mcard"><div class="mcard-label">Web sources</div><div class="mcard-val">{ns}</div></div>
          <div class="mcard"><div class="mcard-label">Chunks</div><div class="mcard-val">{nc}</div></div>
          <div class="mcard"><div class="mcard-label">Vectors</div><div class="mcard-val">{nv}</div></div>
          <div class="mcard"><div class="mcard-label">Sections</div><div class="mcard-val">{len(st.session_state.research)}</div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">🔄 Agent pipeline</div>', unsafe_allow_html=True)
        _pipeline_bar()
        st.markdown('</div>', unsafe_allow_html=True)
        _render_report("general")


# ── REPORT RENDERER ────────────────────────────────────────────────────────────
def _render_report(mode):
    if st.session_state.plan:
        with st.expander("View Research Plan", expanded=False):
            st.markdown(st.session_state.plan)

    if not st.session_state.research:
        icon = "📄" if mode == "pdf" else "🌐"
        label = ("Upload & process a PDF, then click Run Full Pipeline"
                 if mode == "pdf" else "Enter a topic and click One-Click Research")
        st.markdown(f"""
        <div class="empty">
          <div class="empty-icon">{icon}</div>
          <div class="empty-title">No report yet</div>
          <div class="empty-desc">{label}</div>
        </div>""", unsafe_allow_html=True)
        return

    for section, content in st.session_state.research.items():
        st.markdown(f"""
        <div class="sc">
          <div class="sc-label">{section}</div>
          <div class="sc-text">{content}</div>
        </div>""", unsafe_allow_html=True)

    if st.session_state.summary:
        st.markdown(f"""
        <div class="sumbox">
          <div class="sumbox-label">⚡ Executive Summary</div>
          <div class="sumbox-text">{st.session_state.summary}</div>
        </div>""", unsafe_allow_html=True)

    sources = st.session_state.web_sources if mode == "general" else []
    pdf_bytes = build_research_pdf(
        st.session_state.topic, st.session_state.research,
        st.session_state.summary, sources, mode=mode)
    st.download_button("⬇️ Download Report as PDF", data=pdf_bytes,
        file_name=f"{st.session_state.topic.replace(' ','_')}_Report.pdf",
        mime="application/pdf", use_container_width=True)


# ── CHAT TAB ───────────────────────────────────────────────────────────────────
def tab_chat():
    mode = st.session_state.mode
    ag = get_agents()
    vs = st.session_state.vector_store if mode == "pdf" else st.session_state.general_vector_store

    if not vs:
        hint = "Process a PDF first" if mode == "pdf" else "Run a web search first"
        st.markdown(f"""
        <div class="empty"><div class="empty-icon">💬</div>
          <div class="empty-title">No source loaded</div>
          <div class="empty-desc">{hint} to enable chat.</div>
        </div>""", unsafe_allow_html=True)
        return

    src = f"📄 {st.session_state.last_pdf_name}" if mode == "pdf" else f"🌐 {st.session_state.topic}"
    st.markdown(f"""
    <div class="chat-outer">
      <div class="chat-head"><div class="chat-head-label">Chatting with: {src}</div></div>
      <div class="chat-body">""", unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown('<div style="text-align:center;padding:28px;color:var(--muted);font-size:0.82rem;">'
                    'Ask anything about your source material ↓</div>', unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            is_u = msg["role"] == "user"
            cls = "u" if is_u else "a"
            av = "U" if is_u else "AI"
            lbl = "You" if is_u else "ResearchMind"
            st.markdown(f"""
            <div class="cmsg {cls}">
              <div class="cav {cls}">{av}</div>
              <div class="cbody">
                <div class="crole">{lbl}</div>
                <div class="ctext">{msg["content"]}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

    q = st.chat_input("Ask a question about your source…")
    if q:
        st.session_state.chat_history.append({"role": "user", "content": q})
        q_emb = ag["embedding"].embed_query(q)
        chunks = ag["retriever"].retrieve(vs, q_emb, top_k=st.session_state.top_k)
        ans = ag["chat"].answer(q, chunks, mode=mode)
        txt = ans.text if ans.ok else "Could not generate a response. Please try again."
        st.session_state.chat_history.append({"role": "assistant", "content": txt})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat", key="btn_cc"):
            st.session_state.chat_history = []
            st.rerun()


# ── REFERENCES TAB ─────────────────────────────────────────────────────────────
def tab_references():
    mode = st.session_state.mode
    if mode == "pdf":
        if not st.session_state.pdf_text:
            st.markdown('<div class="empty"><div class="empty-icon">📄</div>'
                        '<div class="empty-title">No PDF loaded</div>'
                        '<div class="empty-desc">Process a PDF to view extracted content.</div></div>',
                        unsafe_allow_html=True)
            return
        t1, t2 = st.tabs(["📝 Extracted Text", "🧩 Chunks"])
        with t1:
            st.text_area("Raw text", st.session_state.pdf_text[:8000],
                         height=380, label_visibility="collapsed")
        with t2:
            for c in st.session_state.chunks[:25]:
                with st.expander(f"Chunk {c.id}  —  chars {c.start} to {c.end}"):
                    st.caption(c.text)
    else:
        if not st.session_state.web_results:
            st.markdown('<div class="empty"><div class="empty-icon">🌐</div>'
                        '<div class="empty-title">No web search yet</div>'
                        '<div class="empty-desc">Run General Web Research to see sources.</div></div>',
                        unsafe_allow_html=True)
            return

        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;">
          <div style="font-size:0.98rem;font-weight:800;color:var(--text);">
            {len(st.session_state.web_results)} Sources Retrieved
          </div>
          <div style="font-size:0.72rem;color:var(--muted);">
            Topic: <strong style="color:var(--text);">{st.session_state.topic}</strong>
          </div>
        </div>""", unsafe_allow_html=True)

        for i, r in enumerate(st.session_state.web_results, 1):
            score_pct = int(r.score * 100)
            safe_title = r.title.replace("<","&lt;").replace(">","&gt;")
            safe_snip = r.content[:300].replace("<","&lt;").replace(">","&gt;")
            url_disp = r.url[:72] + ("…" if len(r.url) > 72 else "")
            st.markdown(f"""
            <div class="ref-card">
              <div class="ref-num">{i}</div>
              <div style="flex:1;">
                <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:8px;margin-bottom:4px;">
                  <div class="ref-title">{safe_title}</div>
                  <div class="ref-score">{score_pct}%</div>
                </div>
                <div class="ref-url">🔗 <a href="{r.url}" target="_blank"
                  style="color:var(--brand);text-decoration:none;">{url_disp}</a></div>
                <div class="ref-snip">{safe_snip}{"…" if len(r.content)>300 else ""}</div>
              </div>
            </div>""", unsafe_allow_html=True)


# ── ARCHITECTURE TAB ───────────────────────────────────────────────────────────
def tab_architecture():
    c1, c2 = st.columns([1, 1], gap="large")
    mode = st.session_state.mode

    with c1:
        st.markdown('<div class="card"><div class="card-title">🤖 Agent registry</div>', unsafe_allow_html=True)
        icons = {"PDF Reader Agent":"📄","Chunking Agent":"✂️","Embedding Agent":"🔢",
                 "Vector Store Agent":"🗄️","Web Search Agent":"🌐","Retriever Agent":"🔍",
                 "Planner Agent":"📋","Research Agent":"✍️","Summarizer Agent":"📊","Chat Agent":"💬"}
        for role in AGENT_ROLES:
            ic = icons.get(role["name"], "🤖")
            st.markdown(f"""
            <div class="agent-row">
              <div class="agent-icon">{ic}</div>
              <div class="agent-body">
                <div class="agent-name">{role['name']}</div>
                <div class="agent-goal">{role['goal']}</div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card"><div class="card-title">🔧 Runtime status</div>', unsafe_allow_html=True)
        from config import GROQ_API_KEY, TAVILY_API_KEY, GROQ_MODEL
        checks = [
            ("Groq LLM", bool(GROQ_API_KEY), GROQ_MODEL),
            ("Tavily Search", bool(TAVILY_API_KEY), "Web search"),
            ("FAISS Vector DB", True, "CPU index"),
            ("SentenceTransformers", True, "all-MiniLM-L6-v2"),
        ]
        for name, ok, detail in checks:
            color = "var(--green)" if ok else "var(--red)"
            bg = "var(--green-bg)" if ok else "var(--red-bg)"
            border = "var(--green-border)" if ok else "#FCA5A5"
            icon = "✓" if ok else "✗ Missing — add to .env"
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {border};border-radius:8px;
                        padding:9px 14px;margin-bottom:6px;
                        display:flex;justify-content:space-between;align-items:center;">
              <div>
                <span style="font-size:0.82rem;font-weight:700;color:var(--text);">{name}</span>
                <span style="font-size:0.7rem;color:var(--muted);margin-left:8px;">{detail}</span>
              </div>
              <span style="color:{color};font-weight:800;font-size:0.85rem;">{icon}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">🗺️ Workflow — current mode</div>', unsafe_allow_html=True)
        if mode == "pdf":
            flow = ("User → PDF Reader → Chunking → Embedding\n"
                    "→ FAISS Vector Store → Retriever\n"
                    "→ Planner → Research Agent → Summarizer\n"
                    "→ Report PDF  |  Retriever → Chat Agent")
        else:
            flow = ("User → Web Search (Tavily) → Chunking → Embedding\n"
                    "→ FAISS Vector Store → Retriever\n"
                    "→ Planner → Research Agent → Summarizer\n"
                    "→ Report PDF  |  Retriever → Chat Agent")
        st.code(flow, language=None)
        st.markdown('</div>', unsafe_allow_html=True)


# ── MAIN ───────────────────────────────────────────────────────────────────────
_init()
render_nav()

tabs = st.tabs(["🏠 Workspace", "💬 Chat", "📚 References", "🏗️ Architecture"])
with tabs[0]:
    if st.session_state.mode == "pdf":
        tab_workspace_pdf()
    else:
        tab_workspace_general()
with tabs[1]:
    tab_chat()
with tabs[2]:
    tab_references()
with tabs[3]:
    tab_architecture()
