import streamlit as st
import os
import tempfile
import uuid
import shutil
import time
from typing import TypedDict, List, Literal

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from langgraph.graph import StateGraph, END, START
from dotenv import load_dotenv

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="DocuChat AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 2.5rem;
        color: #4A90E2;
        text-align: center;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

os.environ["ANONYMIZED_TELEMETRY"] = "False"
load_dotenv()

st.markdown('<div class="main-header">🧠 DocuChat AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Converse com seus documentos PDF usando Llama 3 & RAG</div>', unsafe_allow_html=True)


# --- GARBAGE COLLECTOR ---
def cleanup_old_sessions():
    current_time = time.time()
    for filename in os.listdir("."):
        if filename.startswith("chroma_db_"):
            dir_path = os.path.join(".", filename)
            if os.path.isdir(dir_path):
                try:
                    if (current_time - os.path.getctime(dir_path)) > 3600:
                        shutil.rmtree(dir_path)
                except Exception:
                    pass

cleanup_old_sessions()


# --- CACHED RESOURCES ---
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_llm(model_name: str, temperature: float):
    return ChatGroq(temperature=temperature, model_name=model_name)

@st.cache_resource
def get_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# --- PDF PROCESSING ---
def process_pdf(file_bytes, file_name, persist_dir):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source_file"] = file_name

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(docs)

        embedding_model = get_embedding_model()

        if os.path.exists(persist_dir):
            vector_store = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
            vector_store.add_documents(chunks)
        else:
            Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory=persist_dir
            )

        return chunks

    except Exception as e:
        st.error(f"Erro ao processar '{file_name}': {e}")
        return None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def generate_summary(file_name, chunks, model_name, temperature):
    sample = "\n\n".join([c.page_content for c in chunks[:6]])
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Você é um assistente especializado em análise de documentos. "
         "Crie resumos concisos e informativos em português do Brasil."),
        ("human",
         f"Faça um resumo executivo do documento **{file_name}** com base no trecho abaixo. "
         f"Inclua: tema principal, pontos-chave e qualquer dado relevante encontrado.\n\n{sample}")
    ])
    chain = prompt | get_llm(model_name, temperature) | StrOutputParser()
    return chain.invoke({})


def build_base_retriever(persist_dir, all_chunks):
    embedding_model = get_embedding_model()
    vector_store = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)

    vector_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7}
    )

    if all_chunks:
        bm25_retriever = BM25Retriever.from_documents(all_chunks)
        bm25_retriever.k = 6
        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6]
        )

    return vector_retriever


def rerank(query, docs, top_n=4):
    reranker = get_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]


# ---------------------------------------------------------------------------
# LANGGRAPH — Retrieval Agent
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    question: str           # original user question
    query: str              # current query (may be rewritten)
    chat_history: list      # LangChain HumanMessage / AIMessage objects
    documents: List[Document]
    retries: int            # how many rewrite attempts so far


def build_rag_graph(model_name: str, temperature: float, persist_dir: str, all_chunks: list):
    """
    Builds and compiles a LangGraph retrieval agent.

    Graph flow:
        contextualize → retrieve → rerank → grade
                                              │
                               "relevant" → END   (documents ready for answer)
                               "rewrite"  → rewrite → retrieve (max 1 retry)
    """
    llm = get_llm(model_name, temperature)
    retriever = build_base_retriever(persist_dir, all_chunks)

    # Node 1: rewrite follow-up questions into standalone queries
    def contextualize(state: RAGState) -> RAGState:
        if not state["chat_history"]:
            return {**state, "query": state["question"]}

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Dado o histórico da conversa e a última pergunta do usuário, "
             "reformule-a como uma questão independente. NÃO responda, apenas reformule."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        chain = prompt | llm | StrOutputParser()
        new_query = chain.invoke({"input": state["question"], "chat_history": state["chat_history"]})
        return {**state, "query": new_query}

    # Node 2: hybrid BM25 + vector retrieval
    def retrieve(state: RAGState) -> RAGState:
        docs = retriever.invoke(state["query"])
        return {**state, "documents": docs}

    # Node 3: cross-encoder reranking
    def rerank_node(state: RAGState) -> RAGState:
        docs = rerank(state["query"], state["documents"], top_n=4)
        return {**state, "documents": docs}

    # Node 4: rewrite the query differently if context was insufficient
    def rewrite_query(state: RAGState) -> RAGState:
        prompt = ChatPromptTemplate.from_template(
            "A busca pela pergunta '{question}' não retornou contexto suficiente. "
            "Reformule-a de forma diferente para melhorar a recuperação, "
            "mantendo o mesmo objetivo. Retorne apenas a nova pergunta."
        )
        chain = prompt | llm | StrOutputParser()
        new_query = chain.invoke({"question": state["query"]})
        return {**state, "query": new_query, "retries": state["retries"] + 1}

    # Conditional edge: grade whether retrieved docs are relevant enough
    def grade_documents(state: RAGState) -> Literal["generate", "rewrite"]:
        # After 1 retry, proceed regardless to avoid infinite loops
        if state["retries"] >= 1:
            return "generate"

        grader_prompt = ChatPromptTemplate.from_template(
            "Pergunta: {question}\n\n"
            "Trecho do documento:\n{document}\n\n"
            "Este trecho contém informação útil para responder a pergunta? "
            "Responda apenas 'sim' ou 'não'."
        )
        grader = grader_prompt | llm | StrOutputParser()

        for doc in state["documents"]:
            result = grader.invoke({
                "question": state["query"],
                "document": doc.page_content[:500]
            })
            if "sim" in result.lower():
                return "generate"

        return "rewrite"

    # Build the graph
    graph = StateGraph(RAGState)

    graph.add_node("contextualize", contextualize)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank_node)
    graph.add_node("rewrite", rewrite_query)

    graph.add_edge(START, "contextualize")
    graph.add_edge("contextualize", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_conditional_edges(
        "rerank",
        grade_documents,
        {"generate": END, "rewrite": "rewrite"}
    )
    graph.add_edge("rewrite", "retrieve")

    return graph.compile()


# --- SESSION INIT ---
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "summarized_files" not in st.session_state:
    st.session_state.summarized_files = set()

persist_dir = f"./chroma_db_{st.session_state.session_id}"

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("❌ A chave da API Groq não foi encontrada. Verifique o arquivo .env")
    st.stop()


# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Configurações")

    model_option = st.selectbox(
        "Escolha o Modelo:",
        ("llama-3.3-70b-versatile", "llama-3.1-8b-instant"),
        format_func=lambda x: "Llama 3.3 70B (Mais Inteligente)" if "70b" in x else "Llama 3.1 8B (Mais Rápido)"
    )

    temperature = st.slider(
        "Temperatura:",
        min_value=0.0, max_value=1.0, value=0.2, step=0.1,
        help="0 = mais preciso e factual | 1 = mais criativo e exploratório"
    )

    st.markdown("---")
    st.header("📂 Documentos")

    uploaded_files = st.file_uploader(
        "Arraste seus PDFs aqui",
        type="pdf",
        accept_multiple_files=True
    )

    if st.session_state.processed_files:
        st.markdown("**Indexados:**")
        for fname in sorted(st.session_state.processed_files):
            st.markdown(f"- {fname}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Limpar Chat"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        chat_text = "\n".join([
            f"{m['role'].upper()}: {m['content']}"
            for m in st.session_state.get("messages", [])
        ])
        st.download_button(
            label="📥 Baixar Chat",
            data=chat_text,
            file_name="historico_chat.txt",
            mime="text/plain"
        )


# --- PROCESS NEW UPLOADS ---
if uploaded_files:
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]

    if new_files:
        with st.status(f"⚙️ Indexando {len(new_files)} documento(s)...", expanded=True) as status:
            all_ok = True
            newly_indexed = []

            for f in new_files:
                st.write(f"Processando: {f.name}...")
                chunks = process_pdf(f.getvalue(), f.name, persist_dir)
                if chunks is not None:
                    st.session_state.processed_files.add(f.name)
                    st.session_state.all_chunks.extend(chunks)
                    newly_indexed.append((f.name, chunks))
                    st.write(f"✓ {f.name} — {len(chunks)} chunks indexados")
                else:
                    all_ok = False

            if newly_indexed:
                st.write("Gerando resumos...")
                for file_name, chunks in newly_indexed:
                    if file_name not in st.session_state.summarized_files:
                        try:
                            summary = generate_summary(file_name, chunks, model_option, temperature)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"**Resumo de _{file_name}_**\n\n{summary}"
                            })
                            st.session_state.summarized_files.add(file_name)
                        except Exception as e:
                            st.warning(f"Não foi possível gerar resumo para {file_name}: {e}")

            label = "✅ Tudo pronto! Pode perguntar." if all_ok else "⚠️ Alguns arquivos falharam."
            status.update(label=label, state="complete" if all_ok else "error", expanded=False)

has_documents = bool(st.session_state.processed_files)


# --- CHAT UI ---
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Ex: Qual o resumo executivo deste documento?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    if has_documents:
        with st.chat_message("assistant", avatar="🤖"):
            status_placeholder = st.empty()
            response_placeholder = st.empty()
            full_response = ""

            try:
                # Build LangChain-format message history (last 6 = 3 exchanges)
                lc_history = []
                for m in st.session_state.messages[:-1][-6:]:
                    if m["role"] == "user":
                        lc_history.append(HumanMessage(content=m["content"]))
                    else:
                        lc_history.append(AIMessage(content=m["content"]))

                # --- Run the LangGraph retrieval agent ---
                rag_graph = build_rag_graph(
                    model_option, temperature, persist_dir, st.session_state.all_chunks
                )

                initial_state: RAGState = {
                    "question": prompt,
                    "query": prompt,
                    "chat_history": lc_history,
                    "documents": [],
                    "retries": 0,
                }

                # Stream graph updates so user sees what's happening
                final_state = initial_state
                for update in rag_graph.stream(initial_state, stream_mode="updates"):
                    node_name = next(iter(update))
                    node_state = update[node_name]

                    if node_name == "contextualize" and node_state.get("query") != prompt:
                        status_placeholder.caption(f"🔎 Query reformulada: _{node_state['query']}_")
                    elif node_name == "rewrite":
                        status_placeholder.warning(
                            f"⚠️ Contexto insuficiente. Reformulando query para: _{node_state.get('query', '')}_"
                        )
                    elif node_name == "rerank":
                        status_placeholder.caption(
                            f"✅ {len(node_state.get('documents', []))} chunks selecionados pelo reranker"
                        )

                    final_state = {**final_state, **node_state}

                status_placeholder.empty()

                docs = final_state["documents"]
                context_text = "\n\n".join([doc.page_content for doc in docs])

                # --- Stream the final answer with LCEL ---
                answer_prompt = ChatPromptTemplate.from_messages([
                    ("system",
                     "Você é um assistente profissional analisando documentos. "
                     "Responda de forma direta e em português do Brasil. "
                     "Use o contexto abaixo. Se a informação não estiver no contexto, diga que não encontrou.\n\n"
                     "CONTEXTO:\n{context}"),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])

                chain = answer_prompt | get_llm(model_option, temperature) | StrOutputParser()

                for chunk in chain.stream({
                    "context": context_text,
                    "input": final_state["query"],
                    "chat_history": lc_history
                }):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

                with st.expander("🔍 Fontes consultadas"):
                    for i, doc in enumerate(docs):
                        page_num = int(doc.metadata.get("page", 0)) + 1
                        source_file = doc.metadata.get("source_file", "documento")
                        st.markdown(f"**Fonte {i+1} — {source_file} (Pág. {page_num}):**")
                        st.caption(f"...{doc.page_content[:200]}...")

            except Exception as e:
                st.error(f"Erro na comunicação com a IA: {e}")
    else:
        st.warning("⚠️ Por favor, faça o upload de um PDF no menu lateral para começar.")
