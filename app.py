import streamlit as st
import os
import tempfile
import uuid
import shutil
import time
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
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

# #2: Cache LLM per model name — avoids recreating on every message
@st.cache_resource
def get_llm(model_name: str):
    return ChatGroq(temperature=0.2, model_name=model_name)


# --- PDF PROCESSING ---
def process_pdf(file_bytes, file_name, persist_dir):
    """Load, split, embed and store a PDF. Returns the list of chunks."""
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


# #3: Hybrid retriever — BM25 (keyword) + vector (semantic) via EnsembleRetriever
def build_retriever(persist_dir, all_chunks):
    embedding_model = get_embedding_model()
    vector_store = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)

    vector_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.7}
    )

    if all_chunks:
        bm25_retriever = BM25Retriever.from_documents(all_chunks)
        bm25_retriever.k = 4
        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6]
        )

    return vector_retriever


# --- SESSION INIT ---
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []

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

    st.markdown("---")
    st.header("📂 Documentos")

    # #4: Multiple PDF uploads
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
            for f in new_files:
                st.write(f"Processando: {f.name}...")
                chunks = process_pdf(f.getvalue(), f.name, persist_dir)
                if chunks is not None:
                    st.session_state.processed_files.add(f.name)
                    st.session_state.all_chunks.extend(chunks)
                    st.write(f"✓ {f.name} — {len(chunks)} chunks indexados")
                else:
                    all_ok = False

            label = "✅ Tudo pronto! Pode perguntar." if all_ok else "⚠️ Alguns arquivos falharam."
            state = "complete" if all_ok else "error"
            status.update(label=label, state=state, expanded=False)

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
            response_placeholder = st.empty()
            full_response = ""

            try:
                llm = get_llm(model_option)

                # #3: Hybrid retriever (BM25 + vector)
                retriever = build_retriever(persist_dir, st.session_state.all_chunks)

                # Build LangChain-format message history (last 6 messages = 3 exchanges)
                lc_history = []
                for m in st.session_state.messages[:-1][-6:]:
                    if m["role"] == "user":
                        lc_history.append(HumanMessage(content=m["content"]))
                    else:
                        lc_history.append(AIMessage(content=m["content"]))

                # #1: History-aware retriever — reformulates follow-up questions
                # before hitting the retriever, so "explain the second point" works
                contextualize_prompt = ChatPromptTemplate.from_messages([
                    ("system",
                     "Dado o histórico da conversa e a última pergunta do usuário, "
                     "reformule a pergunta como uma questão independente que possa ser entendida "
                     "sem o histórico. NÃO responda a pergunta, apenas reformule-a se necessário."),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])
                history_aware_retriever = create_history_aware_retriever(
                    llm, retriever, contextualize_prompt
                )

                docs = history_aware_retriever.invoke({
                    "input": prompt,
                    "chat_history": lc_history
                })
                context_text = "\n\n".join([doc.page_content for doc in docs])

                answer_prompt = ChatPromptTemplate.from_messages([
                    ("system",
                     "Você é um assistente profissional analisando documentos. "
                     "Responda de forma direta e em português do Brasil. "
                     "Use o contexto abaixo. Se a informação não estiver no contexto, diga que não encontrou.\n\n"
                     "CONTEXTO:\n{context}"),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])

                chain = answer_prompt | llm | StrOutputParser()

                for chunk in chain.stream({
                    "context": context_text,
                    "input": prompt,
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
