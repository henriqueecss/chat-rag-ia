import streamlit as st
import os
import tempfile
import uuid
import shutil
import time
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="DocuChat AI", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILIZAÇÃO CSS CUSTOMIZADA ---
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
    /* Melhorando a aparência dos cards de mensagem */
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SETUP INICIAL ---
os.environ["ANONYMIZED_TELEMETRY"] = "False"
load_dotenv()

# --- CABEÇALHO VISUAL ---
st.markdown('<div class="main-header">🧠 DocuChat AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Converse com seus documentos PDF usando Llama 3 & RAG</div>', unsafe_allow_html=True)

# --- FUNÇÃO DE FAXINA (GARBAGE COLLECTOR) ---
def cleanup_old_sessions():
    base_dir = "."
    current_time = time.time()
    max_age = 3600 # 1 hora
    
    for filename in os.listdir(base_dir):
        if filename.startswith("chroma_db_"):
            dir_path = os.path.join(base_dir, filename)
            if os.path.isdir(dir_path):
                try:
                    creation_time = os.path.getctime(dir_path)
                    if (current_time - creation_time) > max_age:
                        shutil.rmtree(dir_path)
                except Exception:
                    pass

cleanup_old_sessions()

# --- CACHE DO MODELO DE EMBEDDING ---
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- PROCESSAMENTO DO PDF ---
def process_pdf(file_bytes):
    session_id = uuid.uuid4().hex
    persist_dir = f"./chroma_db_{session_id}"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, 
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(docs)
        
        embedding_model = get_embedding_model()
        
        vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=embedding_model,
            persist_directory=persist_dir 
        )
        return vector_store, persist_dir
        
    except Exception as e:
        st.error(f"Erro crítico ao processar PDF: {e}")
        return None, None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- CARREGAMENTO DE SESSÃO ---
def load_existing_vector_store(persist_dir):
    if os.path.exists(persist_dir):
        embedding_model = get_embedding_model()
        return Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    return None

# --- SIDEBAR ---
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("❌ A chave da API Groq não foi encontrada. Verifique o arquivo .env")
    st.stop()

with st.sidebar:
    st.title("⚙️ Configurações")
    
    # 1. SELETOR DE MODELO 
    model_option = st.selectbox(
        "Escolha o Modelo:",
        ("llama-3.3-70b-versatile", "llama-3.1-8b-instant"),
        format_func=lambda x: "Llama 3.3 70B (Mais Inteligente)" if "70b" in x else "Llama 3.1 8B (Mais Rápido)"
    )
    
    st.markdown("---")
    st.header("📂 Documento")
    uploaded_file = st.file_uploader("Arraste seu PDF aqui", type="pdf")
    
    st.markdown("---")
    
    # 2. BOTÕES DE AÇÃO
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Limpar Chat"):
            st.session_state.messages = []
            st.rerun()
            
    with col2:
        # Função para transformar chat em texto
        chat_text = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.get('messages', [])])
        st.download_button(
            label="📥 Baixar Chat",
            data=chat_text,
            file_name="historico_chat.txt",
            mime="text/plain"
        )

# --- LÓGICA PRINCIPAL ---
vector_store = None
if uploaded_file:
    if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        with st.status("⚙️ Indexando documento...", expanded=True) as status:
            st.write("Lendo arquivo PDF...")
            vector_store, new_dir = process_pdf(uploaded_file.getvalue())
            
            if vector_store:
                st.session_state.vector_store_dir = new_dir
                st.session_state.last_uploaded = uploaded_file.name
                status.update(label="✅ Tudo pronto! Pode perguntar.", state="complete", expanded=False)
            else:
                status.update(label="❌ Falha no processamento", state="error")
    
    elif "vector_store_dir" in st.session_state:
        vector_store = load_existing_vector_store(st.session_state.vector_store_dir)

# --- CHAT UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Ex: Qual o resumo executivo deste documento?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    if vector_store:
        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                # Usa o modelo selecionado no Sidebar
                llm = ChatGroq(temperature=0.2, model_name=model_option)
                retriever = vector_store.as_retriever(search_kwargs={"k": 4})

                docs = retriever.invoke(prompt)
                context_text = "\n\n".join([doc.page_content for doc in docs])
                
                # Prompt Engineering 
                prompt_template = ChatPromptTemplate.from_template(
                    """Você é um assistente profissional analisando um documento.
                    Responda à pergunta de forma direta e em português do Brasil.
                    Use o contexto fornecido abaixo. Se a informação não estiver no texto, diga que não encontrou.
                    
                    CONTEXTO:
                    {context}
                    
                    PERGUNTA: {question}
                    """
                )
                
                chain = prompt_template | llm | StrOutputParser()
                
                for chunk in chain.stream({"context": context_text, "question": prompt}):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

                # Feature "Auditabilidade"
                with st.expander("🔍 Fontes consultadas (Páginas)"):
                    for i, doc in enumerate(docs):
                        page_num = int(doc.metadata.get('page', 0)) + 1
                        st.markdown(f"**Fonte {i+1} (Pág. {page_num}):**")
                        st.caption(f"...{doc.page_content[:200]}...")

            except Exception as e:
                st.error(f"Erro na comunicação com a IA: {e}")
    else:
        if not uploaded_file:
            st.warning("⚠️ Por favor, faça o upload de um PDF no menu lateral para começar.")