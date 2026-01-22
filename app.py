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

# --- CONFIGURAÇÃO INICIAL ---
os.environ["ANONYMIZED_TELEMETRY"] = "False"
load_dotenv()

st.set_page_config(page_title="Chat PDF", page_icon="🤖", layout="wide")
st.title("🤖 Chat com seus Arquivos (Arquitetura de Sessão Isolada)")

# --- FUNÇÃO DE FAXINA (GARBAGE COLLECTOR DE PASTAS) ---
# Esta função roda no início para limpar pastas velhas criadas em execuções anteriores
def cleanup_old_sessions():
    base_dir = "."
    current_time = time.time()
    # Define "velho" como pastas criadas há mais de 1 hora (3600 segundos)
    # Isso evita apagar a pasta que você está usando agora
    max_age = 3600 

    for filename in os.listdir(base_dir):
        if filename.startswith("chroma_db_"):
            dir_path = os.path.join(base_dir, filename)
            # Verifica a idade da pasta
            if os.path.isdir(dir_path):
                creation_time = os.path.getctime(dir_path)
                if (current_time - creation_time) > max_age:
                    try:
                        shutil.rmtree(dir_path)
                        print(f"Faxina: Pasta antiga {filename} removida.")
                    except Exception:
                        pass # Se o Windows bloquear, ignora silenciosamente

# Executa a faxina ao iniciar o script
cleanup_old_sessions()

# --- CACHE DO MODELO ---
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- PROCESSAMENTO COM UUID (SESSÃO ÚNICA) ---
def process_pdf(file_bytes):
    # Cria ID único para garantir que NUNCA misture dados
    session_id = uuid.uuid4().hex
    persist_dir = f"./chroma_db_{session_id}"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        
        embedding_model = get_embedding_model()
        
        vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=embedding_model,
            persist_directory=persist_dir 
        )
        return vector_store, persist_dir
        
    except Exception as e:
        st.error(f"Erro ao processar: {e}")
        return None, None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- CARREGAR BANCO ---
def load_existing_vector_store(persist_dir):
    if os.path.exists(persist_dir):
        embedding_model = get_embedding_model()
        return Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    return None

# --- SIDEBAR ---
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("❌ Chave GROQ_API_KEY faltando.")
    st.stop()

with st.sidebar:
    st.header("📂 Upload")
    uploaded_file = st.file_uploader("Envie seu PDF", type="pdf")
    
    st.markdown("---")
    # Este botão agora apenas reinicia a interface, a faxina pesada ocorre no boot
    if st.button("🔄 Novo Chat"):
        st.session_state.messages = []
        st.session_state.pop("vector_store_dir", None)
        st.session_state.pop("last_uploaded", None)
        st.rerun()

# --- ESTADO ---
vector_store = None
if uploaded_file:
    if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        with st.spinner("Criando ambiente isolado..."):
            vector_store, new_dir = process_pdf(uploaded_file.getvalue())
            if vector_store:
                st.session_state.vector_store_dir = new_dir
                st.session_state.last_uploaded = uploaded_file.name
                st.success(f"✅ Documento carregado!")
    
    elif "vector_store_dir" in st.session_state:
        vector_store = load_existing_vector_store(st.session_state.vector_store_dir)

# --- CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Pergunte..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    if vector_store:
        try:
            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
                    retriever = vector_store.as_retriever()

                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)

                    prompt_template = ChatPromptTemplate.from_template(
                        """Responda EXCLUSIVAMENTE com base no contexto abaixo:
                        {context}
                        Pergunta: {question}
                        """
                    )

                    rag_chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt_template
                        | llm
                        | StrOutputParser()
                    )
                    
                    response = rag_chain.invoke(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
        except Exception as e:
            st.error(f"Erro: {e}")
    else:
        st.warning("⚠️ O banco de dados ainda não foi gerado.")