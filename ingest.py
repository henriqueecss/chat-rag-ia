import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Carregar o PDF
caminho_arquivo = "documento.pdf"

if not os.path.exists(caminho_arquivo):
    print("ERRO: O arquivo 'documento.pdf' não foi encontrado na pasta.")
    exit()

print(f"📄 Carregando o arquivo {caminho_arquivo}...")
loader = PyPDFLoader(caminho_arquivo)
documentos = loader.load()
print(f"   Sucesso! {len(documentos)} páginas carregadas.")

# 2. Quebrar em Chunks (Pedaços)
# Chunk size 1000: Cada pedaço terá +- 1000 caracteres.
# Overlap 200: O final de um pedaço repete no começo do próximo (para não cortar frases no meio).
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documentos)
print(f"🧩 Texto dividido em {len(chunks)} pedaços (chunks).")

# 3. Criar Embeddings e Salvar no Banco (ChromaDB)
print("💾 Gerando embeddings e salvando no banco de dados (isso pode demorar um pouco)...")

# Usamos um modelo leve da HuggingFace que roda no seu PC
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Cria o banco local na pasta "chroma_db"
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

print("✅ Tudo pronto! O banco de dados vetorial foi criado na pasta 'chroma_db'.")