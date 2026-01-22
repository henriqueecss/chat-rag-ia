import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Configuração Inicial
load_dotenv()

print("🤖 Iniciando o Chatbot... (Carregando banco de dados)")

# 2. Carregar o Banco de Dados (Memória)
# Precisamos da mesma função de embeddings usada na ingestão
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

# Define o 'retriever' (o buscador)
# k=3 significa: "traga os 3 trechos mais parecidos com a pergunta"
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 3. Configurar o Cérebro (LLM)
llm = ChatGroq(
    temperature=0, # 0 = mais preciso, 1 = mais criativo
    model_name="llama-3.3-70b-versatile" 
)

# 4. O Prompt (As regras do jogo)
system_prompt = (
    "Você é um assistente especializado em responder perguntas sobre o documento fornecido."
    "Use APENAS o contexto abaixo para responder."
    "Se a resposta não estiver no contexto, diga: 'Não encontrei essa informação no documento'."
    "Seja direto e útil."
    "\n\n"
    "--- CONTEXTO ---\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 5. Criar a "Corrente" (Chain)
# 'stuff_documents' pega os textos encontrados e coloca dentro do prompt
question_answer_chain = create_stuff_documents_chain(llm, prompt)
# 'retrieval_chain' gerencia a busca + a resposta
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 6. O Loop da Conversa (Terminal)
print("\n✅ Chat pronto! Pergunte algo sobre seu PDF (digite 'sair' para encerrar).")
print("----------------------------------------------------------------------")

while True:
    pergunta = input("\nVocê: ")
    
    if pergunta.lower() in ['sair', 'exit', 'quit']:
        print("Encerrando...")
        break
    
    if not pergunta:
        continue

    print("🤖 Pesquisando e respondendo...")
    
    try:
        # Invoca a chain
        resposta = rag_chain.invoke({"input": pergunta})
        print(f"IA: {resposta['answer']}")
    except Exception as e:
        print(f"❌ Ocorreu um erro: {e}")