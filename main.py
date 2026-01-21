import os
# Importa a função para carregar o arquivo .env
from dotenv import load_dotenv
# Importa a integração da Groq com o LangChain
from langchain_groq import ChatGroq

# 1. Carrega as variáveis de ambiente (sua senha segura)
load_dotenv()

# Verifica se a chave foi carregada (boa prática de debug)
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("ERRO: A chave da API não foi encontrada. Verifique o arquivo .env")
    exit()

# 2. Inicializa o Modelo (LLM)
# 'temperature=0' deixa a IA mais precisa e menos "criativa" (bom para sistemas técnicos)
# 'model_name' define qual cérebro vamos usar. O Llama 3 é excelente.
chat = ChatGroq(
    temperature=0, 
    model_name="llama-3.3-70b-versatile",
    api_key=api_key
)

# 3. A Pergunta (Prompt)
pergunta = "Explique para um iniciante em computação o que é um LLM em uma frase curta."

print("🤖 Pensando...")

# 4. A Chamada (Invocação)
# O método .invoke() envia a pergunta para a nuvem da Groq e espera a resposta
resposta = chat.invoke(pergunta)

# 5. O Resultado
# A resposta vem cheia de metadados. Queremos apenas o conteúdo (.content)
print(f"\nRESPOSTA: {resposta.content}")