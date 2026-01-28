# 🧠 Chat RAG AI: RAG Avançado com Persistência Gerenciada

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Llama 3](https://img.shields.io/badge/Model-Llama%203.3-orange?style=for-the-badge)

> **Uma aplicação de Chat com PDF que resolve problemas reais de concorrência e gerenciamento de memória em sistemas RAG locais.**

---

## 📸 Demonstração
*(Cole aqui o print ou GIF da sua aplicação rodando)*

## 🚀 Sobre o Projeto
Este projeto não é apenas um wrapper da API da OpenAI. É uma implementação completa de uma arquitetura **RAG (Retrieval-Augmented Generation)** projetada para rodar localmente com robustez.

O diferencial deste projeto é o foco na **Engenharia de Software** por trás do chat:
1.  **Isolamento de Sessão:** Cada documento carregado cria um banco vetorial único (hash UUID), impedindo que dados de usuários diferentes se misturem (*Context Poisoning*).
2.  **Gerenciamento de Recursos:** Implementação de um *Garbage Collector* customizado para lidar com arquivos temporários que o Windows bloqueia (*File Locking*), limpando o disco automaticamente no boot.
3.  **UX Profissional:** Interface com feedback de carregamento, streaming de respostas, alternância de modelos (Llama 70B vs 8B) e citação de fontes.

## 🛠️ Arquitetura e Tech Stack

| Componente | Tecnologia | Função |
| :--- | :--- | :--- |
| **Frontend** | Streamlit | Interface reativa com gestão de estado (Session State) e CSS customizado. |
| **Orquestração** | LangChain (LCEL) | Pipelines de processamento de texto e conexão com LLM. |
| **Vector Store** | ChromaDB | Banco de dados vetorial persistente em disco. |
| **Embeddings** | HuggingFace | Modelo `all-MiniLM-L6-v2` (rápido e eficiente para CPU). |
| **Inference** | Groq API | Acesso ultra-rápido ao modelo **Llama 3.3 70B**. |

## 📂 Estrutura do Projeto
```text
📂 chat-rag-ai
├── 📄 app.py              # Código principal (Frontend + Lógica RAG + Garbage Collector)
├── 📄 requirements.txt    # Dependências do projeto
├── 📄 .env                # Variáveis de ambiente (API Keys)
├── 📄 .gitignore          # Configuração do Git
└── 📂 .venv               # Ambiente virtual Python

## 🧠 Desafios de Engenharia & Soluções

### O Problema: File Locking no Windows
Durante o desenvolvimento, ao tentar deletar o banco de dados `ChromaDB` imediatamente após o uso, o Windows retornava erros de `PermissionError`, pois o processo do Python ainda mantinha o arquivo aberto (File Locking). Isso causava acúmulo de lixo no disco e falhas na aplicação.

### A Solução: Lazy Garbage Collection
Implementei um algoritmo de limpeza "preguiçosa" (`cleanup_old_sessions`). Ao iniciar a aplicação, o sistema verifica pastas `chroma_db_` criadas há mais de 1 hora e as remove com segurança. Isso garante que:
1.  O usuário nunca veja um erro de "Acesso Negado".
2.  O disco não fique cheio de lixo (bancos vetoriais antigos).
3.  A performance do app se mantém estável sem travamentos.

## ⚙️ Como Rodar Localmente

**Pré-requisitos:** Python 3.10+ e uma chave de API da [Groq](https://console.groq.com/).

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/SEU_USUARIO/chat-rag-ai.git](https://github.com/SEU_USUARIO/chat-rag-ai.git)
   cd chat-rag-ai

2. **Configure o Ambiente:**
   ```bash
   python -m venv .venv
   # Linux/Mac:
   source .venv/bin/activate
   # Windows:
   .venv\Scripts\activate
   
   pip install -r requirements.txt

3. **Configure as Chaves: Renomeie o .env.example para .env ou crie um novo na raiz do projeto:**
    GROQ_API_KEY=gsk_sua_chave_aqui...

4. **Execute:**
    streamlit run app.py

**🔮 Melhorias Futuras (Roadmap)**

[ ] Implementar suporte para múltiplos arquivos PDF simultâneos.

[ ] Adicionar persistência de histórico de chat em banco SQL.

[ ] Containerização com Docker para deploy simplificado.