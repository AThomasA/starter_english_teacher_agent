# 🎓 SOS English — Starter Teacher Agent

Agente professor de inglês com RAG, treinado com o livro Starter da SOS English.  
Projeto de portfólio demonstrando uso de RAG + LLM local com Ollama.

## 🛠️ Stack

- **LLM:** Llama 3.2 via Ollama (local, sem API key)
- **RAG:** LangChain + ChromaDB
- **Embeddings:** sentence-transformers
- **Interface:** Streamlit

## 🚀 Como rodar

### 1. Instale o Ollama
Acesse [ollama.com](https://ollama.com) e instale.  
Depois rode no terminal:
```bash
ollama pull llama3.2
```

### 2. Clone o repositório
```bash
git clone https://github.com/seu-usuario/starter-english-teacher-agent.git
cd starter-english-teacher-agent/agent-test
```

### 3. Instale as dependências
```bash
uv venv
uv sync
```

### 4. Adicione os PDFs
Coloque o Student Book e o Workbook em PDF dentro da pasta `data/`

### 5. Processe os PDFs
```bash
python ingest.py
```

### 6. Rode o agente
```bash
streamlit run app.py
```

## 📁 Estrutura do projeto
```
agent-test/
├── data/          ← PDFs dos livros (não incluídos, direitos autorais)
├── chroma_db/     ← banco vetorial gerado localmente
├── ingest.py      ← processa PDFs e cria o RAG
├── agent.py       ← lógica do agente
├── app.py         ← interface Streamlit
├── pyproject.toml
├── .env
└── README.md
```

## 🏫 Sobre a SOS English

Projeto desenvolvido como portfólio técnico e demonstração do uso de IA  
aplicada ao ensino de inglês online da [SOS English](https://instagram.com/sabino.onlineschool).

Caso queira acesso ao PDF utilizado para testar o modelo, entre em contato pelo Instagram.  
E se você quiser aulas de inglês online para aprender a se comunicar de verdade, conversa com a gente por lá que vamos definir o melhor horário para você.