# English Teacher Agent — Documentação do Projeto

> Este arquivo explica como o projeto funciona, o que cada parte faz,
> e como tudo se conecta. Escrito para você estudar seu próprio projeto.

---

## O que é esse projeto

Um agente de professora de inglês (a Bianca) que roda localmente via Streamlit.
Ele usa os livros da SOS English como base de conhecimento (RAG),
mantém memória das aulas no Obsidian, e registra custo de tokens a cada sessão.

---

## Fluxo Completo

```
Você digita uma mensagem no Streamlit
        ↓
agent.py recebe a mensagem
        ↓
_get_rag_context() busca conteúdo relevante do livro
(md_retriever → ChromaDB → retorna trecho do nível ativo)
        ↓
_compress_history() monta o histórico comprimido
(mantém só as últimas 4 trocas completas + resumo das anteriores)
        ↓
_build_system_prompt() já foi montado ao iniciar
(BIANCA_PERSONALITY + arquivos de 00-bianca-core/ do vault)
        ↓
_call_llm() envia tudo para o modelo (Groq / OpenAI / Anthropic)
        ↓
Resposta volta, token_tracker registra métricas
        ↓
Streamlit exibe a resposta + custo em tempo real na sidebar
        ↓
Ao clicar "Salvar no Obsidian":
  → save_cost_report() → 06-session-reports/costs/
  → save_progress_report() → 06-session-reports/progress/
```

---

## Estrutura de Pastas do Projeto

```
english-teacher-agent/
│
├── agent/
│   ├── core/
│   │   └── agent.py              ← Cérebro do agente. Chat, histórico, RAG
│   ├── models/
│   │   └── model_config.py       ← Troca de modelo via .env
│   └── tools/
│       ├── token_tracker.py      ← Conta tokens e calcula custo por mensagem
│       └── obsidian_writer.py    ← Lê vault, salva relatórios .md
│
├── rag/
│   ├── md_indexer.py             ← Lê .md do vault e indexa no ChromaDB
│   ├── md_retriever.py           ← Busca contexto relevante no ChromaDB
│   └── chroma_db/                ← Banco vetorial (gerado automaticamente)
│
├── ui/
│   └── app.py                    ← Interface Streamlit completa
│
├── obsidian-vault/               ← Abrir como vault no Obsidian
│   ├── 00-bianca-core/           ← Personalidade e método da Bianca
│   ├── 01-curriculum/            ← Estrutura dos níveis da escola
│   ├── 02-books/                 ← MDs dos livros (fonte do RAG)
│   │   ├── starter/
│   │   ├── level-1/
│   │   └── ...
│   ├── 03-teaching-metrics/      ← Rubricas e métricas de avaliação
│   ├── 04-session-memory/        ← Memória persistente do aluno
│   │   ├── allan-profile.md      ← Perfil personalizado do Allan
│   │   ├── level-assessment.md   ← Diagnóstico e plano de estudo
│   │   ├── student-profile.md    ← Progresso acumulado
│   │   ├── error-patterns.md     ← Padrões de erro identificados
│   │   └── vocabulary-learned.md ← Vocabulário aprendido por sessão
│   └── 06-session-reports/       ← Gerado automaticamente por aula
│       ├── costs/                ← Custo de tokens (YYYY-MM-DD_ID_cost.md)
│       └── progress/             ← Relatório da aula (YYYY-MM-DD_ID_progress.md)
│
├── .env                          ← API keys e configurações
├── pyproject.toml                ← Dependências (gerenciado pelo uv)
└── README.md
```

---

## O que cada arquivo de código faz

### `agent/core/agent.py`
O arquivo mais importante. Contém a classe `BiancaAgent`.

- `__init__`: cria sessão, carrega config do modelo, monta system prompt
- `_build_system_prompt()`: junta BIANCA_PERSONALITY + arquivos de 00-bianca-core/
- `_get_rag_context(query)`: chama md_retriever para buscar trecho do livro
- `_compress_history()`: mantém histórico enxuto — resume mensagens antigas
- `chat(user_message)`: orquestra tudo — RAG + histórico + LLM + métricas
- `_call_llm()`: chamada real para Groq/OpenAI/Anthropic com histórico comprimido

### `agent/models/model_config.py`
Centraliza a configuração dos modelos.
Para trocar de modelo: mude `ACTIVE_MODEL` no `.env`.
- `groq` → Llama 3.3 70B (gratuito)
- `openai` → GPT-4o mini (pago)
- `anthropic` → Claude Haiku (pago)

### `agent/tools/token_tracker.py`
Rastreia tokens e calcula custo por mensagem e por sessão.
Cada chamada ao LLM registra: tokens input, tokens output, tempo de resposta, custo em USD.
`get_summary()` retorna o resumo completo da sessão para a sidebar.

### `agent/tools/obsidian_writer.py`
Lê e escreve no vault Obsidian.
- `load_bianca_context()`: lê 00-bianca-core/ + 01-curriculum/ + 03-teaching-metrics/
- `save_cost_report()`: salva custo de tokens em 06-session-reports/costs/
- `save_progress_report()`: salva histórico da conversa em 06-session-reports/progress/

### `rag/md_indexer.py`
Lê os arquivos .md de obsidian-vault/02-books/ e indexa no ChromaDB.
Usa LlamaIndex HierarchicalNodeParser — cria 3 camadas:
- Nós pai (2048 tokens): contexto geral do documento
- Nós intermediários (512 tokens): seções
- Nós folha (128 tokens): chunks pequenos — esses são indexados e buscados

### `rag/md_retriever.py`
Busca os chunks mais relevantes para uma query no ChromaDB.
Usa AutoMergingRetriever: se vários chunks da mesma seção baterem,
retorna a seção pai em vez de chunks fragmentados.
Retorna contexto formatado com header de origem (livro, unidade, nível).

### `ui/app.py`
Interface Streamlit. Sidebar com dashboard de tokens, RAG controls, botões de salvar.
Chat principal com histórico de mensagens e métricas por resposta.

---

## Sistema de Custo de Tokens

A cada mensagem enviada ao LLM, o sistema registra:

```
Tokens input  = system prompt + histórico + mensagem + contexto RAG
Tokens output = resposta do modelo
Custo         = (input_tokens / 1_000_000) × preço_input
              + (output_tokens / 1_000_000) × preço_output
```

Preços por modelo (atualize em model_config.py se mudarem):
- Groq Llama 3.3 70B: $0.59/M input · $0.79/M output
- GPT-4o mini: $0.15/M input · $0.60/M output
- Claude Haiku: $0.80/M input · $4.00/M output

Os relatórios de custo ficam em `06-session-reports/costs/` no vault.
Cada arquivo tem: modelo usado, tokens input/output, custo total, % da janela usada.

---

## Sistema RAG — Como o Agente Acessa os Livros

```
1. Você indexa os .md de um nível (botão "Indexar MDs do Vault")
   → md_indexer.py lê os arquivos de 02-books/<nivel>/
   → LlamaIndex divide em nós hierárquicos
   → Nós folha são vetorizados e salvos no ChromaDB

2. Você ativa o RAG para aquele nível (botão "Ativar RAG")
   → agent.py passa a chamar md_retriever em cada mensagem

3. A cada mensagem:
   → md_retriever.retrieve(query) busca top-2 nós mais similares
   → AutoMergingRetriever decide: retorna folhas ou promove para pai
   → Resultado: ~100-200 tokens de contexto cirúrgico
   → Contexto é injetado na mensagem antes de ir para o LLM
```

---

## Compressão de Histórico

Sem compressão, o histórico cresce infinitamente e estoura a janela de contexto.

Com a compressão implementada:
- Até 6 trocas: envia histórico completo
- Acima de 6: resume as mensagens antigas em ~150 tokens + mantém as últimas 4 trocas completas

Configurável no `.env`:
```env
HISTORY_KEEP_LAST=4
HISTORY_COMPRESS_AFTER=6
```

---

## Memória Entre Sessões (Vault Obsidian)

O Obsidian funciona como memória de longo prazo do agente.
Arquivos lidos no início de cada sessão (via load_bianca_context()):
- 00-bianca-core/ → personalidade e método da Bianca
- 04-session-memory/allan-profile.md → perfil e preferências do aluno
- 04-session-memory/level-assessment.md → nível atual e plano de estudo

Arquivos gerados automaticamente ao salvar:
- 06-session-reports/costs/ → custo de cada sessão
- 06-session-reports/progress/ → histórico completo da conversa

---

## Dependências Instaladas (uv)

```bash
uv add streamlit
uv add groq openai anthropic
uv add python-dotenv tiktoken
uv add chromadb
uv add llama-index
uv add llama-index-embeddings-huggingface
uv add llama-index-vector-stores-chroma
uv add llama-index-readers-file
uv add pymupdf pillow pytesseract   # mantidos mas não mais usados no RAG
```

---

## Como Rodar

```bash
# 1. Ativa o ambiente virtual
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows

# 2. Copia e preenche o .env
cp .env.example .env
# edita o .env com sua chave Groq (ou OpenAI / Anthropic)

# 3. Roda o Streamlit
streamlit run ui/app.py
```

---

## Próximos Passos Planejados

- [ ] Integração com Telegram via n8n (mesmo agente, nova interface)
- [ ] Análise automática do progresso do aluno (agente separado)
- [ ] GraphRAG — grafo de conhecimento sobre os conteúdos dos livros
- [ ] Compressão de contexto com LLMLingua antes de enviar ao LLM
- [ ] Questionário real da Bianca para refinar a personalidade do agente

---

## Links do Vault

- [[personality]]
- [[teaching-philosophy]]
- [[allan-profile]]
- [[level-assessment]]
- [[student-profile]]
