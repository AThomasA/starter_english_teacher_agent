# agent/tools/obsidian_writer.py
# ============================================================
# LEITOR E ESCRITOR DO VAULT OBSIDIAN
# ============================================================
# Responsável por:
# 1. Ler arquivos de conhecimento do vault (personalidade Bianca, conteúdos)
# 2. Salvar relatório de custo de tokens por sessão
# 3. Salvar relatório de progresso do aluno por sessão

import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

VAULT_PATH = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./obsidian-vault"))

# Subpastas dentro do vault
REPORTS_COST_PATH    = VAULT_PATH / "05-session-reports" / "costs"
REPORTS_PROGRESS_PATH = VAULT_PATH / "05-session-reports" / "progress"
BIANCA_CORE_PATH     = VAULT_PATH / "00-bianca-core"


def _ensure_dir(path: Path):
    """Garante que o diretório existe."""
    path.mkdir(parents=True, exist_ok=True)


# ============================================================
# LEITURA DO VAULT
# ============================================================

def read_file(relative_path: str) -> str:
    """
    Lê um arquivo .md do vault pelo caminho relativo.
    Ex: read_file("00-bianca-core/personality.md")
    """
    full_path = VAULT_PATH / relative_path
    if not full_path.exists():
        return ""
    return full_path.read_text(encoding="utf-8")


def read_all_files_in_folder(relative_folder: str) -> str:
    """
    Lê todos os arquivos .md de uma pasta do vault e concatena o conteúdo.
    Útil para carregar todo o conhecimento da Bianca de uma vez.
    """
    folder = VAULT_PATH / relative_folder
    if not folder.exists():
        return ""

    contents = []
    for md_file in sorted(folder.rglob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        contents.append(f"## [{md_file.stem}]\n{text}")

    return "\n\n---\n\n".join(contents)


def load_student_context() -> str:
    """Carrega perfil e avaliação de nível do aluno."""
    parts = []
    for f in [
        "05-session-memory/allan-profile.md",
        "05-session-memory/level-assessment.md",
        "05-session-memory/student-profile.md",
    ]:
        content = read_file(f)
        if content:
            parts.append(f"## [{Path(f).stem}]\n{content}")
    return "\n\n---\n\n".join(parts)


def load_bianca_context() -> str:
    """
    Carrega todo o contexto da Bianca (personalidade + currículo + métricas).
    Este é o contexto que vai no system prompt do agente.
    """
    context_parts = []

    # Lê as pastas de conhecimento principais
    for folder in ["00-bianca-core", "01-curriculum", "03-teaching-metrics"]:
        content = read_all_files_in_folder(folder)
        if content:
            context_parts.append(content)

    student_ctx = load_student_context()
    if student_ctx:
        context_parts.append(student_ctx)

    return "\n\n===\n\n".join(context_parts)


# ============================================================
# RELATÓRIO DE CUSTO DE TOKENS
# ============================================================

def save_cost_report(session_summary: dict, session_id: str) -> str:
    """
    Salva o relatório de custo de tokens da sessão como arquivo .md no vault.
    Retorna o caminho do arquivo criado.
    """
    _ensure_dir(REPORTS_COST_PATH)

    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M")
    filename  = f"{date_str}_{session_id}_cost.md"
    filepath  = REPORTS_COST_PATH / filename

    content = f"""---
tags: [cost-report, session]
date: {date_str}
session_id: {session_id}
model: {session_summary.get('model', 'N/A')}
provider: {session_summary.get('provider', 'N/A')}
---

# 💰 Relatório de Custo — {date_str} às {time_str}

## Modelo Utilizado
- **Provider:** {session_summary.get('provider', 'N/A')}
- **Modelo:** `{session_summary.get('model', 'N/A')}`

## Consumo de Tokens
| Métrica | Valor |
|---|---|
| Tokens de Input | {session_summary.get('total_input_tokens', 0):,} |
| Tokens de Output | {session_summary.get('total_output_tokens', 0):,} |
| **Total de Tokens** | **{session_summary.get('total_tokens', 0):,}** |
| Janela de Contexto | {session_summary.get('context_window', 0):,} |
| % da Janela Usada | {session_summary.get('context_window_used_pct', 0):.2f}% |

## Custo
| Métrica | Valor |
|---|---|
| **Custo Total** | **${session_summary.get('total_cost_usd', 0):.6f} USD** |

## Performance
| Métrica | Valor |
|---|---|
| Trocas na conversa | {session_summary.get('total_exchanges', 0)} |
| Duração da sessão | {session_summary.get('session_duration_minutes', 0):.1f} min |
| Tempo médio de resposta | {session_summary.get('avg_response_time_seconds', 0):.2f}s |

---
*Gerado automaticamente pelo English Teacher Agent*
"""

    filepath.write_text(content, encoding="utf-8")
    return str(filepath)


# ============================================================
# RELATÓRIO DE PROGRESSO DO ALUNO
# ============================================================

def save_progress_report(
    session_id: str,
    student_name: str,
    conversation_history: list,
    session_summary: dict,
) -> str:
    """
    Gera e salva um relatório de progresso do aluno baseado na conversa.
    Retorna o caminho do arquivo criado.
    """
    _ensure_dir(REPORTS_PROGRESS_PATH)

    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M")
    filename  = f"{date_str}_{session_id}_progress.md"
    filepath  = REPORTS_PROGRESS_PATH / filename

    # Monta o histórico da conversa formatado
    conversation_md = ""
    for msg in conversation_history:
        role = "🧑 Aluno" if msg["role"] == "user" else "👩‍🏫 Bianca"
        conversation_md += f"**{role}:** {msg['content']}\n\n"

    content = f"""---
tags: [progress-report, session, student]
date: {date_str}
session_id: {session_id}
student: {student_name}
---

# 📚 Relatório de Aula — {student_name}
**Data:** {date_str} às {time_str}
**Sessão:** `{session_id}`

---

## 📝 Histórico da Conversa

{conversation_md}

---

## 📊 Análise da Sessão

> ⚠️ **TODO:** Esta seção será gerada automaticamente por um agente de análise
> na próxima fase do projeto. Por enquanto, preencha manualmente após a aula.

### ✅ Pontos Fortes (onde foi bem)
- 

### ⚠️ Dificuldades Identificadas (onde precisa melhorar)
- 

### 🎯 Tópicos Trabalhados
- 

### 💡 Observações da Bianca
- 

### 📈 Progresso Geral
- [ ] Iniciante
- [ ] Desenvolvendo
- [ ] Intermediário
- [ ] Avançado

---

## 🔢 Dados da Sessão
- **Trocas na conversa:** {session_summary.get('total_exchanges', 0)}
- **Duração:** {session_summary.get('session_duration_minutes', 0):.1f} min
- **Modelo usado:** `{session_summary.get('model', 'N/A')}`

---
*Gerado automaticamente pelo English Teacher Agent*
*Relatório de custo relacionado: [[costs/{date_str}_{session_id}_cost]]*
"""

    filepath.write_text(content, encoding="utf-8")
    return str(filepath)