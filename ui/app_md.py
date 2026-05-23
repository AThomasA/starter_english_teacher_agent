# ui/app_md.py
# ============================================================
# ENGLISH TEACHER AGENT — Interface Streamlit (versão MD/RAG)
# ============================================================
# Para rodar: streamlit run ui/app_md.py
# (app.py é alias — mesmo código)

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime

import streamlit as st

from agent.core.agent import BiancaAgent
from agent.models.model_config import get_active_model_config
from agent.tools.obsidian_writer import save_cost_report, save_progress_report

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="SOS English — Bianca",
    page_icon="🇺🇸",
    layout="wide",
)

# ============================================================
# ESTADO DA SESSÃO
# ============================================================

if "agent" not in st.session_state:
    st.session_state.agent = BiancaAgent()

if "messages_display" not in st.session_state:
    st.session_state.messages_display = []

if "student_name" not in st.session_state:
    st.session_state.student_name = "Aluno"

if "session_saved" not in st.session_state:
    st.session_state.session_saved = False

if "rag_active_level" not in st.session_state:
    st.session_state.rag_active_level = None

if "ui_notice" not in st.session_state:
    st.session_state.ui_notice = None

if "ui_notice_type" not in st.session_state:
    st.session_state.ui_notice_type = "info"

agent: BiancaAgent = st.session_state.agent


def _show_notice():
    """Exibe aviso persistido após rerun (indexação, nome, etc.)."""
    msg = st.session_state.ui_notice
    if not msg:
        return
    kind = st.session_state.ui_notice_type
    if kind == "success":
        st.success(msg)
    elif kind == "error":
        st.error(msg)
    elif kind == "warning":
        st.warning(msg)
    else:
        st.info(msg)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("⚙️ Configurações")

    _show_notice()

    st.caption("👤 Aluno")
    name_input = st.text_input(
        "Seu nome",
        value=st.session_state.student_name,
        key="student_name_input",
        placeholder="Ex.: Allan",
    )
    if st.button("💾 Salvar nome", use_container_width=True, key="save_name_btn"):
        nome = name_input.strip()
        if nome:
            st.session_state.student_name = nome
            st.session_state.ui_notice = f"Nome salvo: **{nome}**"
            st.session_state.ui_notice_type = "success"
            st.toast(f"Nome salvo: {nome}", icon="✅")
            st.rerun()
        else:
            st.session_state.ui_notice = "Digite um nome antes de salvar."
            st.session_state.ui_notice_type = "warning"
            st.rerun()

    st.caption(f"Atual: **{st.session_state.student_name}**")

    st.divider()

    try:
        model_cfg = get_active_model_config()
        st.caption("🤖 Modelo Ativo")
        st.info(f"**{model_cfg['provider']}**\n\n`{model_cfg['model_id']}`")
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")

    st.divider()

    st.caption("📊 Consumo da Sessão")
    summary = agent.get_session_summary()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Input", f"{summary['total_input_tokens']:,}")
        st.metric("Output", f"{summary['total_output_tokens']:,}")
    with col2:
        st.metric("Total", f"{summary['total_tokens']:,}")
        st.metric("Trocas", summary["total_exchanges"])

    ctx_pct = summary["context_window_used_pct"]
    st.caption(f"🪟 Janela — {ctx_pct:.1f}% usada")
    st.progress(min(ctx_pct / 100, 1.0))

    st.divider()

    st.caption("💰 Custo e Performance")
    st.metric("Custo Total", f"${summary['total_cost_usd']:.6f} USD")
    st.metric("Duração", f"{summary['session_duration_minutes']:.1f} min")
    st.metric("Resp. média", f"{summary['avg_response_time_seconds']:.2f}s")

    st.divider()

    st.caption("📚 Base de Conhecimento")

    selected_level = st.selectbox(
        "Nível",
        ["starter", "level-1", "level-2", "level-3", "level-4"],
        key="rag_level_select",
    )

    try:
        from rag.md_indexer import get_collection_info

        info = get_collection_info(selected_level)
        if info["exists"]:
            st.success(
                f"✅ {info['count']} nós indexados\n"
                f"({info.get('documents', '?')} arquivos .md)"
            )
        else:
            st.warning("⚠️ Não indexado ainda — clique em Indexar abaixo")
    except Exception as e:
        st.error(f"Erro ao ler índice: {e}")

    if st.button("📥 Indexar MDs do Vault", use_container_width=True, key="index_btn"):
        with st.status(f"Indexando **{selected_level}**… (pode levar alguns minutos)", expanded=True) as status:
            try:
                from rag.md_indexer import run_indexing
                from rag.md_retriever import invalidate_cache

                stats = run_indexing(level=selected_level)
                invalidate_cache(selected_level)
                msg = (
                    f"Indexação concluída: **{stats['leaf_nodes']}** nós folha "
                    f"de **{stats['documents']}** arquivos .md (`{selected_level}`)."
                )
                st.session_state.ui_notice = msg
                st.session_state.ui_notice_type = "success"
                status.update(label="Indexação concluída", state="complete")
                st.toast("Indexação concluída!", icon="✅")
            except FileNotFoundError as e:
                st.session_state.ui_notice = str(e)
                st.session_state.ui_notice_type = "error"
                status.update(label="Falha na indexação", state="error")
            except Exception as e:
                err = str(e)
                if "config_sentence_transformers" in err or "No such file" in err:
                    err += (
                        "\n\n**Cache do modelo corrompido.** No PowerShell, rode:\n"
                        "```\n"
                        "Remove-Item -Recurse -Force "
                        "$env:LOCALAPPDATA\\llama_index\\llama_index\\Cache\\"
                        "models--sentence-transformers--all-MiniLM-L6-v2 -ErrorAction SilentlyContinue\n"
                        "```\n"
                        "Depois clique em Indexar novamente (vai baixar o modelo)."
                    )
                st.session_state.ui_notice = f"Erro na indexação: {err}"
                st.session_state.ui_notice_type = "error"
                status.update(label="Falha na indexação", state="error")
        st.rerun()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🟢 Ativar RAG", use_container_width=True):
            st.session_state.rag_active_level = selected_level
            agent.set_rag_level(selected_level)
            st.session_state.ui_notice = f"RAG ativo no nível **{selected_level}**"
            st.session_state.ui_notice_type = "success"
            st.toast(f"RAG: {selected_level}", icon="🟢")
            st.rerun()
    with col_b:
        if st.button("⚪ Desativar", use_container_width=True):
            st.session_state.rag_active_level = None
            agent.set_rag_level(None)
            st.session_state.ui_notice = "RAG desativado."
            st.session_state.ui_notice_type = "info"
            st.rerun()

    if st.session_state.rag_active_level:
        st.info(f"🟢 RAG ativo: **{st.session_state.rag_active_level}**")
    else:
        st.caption("⚪ RAG desativado")

    st.divider()

    st.caption("🗜️ Histórico")
    total_pairs = len(agent.conversation_history) // 2
    compress_after = int(os.getenv("HISTORY_COMPRESS_AFTER", "6"))
    keep_last = int(os.getenv("HISTORY_KEEP_LAST", "4"))

    if total_pairs <= compress_after:
        st.caption(f"{total_pairs}/{compress_after} trocas (completo)")
    else:
        st.caption(f"✅ {total_pairs - keep_last} resumidas | {keep_last} completas")

    st.divider()

    if st.button("💾 Salvar no Obsidian", use_container_width=True):
        if summary["total_exchanges"] == 0:
            st.warning("Nenhuma conversa para salvar.")
        else:
            try:
                cost_path = save_cost_report(summary, agent.session_id)
                prog_path = save_progress_report(
                    session_id=agent.session_id,
                    student_name=st.session_state.student_name,
                    conversation_history=agent.conversation_history,
                    session_summary=summary,
                )
                st.session_state.ui_notice = "Relatórios salvos no Obsidian."
                st.session_state.ui_notice_type = "success"
                st.toast("Relatórios salvos!", icon="💾")
                st.caption(f"📁 {cost_path}")
                st.caption(f"📁 {prog_path}")
            except Exception as e:
                st.error(f"Erro: {e}")

    if st.button("🔄 Nova Sessão", use_container_width=True):
        preserved_name = st.session_state.student_name
        st.session_state.clear()
        st.session_state.student_name = preserved_name
        st.rerun()

    if st.session_state.ui_notice and st.button(
        "✕ Limpar aviso", use_container_width=True
    ):
        st.session_state.ui_notice = None
        st.rerun()

# ============================================================
# CHAT PRINCIPAL
# ============================================================

st.title("🇺🇸 SOS English — Bianca")
st.caption(
    f"Aluno: **{st.session_state.student_name}** | "
    f"Sessão: `{agent.session_id}` | "
    f"{datetime.now().strftime('%d/%m/%Y')} | "
    f"RAG: {st.session_state.rag_active_level or 'desativado'}"
)

for msg in st.session_state.messages_display:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metrics" in msg:
            m = msg["metrics"]
            st.caption(
                f"⚡ {m.response_time_seconds:.2f}s | "
                f"↑{m.input_tokens} ↓{m.output_tokens} tokens | "
                f"💰 ${m.total_cost_usd:.6f}"
            )

if prompt := st.chat_input("Digite sua mensagem em inglês ou português..."):
    st.session_state.messages_display.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Bianca está pensando..."):
            try:
                response, metrics = agent.chat(prompt)
                st.markdown(response)
                st.caption(
                    f"⚡ {metrics.response_time_seconds:.2f}s | "
                    f"↑{metrics.input_tokens} ↓{metrics.output_tokens} tokens | "
                    f"💰 ${metrics.total_cost_usd:.6f} | "
                    f"`{metrics.model_id}`"
                )
                st.session_state.messages_display.append({
                    "role": "assistant",
                    "content": response,
                    "metrics": metrics,
                })
            except Exception as e:
                st.error(f"Erro: {e}")
                st.info("Verifique a API key no `.env` e o modelo selecionado.")
