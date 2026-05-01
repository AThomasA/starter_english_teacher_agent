# ui/app.py
# ============================================================
# ENGLISH TEACHER AGENT — Interface Streamlit
# ============================================================
# Para rodar: streamlit run ui/app.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from datetime import datetime

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
# INICIALIZAÇÃO DO ESTADO DA SESSÃO
# ============================================================

if "agent" not in st.session_state:
    st.session_state.agent = BiancaAgent()

if "messages_display" not in st.session_state:
    st.session_state.messages_display = []  # lista para exibir no chat

if "student_name" not in st.session_state:
    st.session_state.student_name = "Aluno"

if "session_saved" not in st.session_state:
    st.session_state.session_saved = False

agent: BiancaAgent = st.session_state.agent

# ============================================================
# SIDEBAR — Dashboard de Tokens e Configurações
# ============================================================

with st.sidebar:
    st.title("⚙️ Configurações")

    # Nome do aluno
    student_name = st.text_input("Seu nome", value=st.session_state.student_name)
    st.session_state.student_name = student_name

    st.divider()

    # Modelo ativo
    try:
        model_cfg = get_active_model_config()
        st.caption("🤖 Modelo Ativo")
        st.info(
            f"**{model_cfg['provider']}**\n\n"
            f"`{model_cfg['model_id']}`"
        )
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")

    st.divider()

    # Dashboard de tokens da sessão atual
    st.caption("📊 Consumo da Sessão")
    summary = agent.get_session_summary()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Tokens Input", f"{summary['total_input_tokens']:,}")
        st.metric("Tokens Output", f"{summary['total_output_tokens']:,}")
    with col2:
        st.metric("Total Tokens", f"{summary['total_tokens']:,}")
        st.metric("Trocas", summary['total_exchanges'])

    # Barra de contexto
    ctx_pct = summary['context_window_used_pct']
    st.caption(f"🪟 Janela de Contexto — {ctx_pct:.1f}% usada")
    st.progress(min(ctx_pct / 100, 1.0))
    st.caption(f"Janela total: {summary['context_window']:,} tokens")

    st.divider()

    # Custo e tempo
    st.caption("💰 Custo e Performance")
    st.metric("Custo Total", f"${summary['total_cost_usd']:.6f} USD")
    st.metric("Duração", f"{summary['session_duration_minutes']:.1f} min")
    st.metric("Tempo médio resposta", f"{summary['avg_response_time_seconds']:.2f}s")

    st.divider()

    # Botão para salvar relatórios no Obsidian
    if st.button("💾 Salvar Relatórios no Obsidian", use_container_width=True):
        if summary['total_exchanges'] == 0:
            st.warning("Nenhuma conversa para salvar ainda.")
        else:
            try:
                cost_path = save_cost_report(summary, agent.session_id)
                progress_path = save_progress_report(
                    session_id=agent.session_id,
                    student_name=st.session_state.student_name,
                    conversation_history=agent.conversation_history,
                    session_summary=summary,
                )
                st.success("✅ Relatórios salvos!")
                st.caption(f"📁 Custo: `{cost_path}`")
                st.caption(f"📁 Progresso: `{progress_path}`")
                st.session_state.session_saved = True
            except Exception as e:
                st.error(f"Erro ao salvar: {e}")

    
    # Base de Conhecimento
    st.divider()

    st.caption("📚 Base de Conhecimento")

    selected_level = st.selectbox(
    "Selecionar nível",
    ["starter", "level-1", "level-2"],
    key="rag_level"
    )

    if st.button("📥 Processar PDFs") and not st.session_state.get("processing", False):
        st.session_state.processing = True

        try:
            from rag.rag_pipeline import run_ingestion

            run_ingestion(level=selected_level)
            
            st.success(f"✅ PDFs processados para {selected_level}")
        except Exception as e:
            st.error(f"Erro: {e}")
        finally:
            st.session_state.processing = False

    # Botão para nova sessão
    if st.button("🔄 Nova Sessão", use_container_width=True):
        for key in ["agent", "messages_display", "session_saved"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ============================================================
# ÁREA PRINCIPAL — Chat
# ============================================================

st.title("🇺🇸 SOS English — Bianca")
st.caption(f"Sessão: `{agent.session_id}` | {datetime.now().strftime('%d/%m/%Y')}")

# Exibir histórico de mensagens
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

# Input do usuário
if prompt := st.chat_input("Digite sua mensagem em inglês ou português..."):

    # Exibe mensagem do usuário
    st.session_state.messages_display.append({
        "role": "user",
        "content": prompt,
    })
    with st.chat_message("user"):
        st.markdown(prompt)

    # Chama o agente
    with st.chat_message("assistant"):
        with st.spinner("Bianca está pensando..."):
            try:
                response, metrics = agent.chat(prompt)
                st.markdown(response)
                st.caption(
                    f"⚡ {metrics.response_time_seconds:.2f}s | "
                    f"↑{metrics.input_tokens} ↓{metrics.output_tokens} tokens | "
                    f"💰 ${metrics.total_cost_usd:.6f} | "
                    f"Modelo: `{metrics.model_id}`"
                )
                # Salva no histórico de exibição
                st.session_state.messages_display.append({
                    "role": "assistant",
                    "content": response,
                    "metrics": metrics,
                })
            except Exception as e:
                st.error(f"Erro ao chamar o agente: {e}")
                st.info(
                    "Verifique se a API key está correta no arquivo `.env` "
                    "e se o modelo selecionado está disponível."
                )