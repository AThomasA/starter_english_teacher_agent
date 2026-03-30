import streamlit as st
from agent import create_agent

st.set_page_config(
    page_title="SOS English - Starter Teacher Agent",
    page_icon="🎓",
    layout="centered",
)

st.title("🎓 Agente English Teacher")
st.subheader("SOS English — Starter Level")
st.markdown(
    """
    Esse é um agente professor da escola de inglês online **SOS English**.
    Ele foi treinado com o livro Starter, que é o nível básico do curso.
    Fique à vontade para praticar inglês, tirar suas dúvidas e aprender
    de forma leve e descontraída.
    """
)

st.divider()

if "agent" not in st.session_state:
    with st.spinner("Carregando agente..."):
        st.session_state.agent = create_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Digite sua pergunta em inglês ou português..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    chat_history = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}"
        for m in st.session_state.messages[:-1]
    )

    with st.chat_message("assistant"):
        with st.spinner("Bi está pensando..."):
            response = st.session_state.agent.invoke({
                "question": question,
                "chat_history": chat_history,
            })
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})