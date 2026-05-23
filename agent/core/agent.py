# agent/core/agent.py
# ============================================================
# AGENTE PRINCIPAL — BIANCA, ENGLISH TEACHER
# ============================================================

import os
import time
import uuid
from typing import Tuple
from dotenv import load_dotenv
from pathlib import Path

from agent.models.model_config import get_llm_client, get_active_model_config
from agent.tools.obsidian_writer import load_bianca_context
from agent.tools.token_tracker import TokenTracker, MessageMetrics

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

# ============================================================
# PERSONALIDADE DA BIANCA — compacta (detalhes ficam no vault)
# ============================================================
# TODO: substituir pelo questionário real da Bianca quando disponível.

BIANCA_PERSONALITY = """You are Bianca, a warm and practical English teacher at SOS English (Brazil).

PERSONALITY: Encouraging, patient, genuinely curious about each student's life.
Students feel safe making mistakes. You celebrate small wins, correct gently,
and never make anyone feel judged.

TEACHING APPROACH:
- Always personalize using what you know about the student's life, interests and goals.
- Believe in active immersion: make the student USE English immediately, not just receive it.
- Use examples from the student's interests (RPG, fantasy, Lord of the Rings, Final Fantasy, magic).
- Always use CURRENT, natural English. Never use old-fashioned or formal expressions nobody uses today.
- Prefer examples from real conversations, series, YouTube, podcasts — not textbook English.

STUDENT PROFILE — ALLAN:
- Passive-receptive learner: understands well when reading/listening, struggles to BUILD sentences.
- The main focus must always be on ACTIVE PRODUCTION — making Allan construct sentences himself.
- Less explanation, more guided practice. Make him try first, then correct.
- Engages well with RPG/fantasy examples. Use them naturally in grammar and vocabulary practice.

CURRICULUM — FOLLOW THE SCHOOL BOOKS IN ORDER:
- Always follow SOS English curriculum: Starter → Level 1 → Level 2 → Level 3 → Level 4
- Even if the student seems advanced, start from the BEGINNING of the assessed level.
- Content already mastered: accelerate through it, don't waste time.
- Content with difficulty: slow down, change approach, give more practice.
- NEVER jump levels or skip units without completing the current one.
- At the start of each session: check where we left off and continue from there.

FIRST SESSION — LEVEL ASSESSMENT:
If no level has been assessed yet (check level-assessment.md):
1. Run a short diagnostic: ask strategic questions, give short sentence-building challenges.
2. Test both comprehension AND production (production is where Allan struggles most).
3. Define the starting level and unit.
4. Save the full assessment to obsidian-vault/05-session-memory/level-assessment.md
5. Only then start teaching.

SESSION STRUCTURE:
1. Check-in (2 min): connect personally, ask if they used English since last session
2. Review (5 min): revisit last session's content — make Allan produce, not just remember
3. New content (20 min): follow the current book unit
4. Active production (10 min): Allan builds sentences, does challenges, no passive receiving
5. Wrap-up (3 min): highlight one win, one focus point, optional challenge for next session

CORRECTION RULES:
- Max ONE correction per message — choose the most important.
- Use recast: repeat the correct form naturally in your response, don't just point out the error.
- Never correct while the student is building confidence mid-production.

SESSION RECORDS:
- After each session, the student will save reports to the vault.
- You should remind them at the end of the session to click "Salvar no Obsidian".
- Track vocabulary learned, errors identified, and progress notes mentally during the session.

LANGUAGE MIX:
- Starter/Level 1: mostly Portuguese, English introduced gradually
- Level 2: 50/50, push Allan to respond more in English
- Level 3+: mostly English, Portuguese only for complex grammar explanations

ALWAYS: End with an encouraging question or small sentence-building challenge.
NEVER: Generic explanations without personalization. Correct every mistake at once.
       Use old-fashioned English expressions. Advance without consolidating."""

# ============================================================
# CONFIGURAÇÕES DE COMPRESSÃO DE HISTÓRICO
# ============================================================

# Mantém as últimas N trocas completas (1 troca = 1 user + 1 assistant)
HISTORY_KEEP_LAST = int(os.getenv("HISTORY_KEEP_LAST", "4"))

# A partir de quantas trocas começa a comprimir
HISTORY_COMPRESS_AFTER = int(os.getenv("HISTORY_COMPRESS_AFTER", "6"))


# ============================================================
# CLASSE DO AGENTE
# ============================================================

class BiancaAgent:
    """
    Agente da Bianca com:
    - System prompt compacto (sem vault inteiro no prompt)
    - Compressão automática do histórico
    - RAG conectado ao chat()
    """

    def __init__(self):
        self.session_id = str(uuid.uuid4())[:8]
        self.tracker = TokenTracker()
        self.model_config = get_active_model_config()

        # Histórico completo (para salvar no relatório)
        self.conversation_history: list = []

        # Resumo comprimido das mensagens antigas (cache)
        self._history_summary: str = ""

        # Nível RAG ativo (definido via sidebar no Streamlit)
        self.rag_level: str | None = None

        # Monta system prompt — só 00-bianca-core/, sem vault inteiro
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """
        Monta o system prompt final.
        Carrega APENAS os arquivos de 00-bianca-core/ do vault,
        não o vault inteiro — evita tokens desnecessários.
        """
        prompt = BIANCA_PERSONALITY

        vault_context = load_bianca_context()
        if vault_context:
            # Limita o contexto do vault a 8000 chars (~2000 tokens)
            max_vault_chars = 8000
            if len(vault_context) > max_vault_chars:
                vault_context = (
                    vault_context[:max_vault_chars]
                    + "\n[...vault context truncated to save tokens]"
                )
            prompt += f"\n\n--- CONTEXT FROM VAULT ---\n{vault_context}"

        return prompt

    def set_rag_level(self, level: str):
        """Define qual nível RAG usar nas consultas."""
        self.rag_level = level

    def _get_rag_context(self, query: str) -> str:
        """
        Busca contexto RAG relevante para a query.
        Retorna string vazia se RAG não estiver configurado ou falhar.
        """
        if not self.rag_level:
            return ""

        try:
            from rag.md_retriever import retrieve_context
            context = retrieve_context(
                query=query,
                level=self.rag_level,
                k=int(os.getenv("RAG_TOP_K", "2")),  # 2 chunks = ~250 tokens
            )
            return context
        except Exception:
            return ""

    def _compress_history(self) -> list:
        """
        Retorna o histórico preparado para enviar ao LLM.

        Lógica:
        - Se total de trocas <= HISTORY_COMPRESS_AFTER: manda tudo
        - Se maior: resumo simples das antigas + últimas HISTORY_KEEP_LAST trocas completas
        """
        total_pairs = len(self.conversation_history) // 2

        if total_pairs <= HISTORY_COMPRESS_AFTER:
            return list(self.conversation_history)

        keep_messages = HISTORY_KEEP_LAST * 2
        old_messages = self.conversation_history[:-keep_messages]
        recent_messages = self.conversation_history[-keep_messages:]

        # Gera resumo só uma vez por bloco de mensagens antigas
        if not self._history_summary:
            lines = []
            for msg in old_messages:
                role = "Student" if msg["role"] == "user" else "Bianca"
                preview = msg["content"][:100].replace("\n", " ")
                lines.append(f"{role}: {preview}...")
            self._history_summary = (
                "Summary of earlier conversation:\n" + "\n".join(lines)
            )

        compressed = [
            {"role": "user", "content": self._history_summary},
            {
                "role": "assistant",
                "content": "Got it, I have context from our earlier conversation.",
            },
        ] + list(recent_messages)

        return compressed

    def chat(self, user_message: str) -> Tuple[str, MessageMetrics]:
        """
        Envia uma mensagem para o agente e retorna (resposta, métricas).
        """
        # Busca contexto RAG ANTES de adicionar ao histórico
        rag_context = self._get_rag_context(user_message)

        # Mensagem que vai pro LLM: user + RAG inline se houver
        if rag_context:
            message_to_llm = (
                f"{user_message}\n\n"
                f"[Relevant content from the book:\n{rag_context}]"
            )
        else:
            message_to_llm = user_message

        # Histórico salva a mensagem limpa (sem RAG) — para o relatório
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        client, model_id, provider = get_llm_client()

        start_time = time.time()
        response_text, input_tokens, output_tokens = self._call_llm(
            client, model_id, provider, message_to_llm
        )
        elapsed = time.time() - start_time

        self.conversation_history.append({
            "role": "assistant",
            "content": response_text,
        })

        # Invalida o cache do resumo para recalcular na próxima compressão
        self._history_summary = ""

        metrics = self.tracker.record(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            response_time_seconds=elapsed,
        )

        return response_text, metrics

    def _call_llm(self, client, model_id: str, provider: str, user_message: str):
        """
        Chamada ao LLM com histórico comprimido.
        Retorna: (response_text, input_tokens, output_tokens)
        """
        max_tokens = int(os.getenv("MAX_TOKENS", "800"))
        temperature = float(os.getenv("TEMPERATURE", "0.7"))

        # Histórico comprimido (sem a última mensagem do user — vai separada)
        compressed = self._compress_history()

        # Remove a última mensagem do histórico comprimido (user atual)
        # pois vamos injetar a versão com RAG no lugar
        if compressed and compressed[-1]["role"] == "user":
            history_to_send = compressed[:-1]
        else:
            history_to_send = compressed

        # Adiciona a mensagem atual (com RAG se houver)
        history_to_send = history_to_send + [
            {"role": "user", "content": user_message}
        ]

        # --- GROQ ou OPENAI ---
        if provider in ("groq", "openai"):
            messages = [{"role": "system", "content": self.system_prompt}]
            messages += history_to_send

            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = response.choices[0].message.content
            input_tok = response.usage.prompt_tokens
            output_tok = response.usage.completion_tokens
            return text, input_tok, output_tok

        # --- ANTHROPIC ---
        elif provider == "anthropic":
            response = client.messages.create(
                model=model_id,
                system=self.system_prompt,
                messages=history_to_send,
                max_tokens=max_tokens,
            )
            text = response.content[0].text
            input_tok = response.usage.input_tokens
            output_tok = response.usage.output_tokens
            return text, input_tok, output_tok

        else:
            raise ValueError(f"Provider desconhecido: {provider}")

    def get_session_summary(self) -> dict:
        return self.tracker.get_summary()


        