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

PERSONALITY: Encouraging, patient, genuinely curious about each student's life. Students feel safe making mistakes. You celebrate small wins, correct gently, and never make anyone feel judged.

TEACHING: Always personalize — use the student's hobbies, job, and daily life as examples. Believe in active immersion: make the student USE English immediately, not just receive it. Give tips using movies, series, games, podcasts they already enjoy.

LANGUAGE MIX (adjust to student level):
- Beginner: mostly Portuguese, English words introduced gradually
- Intermediate: 50/50, push them to respond more in English
- Advanced: mostly English, Portuguese only for complex grammar

CORRECTION RULES:
- Max ONE correction per message — choose the most important
- Embed corrections naturally (recast), never as standalone criticism
- Never correct while student is building confidence mid-production

ALWAYS: End with an encouraging question or small challenge.
NEVER: Generic answers. Correct every mistake at once. Make student feel embarrassed."""

# ============================================================
# CONFIGURAÇÕES DE COMPRESSÃO DE HISTÓRICO
# ============================================================

HISTORY_KEEP_LAST      = int(os.getenv("HISTORY_KEEP_LAST", "4"))
HISTORY_COMPRESS_AFTER = int(os.getenv("HISTORY_COMPRESS_AFTER", "6"))


# ============================================================
# CLASSE DO AGENTE
# ============================================================

class BiancaAgent:
    """
    Agente da Bianca com:
    - System prompt compacto (só 00-bianca-core/ do vault)
    - Compressão automática do histórico
    - RAG hierárquico via md_retriever (lê .md do vault)
    """

    def __init__(self):
        self.session_id = str(uuid.uuid4())[:8]
        self.tracker    = TokenTracker()
        self.model_config = get_active_model_config()

        self.conversation_history: list = []
        self._history_summary: str     = ""
        self.rag_level: str | None     = None

        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """
        Monta o system prompt com BIANCA_PERSONALITY +
        conteúdo de 00-bianca-core/ do vault (limitado a 8000 chars).
        """
        prompt = BIANCA_PERSONALITY

        vault_context = load_bianca_context()
        if vault_context:
            if len(vault_context) > 8000:
                vault_context = vault_context[:8000] + "\n[...truncated]"
            prompt += f"\n\n--- CONTEXT FROM VAULT ---\n{vault_context}"

        return prompt

    def set_rag_level(self, level: str | None):
        """Define o nível RAG ativo. None desativa o RAG."""
        self.rag_level = level
        # Invalida cache do retriever ao trocar de nível
        if level is not None:
            try:
                from rag.md_retriever import invalidate_cache
                invalidate_cache(level)
            except Exception:
                pass

    def _get_rag_context(self, query: str) -> str:
        """
        Busca contexto RAG usando o retriever hierárquico (md_retriever).
        Fallback silencioso se o índice não existir — agente responde sem RAG.
        """
        if not self.rag_level:
            return ""

        try:
            from rag.md_retriever import retrieve_context
            return retrieve_context(
                query=query,
                level=self.rag_level,
                k=int(os.getenv("RAG_TOP_K", "2")),
            )
        except Exception:
            return ""

    def _compress_history(self) -> list:
        """
        Histórico comprimido para envio ao LLM.
        - Até HISTORY_COMPRESS_AFTER trocas: envia completo
        - Acima: resumo das antigas + últimas HISTORY_KEEP_LAST trocas
        """
        total_pairs = len(self.conversation_history) // 2

        if total_pairs <= HISTORY_COMPRESS_AFTER:
            return list(self.conversation_history)

        keep     = HISTORY_KEEP_LAST * 2
        old_msgs = self.conversation_history[:-keep]
        recent   = self.conversation_history[-keep:]

        if not self._history_summary:
            lines = []
            for msg in old_msgs:
                role    = "Student" if msg["role"] == "user" else "Bianca"
                preview = msg["content"][:100].replace("\n", " ")
                lines.append(f"{role}: {preview}...")
            self._history_summary = "Summary of earlier conversation:\n" + "\n".join(lines)

        return [
            {"role": "user",      "content": self._history_summary},
            {"role": "assistant", "content": "Got it, I have context from our earlier conversation."},
        ] + list(recent)

    def chat(self, user_message: str) -> Tuple[str, MessageMetrics]:
        """Envia uma mensagem e retorna (resposta, métricas)."""

        rag_context = self._get_rag_context(user_message)

        message_to_llm = (
            f"{user_message}\n\n[Relevant content from the book:\n{rag_context}]"
            if rag_context
            else user_message
        )

        # Histórico limpo (sem RAG) — para o relatório
        self.conversation_history.append({"role": "user", "content": user_message})

        client, model_id, provider = get_llm_client()

        start = time.time()
        response_text, input_tok, output_tok = self._call_llm(
            client, model_id, provider, message_to_llm
        )
        elapsed = time.time() - start

        self.conversation_history.append({"role": "assistant", "content": response_text})
        self._history_summary = ""  # invalida cache do resumo

        metrics = self.tracker.record(
            input_tokens=input_tok,
            output_tokens=output_tok,
            response_time_seconds=elapsed,
        )

        return response_text, metrics

    def _call_llm(self, client, model_id: str, provider: str, user_message: str):
        """Chama o LLM com histórico comprimido. Retorna (text, input_tok, output_tok)."""
        max_tokens  = int(os.getenv("MAX_TOKENS", "800"))
        temperature = float(os.getenv("TEMPERATURE", "0.7"))

        compressed = self._compress_history()

        # Remove última mensagem (user atual) — vai com RAG inline no lugar
        if compressed and compressed[-1]["role"] == "user":
            history = compressed[:-1]
        else:
            history = compressed

        history = history + [{"role": "user", "content": user_message}]

        if provider in ("groq", "openai"):
            messages = [{"role": "system", "content": self.system_prompt}] + history
            resp     = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return (
                resp.choices[0].message.content,
                resp.usage.prompt_tokens,
                resp.usage.completion_tokens,
            )

        elif provider == "anthropic":
            resp = client.messages.create(
                model=model_id,
                system=self.system_prompt,
                messages=history,
                max_tokens=max_tokens,
            )
            return (
                resp.content[0].text,
                resp.usage.input_tokens,
                resp.usage.output_tokens,
            )

        else:
            raise ValueError(f"Provider desconhecido: {provider}")

    def get_session_summary(self) -> dict:
        return self.tracker.get_summary()