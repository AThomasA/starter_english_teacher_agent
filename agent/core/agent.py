# agent/core/agent.py
# ============================================================
# AGENTE PRINCIPAL — BIANCA, ENGLISH TEACHER
# ============================================================

import os
import time
import uuid
from typing import Tuple
from dotenv import load_dotenv

from agent.models.model_config import get_llm_client, get_active_model_config
from agent.tools.obsidian_writer import load_bianca_context
from agent.tools.token_tracker import TokenTracker, MessageMetrics

load_dotenv()

# ============================================================
# PERSONALIDADE DA BIANCA
# ============================================================
# TODO: Quando a Bianca responder o questionário de personalidade,
#       substitua o conteúdo de BIANCA_PERSONALITY pelo resultado refinado.
#       Por enquanto, usamos este perfil base construído a partir da descrição
#       do Allan como referência inicial.

BIANCA_PERSONALITY = """
You are Bianca, an English teacher at SOS English school in Brazil.

YOUR PERSONALITY:
- Warm, encouraging, and patient. Students feel safe to make mistakes with you.
- You actively listen and adapt your teaching to what each student shares about their life.
- You celebrate small wins genuinely. Never make students feel embarrassed about errors.
- You're direct and practical — you don't waste time with abstract theory when a real example works better.

YOUR TEACHING APPROACH:
- You ALWAYS bring English into the student's real daily life. If they like gaming, you use game examples. If they watch series, you use series dialogues. If they work in logistics, you use logistics vocabulary.
- You ask about the student's hobbies, routines, and interests at the start and you weave them naturally into the lesson.
- You believe in ACTIVE immersion: you don't just explain, you make the student USE the language immediately.
- You use metrics to help students understand their own progress: you note patterns in their mistakes and point them out gently.
- You give tips with movies, series, games, and podcasts — always aligned with what the student enjoys.
- Corrections are gentle and embedded in your response naturally, not as a standalone criticism.

YOUR COMMUNICATION STYLE:
- Speak in a mix of Portuguese and English depending on the student's level.
  - Beginners: mostly Portuguese with English words introduced gradually.
  - Intermediate: half and half, pushing them to respond more in English.
  - Advanced: mostly English, with Portuguese only for complex grammar explanations.
- Use informal, friendly language. You're a teacher, not a robot.
- Keep responses focused. Don't overwhelm the student with too much at once.
- Always end with an encouraging question or a small challenge to keep the student practicing.

IMPORTANT RULES:
- NEVER give generic answers. Always personalize based on what you know about the student.
- NEVER correct every single mistake at once — choose the most important one per message.
- ALWAYS encourage the student to try speaking/writing in English, even if imperfect.
- If a student seems discouraged, acknowledge the feeling first before teaching.
"""

# ============================================================
# CLASSE DO AGENTE
# ============================================================

class BiancaAgent:
    """
    Agente da Bianca. Mantém o histórico de conversa da sessão
    e centraliza as chamadas ao LLM.
    """

    def __init__(self):
        self.session_id = str(uuid.uuid4())[:8]
        self.tracker = TokenTracker()
        self.conversation_history = []  # lista de {"role": ..., "content": ...}
        self.model_config = get_active_model_config()

        # Carrega contexto do vault Obsidian (se houver arquivos lá)
        vault_context = load_bianca_context()
        self.system_prompt = BIANCA_PERSONALITY
        if vault_context:
            self.system_prompt += (
                "\n\n===\nADDITIONAL KNOWLEDGE FROM VAULT:\n" + vault_context
            )

    def chat(self, user_message: str) -> Tuple[str, MessageMetrics]:
        """
        Envia uma mensagem para o agente e retorna (resposta, métricas).
        """
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        client, model_id, provider = get_llm_client()

        start_time = time.time()
        response_text, input_tokens, output_tokens = self._call_llm(
            client, model_id, provider
        )
        elapsed = time.time() - start_time

        self.conversation_history.append({
            "role": "assistant",
            "content": response_text,
        })

        metrics = self.tracker.record(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            response_time_seconds=elapsed,
        )

        return response_text, metrics

    def _call_llm(self, client, model_id: str, provider: str):
        """
        Chamada ao LLM de acordo com o provider ativo.
        Retorna: (response_text, input_tokens, output_tokens)
        """
        max_tokens = int(os.getenv("MAX_TOKENS", 2000))
        temperature = float(os.getenv("TEMPERATURE", 0.7))

        # --- GROQ ou OPENAI (mesma interface OpenAI-compatible) ---
        if provider in ("groq", "openai"):
            messages = [{"role": "system", "content": self.system_prompt}]
            messages += self.conversation_history

            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = response.choices[0].message.content
            input_tok  = response.usage.prompt_tokens
            output_tok = response.usage.completion_tokens
            return text, input_tok, output_tok

        # --- ANTHROPIC ---
        elif provider == "anthropic":
            response = client.messages.create(
                model=model_id,
                system=self.system_prompt,
                messages=self.conversation_history,
                max_tokens=max_tokens,
            )
            text = response.content[0].text
            input_tok  = response.usage.input_tokens
            output_tok = response.usage.output_tokens
            return text, input_tok, output_tok

        else:
            raise ValueError(f"Provider desconhecido: {provider}")

    def get_session_summary(self) -> dict:
        return self.tracker.get_summary()