# agent/models/model_config.py
# ============================================================
# CONFIGURAÇÃO DE MODELOS
# ============================================================
# Para trocar de modelo:
# 1. Abra o arquivo .env
# 2. Mude ACTIVE_MODEL para: groq | openai | anthropic
# 3. Certifique-se que a API key do modelo escolhido está preenchida
# 4. Reinicie o Streamlit (Ctrl+C e rode novamente)
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv()

# Preço por 1 milhão de tokens (em USD) - atualizado manualmente conforme necessário
MODEL_PRICING = {
    "groq": {
        "model_id": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "input_price_per_1m": 0.59,   # $0.59 por 1M tokens de input
        "output_price_per_1m": 0.79,  # $0.79 por 1M tokens de output
        "provider": "Groq",
        "context_window": 128000,
    },
    "openai": {
        "model_id": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "input_price_per_1m": 0.15,   # $0.15 por 1M tokens de input
        "output_price_per_1m": 0.60,  # $0.60 por 1M tokens de output
        "provider": "OpenAI",
        "context_window": 128000,
    },
    "anthropic": {
        "model_id": os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
        "input_price_per_1m": 0.80,   # $0.80 por 1M tokens de input
        "output_price_per_1m": 4.00,  # $4.00 por 1M tokens de output
        "provider": "Anthropic",
        "context_window": 200000,
    },
}

def get_active_model_config() -> dict:
    """Retorna a configuração do modelo ativo definido no .env."""
    active = os.getenv("ACTIVE_MODEL", "groq").lower()
    if active not in MODEL_PRICING:
        raise ValueError(
            f"Modelo '{active}' não reconhecido. "
            f"Opções válidas: {list(MODEL_PRICING.keys())}"
        )
    config = MODEL_PRICING[active].copy()
    config["active_key"] = active
    return config


def get_llm_client():
    """
    Retorna o cliente LLM correto baseado em ACTIVE_MODEL.
    Retorna uma tupla: (client, model_id, provider)
    """
    config = get_active_model_config()
    provider = config["active_key"]

    if provider == "groq":
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        return client, config["model_id"], "groq"

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return client, config["model_id"], "openai"

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return client, config["model_id"], "anthropic"