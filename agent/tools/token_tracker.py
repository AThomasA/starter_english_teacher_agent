# agent/tools/token_tracker.py
# ============================================================
# RASTREADOR DE TOKENS E CUSTOS POR SESSÃO
# ============================================================

import time
from dataclasses import dataclass, field
from typing import List
from agent.models.model_config import get_active_model_config


@dataclass
class MessageMetrics:
    """Métricas de uma única mensagem."""
    timestamp: float
    input_tokens: int
    output_tokens: int
    response_time_seconds: float
    model_id: str
    provider: str
    input_cost_usd: float
    output_cost_usd: float

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def total_cost_usd(self) -> float:
        return self.input_cost_usd + self.output_cost_usd


class TokenTracker:
    """
    Rastreia o consumo de tokens e custo durante uma sessão de aula.
    Uma sessão = uma conversa completa no Streamlit.
    """

    def __init__(self):
        self.messages: List[MessageMetrics] = []
        self.session_start = time.time()
        self.model_config = get_active_model_config()

    def _calculate_cost(self, tokens: int, price_per_1m: float) -> float:
        """Calcula custo em USD dado número de tokens e preço por milhão."""
        return (tokens / 1_000_000) * price_per_1m

    def record(
        self,
        input_tokens: int,
        output_tokens: int,
        response_time_seconds: float,
    ) -> MessageMetrics:
        """
        Registra as métricas de uma mensagem.
        Retorna o objeto MessageMetrics para uso imediato na UI.
        """
        input_cost = self._calculate_cost(
            input_tokens,
            self.model_config["input_price_per_1m"]
        )
        output_cost = self._calculate_cost(
            output_tokens,
            self.model_config["output_price_per_1m"]
        )

        metrics = MessageMetrics(
            timestamp=time.time(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            response_time_seconds=response_time_seconds,
            model_id=self.model_config["model_id"],
            provider=self.model_config["provider"],
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
        )
        self.messages.append(metrics)
        return metrics

    # ---- Totais acumulados da sessão ----

    @property
    def total_input_tokens(self) -> int:
        return sum(m.input_tokens for m in self.messages)

    @property
    def total_output_tokens(self) -> int:
        return sum(m.output_tokens for m in self.messages)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_cost_usd(self) -> float:
        return sum(m.total_cost_usd for m in self.messages)

    @property
    def context_window_used_pct(self) -> float:
        """Percentual da janela de contexto usada (tokens acumulados)."""
        window = self.model_config["context_window"]
        return min((self.total_tokens / window) * 100, 100.0)

    @property
    def session_duration_minutes(self) -> float:
        return (time.time() - self.session_start) / 60

    @property
    def total_exchanges(self) -> int:
        return len(self.messages)

    def get_summary(self) -> dict:
        """Retorna um dicionário com o resumo completo da sessão."""
        return {
            "model": self.model_config["model_id"],
            "provider": self.model_config["provider"],
            "total_exchanges": self.total_exchanges,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "context_window": self.model_config["context_window"],
            "context_window_used_pct": round(self.context_window_used_pct, 2),
            "session_duration_minutes": round(self.session_duration_minutes, 2),
            "avg_response_time_seconds": round(
                sum(m.response_time_seconds for m in self.messages) / max(len(self.messages), 1), 2
            ),
        }