"""Подсчёт токенов через tiktoken."""

import tiktoken

# Используем cl100k_base (GPT-4) — универсальный энкодер
_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Подсчитать количество токенов в тексте."""
    return len(_encoder.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Обрезать текст до указанного количества токенов."""
    tokens = _encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _encoder.decode(tokens[:max_tokens])


def fit_messages_budget(
    messages: list[dict],
    max_system_tokens: int = 3000,
    max_total_tokens: int = 8192,
) -> list[dict]:
    """Обрезать messages[] чтобы вписаться в бюджет токенов."""
    # Считаем system prompt
    system_tokens = 0
    for msg in messages:
        if msg["role"] == "system":
            system_tokens += count_tokens(msg["content"])
    
    if system_tokens > max_system_tokens:
        # Обрезаем system prompt
        for msg in messages:
            if msg["role"] == "system":
                msg["content"] = truncate_to_tokens(msg["content"], max_system_tokens)
                break
    
    # Оставшееся место для user/assistant
    remaining = max_total_tokens - min(system_tokens, max_system_tokens)
    
    # Обрезаем с конца (старые сообщения)
    result = []
    used = 0
    for msg in reversed(messages):
        if msg["role"] == "system":
            result.insert(0, msg)
            continue
        tokens = count_tokens(msg["content"])
        if used + tokens > remaining:
            break
        result.insert(0, msg)
        used += tokens
    
    return result
