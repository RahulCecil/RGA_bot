from dataclasses import dataclass

@dataclass
class TokenMetrics:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    bytes_in: int
    bytes_out: int

def calculate_bytes(text: str) -> int:
    return len(text.encode('utf-8'))