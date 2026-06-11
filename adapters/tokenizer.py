"""SentencePiece tokenizer wrapper and Mistral 7B model configuration."""

import os
from dataclasses import dataclass
from typing import List, Optional

import sentencepiece as spm

_DEFAULT_MODEL_DIR = os.path.join(
    os.path.expanduser("~"),
    ".cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1",
    "snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da",
)


@dataclass
class MistralConfig:
    vocab_size: int = 32000
    hidden_dim: int = 4096
    intermediate_dim: int = 14336
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 32768
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = False

    @property
    def num_heads_per_kv(self) -> int:
        return self.num_heads // self.num_kv_heads


class Tokenizer:
    """Thin wrapper around SentencePiece for Mistral's tokenizer."""

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = os.path.join(_DEFAULT_MODEL_DIR, "tokenizer.model")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.vocab_size = self.sp.GetPieceSize()

    def encode(self, text: str, add_bos: bool = True) -> List[int]:
        tokens = self.sp.Encode(text)
        if add_bos:
            tokens = [self.bos_id] + tokens
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        return self.sp.Decode(token_ids)

    def encode_instruct(self, user_message: str) -> List[int]:
        """Encode with Mistral instruct format: [INST] ... [/INST]"""
        formatted = f"[INST] {user_message} [/INST]"
        return self.encode(formatted, add_bos=True)
