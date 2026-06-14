# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""Chunker Factory for GraphRAG Pipeline."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Callable
from .factory import Factory

@dataclass
class TextChunk:
    """标准文本块数据模型，对齐 GraphRAG 内部协议"""
    text: str
    token_count: Optional[int] = None

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> List[TextChunk]:
        pass

class SentenceChunker(BaseChunker):
    def __init__(self, **kwargs): pass  # 兼容工厂透传空参数
    def chunk(self, text: str) -> List[TextChunk]:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [TextChunk(text=s.strip(), token_count=len(s.split())) for s in sentences if s.strip()]

class TokenChunker(BaseChunker):
    def __init__(self, size: int = 1024, overlap: int = 128,
                 encode: Optional[Callable] = None, decode: Optional[Callable] = None):
        self.size = size
        self.overlap = overlap
        self.encode = encode or (lambda x: x.split())
        self.decode = decode or (lambda x: " ".join(x))

    def chunk(self, text: str) -> List[TextChunk]:
        tokens = self.encode(text)
        chunks, start = [], 0
        while start < len(tokens):
            end = min(start + self.size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(TextChunk(text=self.decode(chunk_tokens), token_count=len(chunk_tokens)))
            start += self.size - self.overlap
        return chunks

# 🏭 单例工厂实例
class ChunkerFactory(Factory[BaseChunker]): pass
chunker_factory = ChunkerFactory()

# 📝 注册策略（惰性初始化，生产环境按需加载）
chunker_factory.register("sentence", SentenceChunker, scope="transient")
chunker_factory.register("token", TokenChunker, scope="transient")
