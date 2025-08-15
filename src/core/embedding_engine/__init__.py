#Embedding engine package for converting text into vector representations.
#Supports multiple embedding models and backends for semantic search.

from .base import EmbeddingEngine
from .sentence_transformer_backend import SentenceTransformerBackend
from .tensorflow_backend import TensorFlowBackend
from .huggingface_backend import HuggingFaceBackend
from .similarity import SimilarityComputer
from .storage import EmbeddingStorage

__all__ = [
    'EmbeddingEngine',
    'SentenceTransformerBackend',
    'TensorFlowBackend', 
    'HuggingFaceBackend',
    'SimilarityComputer',
    'EmbeddingStorage'
]