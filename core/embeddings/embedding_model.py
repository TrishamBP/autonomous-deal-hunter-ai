"""
Embedding Model Manager

This module provides unified interface for generating embeddings.
Uses sentence-transformers as default embedding model.
"""

import logging
from typing import List, Union
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wrapper for embedding generation using sentence-transformers.
    
    This provides a clean interface for:
    - Loading pre-trained embedding models
    - Generating embeddings for texts
    - Batch processing
    """
    
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    def __init__(self, model_name: str = None):
        """
        Initialize embedding model.
        
        Args:
            model_name: Name of the sentence-transformer model to use.
                       Defaults to all-MiniLM-L6-v2 (fast, lightweight)
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        logger.info(f"Loading embedding model: {self.model_name}")
        
        try:
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded. Dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings for given texts.
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            List of embedding vectors (each is a list of floats)
            
        Example:
            >>> model = EmbeddingModel()
            >>> embeddings = model.embed("Hello world")
            >>> batch_embeddings = model.embed(["text1", "text2"])
        """
        # Convert single string to list
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Single embedding vector
        """
        embeddings = self.embed([text])
        return embeddings[0]
    
    def get_dimension(self) -> int:
        """Get the dimension of embedding vectors."""
        return self.dimension


# Global instance for convenience
_embedding_model = None


def get_embedding_model(model_name: str = None) -> EmbeddingModel:
    """
    Get or create global embedding model instance.
    
    Args:
        model_name: Optional model name to override default
        
    Returns:
        EmbeddingModel instance
    """
    global _embedding_model
    
    if _embedding_model is None:
        _embedding_model = EmbeddingModel(model_name)
    
    return _embedding_model
