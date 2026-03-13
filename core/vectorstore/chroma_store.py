"""
ChromaDB Vector Store Manager

This module provides a wrapper around ChromaDB for storing and retrieving
document embeddings with metadata.
"""

import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    Wrapper for ChromaDB vector database.
    
    Provides clean interface for:
    - Creating/connecting to collections
    - Adding documents with embeddings and metadata
    - Searching/querying documents
    - Retrieving vectors for visualization
    """
    
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "documents"):
        """
        Initialize ChromaDB vector store.
        
        Args:
            persist_directory: Directory to persist database
            collection_name: Name of the collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        logger.info(f"Initializing ChromaDB with persist_directory: {persist_directory}")
        
        # Configure ChromaDB with persistence
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory,
            anonymized_telemetry=False,
        )
        
        try:
            self.client = chromadb.Client(settings)
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"ChromaDB initialized. Collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]] = None,
        ids: List[str] = None,
    ) -> None:
        """
        Add documents with embeddings to the collection.
        
        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: Optional list of metadata dicts
            ids: Optional list of document IDs
            
        Example:
            >>> store = ChromaVectorStore()
            >>> docs = ["Document 1", "Document 2"]
            >>> embeddings = [[0.1, 0.2, ...], [0.3, 0.4, ...]]
            >>> store.add_documents(docs, embeddings)
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        try:
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas or [{"source": f"doc_{i}"} for i in range(len(documents))],
                ids=ids or [f"doc_{i}" for i in range(len(documents))]
            )
            logger.info(f"Added {len(documents)} documents to collection")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def query(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents using embedding.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            
        Returns:
            Dictionary with ids, distances, documents, and metadatas
            
        Example:
            >>> results = store.query(query_embedding, n_results=5)
            >>> print(results['documents'])  # Retrieved document texts
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"Failed to query: {e}")
            raise
    
    def get_all_embeddings(self) -> tuple:
        """
        Get all embeddings and metadata from collection.
        
        Returns:
            Tuple of (embeddings list, ids list, documents list, metadatas list)
            
        Useful for visualization and analysis.
        """
        try:
            all_results = self.collection.get()
            embeddings = all_results.get("embeddings", [])
            ids = all_results.get("ids", [])
            documents = all_results.get("documents", [])
            metadatas = all_results.get("metadatas", [])
            
            logger.info(f"Retrieved {len(embeddings)} embeddings from collection")
            return embeddings, ids, documents, metadatas
        except Exception as e:
            logger.error(f"Failed to get all embeddings: {e}")
            raise
    
    def count(self) -> int:
        """Get number of documents in collection."""
        return self.collection.count()
    
    def delete_all(self) -> None:
        """Delete all documents in collection."""
        try:
            # Get all IDs and delete them
            all_data = self.collection.get()
            if all_data["ids"]:
                self.collection.delete(ids=all_data["ids"])
            logger.info("Deleted all documents from collection")
        except Exception as e:
            logger.error(f"Failed to delete all documents: {e}")
            raise
    
    def persist(self) -> None:
        """Persist collection to disk."""
        try:
            self.client.persist()
            logger.info("ChromaDB persisted to disk")
        except Exception as e:
            logger.error(f"Failed to persist: {e}")
            raise
