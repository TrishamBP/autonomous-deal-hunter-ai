"""
Document Loading and Ingestion

This module provides utilities for loading documents from various sources
and splitting them into chunks for embedding and storage.
"""

import logging
from typing import List, Dict, Any
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class Document:
    """Represents a document with text and metadata."""
    
    def __init__(self, text: str, metadata: Dict[str, Any] = None):
        """
        Initialize document.
        
        Args:
            text: Document text content
            metadata: Optional metadata dictionary
        """
        self.text = text
        self.metadata = metadata or {}


class TextSplitter:
    """
    Simple text splitter for chunking documents.
    
    Splits documents into chunks of specified size with optional overlap.
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize text splitter.
        
        Args:
            chunk_size: Size of each chunk in characters
            overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def split(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.overlap
        
        return chunks


class DocumentLoader:
    """
    Load documents from various sources.
    
    Supports:
    - Plain text files
    - JSON files
    - Multiple files from directory
    """
    
    def __init__(self, splitter: TextSplitter = None):
        """
        Initialize document loader.
        
        Args:
            splitter: Optional TextSplitter for chunking
        """
        self.splitter = splitter or TextSplitter()
    
    def load_text_file(self, filepath: str, chunk: bool = True) -> List[Document]:
        """
        Load a text file.
        
        Args:
            filepath: Path to text file
            chunk: Whether to chunk the document
            
        Returns:
            List of Document objects
            
        Example:
            >>> loader = DocumentLoader()
            >>> docs = loader.load_text_file("sample.txt")
        """
        try:
            path = Path(filepath)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {filepath}")
            
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            logger.info(f"Loaded text file: {filepath}")
            
            if chunk:
                chunks = self.splitter.split(text)
                documents = [
                    Document(
                        text=chunk,
                        metadata={"source": filepath, "chunk_index": i}
                    )
                    for i, chunk in enumerate(chunks)
                ]
            else:
                documents = [Document(text=text, metadata={"source": filepath})]
            
            return documents
        except Exception as e:
            logger.error(f"Failed to load text file {filepath}: {e}")
            raise
    
    def load_json_file(self, filepath: str, text_field: str = "text", chunk: bool = True) -> List[Document]:
        """
        Load a JSON file with documents.
        
        Args:
            filepath: Path to JSON file
            text_field: JSON field containing document text
            chunk: Whether to chunk documents
            
        Returns:
            List of Document objects
        """
        try:
            path = Path(filepath)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {filepath}")
            
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both list of objects and single object
            if not isinstance(data, list):
                data = [data]
            
            documents = []
            for item in data:
                if isinstance(item, dict) and text_field in item:
                    text = item[text_field]
                    metadata = {k: v for k, v in item.items() if k != text_field}
                    metadata["source"] = filepath
                    
                    if chunk:
                        chunks = self.splitter.split(text)
                        for i, chunk in enumerate(chunks):
                            documents.append(
                                Document(
                                    text=chunk,
                                    metadata={**metadata, "chunk_index": i}
                                )
                            )
                    else:
                        documents.append(Document(text=text, metadata=metadata))
            
            logger.info(f"Loaded {len(documents)} documents from JSON: {filepath}")
            return documents
        except Exception as e:
            logger.error(f"Failed to load JSON file {filepath}: {e}")
            raise
    
    def load_directory(self, directory: str, pattern: str = "*.txt", chunk: bool = True) -> List[Document]:
        """
        Load all files matching pattern from directory.
        
        Args:
            directory: Path to directory
            pattern: File pattern to match (e.g., "*.txt", "*.json")
            chunk: Whether to chunk documents
            
        Returns:
            List of Document objects from all files
        """
        try:
            path = Path(directory)
            if not path.is_dir():
                raise NotADirectoryError(f"Not a directory: {directory}")
            
            documents = []
            for filepath in path.glob(pattern):
                if pattern == "*.json":
                    docs = self.load_json_file(str(filepath), chunk=chunk)
                else:
                    docs = self.load_text_file(str(filepath), chunk=chunk)
                documents.extend(docs)
            
            logger.info(f"Loaded {len(documents)} documents from directory: {directory}")
            return documents
        except Exception as e:
            logger.error(f"Failed to load directory {directory}: {e}")
            raise


# Convenience function for loading simple text
def load_documents_simple(texts: List[str], sources: List[str] = None) -> List[Document]:
    """
    Create Document objects from simple list of texts.
    
    Args:
        texts: List of text strings
        sources: Optional list of source names
        
    Returns:
        List of Document objects
    """
    documents = []
    sources = sources or [f"source_{i}" for i in range(len(texts))]
    
    for text, source in zip(texts, sources):
        documents.append(Document(text=text, metadata={"source": source}))
    
    return documents
