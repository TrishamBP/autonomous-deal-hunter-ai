"""
LangChain RAG Pipeline (Primary Implementation)

This is the main RAG implementation using LangChain framework.
It combines:
- Document loading and chunking
- Embedding generation
- ChromaDB vector storage
- LangChain retrieval and QA

This is our PRIMARY implementation.
Alternative implementations exist in llamaindex_rag.py and haystack_rag.py
"""

import logging
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI

# Handle different langchain versions for Document class
try:
    from langchain_core.documents import Document as LangChainDocument
except ImportError:
    from langchain.schema import Document as LangChainDocument

# Optional import - RetrievalQA may not be available in all versions
try:
    from langchain.chains import RetrievalQA
except ImportError:
    RetrievalQA = None
    logging.warning("RetrievalQA not available - QA chain will be disabled")

from core.embeddings.embedding_model import get_embedding_model
from core.vectorstore.chroma_store import ChromaVectorStore
from core.ingestion.document_loader import DocumentLoader, Document

logger = logging.getLogger(__name__)


class LangChainRAGPipeline:
    """
    LangChain-based RAG (Retrieval-Augmented Generation) pipeline.
    
    This pipeline:
    1. Loads and chunks documents
    2. Generates embeddings
    3. Stores in ChromaDB
    4. Retrieves relevant documents for queries
    5. Uses LLM to generate answers
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "documents",
        model_name: str = "gpt-3.5-turbo",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the collection
            model_name: LLM model name (for QA)
            embedding_model: Embedding model name
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.model_name = model_name
        
        logger.info("Initializing LangChain RAG Pipeline")
        
        # Initialize components
        self.doc_loader = DocumentLoader()
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = None
        self.qa_chain = None
        
        logger.info("LangChain RAG Pipeline initialized")
    
    def add_documents(
        self,
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        """
        Add documents to the RAG pipeline.
        
        Args:
            documents: List of Document objects
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        logger.info(f"Adding {len(documents)} documents to RAG pipeline")
        
        # Convert to LangChain documents
        langchain_docs = []
        for doc in documents:
            langchain_docs.append(
                LangChainDocument(
                    page_content=doc.text,
                    metadata=doc.metadata
                )
            )
        
        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        split_docs = splitter.split_documents(langchain_docs)
        logger.info(f"Split into {len(split_docs)} chunks")
        
        # Create or update vectorstore
        self.vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )
        
        # Persist
        self.vectorstore.persist()
        logger.info(f"Added {len(split_docs)} documents to vectorstore")
    
    def load_existing_vectorstore(self) -> None:
        """Load existing vectorstore from disk."""
        logger.info(f"Loading vectorstore from {self.persist_directory}")
        
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )
            logger.info("Vectorstore loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load vectorstore: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results
            
        Returns:
            List of similar documents with metadata
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized. Add documents first.")
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            formatted_results = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score,
                }
                for doc, score in results
            ]
            
            logger.info(f"Similarity search returned {len(formatted_results)} results")
            return formatted_results
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise
    
    def setup_qa_chain(self, llm_kwargs: Dict[str, Any] = None) -> None:
        """
        Setup QA chain for RAG.
        
        Args:
            llm_kwargs: Optional LLM kwargs
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized. Add documents first.")
        
        logger.info("Setting up QA chain")
        
        # Initialize LLM
        llm_kwargs = llm_kwargs or {}
        llm = OpenAI(**llm_kwargs)
        
        # Create retriever and QA chain
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )
        
        logger.info("QA chain setup complete")
    
    def query(self, query: str) -> Dict[str, Any]:
        """
        Run RAG query.
        
        Args:
            query: Query text
            
        Returns:
            Dictionary with answer and source documents
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not setup. Call setup_qa_chain() first.")
        
        try:
            result = self.qa_chain({"query": query})
            
            return {
                "answer": result["result"],
                "sources": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                    for doc in result.get("source_documents", [])
                ],
            }
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
    
    def get_all_vectors(self) -> tuple:
        """
        Get all vectors and metadata for visualization.
        
        Returns:
            Tuple of (embeddings, ids, documents, metadatas)
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized.")
        
        # Access the underlying Chroma collection
        collection = self.vectorstore._collection
        all_data = collection.get()
        
        embeddings = all_data.get("embeddings", [])
        ids = all_data.get("ids", [])
        documents = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])
        
        return embeddings, ids, documents, metadatas


# Convenience function for quick RAG setup
def create_langchain_rag(
    documents: List[Document],
    persist_directory: str = "./chroma_db",
    collection_name: str = "documents",
) -> LangChainRAGPipeline:
    """
    Create and initialize a LangChain RAG pipeline.
    
    Args:
        documents: List of documents
        persist_directory: Where to persist vectors
        collection_name: Name of the collection
        
    Returns:
        Initialized LangChainRAGPipeline
        
    Example:
        >>> from core.ingestion.document_loader import load_documents_simple
        >>> docs = load_documents_simple(["text1", "text2"])
        >>> rag = create_langchain_rag(docs)
        >>> results = rag.similarity_search("query")
    """
    rag = LangChainRAGPipeline(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    rag.add_documents(documents)
    return rag
