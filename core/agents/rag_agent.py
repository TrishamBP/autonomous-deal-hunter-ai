"""
RAG Agent

Agent that orchestrates the RAG (Retrieval-Augmented Generation) pipeline.

This agent:
- Manages document ingestion
- Performs semantic search
- Integrates with LLM for answer generation
- Tracks metrics and traces
"""

import logging
from typing import Optional, Dict, Any, List

from core.agents.base_agent import Agent
from core.rag.langchain_rag import LangChainRAGPipeline, create_langchain_rag
from core.ingestion.document_loader import DocumentLoader, Document
from core.observability.metrics_prometheus import get_metrics_collector
from core.observability.tracing_jaeger import get_jaeger_tracer

logger = logging.getLogger(__name__)


class RAGAgent(Agent):
    """
    Agent that manages RAG operations.
    
    Responsibilities:
    - Load and ingest documents
    - Build vector database
    - Perform retrieval
    - Generate answers using LLM
    - Track metrics and performance
    """
    
    name = "RAG Agent"
    color = Agent.CYAN
    
    def __init__(self, persist_directory: str = "./chroma_db", enable_tracing: bool = True):
        """
        Initialize RAG agent.
        
        Args:
            persist_directory: Where to persist vector DB
            enable_tracing: Enable distributed tracing
        """
        self.log("RAG Agent is initializing")
        
        self.persist_directory = persist_directory
        self.enable_tracing = enable_tracing
        self.rag_pipeline = None
        self.metrics = get_metrics_collector()
        self.tracer = get_jaeger_tracer("RAGAgent") if enable_tracing else None
        self.doc_loader = DocumentLoader()
        
        self.log("RAG Agent initialized successfully")
    
    def load_documents(self, documents: List[Document]) -> None:
        """
        Load documents into RAG pipeline.
        
        Args:
            documents: List of Document objects
        """
        self.log(f"Loading {len(documents)} documents")
        
        if self.enable_tracing and self.tracer:
            with self.tracer.trace_agent_call("RAGAgent", "load_documents"):
                self._load_documents_impl(documents)
        else:
            self._load_documents_impl(documents)
    
    def _load_documents_impl(self, documents: List[Document]) -> None:
        """Implementation of document loading."""
        import time
        start_time = time.time()
        
        try:
            self.rag_pipeline = create_langchain_rag(
                documents,
                persist_directory=self.persist_directory
            )
            
            latency = time.time() - start_time
            self.metrics.record_document_ingestion(len(documents), latency)
            self.log(f"Loaded {len(documents)} documents in {latency:.3f}s")
        except Exception as e:
            self.log(f"Error loading documents: {e}")
            raise
    
    def retrieval(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve similar documents for a query.
        
        Args:
            query: Query text
            k: Number of results
            
        Returns:
            List of retrieved documents with scores
        """
        self.log(f"Retrieving {k} documents for: {query}")
        
        if self.rag_pipeline is None:
            raise ValueError("RAG pipeline not initialized. Load documents first.")
        
        import time
        start_time = time.time()
        
        try:
            if self.enable_tracing and self.tracer:
                with self.tracer.trace_agent_call("RAGAgent", "retrieval", {"query": query}):
                    results = self.rag_pipeline.similarity_search(query, k=k)
            else:
                results = self.rag_pipeline.similarity_search(query, k=k)
            
            latency = time.time() - start_time
            self.metrics.record_vector_search(len(results), latency)
            self.log(f"Retrieved {len(results)} documents in {latency:.3f}s")
            
            return results
        except Exception as e:
            self.log(f"Error in retrieval: {e}")
            raise
    
    def answer(self, query: str) -> Dict[str, Any]:
        """
        Generate answer using RAG.
        
        Args:
            query: Query text
            
        Returns:
            Dictionary with answer and sources
        """
        self.log(f"Generating answer for: {query}")
        
        if self.rag_pipeline is None:
            raise ValueError("RAG pipeline not initialized. Load documents first.")
        
        import time
        start_time = time.time()
        
        try:
            # Setup QA chain if not already done
            if self.rag_pipeline.qa_chain is None:
                self.rag_pipeline.setup_qa_chain()
            
            if self.enable_tracing and self.tracer:
                with self.tracer.trace_rag_pipeline(query):
                    result = self.rag_pipeline.query(query)
            else:
                result = self.rag_pipeline.query(query)
            
            latency = time.time() - start_time
            self.metrics.record_rag_query(latency, status="success")
            self.log(f"Generated answer in {latency:.3f}s")
            
            return result
        except Exception as e:
            self.metrics.record_rag_query(0, status="error")
            self.log(f"Error generating answer: {e}")
            raise
    
    def run(self, documents: List[Document], query: str) -> Dict[str, Any]:
        """
        Run complete RAG pipeline.
        
        Args:
            documents: Documents to ingest
            query: Query to answer
            
        Returns:
            RAG result with answer and sources
        """
        self.log("Running complete RAG pipeline")
        
        # Load documents
        self.load_documents(documents)
        
        # Get answer
        result = self.answer(query)
        
        return result
    
    def get_embeddings_visualization(self, save_path: Optional[str] = None):
        """
        Get embeddings for visualization.
        
        Args:
            save_path: Optional path to save visualization
            
        Returns:
            File path to saved visualization (if save_path provided), else Plotly figure
        """
        self.log("Preparing embeddings visualization")
        
        if self.rag_pipeline is None:
            raise ValueError("RAG pipeline not initialized")
        
        try:
            from core.visualization.tsne_visualizer import visualize_embeddings
            
            embeddings, ids, documents, metadatas = self.rag_pipeline.get_all_vectors()
            
            fig = visualize_embeddings(
                embeddings,
                documents,
                metadatas,
                dimensions=3,
                save_path=save_path
            )
            
            self.log(f"Visualization prepared: {save_path or 'interactive'}")
            return save_path if save_path else fig
        except Exception as e:
            self.log(f"Error preparing visualization: {e}")
            raise
