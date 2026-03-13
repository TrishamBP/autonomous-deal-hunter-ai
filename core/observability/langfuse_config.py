"""
LangFuse Integration

LangFuse is a production-grade LLM tracing and monitoring platform.

This module enables:
- LLM API call tracing
- Agent orchestration tracking
- Embeddings performance monitoring
- Cost tracking
- Performance analytics
"""

import logging
import os
from typing import Optional, Dict, Any
from functools import wraps

logger = logging.getLogger(__name__)

# LangFuse would be installed separately
# from langfuse import Langfuse


class LangFuseConfig:
    """
    LangFuse configuration and context manager.
    
    Sets up LangFuse for tracing:
    - LLM calls
    - Agent interactions
    - Performance metrics
    """
    
    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: str = "https://cloud.langfuse.com",
        enabled: bool = True,
    ):
        """
        Initialize LangFuse configuration.
        
        Args:
            public_key: LangFuse public key (or env LANGFUSE_PUBLIC_KEY)
            secret_key: LangFuse secret key (or env LANGFUSE_SECRET_KEY)
            host: LangFuse host URL
            enabled: Whether to enable tracing
        """
        self.public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.host = host
        self.enabled = enabled
        self.client = None
        
        logger.info(f"LangFuse configured. Enabled: {enabled}")
        
        if enabled and (self.public_key and self.secret_key):
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize LangFuse client."""
        try:
            from langfuse import Langfuse
            
            self.client = Langfuse(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host,
            )
            logger.info("LangFuse client initialized")
        except ImportError:
            logger.warning("langfuse not installed. Install with: pip install langfuse")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize LangFuse: {e}")
            self.enabled = False
    
    def trace_llm_call(self, name: str = "llm_call"):
        """
        Decorator for tracing LLM calls.
        
        Usage:
            @langfuse_config.trace_llm_call("price_estimation")
            def estimate_price(description: str) -> float:
                ...
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled or not self.client:
                    return func(*args, **kwargs)
                
                try:
                    with self.client.trace(name=name) as trace:
                        trace.generation(
                            name=name,
                            model="gpt-3.5-turbo",
                            input=str(args + tuple(kwargs.values())),
                        )
                        result = func(*args, **kwargs)
                        trace.generation(
                            name=f"{name}_result",
                            output=str(result),
                        )
                        return result
                except Exception as e:
                    logger.error(f"Error in LangFuse tracing: {e}")
                    return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def trace_embedding(self, name: str = "embedding"):
        """
        Decorator for tracing embedding generation.
        
        Tracks:
        - Number of texts
        - Embedding dimension
        - Execution time
        """
        def decorator(func):
            @wraps(func)
            def wrapper(self_inner, texts, *args, **kwargs):
                if not self.enabled or not self.client:
                    return func(self_inner, texts, *args, **kwargs)
                
                num_texts = len(texts) if isinstance(texts, list) else 1
                
                try:
                    with self.client.trace(name=name) as trace:
                        trace.span(
                            name="embedding_generation",
                            input={"num_texts": num_texts},
                        )
                        result = func(self_inner, texts, *args, **kwargs)
                        trace.span(
                            name="embedding_result",
                            output={"dimension": len(result[0]) if result else 0},
                        )
                        return result
                except Exception as e:
                    logger.error(f"Error in embedding tracing: {e}")
                    return func(self_inner, texts, *args, **kwargs)
            
            return wrapper
        return decorator
    
    def trace_rag_query(self, query: str):
        """
        Trace RAG query execution.
        
        Logs:
        - Query text
        - Retrieved documents
        - Final answer
        """
        if not self.enabled or not self.client:
            return
        
        try:
            with self.client.trace(name="rag_query") as trace:
                trace.span(name="rag_input", input={"query": query})
                return trace
        except Exception as e:
            logger.error(f"Error tracing RAG query: {e}")
            return None
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """
        Log custom metrics to LangFuse.
        
        Args:
            metrics: Dictionary of metric names and values
        """
        if not self.enabled or not self.client:
            return
        
        try:
            for name, value in metrics.items():
                logger.info(f"LangFuse metric: {name}={value}")
                # Would send to LangFuse: self.client.custom_metric(...)
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
    
    def flush(self):
        """Flush any pending traces."""
        if self.client:
            try:
                self.client.flush()
                logger.info("LangFuse flushed")
            except Exception as e:
                logger.error(f"Error flushing LangFuse: {e}")


# Global instance
_langfuse_config = None


def get_langfuse_config() -> LangFuseConfig:
    """Get global LangFuse config."""
    global _langfuse_config
    if _langfuse_config is None:
        _langfuse_config = LangFuseConfig()
    return _langfuse_config


# CONFIGURATION GUIDE
"""
To use LangFuse:

1. Install: pip install langfuse

2. Sign up at: https://langfuse.com

3. Set environment variables:
   export LANGFUSE_PUBLIC_KEY=pk_...
   export LANGFUSE_SECRET_KEY=sk_...

4. In your code:
   from core.observability.langfuse_config import get_langfuse_config
   
   langfuse = get_langfuse_config()
   
   @langfuse.trace_llm_call("my_task")
   def my_function():
       pass

5. View traces at: https://cloud.langfuse.com
"""
