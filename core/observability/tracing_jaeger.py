"""
Jaeger Distributed Tracing

Jaeger provides distributed tracing for tracking requests across multiple services.

This module enables:
- Distributed trace collection
- Span tracking across agents
- Performance bottleneck identification
- Service dependency mapping
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Jaeger client would be imported separately
# from jaeger_client import Config
# from opentelemetry import trace
# from opentelemetry.exporter.jaeger.thrift import JaegerExporter


class JaegerTracer:
    """
    Distributed tracer using Jaeger.
    
    Enables tracing of:
    - Agent interactions
    - RAG pipeline steps
    - LLM calls
    - Vector operations
    """
    
    def __init__(
        self,
        service_name: str = "rag_system",
        jaeger_host: str = "localhost",
        jaeger_port: int = 6831,
        enabled: bool = True,
    ):
        """
        Initialize Jaeger tracer.
        
        Args:
            service_name: Name of the service
            jaeger_host: Jaeger agent host
            jaeger_port: Jaeger agent port
            enabled: Whether to enable tracing
        """
        self.service_name = service_name
        self.jaeger_host = jaeger_host
        self.jaeger_port = jaeger_port
        self.enabled = enabled
        self.tracer = None
        
        logger.info(f"Initializing Jaeger tracer for service: {service_name}")
        
        if enabled:
            self._initialize_tracer()
    
    def _initialize_tracer(self):
        """Initialize Jaeger tracer."""
        try:
            from jaeger_client import Config
            
            config = Config(
                config={
                    'sampler': {
                        'type': 'const',
                        'param': 1,  # Sample all traces
                    },
                    'local_agent': {
                        'reporting_host': self.jaeger_host,
                        'reporting_port': self.jaeger_port,
                    },
                    'logging': True,
                },
                service_name=self.service_name,
            )
            
            self.tracer = config.initialize_tracer()
            logger.info(f"Jaeger tracer initialized: {self.jaeger_host}:{self.jaeger_port}")
        except ImportError:
            logger.warning("jaeger-client not installed. Install with: pip install jaeger-client")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize Jaeger tracer: {e}")
            self.enabled = False
    
    def start_span(self, operation_name: str, tags: Dict[str, Any] = None) -> 'Span':
        """
        Start a new span.
        
        Args:
            operation_name: Name of the operation
            tags: Optional tags for the span
            
        Returns:
            Span object
            
        Example:
            span = tracer.start_span("rag_query", tags={"query": "what is RAG?"})
            span.set_tag("status", "success")
            span.finish()
        """
        if not self.enabled or not self.tracer:
            return NoOpSpan()
        
        try:
            span = self.tracer.start_span(operation_name)
            if tags:
                for key, value in tags.items():
                    span.set_tag(key, value)
            logger.info(f"Started span: {operation_name}")
            return span
        except Exception as e:
            logger.error(f"Error starting span: {e}")
            return NoOpSpan()
    
    def trace_agent_call(self, agent_name: str, operation: str, tags: Dict[str, Any] = None):
        """
        Trace an agent call.
        
        Args:
            agent_name: Name of the agent
            operation: Operation being performed
            tags: Optional tags
            
        Usage:
            with tracer.trace_agent_call("SpecialistAgent", "price_estimation"):
                estimate_price("laptop")
        """
        class TracingContext:
            def __init__(ctx_self, tracer_inner, agent, op, tags_inner):
                ctx_self.tracer_inner = tracer_inner
                ctx_self.agent = agent
                ctx_self.op = op
                ctx_self.tags = tags_inner or {}
                ctx_self.span = None
            
            def __enter__(ctx_self):
                ctx_self.span = ctx_self.tracer_inner.start_span(
                    f"{ctx_self.agent}.{ctx_self.op}",
                    tags={**ctx_self.tags, "agent": ctx_self.agent}
                )
                return ctx_self.span
            
            def __exit__(ctx_self, exc_type, exc_val, exc_tb):
                if ctx_self.span:
                    if exc_type:
                        ctx_self.span.set_tag("error", True)
                        ctx_self.span.set_tag("error.message", str(exc_val))
                    ctx_self.span.finish()
        
        return TracingContext(self, agent_name, operation, tags)
    
    def trace_rag_pipeline(self, query: str):
        """
        Trace RAG pipeline execution.
        
        Args:
            query: Query text
            
        Usage:
            with tracer.trace_rag_pipeline("what is RAG?") as span:
                # RAG execution
                pass
        """
        return self.trace_agent_call(
            "RAGPipeline",
            "query",
            tags={"query": query[:100]}  # Truncate long queries
        )
    
    def flush(self):
        """Flush any pending spans."""
        if self.tracer:
            try:
                self.tracer.close()
                logger.info("Jaeger tracer flushed")
            except Exception as e:
                logger.error(f"Error flushing Jaeger tracer: {e}")


class NoOpSpan:
    """No-op span for when tracing is disabled."""
    
    def set_tag(self, key: str, value: Any):
        pass
    
    def log_kv(self, kwargs: Dict[str, Any]):
        pass
    
    def finish(self):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# Global tracer instance
_jaeger_tracer = None


def get_jaeger_tracer(service_name: str = "rag_system") -> JaegerTracer:
    """Get global Jaeger tracer."""
    global _jaeger_tracer
    if _jaeger_tracer is None:
        _jaeger_tracer = JaegerTracer(service_name=service_name)
    return _jaeger_tracer


# CONFIGURATION GUIDE
"""
To use Jaeger distributed tracing:

1. Install: pip install jaeger-client

2. Start Jaeger container:
   docker run -d --name jaeger \\
     -e COLLECTOR_ZIPKIN_HOST_PORT=:9411 \\
     -p 6831:6831/udp \\
     -p 16686:16686 \\
     jaegertracing/all-in-one:latest

3. In your code:
   from core.observability.tracing_jaeger import get_jaeger_tracer
   
   tracer = get_jaeger_tracer("my_service")
   
   with tracer.trace_rag_pipeline("what is RAG?") as span:
       # Your RAG code here
       pass
   
   tracer.flush()

4. View traces at: http://localhost:16686

Key concepts:
- Trace: Complete request flow
- Span: Individual operation
- Tags: Metadata on spans
- Logs: Events within spans
"""
