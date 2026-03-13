#!/usr/bin/env python3
"""
Price Is Right - RAG System - Main Entry Point

This module contains both the DealAgentSystem orchestrator and the application entry point.
It initializes the RAG pipeline and launches the Gradio web interface.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# CRITICAL: Add project root to sys.path FIRST so absolute imports work
# when running `python core/main.py` directly (not as module)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Load environment variables
load_dotenv(override=True)

# Configure logging  
def setup_logging():
    """Setup logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)


class DealAgentSystem:
    """
    Main orchestrator for the Price Is Right Deal Agent System
    
    This system combines multiple agents to create a complete RAG-based solution.
    """
    
    def __init__(self):
        """Initialize the agent system"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing DealAgentSystem")
        self.db_name = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
        self.logger.info(f"Vector DB path: {self.db_name}")
    
    def setup_environment(self):
        """Setup environment variables"""
        self.logger.info("Setting up environment variables")
        self.logger.info("Environment setup complete")
    
    def load_rag_agent(self):
        """Load and initialize RAG agent"""
        self.logger.info("Loading RAG Agent")
        try:
            from core.agents.rag_agent import RAGAgent
            agent = RAGAgent(persist_directory="./chroma_db", enable_tracing=False)
            self.logger.info("RAG Agent loaded successfully")
            return agent
        except Exception as e:
            self.logger.error(f"Failed to load RAG Agent: {e}")
            return None
    
    def initialize_all_components(self):
        """Initialize and load all system components"""
        self.logger.info("=" * 60)
        self.logger.info("INITIALIZING SYSTEM COMPONENTS")
        self.logger.info("=" * 60)
        
        self.setup_environment()
        self.rag_agent = self.load_rag_agent()
        
        components = {
            "rag_agent": self.rag_agent,
        }
        
        self.logger.info("=" * 60)
        self.logger.info("INITIALIZATION COMPLETE")
        self.logger.info("=" * 60)
        
        return components


def main():
    """Main entry point for running the application"""
    logger = setup_logging()
    
    try:
        logger.info("=" * 70)
        logger.info("🚀 Starting Price Is Right - RAG System")
        logger.info("=" * 70)
        
        # Import and launch the Gradio app
        from core.gradio_app import main as launch_gradio
        
        logger.info("\n📊 Initializing RAG pipeline and web interface...")
        launch_gradio()
        
    except Exception as e:
        logger.error(f"Failed to launch application: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
