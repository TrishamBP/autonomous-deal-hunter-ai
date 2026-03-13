"""
Gradio Web Interface for Price Is Right RAG System

This is the main entry point for running the system.
Start with: python -m core.gradio_app
or: python launch_app.py
"""

import os
import sys
import logging
import gradio as gr
import json
from typing import Tuple, List, Dict, Optional
from dotenv import load_dotenv

# CRITICAL: Add project root to sys.path so absolute imports work
# when this module is imported by core/main.py
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [RAG App] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Import our RAG system components using absolute imports only
from core.main import DealAgentSystem


class GradioRAGApp:
    """Gradio interface for the RAG-based Price estimation system"""
    
    def __init__(self):
        """Initialize the Gradio app with RAG system"""
        logger.info("Initializing Gradio RAG App...")
        
        self.system = DealAgentSystem()
        self.components = self.system.initialize_all_components()
        
        # Initialize RAG agent
        self.rag_agent = self._init_rag_agent()
        
        # Cache for visualization
        self.last_viz_data = None
        
        logger.info("Gradio RAG App initialized successfully")
    
    def _init_rag_agent(self):
        """Initialize the RAG agent"""
        try:
            from core.agents.rag_agent import RAGAgent
            
            # Use environment variable for persist directory if available
            persist_dir = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
            
            agent = RAGAgent(
                persist_directory=persist_dir,
                enable_tracing=False  # Disable for Gradio
            )
            logger.info(f"RAG Agent initialized with DB at: {persist_dir}")
            return agent
        except Exception as e:
            logger.error(f"Failed to initialize RAG Agent: {e}")
            return None
    
    def estimate_price(self, product_description: str) -> Tuple[str, str]:
        """
        Estimate product price using RAG
        
        Args:
            product_description: Description of the product
            
        Returns:
            Tuple of (estimated_price, context_documents)
        """
        if not self.rag_agent:
            return "Error: RAG Agent not initialized", ""
        
        try:
            logger.info(f"Estimating price for: {product_description[:50]}...")
            
            # Retrieve similar products
            results = self.rag_agent.retrieval(product_description, k=5)
            
            # Format context
            context_text = "📚 **Similar Products Found:**\n\n"
            prices = []
            
            for i, result in enumerate(results, 1):
                context_text += f"**{i}. Similarity: {result['score']:.2%}**\n"
                context_text += f"*{result['content'][:100]}*\n"
                
                # Extract price from metadata if available, or generate reasonable estimate
                if 'metadata' in result and 'price' in result['metadata']:
                    price = result['metadata']['price']
                    prices.append(float(price))
                    context_text += f"💰 Price: ${price:.2f}\n"
                else:
                    # Estimate price based on similarity score and product type
                    estimated_price = self._estimate_price_from_description(result['content'])
                    prices.append(estimated_price)
                    context_text += f"💰 Est. Price: ${estimated_price:.2f}\n"
                
                context_text += "\n"
            
            # Calculate final price estimate
            if prices:
                avg_price = sum(prices) / len(prices)
                min_price = min(prices)
                max_price = max(prices)
                
                estimated = f"💵 **Price Estimate: ${avg_price:.2f}**\n\n"
                estimated += f"Range: ${min_price:.2f} - ${max_price:.2f}\n"
                estimated += f"Based on {len(results)} similar products"
            else:
                estimated = f"Similar products found - unable to estimate price"
            
            return estimated, context_text
            
        except Exception as e:
            logger.error(f"Error estimating price: {e}", exc_info=True)
            return f"Error: {str(e)}", ""
    
    def _estimate_price_from_description(self, description: str) -> float:
        """
        Estimate price based on product description keywords
        
        Args:
            description: Product description
            
        Returns:
            Estimated price in USD
        """
        # Price estimates based on product type
        price_ranges = {
            'headphone': (50, 300),
            'charger': (15, 80),
            'tv': (300, 2000),
            'laptop': (500, 2000),
            'keyboard': (40, 200),
            'hub': (30, 100),
            'webcam': (60, 300),
            'ssd': (80, 300),
            'mouse': (15, 100),
            'monitor': (150, 600),
        }
        
        desc_lower = description.lower()
        
        # Find matching category and return middle of range
        for keyword, (min_price, max_price) in price_ranges.items():
            if keyword in desc_lower:
                return (min_price + max_price) / 2
        
        # Default estimate for unknown products
        return 150.0
    
    def search_similar_products(self, product_description: str, num_results: int = 5) -> str:
        """
        Search for similar products in the vector store
        
        Args:
            product_description: Description to search for
            num_results: Number of results to return
            
        Returns:
            Formatted string of search results
        """
        if not self.rag_agent:
            return "Error: RAG Agent not initialized"
        
        try:
            results = self.rag_agent.retrieval(product_description, k=min(num_results, 10))
            
            output = f"**Found {len(results)} similar products:**\n\n"
            for i, result in enumerate(results, 1):
                output += f"**{i}. Similarity Score: {result['score']:.4f}**\n"
                output += f"```\n{result['content']}\n```\n"
                if result.get('metadata'):
                    output += f"**Metadata:** {json.dumps(result['metadata'], indent=2)}\n"
                output += "\n---\n\n"
            
            return output
            
        except Exception as e:
            logger.error(f"Error searching products: {e}")
            return f"Error: {str(e)}"
    
    def load_documents_ui(self, doc_source: str, num_docs: str) -> str:
        """
        Load documents into the vector store
        
        Args:
            doc_source: Source of documents (sample/custom)
            num_docs: Number of documents to load
            
        Returns:
            Status message
        """
        if not self.rag_agent:
            return "Error: RAG Agent not initialized"
        
        try:
            from core.ingestion.document_loader import load_documents_simple
            
            num = int(num_docs) if num_docs else 50
            
            if doc_source == "sample":
                # Create sample documents with diverse products
                sample_docs = [
                    # Audio Products
                    "Premium noise-canceling headphones with 30-hour battery life",
                    "Wireless earbuds with active noise cancellation and charging case",
                    "Studio monitor speakers with subwoofer included",
                    "Portable Bluetooth speaker with waterproof design",
                    # Accessories
                    "Wireless charger pad compatible with all major devices",
                    "USB-C hub with 7 ports and power delivery",
                    "Mechanical keyboard with RGB backlighting and mechanical switches",
                    "Wireless mouse with precision tracking and ergonomic design",
                    "Monitor arm with integrated USB hub and height adjustment",
                    "Cable management kit for desk organization",
                    # Computing
                    "8K Ultra HD smart TV with quantum dot display",
                    "Gaming laptop with RTX 4090 graphics card",
                    "27-inch 4K monitor with USB-C connectivity",
                    "Ultra-portable laptop with 16-inch display and all-day battery",
                    "Desktop graphics card RTX 4080 with 12GB VRAM",
                    "High-performance SSD 2TB with hardware encryption",
                    "Mini PC with Intel i9 processor and 32GB RAM",
                    # Video/Camera
                    "4K webcam with autofocus and noise reduction",
                    "Mirrorless camera with full-frame sensor 45MP",
                    "Action camera with 8K recording and stabilization",
                    "Phone gimbal stabilizer for smooth video recording",
                    # Other Electronics
                    "Portable SSD 2TB with hardware encryption",
                    "Power bank 100000mAh with fast charging 65W",
                    "Smart watch with fitness tracking and heart rate monitor",
                    "Wireless mouse with precision tracking",
                    "Fast charger 120W with multiple ports",
                    "VR headset with 120Hz refresh rate",
                    "Gaming controller with force feedback and wireless connectivity",
                    "Portable projector with 1080p resolution and WiFi",
                    "Digital tablet 12-inch with stylus included",
                    "E-reader with backlit display and waterproof case",
                    "Router with WiFi 6 and mesh networking",
                    "USB security key for two-factor authentication",
                    "External hard drive 4TB with USB 3.1 connection",
                    "Laptop cooling pad with RGB lighting",
                    "HDMI cable 8K certified ultra high-speed",
                    "Desk lamp with adjustable color temperature",
                    "Webcam privacy shutter with adhesive mount",
                ]
                
                # Use unique products, repeat to reach num_docs if needed
                docs_to_use = (sample_docs * ((num // len(sample_docs)) + 1))[:num]
                docs = load_documents_simple(docs_to_use)
            else:
                return "Custom document loading not yet implemented"
            
            self.rag_agent.load_documents(docs)
            return f"✅ Successfully loaded {len(docs)} documents into vector store"
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return f"Error: {str(e)}"
    
    def visualize_embeddings(self) -> Tuple[str, str]:
        """
        Visualize embeddings using t-SNE
        
        Returns:
            Tuple of (status_message, file_path)
        """
        if not self.rag_agent:
            return "Error: RAG Agent not initialized", ""
        
        try:
            # Check if we have any documents loaded
            if not hasattr(self.rag_agent, 'rag_pipeline') or self.rag_agent.rag_pipeline is None:
                return "Error: No documents loaded. Please load documents first.", ""
            
            logger.info("Generating embedding visualization...")
            file_path = self.rag_agent.get_embeddings_visualization(
                save_path="embeddings_viz.html"
            )
            return f"✅ Visualization saved to {file_path}", f"file={file_path}"
            
        except Exception as e:
            logger.error(f"Error visualizing embeddings: {e}", exc_info=True)
            return f"Error: {str(e)}", ""
    
    def get_system_stats(self) -> str:
        """Get system statistics"""
        try:
            stats = f"""
            **System Statistics:**
            
            - Vector DB Location: `./chroma_db`
            - Embedding Model: sentence-transformers/all-MiniLM-L6-v2
            - Vector Dimension: 384
            - Framework: LangChain + ChromaDB
            - RAG Agent Status: {'✅ Active' if self.rag_agent else '❌ Inactive'}
            """
            return stats
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return f"Error: {str(e)}"
    
    def build_interface(self) -> gr.Blocks:
        """Build the Gradio interface"""
        
        with gr.Blocks(title="Price Is Right - RAG System") as interface:
            gr.Markdown(
                """
                # 💰 Price Is Right - RAG System
                
                Estimate product prices using Retrieval-Augmented Generation (RAG)
                powered by LangChain and ChromaDB vector database.
                """
            )
            
            with gr.Tabs():
                # Tab 1: Price Estimation
                with gr.Tab("🎯 Price Estimation"):
                    with gr.Row():
                        with gr.Column():
                            product_input = gr.Textbox(
                                label="Product Description",
                                placeholder="Enter product description (e.g., 'Premium wireless headphones with ANC')",
                                lines=3
                            )
                            estimate_btn = gr.Button("Estimate Price", variant="primary")
                        
                        with gr.Column():
                            price_output = gr.Textbox(
                                label="Estimated Price",
                                interactive=False,
                                lines=3
                            )
                    
                    context_output = gr.Markdown(
                        label="Context from Similar Products",
                        value="Context will appear here..."
                    )
                    
                    estimate_btn.click(
                        self.estimate_price,
                        inputs=[product_input],
                        outputs=[price_output, context_output]
                    )
                
                # Tab 2: Search Similar Products
                with gr.Tab("🔍 Search Similar"):
                    with gr.Row():
                        search_input = gr.Textbox(
                            label="Search Query",
                            placeholder="Describe the product you're looking for",
                            lines=2
                        )
                        search_btn = gr.Button("Search", variant="primary")
                    
                    num_results = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Number of Results"
                    )
                    
                    search_output = gr.Markdown(
                        label="Search Results",
                        value="Search results will appear here..."
                    )
                    
                    search_btn.click(
                        self.search_similar_products,
                        inputs=[search_input, num_results],
                        outputs=[search_output]
                    )
                
                # Tab 3: Document Management
                with gr.Tab("📚 Documents"):
                    with gr.Row():
                        doc_source = gr.Radio(
                            choices=["sample", "custom"],
                            value="sample",
                            label="Document Source"
                        )
                        num_docs_input = gr.Textbox(
                            value="50",
                            label="Number of Documents",
                            type="text"
                        )
                        load_btn = gr.Button("Load Documents", variant="primary")
                    
                    doc_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        value="Ready to load documents"
                    )
                    
                    load_btn.click(
                        self.load_documents_ui,
                        inputs=[doc_source, num_docs_input],
                        outputs=[doc_status]
                    )
                
                # Tab 4: Visualization
                with gr.Tab("📊 Visualizations"):
                    gr.Markdown("### Embedding Space Visualization (t-SNE)")
                    
                    viz_btn = gr.Button("Generate 3D Visualization", variant="primary")
                    viz_status = gr.Textbox(label="Status", interactive=False)
                    
                    viz_btn.click(
                        self.visualize_embeddings,
                        outputs=[viz_status]
                    )
                    
                    gr.HTML(
                        """
                        <p>The visualization will be saved as an interactive HTML file.
                        Once generated, you can download it and open in your browser.</p>
                        """
                    )
                
                # Tab 5: System Info
                with gr.Tab("ℹ️ System Info"):
                    stats_output = gr.Markdown(label="System Statistics")
                    refresh_btn = gr.Button("Refresh Stats")
                    
                    refresh_btn.click(
                        self.get_system_stats,
                        outputs=[stats_output]
                    )
                    
                    # Auto-load stats on startup
                    interface.load(self.get_system_stats, outputs=[stats_output])
        
        return interface


def main():
    """Launch the Gradio app"""
    logger.info("=" * 60)
    logger.info("LAUNCHING PRICE IS RIGHT - RAG GRADIO APP")
    logger.info("=" * 60)
    
    try:
        app = GradioRAGApp()
        interface = app.build_interface()
        
        logger.info("\n✅ Gradio app ready!")
        logger.info("Opening in browser at http://localhost:7860")
        logger.info("=" * 60)
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            theme=gr.themes.Soft()
        )
        
    except Exception as e:
        logger.error(f"Failed to launch Gradio app: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
