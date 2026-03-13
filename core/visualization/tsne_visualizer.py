"""
t-SNE Vector Visualization

This module provides visualization of high-dimensional embeddings using t-SNE
and Plotly for interactive 3D/2D scatter plots.

t-SNE (t-Distributed Stochastic Neighbor Embedding) is excellent for:
- Visualizing document clusters
- Understanding embedding space structure
- Identifying similar documents
- Detecting anomalies
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)


class TSNEVisualizer:
    """
    Visualize embeddings using t-SNE dimensionality reduction.
    
    Provides:
    - 2D and 3D visualizations
    - Interactive Plotly plots
    - Metadata display on hover
    - Color coding by metadata
    """
    
    def __init__(self, n_components: int = 2, perplexity: int = 30, random_state: int = 42):
        """
        Initialize t-SNE visualizer.
        
        Args:
            n_components: Number of dimensions for output (2 or 3)
            perplexity: t-SNE perplexity parameter
            random_state: Random seed for reproducibility
        """
        if n_components not in [2, 3]:
            raise ValueError("n_components must be 2 or 3")
        
        self.n_components = n_components
        self.perplexity = perplexity
        self.random_state = random_state
        self.tsne_results = None
        
        logger.info(f"Initialized TSNEVisualizer with {n_components}D")
    
    def fit_transform(self, embeddings: List[List[float]]) -> np.ndarray:
        """
        Fit t-SNE and transform embeddings.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            2D or 3D array of transformed coordinates
        """
        embeddings_array = np.array(embeddings)
        logger.info(f"Fitting t-SNE on {len(embeddings)} embeddings of shape {embeddings_array.shape}")
        
        try:
            tsne = TSNE(
                n_components=self.n_components,
                perplexity=self.perplexity,
                random_state=self.random_state,
                n_iter=1000,
            )
            self.tsne_results = tsne.fit_transform(embeddings_array)
            logger.info(f"t-SNE fit successful. Output shape: {self.tsne_results.shape}")
            return self.tsne_results
        except Exception as e:
            logger.error(f"t-SNE fit failed: {e}")
            raise
    
    def plot_2d(
        self,
        embeddings: List[List[float]],
        documents: List[str] = None,
        metadatas: List[Dict[str, Any]] = None,
        color_by: str = None,
        title: str = "Document Embeddings (t-SNE 2D)",
        width: int = 1200,
        height: int = 800,
        save_path: str = None,
    ) -> go.Figure:
        """
        Create 2D t-SNE visualization.
        
        Args:
            embeddings: List of embedding vectors
            documents: Optional list of document texts
            metadatas: Optional list of metadata dicts
            color_by: Optional metadata key to color by
            title: Plot title
            width: Plot width
            height: Plot height
            save_path: Optional path to save HTML
            
        Returns:
            Plotly figure
        """
        logger.info("Creating 2D t-SNE visualization")
        
        # Transform embeddings
        coords = self.fit_transform(embeddings)
        
        # Prepare hover text
        hover_text = []
        if documents:
            for i, doc in enumerate(documents):
                text = f"<b>Doc {i}</b><br>{doc[:100]}..."
                if metadatas and i < len(metadatas):
                    for key, value in metadatas[i].items():
                        text += f"<br>{key}: {value}"
                hover_text.append(text)
        
        # Prepare colors
        colors = None
        if color_by and metadatas:
            try:
                colors = [meta.get(color_by, "unknown") for meta in metadatas]
            except Exception as e:
                logger.warning(f"Could not color by {color_by}: {e}")
        
        # Create figure
        fig = go.Figure(data=[
            go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                marker=dict(
                    size=8,
                    color=colors,
                    colorscale="Viridis" if color_by else None,
                    showscale=True if color_by else False,
                    line=dict(width=1, color="white"),
                ),
                text=hover_text if hover_text else [f"Doc {i}" for i in range(len(coords))],
                hoverinfo="text",
                hovertemplate="%{text}<extra></extra>",
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="t-SNE Component 1",
            yaxis_title="t-SNE Component 2",
            width=width,
            height=height,
            hovermode="closest",
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved 2D visualization to {save_path}")
        
        return fig
    
    def plot_3d(
        self,
        embeddings: List[List[float]],
        documents: List[str] = None,
        metadatas: List[Dict[str, Any]] = None,
        color_by: str = None,
        title: str = "Document Embeddings (t-SNE 3D)",
        width: int = 1200,
        height: int = 800,
        save_path: str = None,
    ) -> go.Figure:
        """
        Create 3D t-SNE visualization.
        
        Args:
            embeddings: List of embedding vectors
            documents: Optional list of document texts
            metadatas: Optional list of metadata dicts
            color_by: Optional metadata key to color by
            title: Plot title
            width: Plot width
            height: Plot height
            save_path: Optional path to save HTML
            
        Returns:
            Plotly figure
        """
        if self.n_components != 3:
            raise ValueError("n_components must be 3 for 3D plot")
        
        logger.info("Creating 3D t-SNE visualization")
        
        # Transform embeddings
        coords = self.fit_transform(embeddings)
        
        # Prepare hover text
        hover_text = []
        if documents:
            for i, doc in enumerate(documents):
                text = f"<b>Doc {i}</b><br>{doc[:100]}..."
                if metadatas and i < len(metadatas):
                    for key, value in metadatas[i].items():
                        text += f"<br>{key}: {value}"
                hover_text.append(text)
        
        # Prepare colors
        colors = None
        if color_by and metadatas:
            try:
                colors = [meta.get(color_by, "unknown") for meta in metadatas]
            except Exception as e:
                logger.warning(f"Could not color by {color_by}: {e}")
        
        # Create figure
        fig = go.Figure(data=[
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker=dict(
                    size=6,
                    color=colors,
                    colorscale="Viridis" if color_by else None,
                    showscale=True if color_by else False,
                    line=dict(width=1, color="white"),
                ),
                text=hover_text if hover_text else [f"Doc {i}" for i in range(len(coords))],
                hoverinfo="text",
                hovertemplate="%{text}<extra></extra>",
            )
        ])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="t-SNE Component 1",
                yaxis_title="t-SNE Component 2",
                zaxis_title="t-SNE Component 3",
            ),
            width=width,
            height=height,
            hovermode="closest",
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved 3D visualization to {save_path}")
        
        return fig


def visualize_embeddings(
    embeddings: List[List[float]],
    documents: List[str] = None,
    metadatas: List[Dict[str, Any]] = None,
    dimensions: int = 3,
    save_path: str = None,
) -> go.Figure:
    """
    Convenience function to quickly visualize embeddings.
    
    Args:
        embeddings: List of embedding vectors
        documents: Optional document texts
        metadatas: Optional metadata
        dimensions: 2 or 3
        save_path: Optional save path
        
    Returns:
        Plotly figure
        
    Example:
        >>> from core.vectorstore.chroma_store import ChromaVectorStore
        >>> store = ChromaVectorStore()
        >>> embeddings, ids, docs, metadata = store.get_all_embeddings()
        >>> fig = visualize_embeddings(embeddings, docs, metadata, dimensions=3)
        >>> fig.show()
    """
    visualizer = TSNEVisualizer(n_components=dimensions)
    
    if dimensions == 2:
        return visualizer.plot_2d(embeddings, documents, metadatas, save_path=save_path)
    else:
        return visualizer.plot_3d(embeddings, documents, metadatas, save_path=save_path)
