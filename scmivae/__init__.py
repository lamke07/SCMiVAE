"""
SCMiVAE: Sturctural causal modelling and identifiable Variational Autoencoders for single-cell data
"""

__version__ = "0.1.1"
__author__ = "Kevin Lam"
__email__ = "kevin.lam@stat.ubc.ca"

from .vae_models import iVAE, VAEBase
from .vae_trainers import ModelTrainerIVAE, ModelTrainerBase
from .utils_data import CountDataset
from .utils_analysis import plot_pca_screeplot, plot_embedding_pairplot, plot_umap_embeddings, plot_umap_with_highlighted_clusters
from .inferencer import VAEInference


__all__ = [
    "iVAE",
    "VAEBase",
    "ModelTrainerIVAE",
    "ModelTrainerBase",
    "CountDataset",
    "VAEInference",
    "plot_pca_screeplot",
    "plot_embedding_pairplot",
    "plot_umap_embeddings",
    "plot_umap_with_highlighted_clusters",
]
