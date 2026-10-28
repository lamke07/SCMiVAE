import torch
from pathlib import Path
import math
import umap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from typing import Optional, Tuple

UMAP_PARAMS = {
    'default': {'n_components': 2, 'n_neighbors': 15, 'min_dist': 0.1, 'random_state': 53},
    'manual':  {'n_components': 2, 'n_neighbors': 25, 'min_dist': 0.35, 'random_state': 41}
}


# --- PCA scree plot ---
def plot_pca_screeplot(
    pca_transformer: PCA,
    output_path: Optional[Path] = None,
    n_components_show: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 5),
    dpi: int = 150,
) -> None:
    """
    Plot PCA scree plot showing explained variance.
    
    Parameters
    ----------
    pca_transformer : PCA
        Fitted sklearn PCA object.
    output_path : Path, optional
        Path to save the figure.
    n_components_show : int, optional
        Number of components to show. If None, show all.
    figsize : tuple
        Figure size.
    dpi : int
        Resolution.
    """
    explained_var = pca_transformer.explained_variance_ratio_
    cumsum_var = np.cumsum(explained_var)
    
    if n_components_show is None:
        n_components_show = len(explained_var)
    else:
        n_components_show = min(n_components_show, len(explained_var))
    
    explained_var = explained_var[:n_components_show]
    cumsum_var = cumsum_var[:n_components_show]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Explained variance per component
    ax1.bar(range(1, n_components_show + 1), explained_var * 100, 
            color='steelblue', alpha=0.7)
    ax1.set_xlabel('Principal Component', fontsize=11)
    ax1.set_ylabel('Explained Variance (%)', fontsize=11)
    ax1.set_title('PCA Scree Plot - Variance per Component', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add elbow detection heuristic (optional)
    # Find "elbow" using second derivative
    if len(explained_var) > 2:
        second_deriv = np.diff(explained_var, n=2)
        elbow_idx = np.argmax(second_deriv) + 2
        if elbow_idx < n_components_show:
            ax1.axvline(elbow_idx, color='red', linestyle='--', linewidth=2,
                       alpha=0.7, label=f'Elbow at PC{elbow_idx}')
            ax1.legend()
    
    # Plot 2: Cumulative explained variance
    ax2.plot(range(1, n_components_show + 1), cumsum_var * 100, 
             marker='o', markersize=5, linewidth=2, color='steelblue')
    
    # Add reference lines
    for pct in [50, 75, 90, 95]:
        idx = np.searchsorted(cumsum_var * 100, pct)
        if idx < n_components_show:
            ax2.axhline(pct, color='gray', linestyle=':', alpha=0.3, linewidth=1)
            ax2.text(n_components_show * 0.95, pct + 1, f'{pct}%', 
                    fontsize=9, color='gray', ha='right')
            ax2.axvline(idx + 1, color='gray', linestyle=':', alpha=0.2, linewidth=1)
            ax2.text(idx + 1, 10, f'PC{idx+1}', fontsize=8, color='gray', 
                    rotation=90, va='bottom')
    
    ax2.set_xlabel('Number of Components', fontsize=11)
    ax2.set_ylabel('Cumulative Explained Variance (%)', fontsize=11)
    ax2.set_title('PCA Scree Plot - Cumulative Variance', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 105])
    
    plt.tight_layout()
    
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved PCA scree plot to {output_path}")
    
    # plt.show()
    plt.close(fig)


# --- Main Embedding plots ------
def plot_embedding_pairplot(
        embeddings: np.ndarray,
        labels,
        output_path: Path,
        *,
        pca=None,
        label_title = None,
        figsize=(8, 8),
        alpha=0.3,
        s=5,
        dpi=120,
        **kwargs
    ):
    """
    Create and save a Seaborn pairplot of embeddings colored by label.

    Parameters
    ----------
    embeddings : np.ndarray
        Array of shape (n_samples, n_features).
    labels : array-like
        Labels corresponding to each embedding.
    output_path : pathlib.Path
        Path where the figure will be saved.
    pca : None, 'full', or int, optional
        If None, no PCA preprocessing. If 'full', use all PCA components.
        If int, use that many PCA components.
    figsize : tuple, optional
        Figure size in inches.
    alpha : float, optional
        Point transparency.
    dpi : int, optional
        Resolution of the saved figure.

    Returns
    -------
    tuple
        (embeddings_pca, None) where embeddings_pca is the PCA-transformed
        embeddings (or original if pca=None), and None since no UMAP is performed.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to numpy if needed
    if hasattr(embeddings, "detach"):
        X = embeddings.detach().cpu().numpy()
    else:
        X = np.asarray(embeddings)

    # Apply PCA if requested
    if pca is not None:
        if pca == 'full':
            n_components = min(X.shape[0], X.shape[1])
        elif isinstance(pca, int):
            n_components = min(pca, X.shape[0], X.shape[1])
        else:
            raise ValueError(f"pca must be None, 'full', or an int, got {pca}")
        
        pca_transformer = PCA(n_components=n_components)
        X = pca_transformer.fit_transform(X)
        embeddings_pca = X
    else:
        embeddings_pca = X

    df = pd.DataFrame(embeddings_pca)
    df["label"] = np.asarray(labels)

    pairgrid = sns.pairplot(
        data=df,
        hue="label",
        palette="deep",
        corner=True,
        plot_kws={
            "alpha": alpha,
            "s": s,
            "edgecolor": "none",
            "linewidth": 0,
        },
    )
    # Remove axes clutter
    for ax in pairgrid.axes.flatten():
        if ax is not None:
            ax.set_xticks([])
            ax.set_yticks([])

    pairgrid.figure.set_size_inches(*figsize)
    pairgrid.figure.suptitle(label_title)
    pairgrid.figure.savefig(output_path, dpi=dpi, bbox_inches="tight")

    plt.close(pairgrid.figure)
    return embeddings_pca


def plot_umap_embeddings(
        encoder_means,
        labels,
        output_path: Path,
        umap_params: dict = UMAP_PARAMS['default'],
        *,
        pca=None,
        umap_coords=None,
        label_title = None,
        figsize=(8, 8),
        point_size=10,
        alpha=0.4,
        dpi=240,
        **kwargs
    ):
    """
    Create and save a UMAP scatter plot of encoder means colored by label.

    Parameters
    ----------
    encoder_means : torch.Tensor or np.ndarray
        Encoder output of shape (n_samples, n_features).
    labels : array-like
        Labels corresponding to each sample.
    output_path : pathlib.Path
        File path where the figure will be saved.
    umap_params : dict
        Parameters passed to umap.UMAP (used only when umap_coords is None).
    pca : None, 'full', or int, optional
        If None, no PCA preprocessing. If 'full', use all PCA components.
        If int, use that many PCA components. Applied before UMAP.
    umap_coords : array-like, optional
        Precomputed UMAP coordinates of shape (n_samples, 2). If provided, UMAP
        is skipped and these coordinates are used for plotting.
    figsize : tuple, optional
        Figure size in inches.
    point_size : int, optional
        Marker size.
    alpha : float, optional
        Marker transparency.
    dpi : int, optional
        Resolution of the saved figure.
    title : str, optional
        Plot title.

    Returns
    -------
    tuple
        (embeddings_pca, umap_coords) where embeddings_pca is the PCA-transformed
        embeddings (or original if pca=None), and umap_coords are the UMAP coordinates.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to numpy if needed
    if hasattr(encoder_means, "detach"):
        X = encoder_means.detach().cpu().numpy()
    else:
        X = np.asarray(encoder_means)

    # Apply PCA if requested
    if pca is not None:
        if pca == 'full':
            n_components = min(X.shape[0], X.shape[1])
        elif isinstance(pca, int):
            n_components = min(pca, X.shape[0], X.shape[1])
        else:
            raise ValueError(f"pca must be None, 'full', or an int, got {pca}")
        
        pca_transformer = PCA(n_components=n_components)
        X = pca_transformer.fit_transform(X)
        embeddings_pca = X
    else:
        embeddings_pca = X

    if umap_coords is not None:
        embedding = np.asarray(umap_coords)
    else:
        reducer = umap.UMAP(**umap_params)
        embedding = reducer.fit_transform(X)

    df = pd.DataFrame(
        embedding, columns=["umap_1", "umap_2"]
    )
    df["label"] = np.asarray(labels)

    unique_labels = sorted(pd.Series(df["label"]).dropna().unique(), key=str)
    n_labels = len(unique_labels)
    palette = dict(
        zip(unique_labels, sns.color_palette("deep", n_colors=n_labels))
    )

    fig, ax = plt.subplots(figsize=figsize)

    sns.scatterplot(
        data=df,
        x="umap_1",
        y="umap_2",
        hue="label",
        palette=palette,
        hue_order=unique_labels,
        s=point_size,
        alpha=alpha,
        edgecolor="none",
        linewidth=0,
        ax=ax,
    )

    # Remove axis clutter for maximum plot area
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(label_title)

    # Clean legend (match plot_umap_with_highlighted_clusters)
    if ax.legend_ is not None:
        ax.legend_.set_frame_on(False)

    ax.legend(
        title="Label",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return embeddings_pca, embedding


def plot_umap_with_highlighted_clusters(
        encoder_means,
        labels,
        highlight_label: str,
        output_path: Path,
        umap_params: dict,
        *,
        pca=None,
        umap_coords=None,
        ncols: int = 4,
        figsize_per_plot: float = 5.5,
        point_size: int = 10,
        alpha_all: float = 0.05,
        alpha_highlight: float = 0.6,
        dpi: int = 240,
        **kwargs
):
    """
    Plot a UMAP embedding with one global view and individual highlighted clusters.

    Parameters
    ----------
    encoder_means : torch.Tensor or np.ndarray
        Embeddings of shape (n_samples, n_features).
    labels : pandas.DataFrame
        DataFrame containing label columns.
    highlight_label : str
        Column name in `labels` to highlight (e.g. 'Clone').
    output_path : pathlib.Path
        Path to save the figure.
    umap_params : dict
        Parameters passed to umap.UMAP (used only when umap_coords is None).
    pca : None, 'full', or int, optional
        If None, no PCA preprocessing. If 'full', use all PCA components.
        If int, use that many PCA components. Applied before UMAP.
    umap_coords : array-like, optional
        Precomputed UMAP coordinates of shape (n_samples, 2). If provided, UMAP
        is skipped and these coordinates are used for plotting.
    ncols : int, optional
        Number of subplot columns.
    figsize_per_plot : float, optional
        Size scaling for each subplot.
    point_size : int, optional
        Scatter marker size.
    alpha_all : float, optional
        Alpha for non-highlighted (grey) points.
    alpha_highlight : float, optional
        Alpha for highlighted cluster points.
    dpi : int, optional
        Output resolution.

    Returns
    -------
    tuple
        (embeddings_pca, umap_coords) where embeddings_pca is the PCA-transformed
        embeddings (or original if pca=None), and umap_coords are the UMAP coordinates.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert embeddings to numpy
    if hasattr(encoder_means, "detach"):
        X = encoder_means.detach().cpu().numpy()
    else:
        X = np.asarray(encoder_means)

    # Apply PCA if requested
    if pca is not None:
        if pca == 'full':
            n_components = min(X.shape[0], X.shape[1])
        elif isinstance(pca, int):
            n_components = min(pca, X.shape[0], X.shape[1])
        else:
            raise ValueError(f"pca must be None, 'full', or an int, got {pca}")
        
        pca_transformer = PCA(n_components=n_components)
        X = pca_transformer.fit_transform(X)
        embeddings_pca = X
    else:
        embeddings_pca = X

    if umap_coords is not None:
        embedding = np.asarray(umap_coords)
    else:
        reducer = umap.UMAP(**umap_params)
        embedding = reducer.fit_transform(X)

    df = pd.DataFrame(embedding, columns=["umap_1", "umap_2"])
    df["label"] = labels[highlight_label].values

    unique_labels = sorted(pd.Series(df["label"]).dropna().unique(), key=str)
    n_labels = len(unique_labels)
    palette = dict(
        zip(unique_labels, sns.color_palette("deep", n_colors=n_labels))
    )
    n_plots = 1 + n_labels
    nrows = int(np.ceil(n_plots / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * figsize_per_plot, nrows * figsize_per_plot),
    )

    axes = np.asarray(axes).reshape(-1)

    # ---------- Plot 0: full UMAP ----------
    ax = axes[0]
    sns.scatterplot(
        data=df,
        x="umap_1",
        y="umap_2",
        hue="label",
        palette=palette,
        hue_order=unique_labels,
        s=point_size,
        alpha=0.5,
        edgecolor="none",
        linewidth=0,
        ax=ax,
    )

    ax.set_title("UMAP – all labels")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(
        title="Label",
        loc="upper right",
        frameon=False,
    )

    # ---------- Individual highlighted clusters ----------

    for i, label_value in enumerate(unique_labels, start=1):
        ax = axes[i]

        mask = df["label"] == label_value

        # Grey background
        sns.scatterplot(
            data=df.loc[~mask],
            x="umap_1",
            y="umap_2",
            color="lightgray",
            s=point_size,
            alpha=alpha_all,
            edgecolor="none",
            linewidth=0,
            ax=ax,
        )

        # Highlighted cluster
        sns.scatterplot(
            data=df.loc[mask],
            x="umap_1",
            y="umap_2",
            color=palette[label_value],
            s=point_size,
            alpha=alpha_highlight,
            edgecolor="none",
            linewidth=0,
            ax=ax,
        )

        ax.set_title(f"UMAP: {label_value}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.legend([], [], frameon=False)

    # Remove unused axes
    for ax in axes[n_plots:]:
        fig.delaxes(ax)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return embeddings_pca, embedding

# -- decoding plots ------
def unflatten_to_square(x, pad_value=0.0):
    """
    Unflatten a 1D tensor/array into the smallest possible square.
    Pads with pad_value if needed.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    n = x.shape[0]
    side = int(math.ceil(math.sqrt(n)))
    total = side * side

    if total > n:
        pad = np.full(total - n, pad_value, dtype=x.dtype)
        x = np.concatenate([x, pad])

    return x.reshape(side, side)


def unflatten_to_rows(x, num_rows: int, pad_value=0.0):
    """
    Unflatten a 1D tensor/array into a 2D array with a fixed number of columns.
    Pads with pad_value if needed.

    Args:
        x: 1D torch.Tensor or numpy.ndarray
        row_size: number of columns (e.g. 28 for MNIST)
        pad_value: value used for padding

    Returns:
        2D numpy.ndarray of shape (num_rows, row_size)
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    n = x.shape[0]
    row_size = int(math.ceil(n / num_rows))
    total = row_size * num_rows

    if total > n:
        pad = np.full(total - n, pad_value, dtype=x.dtype)
        x = np.concatenate([x, pad])

    return x.reshape(num_rows, row_size)


