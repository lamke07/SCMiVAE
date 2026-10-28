import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import umap
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset

from .utils_analysis import (
    plot_embedding_pairplot,
    plot_umap_embeddings,
    plot_umap_with_highlighted_clusters,
)

def _optional_str_seq(x: Optional[Union[str, Sequence[str]]]) -> List[str]:
    """Normalize ``None`` / str / sequence to a list of strings."""
    if x is None:
        return []
    return [x] if isinstance(x, str) else list(x)


# --- Noise conversion functions ---
def to_noise(x, mu, var, eps=1e-8, device: Optional[Union[str, torch.device]] = None):
    """
    Convert x to normalized noise: (x - mu) / sigma
    var: variance (sigma^2), so we take sqrt
    """
    if device is not None:
        device = torch.device(device)
        x = x.to(device)
        mu = mu.to(device)
        var = var.to(device)
    sigma = torch.sqrt(var + eps)
    noise = (x - mu) / sigma
    return noise


def from_noise(
    eps,
    mu,
    var,
    eps_stability=1e-8,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Reconstruct x from noise: mu + sigma * eps
    var: variance (sigma^2), so we take sqrt
    """
    if device is not None:
        device = torch.device(device)
        eps = eps.to(device)
        mu = mu.to(device)
        var = var.to(device)
    sigma = torch.sqrt(var + eps_stability)
    x = mu + sigma * eps
    return x


# --- Stack mean and var pairs (for encoding processing) ---
def _stack_mean_var_pairs(pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[np.ndarray, np.ndarray]:
    """Stack (mean, var) tensors from one or more encoder/prior heads into numpy arrays."""
    if len(pairs) > 1:
        mean = torch.cat([p[0].cpu() for p in pairs], dim=1).numpy()
        var = torch.cat([p[1].cpu() for p in pairs], dim=1).numpy()
    else:
        mean = pairs[0][0].cpu().numpy()
        var = pairs[0][1].cpu().numpy()
    return mean, var


# -- Inferencer Class ------
class VAEInference:
    """
    VAE inference, covariate / counterfactual variants, downstream analysis, and plots.

    Method groups (in source order):
    - Core pass: dataloader, encode/decode into cached embeddings and priors
    - Dataset variants: batch and manual covariate edits, then re-inference
    - Counterfactual latents under a prior shift
    - PCA / UMAP on embeddings
    - Clustering (Leiden, KMeans)
    - Access and cache: get subsets of embedding columns, retrieve reductions, delete caches
    - Comparative latent shift metrics
    - Figures: pairplot, UMAP, composition bars (single key or many keys)
    """
    
    def __init__(
        self,
        trainer,
        dataset,
        batch_size: int = 512,
        num_workers: int = 0,
        device: str = 'cpu',
        run_inference: bool = False
    ):
        """
        Initialize the VAE inference handler.
        
        Parameters
        ----------
        trainer : VAETrainer
            Trained VAE model trainer object with decode_ivae and infer_ivae methods.
        dataset : torch.utils.data.Dataset
            Original dataset (e.g., scData) with cat_df_tensor attribute.
        batch_size : int, optional
            Batch size for inference.
        num_workers : int, optional
            Number of data loading workers.
        device : str, optional
            Device for inference ('cpu', 'cuda', 'mps').
        """
        self.trainer = trainer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.trainer.model.to(self.device)

        self.original_dataset = dataset
        self.original_dataset_cat_cols = [f"{x}_cat" for x in dataset.cat_cols]
        self.original_dataset_cat = pd.concat([
            dataset.cat_df.reset_index(drop=True),
            pd.DataFrame(dataset.cat_df_tensor.detach().numpy(), columns=self.original_dataset_cat_cols),
        ], axis=1)
        
        # Storage for datasets (original and corrected versions)
        self.datasets: Dict[str, Any] = {'default': dataset}
        
        # Storage for embeddings and priors
        self.embeddings: Dict[str, np.ndarray] = {}
        self.priors: Dict[str, np.ndarray] = {}
        self.decodings: Dict[str, Any] = {}
        self.embeddings_var: Dict[str, np.ndarray] = {}
        self.priors_var: Dict[str, np.ndarray] = {}
        
        # Storage for dimensionality reductions
        # Keys follow pattern: "{dataset_key}_pca" or "{dataset_key}_pca_umap"
        self.pca_embeddings: Dict[str, np.ndarray] = {}
        self.umap_embeddings: Dict[str, np.ndarray] = {}
        self.pca_transformers: Dict[str, PCA] = {}

        # Storage for Leiden and KMeans cluster labels; keys like "{embedding_key}_leiden" /  "{embedding_key}_kmeans"
        self.leiden_labels: Dict[str, np.ndarray] = {}
        self.kmeans_labels: Dict[str, np.ndarray] = {}
        
        # Run inference on the default dataset (produce embeddings, priors, and decodings)
        self._run_inference('default')

        if dataset.cat_df is not None:
            self.add_batch_correction(correction_key='corrected')
        if dataset.cat_df is not None and dataset.cont_df is not None:
            self.add_batch_correction(correction_key='corrected_catcont', one_continuous_column=True)
    
    def _create_dataloader(
        self,
        dataset_key: str = 'default',
        sample_indices: Optional[Sequence[int]] = None,
    ) -> DataLoader:
        """Create a DataLoader for the specified dataset."""
        dataset = self.datasets[dataset_key]
        if sample_indices is not None:
            dataset = Subset(dataset, list(sample_indices))
        return DataLoader(
            dataset=dataset,
            shuffle=False,
            drop_last=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    # --- Core: full encode/decode pass into caches ---
    def _run_inference(self, dataset_key: str = 'default'):
        """
        Run VAE inference on the specified dataset.
        
        Parameters
        ----------
        dataset_key : str
            Key identifying which dataset to use.
        """
        loader = self._create_dataloader(dataset_key)
        
        # Run inference
        decodings = self.trainer.decode_ivae(loader, device=self.device)
        embeddings_list, z, prior_list = self.trainer.infer_ivae(
            loader, device=self.device
        )
        
        embeddings, embeddings_var = _stack_mean_var_pairs(embeddings_list)
        priors, priors_var = _stack_mean_var_pairs(prior_list)
        
        # Store results
        self.embeddings[dataset_key] = embeddings
        self.embeddings_var[dataset_key] = embeddings_var

        self.priors[dataset_key] = priors
        self.priors_var[dataset_key] = priors_var

        self.decodings[dataset_key] = decodings

    # --- Simple and manual interventions (corrections) ---
    def add_batch_correction(
        self,
        correction_key: str = 'corrected',
        zero_out_categoricals: bool = True,
        one_continuous_column: bool = False,
    ):
        """
        Create a batch-corrected version of the dataset.
        
        Parameters
        ----------
        correction_key : str
            Key to store the corrected dataset under.
        zero_out_categoricals : bool
            If True, zero out the categorical covariates (batch correction).
        """
        # Create a copy of the original dataset
        corrected_dataset = copy.deepcopy(self.original_dataset)
        
        if zero_out_categoricals:
            # Zero out categorical covariates for batch correction
            corrected_dataset.cat_df_tensor = torch.zeros_like(corrected_dataset.cat_df_tensor)

        if one_continuous_column:
            corrected_dataset.cont_df_tensor = torch.ones_like(corrected_dataset.cont_df_tensor)    
        
        # Store the corrected dataset and run inference
        self.datasets[correction_key] = corrected_dataset
        self._run_inference(correction_key)

    def add_manual_correction(
        self,
        correction_key: str = None,
        cat_target: List = None,
        cat_vals: List[int] = None,
        cont_cols: List[int] = None,
        cont_vals: List[float] = None,
        mode: str = 'align',
    ):
        """
        Create a batch-corrected version of the dataset.
        
        Parameters
        ----------
        cat_target: List[str] or List[int]
            List of categorical columns to target,
            i.e. which variation to keep (correction happens in remaining columns)
        cat_vals: List[int]
            List of values to set for the categorical columns when mode is 'override'
        cont_cols: List[int]
            List of indices of continuous columns to correct.
        cont_vals: List[float]
            List of values to set for the continuous columns.
        mode: str
            Mode of correction:
            'align': align the remaining columns to the groups of target columns (e.g. time point)
            'override': override the target columns with the values in cat_vals
        """

        # Create a copy of the original dataset
        tmp_dataset = copy.deepcopy(self.original_dataset)
        
        # categorical correction
        if cat_target is not None and mode == 'align':
            cat_cols = self.original_dataset_cat_cols
            tmp_df = self.original_dataset_cat.copy()
            tmp_df[cat_cols] = (
                tmp_df
                .groupby(cat_target)[cat_cols]
                .transform("first")
            )
            tmp_dataset.cat_df_tensor = torch.tensor(tmp_df[cat_cols].values)

        if cat_target is not None and mode == 'override' and cat_vals is not None:
            tmp_dataset.cat_df_tensor[:, cat_target] = (
                torch.tensor(
                    cat_vals,
                    dtype=tmp_dataset.cat_df_tensor.dtype,
                    device=tmp_dataset.cat_df_tensor.device
                )
            )

        # continuous correction (requires cont_vals to be provided)
        if cont_cols is not None and cont_vals is not None:
            tmp_dataset.cont_df_tensor[:, cont_cols] = (
                torch.tensor(
                    cont_vals,
                    dtype=tmp_dataset.cont_df_tensor.dtype,
                    device=tmp_dataset.cont_df_tensor.device
                )
            )


        if cat_target is not None and mode == 'align':
            cat_str = '-'.join(cat_target) if len(cat_target) > 1 else cat_target[0]
        elif cat_target is not None and mode == 'override' and cat_vals is not None:
            cat_str = f"{cat_target}-{cat_vals}"
        if cont_cols is not None:
            cont_str = '-'.join(cont_cols) + f"_{cont_vals}" if len(cont_cols) > 1 else f"{cont_cols[0]}-{cont_vals}"
        
        if correction_key is None:
            if cat_target is not None:
                if cont_cols is None:
                    correction_key = f"corrected_{mode}_{cat_str}"
                else:
                    correction_key = f"corrected_{mode}_{cat_str}_{cont_str}"
            else:
                if cont_cols is not None:
                    correction_key = f"corrected_{mode}_{cont_str}"
                else:
                    correction_key = f"corrected_{mode}"
        
        # Store the corrected dataset and run inference
        self.datasets[correction_key] = tmp_dataset
        self._run_inference(correction_key)

        return tmp_dataset

    # --- Counterfactual latents (prior shift, preserved noise) ---
    def run_counterfactual_inference(
        self,
        dataset_key: str,
        dataset_ref: str = 'default',
        new_dataset_key: str = None,
        n_samples: Optional[int] = None,
        sample_indices: Optional[Sequence[int]] = None,
    ):
        """
        Counterfactual inference under a prior shift: keep latent and observation noise,
        swap in the prior from ``dataset_key`` (intervened covariates).

        Uses the encoder posterior on ``dataset_ref`` as the factual latent. If
        ``n_samples`` is set, draws that many samples per cell from that posterior;
        otherwise uses the stored posterior mean (point estimate).

        Parameters
        ----------
        dataset_key : str
            Dataset whose prior defines the counterfactual (e.g. different batch covariates).
        dataset_ref : str
            Reference dataset whose encoder posterior and observations supply the noise.
        new_dataset_key : str
            Key to store counterfactual latents and decodings under.
        n_samples : Optional[int]
            If set, number of posterior samples per cell before the noise transform; if
            None, use posterior means only.
        sample_indices : Optional[Sequence[int]]
            If set, restrict to these cell indices (must be valid for both datasets).
        """

        if dataset_key == dataset_ref:
            raise ValueError(f"dataset_key and dataset_ref cannot be the same: {dataset_key} == {dataset_ref}")

        index_tensor = None
        if sample_indices is not None:
            index_tensor = torch.as_tensor(sample_indices, dtype=torch.long)
            if index_tensor.numel() == 0:
                raise ValueError("sample_indices must contain at least one index")

            n_dataset = len(self.datasets[dataset_key])
            n_ref = len(self.datasets[dataset_ref])
            if torch.any(index_tensor < 0):
                raise ValueError("sample_indices must be non-negative")
            if torch.any(index_tensor >= n_dataset):
                raise ValueError(
                    f"sample_indices contains values outside dataset '{dataset_key}' "
                    f"(size={n_dataset})"
                )
            if torch.any(index_tensor >= n_ref):
                raise ValueError(
                    f"sample_indices contains values outside dataset '{dataset_ref}' "
                    f"(size={n_ref})"
                )

        embeddings_ref, embeddings_var_ref, priors_ref, priors_var_ref = (
            torch.tensor(self.embeddings[dataset_ref], device=self.device),
            torch.tensor(self.embeddings_var[dataset_ref], device=self.device),
            torch.tensor(self.priors[dataset_ref], device=self.device),
            torch.tensor(self.priors_var[dataset_ref], device=self.device)
        )

        if index_tensor is not None:
            index_tensor = index_tensor.to(self.device)
            embeddings_ref = embeddings_ref[index_tensor]
            embeddings_var_ref = embeddings_var_ref[index_tensor]
            priors_ref = priors_ref[index_tensor]
            priors_var_ref = priors_var_ref[index_tensor]
            x_ref = self.datasets[dataset_ref].gene_expression[index_tensor].to(self.device)
        else:
            x_ref = self.datasets[dataset_ref].gene_expression.to(self.device)

        loader = self._create_dataloader(dataset_key, sample_indices=sample_indices)
        _, _, prior_list = self.trainer.infer_ivae(
            loader, device=self.device
        )
        priors_new = torch.cat([prior[0].to(self.device) for prior in prior_list], dim=1)
        priors_var_new = torch.cat([prior[1].to(self.device) for prior in prior_list], dim=1)

        if n_samples is not None:
            embeddings_ref = self.trainer.model.encoder_dist.sample(
                mu=embeddings_ref,
                v=embeddings_var_ref,
                n_samples=n_samples,
            )

        x_decoded = self.trainer.decode_from_latent(loader, embeddings_ref, device=self.device)
        noise_x = to_noise(x=x_ref, mu=x_decoded[0], var=x_decoded[1], device=self.device)
        noise_z = to_noise(x=embeddings_ref, mu=priors_ref, var=priors_var_ref, device=self.device)

        z_cf = from_noise(eps=noise_z, mu=priors_new, var=priors_var_new, device=self.device)
        x_cf_decoded = self.trainer.decode_from_latent(loader, z_cf, device=self.device)
        x_cf = from_noise(eps=noise_x, mu=x_cf_decoded[0], var=x_cf_decoded[1], device=self.device)

        if new_dataset_key is None:
            new_dataset_key = f"{dataset_key}_ViaPrior"

        self.embeddings[new_dataset_key] = z_cf.cpu().numpy()
        self.decodings[new_dataset_key] = x_cf.cpu().numpy()

    # --- Dimensionality reduction (embeddings → PCA / UMAP) ---
    def compute_pca(
        self,
        dataset_key: str = 'default',
        n_components: Union[str, int] = 'full',
    ) -> np.ndarray:
        """
        Compute PCA on embeddings.
        
        Parameters
        ----------
        dataset_key : str
            Which embedding set to use.
        n_components : 'full' or int
            Number of PCA components. If 'full', use all components.
        
        Returns
        -------
        np.ndarray
            PCA-transformed embeddings.
        """
        embeddings = self.embeddings[dataset_key]
        
        if n_components == 'full':
            n_components = min(embeddings.shape[0], embeddings.shape[1])
        elif isinstance(n_components, int):
            n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
        else:
            raise ValueError(f"n_components must be 'full' or int, got {n_components}")
        
        pca = PCA(n_components=n_components)
        pca_embeddings = pca.fit_transform(embeddings)
        
        # Store results
        pca_key = f"{dataset_key}_pca"
        self.pca_embeddings[pca_key] = pca_embeddings
        self.pca_transformers[pca_key] = pca
        
        return pca_embeddings
    
    def compute_umap(
        self,
        dataset_key: str = 'default',
        use_pca: bool = True,
        pca_components: Union[str, int] = 'full',
        umap_params: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Compute UMAP on embeddings (optionally after PCA).
        
        Parameters
        ----------
        dataset_key : str
            Which embedding set to use.
        use_pca : bool
            Whether to apply PCA before UMAP.
        pca_components : 'full' or int
            Number of PCA components if use_pca is True.
        umap_params : dict, optional
            Parameters to pass to UMAP constructor.
        
        Returns
        -------
        np.ndarray
            UMAP coordinates (n_samples, 2).
        """
        if umap_params is None:
            umap_params = {}
        
        # Determine input for UMAP and construct appropriate key
        if use_pca:
            pca_key = f"{dataset_key}_pca"
            if pca_key not in self.pca_embeddings:
                self.compute_pca(dataset_key, n_components=pca_components)
            X = self.pca_embeddings[pca_key]
            # Use naming convention that shows PCA was applied before UMAP
            umap_key = f"{dataset_key}_pca_umap"
        else:
            X = self.embeddings[dataset_key]
            umap_key = f"{dataset_key}_umap"
        
        # Compute UMAP
        reducer = umap.UMAP(**umap_params)
        umap_coords = reducer.fit_transform(X)
        
        # Store results
        self.umap_embeddings[umap_key] = umap_coords

        return umap_coords

    # --- Cached embeddings: lookup, column subset, eviction ---
    def get_embedding(
        self,
        dataset_key: str = 'default',
        reduction: Optional[str] = None,
    ) -> np.ndarray:
        """
        Retrieve embeddings with optional dimensionality reduction.
        
        Parameters
        ----------
        dataset_key : str
            Which dataset's embeddings to use.
        reduction : str, optional
            Type of reduction: None (raw), 'pca', 'umap', or 'pca_umap'.
        
        Returns
        -------
        np.ndarray
            Requested embeddings.
        """
        if reduction is None:
            return self.embeddings[dataset_key]
        if reduction == 'pca':
            pca_key = f"{dataset_key}_pca"
            if pca_key not in self.pca_embeddings:
                raise ValueError(
                    f"PCA not computed for {dataset_key}. "
                    f"Call compute_pca('{dataset_key}') first."
                )
            return self.pca_embeddings[pca_key]
        if reduction in ('umap', 'pca_umap'):
            umap_key = f"{dataset_key}_pca_umap" if reduction == 'pca_umap' else f"{dataset_key}_umap"
            hint = (
                f"Call compute_umap('{dataset_key}', use_pca=True) first."
                if reduction == 'pca_umap'
                else f"Call compute_umap('{dataset_key}', use_pca=False) first."
            )
            if umap_key not in self.umap_embeddings:
                raise ValueError(
                    f"{reduction} not computed for {dataset_key}. {hint}"
                )
            return self.umap_embeddings[umap_key]
        raise ValueError(
            f"Unknown reduction type: {reduction}. "
            f"Use None, 'pca', 'umap', or 'pca_umap'."
        )

    def delete_embeddings(self, dataset_key: str) -> None:
        """
        Delete all embeddings and associated cached data for a dataset key.

        Removes raw embeddings, priors, decodings, and any derived PCA/UMAP
        entries for the given key. If the key is not 'default', the dataset
        entry is also removed from `self.datasets`.

        Parameters
        ----------
        dataset_key : str
            Key identifying the dataset whose embeddings to delete (e.g. 'default', 'corrected').
        """
        deleted = []
        for name, d in (
            ('embeddings', self.embeddings),
            ('priors', self.priors),
            ('decodings', self.decodings),
        ):
            if dataset_key in d:
                d.pop(dataset_key)
                deleted.append(name)
        if dataset_key != 'default' and dataset_key in self.datasets:
            self.datasets.pop(dataset_key)
            deleted.append('dataset')
        pca_key = f"{dataset_key}_pca"
        if pca_key in self.pca_embeddings:
            self.pca_embeddings.pop(pca_key)
            deleted.append('pca_embeddings')
        if pca_key in self.pca_transformers:
            self.pca_transformers.pop(pca_key)
            deleted.append('pca_transformers')
        for umap_key in (f"{dataset_key}_umap", f"{dataset_key}_pca_umap"):
            if umap_key in self.umap_embeddings:
                self.umap_embeddings.pop(umap_key)
                deleted.append(umap_key)
        leiden_key = f"{dataset_key}_leiden"
        if leiden_key in self.leiden_labels:
            self.leiden_labels.pop(leiden_key)
            deleted.append('leiden_labels')
        kmeans_key = f"{dataset_key}_kmeans"
        if kmeans_key in self.kmeans_labels:
            self.kmeans_labels.pop(kmeans_key)
            deleted.append('kmeans_labels')
        if deleted:
            print(f"Deleted for dataset_key '{dataset_key}': {', '.join(deleted)}.")
        else:
            print(f"No cached data found for dataset_key '{dataset_key}'.")

    def _ensure_embeddings_umap(
        self,
        dataset_key: str,
        use_pca: bool,
        pca_components: Union[str, int],
        umap_params: Optional[dict],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ensure PCA (if requested) and UMAP exist; return raw embeddings and UMAP coordinates."""
        if use_pca:
            pca_key = f"{dataset_key}_pca"
            if pca_key not in self.pca_embeddings:
                print(f"Computing PCA for '{dataset_key}'...")
                self.compute_pca(dataset_key, n_components=pca_components)
            else:
                print(f"PCA already computed for '{dataset_key}'...")

        umap_key = f"{dataset_key}_pca_umap" if use_pca else f"{dataset_key}_umap"
        if umap_key not in self.umap_embeddings:
            print(f"Computing UMAP for '{dataset_key}' (use_pca={use_pca})...")
            self.compute_umap(
                dataset_key,
                use_pca=use_pca,
                pca_components=pca_components,
                umap_params=umap_params,
            )
        return self.embeddings[dataset_key], self.umap_embeddings[umap_key]

    def plot_all(
        self,
        samples: pd.DataFrame,
        plot_configs: List[Tuple[str, str]],
        output_dir: Path,
        dataset_key: str = 'default',
        use_pca: bool = False,
        pca_components: Union[str, int] = 'full',
        umap_params: Optional[dict] = None,
        **plot_kwargs,
    ):
        """
        Generate all plots for a given dataset.

        Parameters
        ----------
        samples : pd.DataFrame
            Metadata DataFrame with label columns.
        plot_configs : list of tuple
            List of (label_column, suffix) pairs for plotting.
            E.g., [('dataset', '_batch'), ('celltype', '_cell')]
        output_dir : Path
            Directory to save plots.
        dataset_key : str
            Which dataset's embeddings to plot.
        use_pca : bool
            Whether to use PCA reduction before UMAP.
        pca_components : 'full' or int
            Number of PCA components.
        umap_params : dict, optional
            UMAP parameters.
        **plot_kwargs
            Additional keyword arguments passed to plotting functions.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        effective_configs = list(plot_configs)
        samples_plot = samples.copy()

        embeddings, umap_coords = self._ensure_embeddings_umap(dataset_key, use_pca, pca_components, umap_params)
        base_kw = {'embeddings': embeddings, 'encoder_means': embeddings, 'pca': None, **plot_kwargs}

        for label_column, suffix in effective_configs:
            print(f"Generating plots for '{label_column}' with suffix '{suffix}'...")
            labels = samples_plot[label_column]

            # Pairplot
            plot_embedding_pairplot(
                **base_kw,
                labels=labels,
                label_title=label_column,
                output_path=output_dir / f'RawEmbeddings{suffix}.png',
            )

            # UMAP scatter
            plot_umap_embeddings(
                **base_kw,
                umap_coords=umap_coords,
                labels=labels,
                label_title=label_column,
                output_path=output_dir / f'UMAPEmbeddings{suffix}.png',
                umap_params=umap_params,
            )

            # UMAP highlighted clusters
            plot_umap_with_highlighted_clusters(
                **base_kw,
                umap_coords=umap_coords,
                labels=samples_plot,
                highlight_label=label_column,
                output_path=output_dir / f'UMAPHighlighted{suffix}.png',
                umap_params=umap_params,
            )

    def plot_embeddings_per_dataset(
        self,
        samples: pd.DataFrame,
        plot_configs: List[Tuple[str, str]],
        output_dir: Path,
        dataset_keys: Sequence[str] = ('default', 'corrected'),
        **kwargs,
    ):
        """
        Run :meth:`plot_all` for each ``dataset_key``, writing to ``output_dir / dataset_key``.

        Use this to compare embeddings across stored dataset variants (e.g. raw vs
        batch-corrected) without duplicating ``plot_all`` calls.
        """
        for dataset_key in dataset_keys:
            if dataset_key not in self.datasets:
                print(f"Warning: Dataset '{dataset_key}' not found, skipping.")
                continue
            print(f"\nProcessing dataset: '{dataset_key}'")
            self.plot_all(
                samples=samples,
                plot_configs=plot_configs,
                output_dir=Path(output_dir) / dataset_key,
                dataset_key=dataset_key,
                **kwargs,
            )
