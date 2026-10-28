import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse


def load_adata(config, root_dir, combine=True):
    """
    Load AnnData from config: single h5ad or per-patient subdirs, optionally concatenated.
    Returns (adata, keep) where keep is the HVG boolean mask (or Series) for genes.
    """
    base = root_dir / "data" / config["data_dir"]
    fname = config["file_name"]

    # ---- Most data sets ----
    if not config["per_patient_subdir"]:
        adata = ad.read_h5ad(base / fname)
        keep = adata.var[config["hvg_col"]] if config["hvg_col"] is not None else None
        return adata, keep

    # ---- datasets with subsets ----
    if combine:
        adatas = {
            p: ad.read_h5ad(base / p / fname)
            for p in config["patients"]
        }
        keep = np.logical_or.reduce([
            a.var[config["hvg_col"]].values
            for a in adatas.values()
        ])

        adata = ad.concat(adatas, label="patient", axis=0)
    else:
        # adata = adatas[p]
        p = config["subset"][0]
        adata = ad.read_h5ad(base / p / fname)
        keep = adata.var[config["hvg_col"]]

    return adata, keep


def apply_subset(adata, run_cfg, subsets_cfg):
    """
    Filter adata to samples listed in run_cfg["subset"] using subsets_cfg.
    run_cfg = data_cfg; subsets_cfg = DATASUBSETCONFIGS. Returns adata unchanged if no subset.
    """
    if not run_cfg.get("use_subset"):
        return adata

    all_samples = []

    for key in run_cfg["subset"]:
        sub = subsets_cfg[key]
        all_samples.extend(sub["samples"])
        obs_col = sub["obs_filter_col"]

    return adata[adata.obs[obs_col].isin(all_samples)]


def get_expression(adata, expr_cfg):
    """
    Extract expression matrix from adata: use X or a layer (expr_cfg["type"] and optional "key").
    If expr_cfg["force_dense"] is True, convert sparse to dense. Returns numpy array.
    """
    # ---- select matrix ----
    if expr_cfg["type"] == "layer":

        key = expr_cfg["key"]

        if key not in adata.layers:
            raise ValueError(f"Layer '{key}' not found")

        X = adata.layers[key]

    elif expr_cfg["type"] == "X":

        X = adata.X

    else:
        raise ValueError("expr_source.type must be 'layer' or 'X'")

    # ---- densify safely ----
    if expr_cfg.get("force_dense", False) and issparse(X):
        X = X.toarray()

    return np.asarray(X)


def build_gene_filter_mask(adata, data_cfg, keep=None):
    """
    Build a boolean gene filter mask from data_cfg["gene_filter"]: HVG (requires keep),
    exclude mitochondrial (MT-), exclude ribosomal (RPS/RPL/MRPS/MRPL). Returns a pandas Series.
    """
    gf = data_cfg.get("gene_filter", {})

    # ---- start with all True but indexed by gene names ----
    mask = pd.Series(
        True,
        index=adata.var_names,
        name="gene_filter"
    )

    # ---- HVG ----
    if gf.get("hvg", False):

        if keep is None:
            raise ValueError("HVG filtering requested but no HVG mask provided")

        # ensure keep is also a Series with same index
        if not isinstance(keep, pd.Series):
            keep = pd.Series(keep, index=adata.var_names)

        mask &= keep

    # ---- mito ----
    if gf.get("mito", False):
        mito_mask = adata.var_names.str.startswith("MT-")
        mask &= ~mito_mask

    # ---- ribo ----
    if gf.get("ribo", False):
        ribo_mask = (
            adata.var_names.str.startswith("RPS") |
            adata.var_names.str.startswith("RPL") |
            adata.var_names.str.startswith("MRPS") |
            adata.var_names.str.startswith("MRPL")
        )
        mask &= ~ribo_mask

    return mask