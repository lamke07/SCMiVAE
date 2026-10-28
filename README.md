# SCMiVAE

Structural causal modelling and identifiable variational autoencoders (iVAE) for single-cell data, built on PyTorch. The prior over the latent space is conditioned on auxiliary variables to give identifiable representations and to perform counterfactual inference.

## Installation

```bash
pip install scmivae
# optional: AnnData support
pip install "scmivae[adata]"
```

From source:

```bash
conda env create -f environment.yml
conda activate scmivae
pip install -e .
```

## Usage

```python
from scmivae import iVAE, ModelTrainerIVAE, VAEInference
from scmivae.distributions import Normal
from scmivae.utils_data import CountDataset
from scmivae.utils_analysis import (
    plot_embedding_pairplot,
    plot_umap_embeddings,
    plot_umap_with_highlighted_clusters,
)
```

## Package structure

```
scmivae/
├── __init__.py          # main exports
├── vae_models.py        # VAEBase, iVAE
├── vae_trainers.py      # ModelTrainerBase, ModelTrainerIVAE
├── inferencer.py        # VAEInference
├── distributions.py     # Normal, Poisson, NB, ZINB
├── utils_nn.py          # MLP, CatEmbedding, weight init
├── utils_data.py        # CountDataset
├── utils_adata.py       # AnnData helpers (needs the `adata` extra)
├── utils_trainer.py     # early stopping, checkpoints, loss plots
└── utils_analysis.py    # embedding / UMAP plots
```

## Dependencies

torch >= 2.0, numpy, pandas, scikit-learn, matplotlib, seaborn, umap-learn. Optional (`adata` extra): anndata, scipy.

## License

MIT. Parts of `distributions.py` and `vae_models.py` are adapted from [iVAE](https://github.com/ilkhem/iVAE) by Ilyes Khemakhem (MIT); see [LICENSE](LICENSE).

## References

Ilyes Khemakhem, Diederik Kingma, Ricardo Monti, and Aapo Hyvarinen. Variational Autoencoders and Nonlinear ICA: A Unifying Framework. In Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, PMLR, June 2020

## Author

Kevin Lam (kevin.lam@stat.ubc.ca)
