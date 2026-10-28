# Portions of this file are adapted from the iVAE implementation by Ilyes Khemakhem
# (https://github.com/ilkhem/iVAE), MIT License, Copyright (c) 2019 Ilyes Khemakhem.
# See the LICENSE file in the project root for the full license text.

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Any, Dict, Union
from numbers import Number
from abc import abstractmethod

# Import distributions and utilities
from .distributions import Normal, Poisson, ZINB, NB
from .utils_nn import MLP, CatEmbedding, weights_init

# Constants
DIST_MAP = {"gaussian": Normal, "poisson": Poisson, "nb": NB, "zinb": ZINB}
VALID_NET_MODES = ['mean', 'variance', 'both', 'none']
VALID_ACTIVATIONS = ['relu', 'lrelu', 'xtanh', 'sigmoid', 'softplus', 'none']
DEFAULT_TOL = 1e-9

class VAEBase(nn.Module):
    """
    Base Variational Autoencoder class providing encoder/decoder wiring and parameter helpers.

    Code layout: ``__init__`` and init helpers, validation, decoder path, auxiliary inputs,
    encoder/prior internals, abstract API, then ``internal_forward`` / ``internal_elbo``.
    
    This class implements the core functionality for variational autoencoders with support for
    multiple decoder distributions and conditional variational autoencoders (cVAE).

    Args:
        device: Device to place tensors on ('cpu' or 'cuda')
        latent_dim: Dimensionality of the latent space
        data_dim: Dimensionality of the input data
        cvae_dim: Dimensionality of conditional inputs for cVAE
        num_cats: List of number of categories for each categorical variable
        cont_dim: Dimensionality of continuous auxiliary variables
        n_layers: Number of layers in MLP networks
        hidden_dim: Hidden layer dimensionality
        activation: Activation function name ('lrelu', 'relu', 'xtanh', etc.)
        dropout: Dropout probability
        slope: Negative slope for LeakyReLU or alpha for Xtanh
        scale: Scale parameter for Xtanh activation
        use_batchnorm: Whether to use batch normalization
        cvae: Whether to use conditional VAE
        cvae_aux: Whether to create separate networks for conditional variables
        prior: Prior distribution (if None, uses Normal)
        net_mode: Prior network mode ('mean', 'variance', 'both', 'none')
        decoder: Decoder distribution type ('gaussian', 'poisson', 'nb', 'zinb')
        decoder_var: Variance parameterization ('dec_nn' for learned, or numeric for fixed)
        decoder_softmax: Whether to apply softmax before library size scaling
        decoder_use_log1p: Whether to apply log1p transformation for Gaussian decoder
        encoder: Encoder distribution (if None, uses Normal)
        verbose: Whether to print verbose output
    
    Important:
        - `decoder` must be one of: 'gaussian', 'poisson', 'nb', 'zinb'
        - `decoder_var` can be 'dec_nn' to learn variance via a network, or a numeric fixed value
        - Subclasses must implement abstract methods: prior_params, encoder_params, return_vars, forward, elbo
    """

    def __init__(
        self,
        device: str = "cpu",
        latent_dim: int=4, data_dim: int=None, cvae_dim: int = 0, num_cats: List[int] = [0], catemb_dim: int = None, cont_dim: int = 0, 
        n_layers: int = 3, hidden_dim: int = 50, activation: str = "lrelu", dropout: float=0.0, slope: float = 0.1, scale: float = 1.0, use_batchnorm: bool = False,
        hidden_channels=None, spatial_shape=None,
        cvae: bool = False, cvae_aux: bool = False,
        prior: Optional[str] = None, net_mode: str = "mean",
        decoder: Optional[str] = "gaussian", decoder_var: Optional[Any] = 1.0, decoder_softmax: bool = False, decoder_use_log1p: bool = False,
        encoder: Optional[str] = None, use_maxpool: bool=False, encoder_use_bottleneck: bool=False, encoder_use_u: bool=True,
        verbose: bool = False,
        branching_params: Optional[Dict[str, Any]] = None, # unused parameter
        **kwargs
    ):
        super().__init__()

        self.device = torch.device(device)
        self._set_core_dimensions(
            latent_dim=latent_dim,
            data_dim=data_dim,
            cvae_dim=cvae_dim,
            num_cats=num_cats,
            catemb_dim=catemb_dim,
            cont_dim=cont_dim,
            hidden_dim=hidden_dim,
            branching_params=None,  # unused parameter
            verbose=bool(verbose),
        )
        self._set_mlp_params(
            n_layers=n_layers,
            activation=activation,
            dropout=dropout,
            slope=slope,
            scale=scale,
            use_batchnorm=use_batchnorm,
            hidden_channels=hidden_channels,
            spatial_shape=spatial_shape,
            use_maxpool=use_maxpool,
            encoder_cnn_HW=None,
        )
        self._set_model_flags(
            net_mode=net_mode,
            cvae=cvae,
            cvae_aux=cvae_aux,
            decoder_var=decoder_var,
            decoder_softmax=decoder_softmax,
            decoder_use_log1p=decoder_use_log1p,
            decoder_use_cnn=False,
            encoder_use_cnn=False,
            encoder_use_bottleneck=encoder_use_bottleneck,
            encoder_use_u=encoder_use_u,
            verbose=verbose,
        )
        self._validate_init_args(
            decoder=decoder,
            net_mode=net_mode,
            data_dim=data_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )
        self.decoder = decoder

        # Categorical encoder for categorical variables
        self.catemb = CatEmbedding(self.num_cats, self.catemb_dim)
        if self.cvae:
            self.catemb_cvae = CatEmbedding(self.num_cats, self.catemb_dim)
        # If we have exactly two categorical columns with matching embedding dims,
        # use additive embedding to make pair representation order-invariant.
        self.use_additive_cat_embedding = self._supports_additive_cat_embedding(self.catemb)
        if self.use_additive_cat_embedding:
            print("Using additive categorical embedding")
        else:
            print("Not using additive categorical embedding")
        self.cat_u_dim = self._cat_aux_dim(self.catemb)
        self.cat_u_cvae_dim = self._cat_aux_dim(self.catemb_cvae) if self.cvae else self.cat_u_dim

        self._init_distributions(prior=prior, encoder=encoder)
        self._init_decoder_networks(decoder_var=decoder_var)
        self._init_prior_networks()
        self._init_encoder_networks()

        # final device placement
        self.to(self.device)

        if self.verbose:
            print(json.dumps(self._init_summary_dict(), indent=2))

    def _set_core_dimensions(
        self,
        latent_dim: int,
        data_dim: int,
        cvae_dim: int,
        num_cats: List[int],
        catemb_dim: Optional[int],
        cont_dim: int,
        hidden_dim: int,
        branching_params: Optional[Dict[str, Any]],
        verbose: bool,
    ) -> None:
        """Store core dimensions and derived latent split metadata."""
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.cvae_dim = cvae_dim # if conditional inputs are needed
        self.num_cats = num_cats
        self.catemb_dim = catemb_dim
        self.cat_dim = len(self.num_cats)
        self.cont_dim = cont_dim
        self.latent_dim1 = self._compute_non_branching_latent_dim(branching_params, verbose)

    def _compute_non_branching_latent_dim(self, branching_params: Optional[Dict[str, Any]], verbose: bool) -> int:
        """
        Return latent dimension not reserved for branching variables.
        Unused function from earlier experiments.
        """
        
        return self.latent_dim

    def _set_mlp_params(
        self,
        n_layers: int,
        activation: str,
        dropout: float,
        slope: float,
        scale: float,
        use_batchnorm: bool,
        hidden_channels: Any,
        spatial_shape: Any,
        use_maxpool: bool,
        encoder_cnn_HW: Any=None,
    ) -> None:
        """Store shared MLP/CNN architectural hyperparameters."""
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.scale = scale
        self.hidden_channels = hidden_channels
        self.spatial_shape = spatial_shape
        self.use_maxpool = use_maxpool
        self.encoder_cnn_HW = None
        self.mlp_params = dict(
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            activation=self.activation,
            dropout=self.dropout,
            slope=self.slope,
            scale=self.scale,
            device=self.device,
            use_batchnorm=self.use_batchnorm,
            hidden_channels=self.hidden_channels,
            spatial_shape=self.spatial_shape,
            use_maxpool=self.use_maxpool,
            encoder_cnn_HW=None,
        )

    def _set_model_flags(
        self,
        net_mode: str,
        cvae: bool,
        cvae_aux: bool,
        decoder_var: Any,
        decoder_softmax: bool,
        decoder_use_log1p: bool,
        encoder_use_bottleneck: bool,
        verbose: bool,
        encoder_use_u: bool,
        decoder_use_cnn: bool=False,
        encoder_use_cnn: bool=False
    ) -> None:
        """Store model behavior flags."""
        self.net_mode = net_mode
        self.cvae = cvae
        self.cvae_aux = cvae_aux
        self.decoder_var = decoder_var
        self.decoder_softmax = decoder_softmax
        self.decoder_use_log1p = decoder_use_log1p
        self.decoder_use_cnn = False # deactivated by default
        self.encoder_use_cnn = False # deactivated by default
        self.encoder_use_bottleneck = encoder_use_bottleneck
        self.encoder_use_u = encoder_use_u
        self.verbose = bool(verbose)

    def _validate_init_args(
        self,
        decoder: Optional[str],
        net_mode: str,
        data_dim: Optional[int],
        latent_dim: int,
        hidden_dim: int,
        n_layers: int,
    ) -> None:
        """Validate constructor arguments for consistency."""
        if decoder is None or decoder not in DIST_MAP:
            raise ValueError(f"decoder must be one of {list(DIST_MAP.keys())}; got {decoder}")
        if net_mode not in VALID_NET_MODES:
            raise ValueError(f"net_mode must be one of {VALID_NET_MODES}; got {net_mode}")
        if data_dim is None:
            raise ValueError("data_dim must be specified")
        self._validate_positive_number(latent_dim, "latent_dim")
        self._validate_positive_number(data_dim, "data_dim")
        self._validate_positive_number(hidden_dim, "hidden_dim")
        self._validate_positive_number(n_layers, "n_layers")

    def _init_distributions(self, prior: Optional[str], encoder: Optional[str]) -> None:
        """Initialize prior, encoder and decoder distribution objects."""
        self.prior_dist = prior if prior is not None else Normal(device=self.device)
        self.encoder_dist = encoder if encoder is not None else Normal(device=self.device)
        self.decoder_dist = DIST_MAP[self.decoder](device=self.device)

    def _init_decoder_networks(self, decoder_var: Any) -> None:
        """Initialize decoder trunk and output heads."""
        self.decoder_in_dim = (
            self.latent_dim
            if not self.cvae
            else self.latent_dim + self.cat_u_cvae_dim + self.cont_dim
        )
        self.decoder_base = MLP(self.decoder_in_dim, self.hidden_dim, **self.mlp_params)
        self.decoder_mean_head = nn.Linear(self.hidden_dim, self.data_dim)

        # decoder variance head (optional, depending on decoder_var)
        if isinstance(decoder_var, str) and decoder_var == "dec_nn":
            if self.verbose:
                print("Using NN decoder variance")
            self.decoder_var_mode = "dec_nn"
            self.decoder_var_head = nn.Linear(self.hidden_dim, self.data_dim)
        elif isinstance(decoder_var, Number):
            self._validate_positive_number(decoder_var, "decoder_var")
            self.decoder_var_mode = "fixed"
            self.decoder_var = torch.tensor(float(decoder_var), device=self.device)
        else:
            raise ValueError("decoder_var must be 'dec_nn' or a positive numeric value")
        # Probability head (for zero inflated case)
        if self.decoder == "zinb":
            self.decoder_zi_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.data_dim),
                nn.Sigmoid(),
            )

    def _init_prior_networks(self) -> None:
        """Initialize prior network trunk and heads."""
        self.prior_base = MLP(self.cat_u_dim + self.cont_dim, self.hidden_dim, **self.mlp_params)
        self.prior_mean_head = nn.Linear(self.hidden_dim, self.latent_dim1)
        self.prior_var_head = nn.Linear(self.hidden_dim, self.latent_dim1)

    def _init_encoder_networks(self) -> None:
        """Initialize encoder network trunk and heads."""
        if self.encoder_use_u:
            self.encoder_base = MLP(
                self.data_dim + self.cat_u_dim + self.cont_dim,
                self.hidden_dim,
                **self.mlp_params,
            )
        else:
            self.encoder_base = MLP(self.data_dim, self.hidden_dim, **self.mlp_params)

        mlp_bottleneck_params = dict(self.mlp_params)
        mlp_bottleneck_params["n_layers"] = 1
        self.encoder_bottleneck = MLP(
            self.hidden_dim + self.cat_u_dim + self.cont_dim,
            self.hidden_dim,
            **mlp_bottleneck_params,
        )
        self.encoder_mean_head = nn.Linear(self.hidden_dim, self.latent_dim1)
        self.encoder_var_head = nn.Linear(self.hidden_dim, self.latent_dim1)

    def _init_summary_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary of the VAE configuration."""
        return {
            # "model": "VAEBase",
            # "device": str(self.device),

            "dimensions": {
                "data_dim": self.data_dim,
                "latent_dim_total": self.latent_dim,
                "latent_dim_non_branching": self.latent_dim1,
                "hidden_dim": self.hidden_dim,
            },

            "architecture": {
                "n_layers": self.n_layers,
                "activation": self.activation,
                "dropout": self.dropout,
                "batchnorm": self.use_batchnorm,
            },

            "decoder": {
                "distribution": self.decoder,
                "variance_mode": self.decoder_var_mode,
                "fixed_variance": (
                    float(self.decoder_var)
                    if self.decoder_var_mode == "fixed"
                    else None
                ),
                "softmax_scaling": self.decoder_softmax,
                "log1p_transform": self.decoder_use_log1p,
            },

            "prior": {
                "net_mode": self.net_mode,
                "distribution": type(self.prior_dist).__name__,
            },

            "conditioning": {
                "cvae": self.cvae,
                "cvae_dim": self.cvae_dim,
                "cvae_aux": self.cvae_aux,
            },

            "auxiliary_variables": {
                "categorical_dims": self.num_cats if any(self.num_cats) else [],
                "continuous_dim": self.cont_dim,
            },
        }

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def _validate_positive_number(self, value: Number, name: str) -> None:
        """
        Validate that a number is positive.
        
        Args:
            value: Number to validate
            name: Name for error messages
        """
        if float(value) <= 0:
            raise ValueError(f"{name} must be positive; got {value}")
    
    # -------------------------------------------------------------------------
    # Decoder
    # -------------------------------------------------------------------------

    def _apply_library_size(self, f_tensor: torch.Tensor, library_size: Optional[torch.Tensor], dim_mode: int = 1) -> torch.Tensor:
        """
        Apply library-size scaling to decoder outputs.
        
        Args:
            f_tensor: Input tensor to scale
            library_size: Library size tensor for scaling (optional)
            dim_mode: Dimension along which to apply softmax if decoder_softmax is True
            
        Returns:
            Scaled tensor. If library_size is None, returns f_tensor unchanged.
            If decoder_softmax is True, applies softmax before scaling.
        """
        if library_size is None:
            return f_tensor
        if self.decoder_softmax:
            return F.softmax(f_tensor, dim=dim_mode) * library_size
        else:
            return f_tensor * library_size

    def decoder_params(self, s_in: torch.Tensor, u_cat: Optional[torch.Tensor], u_cont: Optional[torch.Tensor],
                       aux_info: Optional[torch.Tensor] = None,
                       library_size: Optional[torch.Tensor] = None, dim_mode: int = 1, tol: float = DEFAULT_TOL, use_log1p: Optional[bool] = None) -> Tuple:
        """
        Compute decoder parameters based on the chosen decoder distribution family.
        
        Args:
            s_in: Latent samples input tensor
            aux_info: Optional auxiliary information for conditional VAE
            library_size: Optional library size for scaling (used in scRNA-seq)
            dim_mode: Dimension along which to apply operations
            tol: Small tolerance value to prevent numerical issues
            use_log1p: Whether to apply log1p transformation (overrides class setting if provided)
            
        Returns:
            Tuple containing distribution parameters:
            - gaussian: (mean, variance)
            - poisson: (rate,)
            - nb: (mean, dispersion)
            - zinb: (mean, dispersion, zero_inflation_probability)
        """
        if self.cvae:
            u_cvae = self.build_u(u_cat, u_cont, x=s_in, for_cvae=True)
            decoder_input = torch.cat((s_in, u_cvae), dim=1)
        else:
            decoder_input = s_in

        if use_log1p is None:
            use_log1p = self.decoder_use_log1p

        # Shared trunk (mean always exists, use library size transformation if applicable)
        decoder_hidden = self.decoder_base(decoder_input)
        mean_pre_lib = self.decoder_mean_head(decoder_hidden)
        mean_scaled = self._apply_library_size(mean_pre_lib, library_size, dim_mode)

        # Performance note: avoid redundant softplus when using fixed variance
        if self.decoder_var_mode == "fixed":
            variance = torch.ones_like(mean_pre_lib) * self.decoder_var.to(mean_pre_lib.device)
        else:
            variance = F.softplus(self.decoder_var_head(decoder_hidden)) + tol

        zi_prob = self.decoder_zi_head(decoder_hidden) if self.decoder == "zinb" else None

        if self.decoder == "gaussian":
            # gaussian_mean = torch.log1p(mean_scaled) if use_log1p else F.softplus(mean_scaled)
            gaussian_mean = torch.log1p(mean_scaled) if use_log1p else F.leaky_relu(mean_scaled)
            return gaussian_mean, variance
        elif self.decoder == "poisson":
            return [F.softplus(mean_scaled) + tol]
        elif self.decoder == "nb":
            return F.softplus(mean_scaled), variance
        elif self.decoder == "zinb":
            return F.softplus(mean_scaled), variance, zi_prob
        else:
            # unreachable because of validation earlier
            raise RuntimeError(f"Unhandled decoder type: {self.decoder}")

    # -------------------------------------------------------------------------
    # Auxiliary inputs (categorical / continuous embeddings for encoder & prior)
    # -------------------------------------------------------------------------

    def build_u(
        self,
        u_cat: Optional[torch.Tensor] = None,
        u_cont: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        for_cvae: bool = False,
    ) -> torch.Tensor:
        """Merge categorical embeddings and continuous auxiliary variables."""
        if (u_cat is None and u_cont is None) or (
            u_cat.shape[1] == 0 and u_cont.shape[1] == 0
        ):
            return torch.zeros(x.shape[0], 0, device=self.device)
        parts: List[torch.Tensor] = []
        if u_cat is not None and self.catemb is not None and self.catemb.output_dim > 0:
            cat_embedder = self.catemb_cvae if for_cvae else self.catemb
            cat_embed = self._embed_categorical_additive(u_cat, cat_embedder)
            parts.append(cat_embed)
        if u_cont is not None and self.cont_dim > 0:
            parts.append(u_cont)
        return torch.cat(parts, dim=1)

    def _supports_additive_cat_embedding(self, cat_embedder: CatEmbedding) -> bool:
        """True when additive pair embedding is shape-safe."""
        if self.cat_dim != 2:
            return False
        if len(cat_embedder.embedding_dims) != 2:
            return False
        if cat_embedder.embedding_dims[0] != cat_embedder.embedding_dims[1]:
            return False
        if len(self.num_cats) != 2:
            return False
        return self.num_cats[0] == self.num_cats[1]

    def _cat_aux_dim(self, cat_embedder: CatEmbedding) -> int:
        """Effective categorical aux dimension after optional additive merge."""
        if self._supports_additive_cat_embedding(cat_embedder):
            return int(cat_embedder.embedding_dims[0])
        return int(cat_embedder.output_dim)

    def _embed_categorical_additive(
        self,
        u_cat: torch.Tensor,
        cat_embedder: CatEmbedding,
    ) -> torch.Tensor:
        """
        Build categorical embedding; for two columns with matched dims, use:
          emb = emb1 + emb2
        Otherwise keep the original concatenation behavior.
        """
        if not self._supports_additive_cat_embedding(cat_embedder):
            return cat_embedder(u_cat)
        shared_embedding = cat_embedder.embeddings[0]
        emb1 = shared_embedding(u_cat[:, 0].long())
        emb2 = shared_embedding(u_cat[:, 1].long())
        # Treat category index 0 as control baseline so control contributes zero.
        ctrl_idx = torch.zeros(u_cat.shape[0], dtype=torch.long, device=u_cat.device)
        ctrl_emb = shared_embedding(ctrl_idx)
        return (emb1 - ctrl_emb) + (emb2 - ctrl_emb)

    def encoder_params_x_in(self, x: torch.Tensor, aux_info: Optional[torch.Tensor] = None,
                            library_size: Optional[torch.Tensor] = None, tol: float = DEFAULT_TOL) -> torch.Tensor:
        """
        Prepare the input data for the encoder with optional scaling and auxiliary information.
        
        Args:
            x: Input data tensor
            aux_info: Optional auxiliary information to concatenate
            library_size: Optional library size for normalization
            tol: Small tolerance value to prevent division by zero
            
        Returns:
            Processed input tensor ready for encoder network.
        """
        # [TO DO] make sure broadcasting works: assume library_size shape matches batch cols
        x_in = x if library_size is None else x / (library_size + tol)

        if aux_info is None:
            return x_in
        else:
            return torch.cat((x_in, aux_info), dim=1)

    # -------------------------------------------------------------------------
    # Encoder & prior
    # -------------------------------------------------------------------------

    def _encoder_params(self, x: torch.Tensor, u_cat: Optional[torch.Tensor], u_cont: Optional[torch.Tensor], 
                       aux_info: Optional[torch.Tensor] = None, library_size: Optional[torch.Tensor] = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute encoder distribution parameters.
        
        Args:
            x: Input data tensor
            u_cat: Categorical auxiliary variables
            u_cont: Continuous auxiliary variables
            aux_info: Additional auxiliary information for cVAE
            library_size: Library size for normalization
            
        Returns:
            List of tuples containing (mean, variance) parameters for encoder distribution.
        """
        u_aux = self.build_u(u_cat, u_cont, x=x)
        if self.encoder_use_u:
            x_prepared = self.encoder_params_x_in(x, aux_info, library_size)
            encoder_input = torch.cat((x_prepared, u_aux), dim=1)
        else:
            encoder_input = self.encoder_params_x_in(x, aux_info, library_size)

        encoder_hidden = self.encoder_base(encoder_input)
        if self.encoder_use_bottleneck:
            encoder_hidden = self.encoder_bottleneck(torch.cat((encoder_hidden, u_aux), dim=1))
        enc_mean = self.encoder_mean_head(encoder_hidden)
        enc_var = F.softplus(self.encoder_var_head(encoder_hidden))

        return [(enc_mean, enc_var)]

    def _prior_params(self, u_cat: Optional[torch.Tensor], u_cont: Optional[torch.Tensor], x: Optional[torch.Tensor] = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute prior distribution parameters based on auxiliary variables.
        
        Args:
            u_cat: Categorical auxiliary variables
            u_cont: Continuous auxiliary variables  
            x: Input data tensor (used for batch size inference)
            
        Returns:
            List of tuples containing (mean, variance) parameters for prior distribution.
        """

        if (
            self.net_mode == "none"
            or (u_cat is None and u_cont is None)
            or (self.catemb.output_dim == 0 and self.cont_dim == 0)
        ):
            prior_mean = torch.zeros(x.shape[0], self.latent_dim1, device=self.device)
            prior_var = torch.ones(x.shape[0], self.latent_dim1, device=self.device)
        else:
            u_aux = self.build_u(u_cat, u_cont, x=x)
            prior_hidden = self.prior_base(u_aux)

            if self.net_mode == "mean":
                prior_mean = self.prior_mean_head(prior_hidden)
                prior_var = torch.ones_like(prior_mean)
            elif self.net_mode == "variance":
                prior_var = F.softplus(self.prior_var_head(prior_hidden))
                prior_mean = torch.zeros_like(prior_var)
            elif self.net_mode == "both":
                prior_mean = self.prior_mean_head(prior_hidden)
                prior_var = F.softplus(self.prior_var_head(prior_hidden))
            else:
                raise ValueError(
                    f"Unknown net_mode: {self.net_mode}. Must be one of "
                    "['mean', 'variance', 'both', 'none']"
                )

        return [(prior_mean, prior_var)]

    # -------------------------------------------------------------------------
    # Abstract API (subclasses implement)
    # -------------------------------------------------------------------------
    @abstractmethod
    def prior_params(self, *args, **kwargs) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Compute prior distribution parameters (to be implemented in subclass)."""
        raise NotImplementedError

    @abstractmethod
    def encoder_params(self, *args, **kwargs) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Compute encoder distribution parameters (to be implemented in subclass)."""
        raise NotImplementedError

    @abstractmethod
    def return_vars(self, *args, **kwargs) -> Tuple:
        """Return input variables in a standardized format (to be implemented in subclass)."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tuple:
        """Forward pass returning decoder params, encoder params, latent samples, and prior params (to be implemented in subclass)."""
        raise NotImplementedError

    @abstractmethod
    def elbo(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute ELBO loss components (to be implemented in subclass)."""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Forward & ELBO (shared implementation)
    # -------------------------------------------------------------------------

    def internal_forward(
        self,
        x: torch.Tensor,
        u: Union[Tuple, List],
        aux_info: Optional[torch.Tensor] = None, 
        library_size: Optional[torch.Tensor] = None
    ) -> Tuple:
        """
        Internal forward pass that computes all VAE components.
        
        Args:
            x: Input data tensor
            u: Tuple of auxiliary variables (u_cat, u_cont), u is the auxiliary conditioning that prior_params() expects
            aux_info: Additional auxiliary information for cVAE
            library_size: Library size for scaling
            
        Returns:
            Tuple containing (decoder_params, encoder_params, z_samples_list, prior_params)
        """
        prior_params = self.prior_params(*u, x=x)
        encoder_params = self.encoder_params(x, *u, aux_info, library_size)

        # sample per-encoder output; keep shape ordering consistent with original design
        n_enc = len(encoder_params)
        z_list = [self.encoder_dist.sample(*encoder_params[i]) for i in range(n_enc)]
        z_concat = torch.cat(z_list, dim=1)
        decoder_params = self.decoder_params(z_concat, *u, aux_info, library_size)
        return decoder_params, encoder_params, z_list, prior_params


    def internal_elbo(
        self,
        x: torch.Tensor,
        u: Union[Tuple, List],
        aux_info: Optional[torch.Tensor] = None,
        library_size: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Evidence Lower Bound (ELBO) components.
        
        Args:
            x: Input data tensor
            u: Tuple of auxiliary variables (u_cat, u_cont)
            aux_info: Additional auxiliary information for cVAE
            library_size: Library size for scaling
            
        Returns:
            Tuple containing:
            - recon_loss: Reconstruction loss (-log p(x|z)).mean()
            - kl_terms: KL divergence D[q(z|x,u) || p(z|u)].mean()
            - z_cat: Concatenated latent samples
        """
        decoder_params, encoder_params, z_list, prior_params = self.internal_forward(x, u, aux_info, library_size)
        
        log_px_z = self.decoder_dist.log_pdf(x, *decoder_params)
        n_enc = len(encoder_params)
        log_qz_xu = [self.encoder_dist.log_pdf(z_list[i], *encoder_params[i]) for i in range(n_enc)]
        log_pz_u = [self.prior_dist.log_pdf(z_list[i], *prior_params[i]) for i in range(n_enc)]

        # Compute output terms
        recon_loss = (-log_px_z).mean()
        
        # Performance optimization: compute KL terms efficiently
        # Stack tensors only once and use vectorized operations
        log_pz_u_stacked = torch.stack(log_pz_u, dim=1)
        log_qz_xu_stacked = torch.stack(log_qz_xu, dim=1)
        kl_terms = (log_qz_xu_stacked - log_pz_u_stacked).sum(dim=1).mean()
        
        z_cat = torch.cat(z_list, dim=1)
        
        return recon_loss, kl_terms, z_cat

class iVAE(VAEBase):
    """
    Identifiable Variational Autoencoder (iVAE) implementation.
    
    This class implements the iVAE model which learns identifiable representations
    by conditioning the prior distribution on auxiliary variables.
    
    Args:
        **kwargs: Arguments passed to VAEBase parent class
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.apply(weights_init)

    def prior_params(self, u_cat: Optional[torch.Tensor], u_cont: Optional[torch.Tensor], x: Optional[torch.Tensor] = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Compute prior distribution parameters."""
        return self._prior_params(u_cat, u_cont, x)

    def encoder_params(self, x: torch.Tensor, u_cat: Optional[torch.Tensor], u_cont: Optional[torch.Tensor], 
                      aux_info: Optional[torch.Tensor] = None, library_size: Optional[torch.Tensor] = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Compute encoder distribution parameters."""
        return self._encoder_params(x, u_cat, u_cont, aux_info, library_size)

    def return_vars(self, x: torch.Tensor, u_cat: Optional[torch.Tensor], u_cont: Optional[torch.Tensor], 
                   aux_info: Optional[torch.Tensor] = None, library_size: Optional[torch.Tensor] = None) -> Tuple:
        """Return input variables in standardized format."""
        return x, [u_cat, u_cont], aux_info, library_size

    def forward(self, x: torch.Tensor, u_cat: Optional[torch.Tensor], u_cont: Optional[torch.Tensor], 
               aux_info: Optional[torch.Tensor] = None, library_size: Optional[torch.Tensor] = None) -> Tuple:
        """Forward pass through the iVAE model."""
        decoder_params, encoder_params, z, prior_params = self.internal_forward(x, [u_cat, u_cont], aux_info, library_size)
        return decoder_params, encoder_params, torch.cat(z, 1), prior_params

    def elbo(self, x: torch.Tensor, u_cat: Optional[torch.Tensor], u_cont: Optional[torch.Tensor], 
            aux_info: Optional[torch.Tensor] = None, library_size: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute ELBO loss components."""
        return self.internal_elbo(x, [u_cat, u_cont], aux_info, library_size)
