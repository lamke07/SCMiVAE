# Portions of this file are adapted from the iVAE implementation by Ilyes Khemakhem
# (https://github.com/ilkhem/iVAE), MIT License, Copyright (c) 2019 Ilyes Khemakhem.
# See the LICENSE file in the project root for the full license text.

import torch
import torch.nn as nn
import numpy as np

from torch import distributions as dist


class Dist:
    def __init__(self):
        pass

    def sample(self, *args):
        pass

    def log_pdf(self, *args, **kwargs):
        pass


class Poisson(Dist):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.name = 'poisson'

    def sample(self, poisson_rate):
        sample_model = dist.poisson.Poisson(poisson_rate)
        poisson_sample = sample_model.sample(poisson_rate.size()).squeeze()

        return poisson_sample

    def log_pdf(self, x, poisson_rate, reduction='none'):
        # Return positive log-lik to avoid confusion
        lpdf = -nn.functional.poisson_nll_loss(
            input=poisson_rate, target=x, reduction=reduction, log_input=True
        ).sum(dim=-1)

        return lpdf


class NB(Dist):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.name = 'NB'

    def sample(self, mu, phi):
        total_count = phi
        probs = phi / (mu + phi)

        sample_model = dist.negative_binomial.NegativeBinomial(total_count, probs)
        NB_sample = sample_model.sample(total_count.size()).squeeze()

        return NB_sample

    def log_pdf(self, x, mu, phi, eps=1e-8, reduction='none'):
        # Return positive log-lik to avoid confusion
        # lpdf_unnormalized = x * torch.log(mu/(mu+phi)) + phi * torch.log(phi/(mu+phi))
        lpdf_unnormalized = x * (torch.log(mu + eps) - torch.log(mu + phi + eps)) + \
                            phi * (torch.log(phi + eps) - torch.log(mu + phi + eps))

        lpdf_normalization = torch.lgamma(x + phi + eps) - torch.lgamma(x + 1.) - torch.lgamma(phi + eps)
        # lpdf_normalization[total_count + x == 0.] = 0.

        lpdf = lpdf_unnormalized + lpdf_normalization

        return lpdf


class ZINB(Dist):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.name = 'ZINB'
        self.NB = NB(device=device)

    def sample(self, mu, phi, zinb_probs):
        # [TO BE UPDATED]
        # sample_model = dist.negative_binomial.NegativeBinomial(total_count, probs)
        NB_sample = self.NB.sample(mu, phi)

        return NB_sample

    def log_pdf(self, x, mu, phi, zinb_probs, reduction='none'):
        lpdf_NB = self.NB.log_pdf(x, mu, phi, reduction=reduction)

        lpdf = torch.where(
            x == 0.,
            (zinb_probs + (1 - zinb_probs) * lpdf_NB.exp()).log(),
            (1. - zinb_probs).log() + lpdf_NB
        )

        return lpdf


class Normal(Dist):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.c = 2 * np.pi * torch.ones(1).to(self.device)
        self._dist = dist.normal.Normal(torch.zeros(1).to(self.device), torch.ones(1).to(self.device))
        self.name = 'gauss'

    def sample(self, mu, v, n_samples=None):
        # Use mu's device so distribution follows model.to(device)
        if n_samples is None:
            eps = torch.randn_like(mu)
            scaled = eps.mul(v.sqrt())
            return scaled.add(mu)

        if n_samples < 1:
            raise ValueError("n_samples must be at least 1")

        sample_shape = (n_samples,) + mu.shape
        eps = torch.randn(sample_shape, device=mu.device, dtype=mu.dtype)
        scaled = eps.mul(v.sqrt().unsqueeze(0))
        return scaled.add(mu.unsqueeze(0))

    def log_pdf(self, x, mu, v, reduce=True, param_shape=None):
        """compute the log-pdf of a normal distribution with diagonal covariance"""
        if param_shape is not None:
            mu, v = mu.view(param_shape), v.view(param_shape)
        c = (2 * np.pi) * torch.ones(1, device=mu.device, dtype=mu.dtype)
        lpdf = -0.5 * (torch.log(c) + v.log() + (x - mu).pow(2).div(v))
        if reduce:
            return lpdf.sum(dim=-1)
        else:
            return lpdf

    def log_pdf_full(self, x, mu, v):
        """
        compute the log-pdf of a normal distribution with full covariance
        v is a batch of "pseudo sqrt" of covariance matrices of shape (batch_size, d_latent, d_latent)
        mu is batch of means of shape (batch_size, d_latent)
        """
        batch_size, d = mu.size()
        cov = torch.einsum('bik,bjk->bij', v, v)  # compute batch cov from its "pseudo sqrt"
        assert cov.size() == (batch_size, d, d)
        inv_cov = torch.inverse(cov)  # works on batches
        c = d * torch.log(2 * np.pi * torch.ones(1, device=mu.device, dtype=mu.dtype))
        # matrix log det doesn't work on batches!
        _, logabsdets = self._batch_slogdet(cov)
        xmu = x - mu
        return -0.5 * (c + logabsdets + torch.einsum('bi,bij,bj->b', [xmu, inv_cov, xmu]))

    def _batch_slogdet(self, cov_batch: torch.Tensor):
        """
        compute the log of the absolute value of determinants for a batch of 2D matrices. Uses torch.slogdet
        this implementation is just a for loop, but that is what's suggested in torch forums
        gpu compatible
        """
        batch_size = cov_batch.size(0)
        dev = cov_batch.device
        signs = torch.empty(batch_size, requires_grad=False, device=dev)
        logabsdets = torch.empty(batch_size, requires_grad=False, device=dev)
        for i, cov in enumerate(cov_batch):
            signs[i], logabsdets[i] = torch.slogdet(cov)
        return signs, logabsdets
