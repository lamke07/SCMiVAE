import json
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.func import jacfwd

# --- general utilities ---
def write_status(run_dir, status, msg=None, exit_code=None):
    """
    Write a STATUS file in the run directory.

    Args:
        run_dir (str | Path): Path to the run directory.
        status (str): "OK" or "FAIL".
        msg (str, optional): Extra message (reason, metrics, etc.).
        exit_code (int, optional): If provided, sys.exit(exit_code) is called.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    status_path = run_dir / "STATUS"

    with open(status_path, "w") as f:
        f.write(f"STATUS: {status}\n")
        f.write(f"TIMESTAMP: {datetime.now().isoformat()}\n")
        if msg:
            f.write(f"MESSAGE: {msg}\n")

    if exit_code is not None:
        sys.exit(exit_code)

    return status_path


# --- Loss saving ---
def save_loss(train_loss, valid_loss, test_loss, str_save, decimals=1):
    """Save final losses as JSON."""
    loss_dict = {
        "train_loss": {k: np.round(v[-1], decimals) for k, v in train_loss.items()},
        "valid_loss": {k: np.round(v[-1], decimals) for k, v in valid_loss.items()},
        "test_loss": {k: np.round(v[-1], decimals) for k, v in test_loss.items()},
    }
    with open(str_save, "w") as f:
        json.dump(loss_dict, f, indent=2)


def save_ckp(state, checkpoint_path, best_model_path=None, is_best=False):
    """
    Save a checkpoint and optionally copy it as the best model.
    """
    torch.save(state, checkpoint_path)
    if is_best and best_model_path is not None:
        shutil.copyfile(checkpoint_path, best_model_path)


def load_ckp(checkpoint_path, model, optimizer=None, device="cpu"):
    """
    Load a checkpoint into model (and optimizer if provided).
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint["state_dict"])
    
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return (
        model,
        optimizer,
        checkpoint.get("epoch"),
        checkpoint.get("train_loss"),
        checkpoint.get("valid_loss"),
    )


# --- Jacobian loss ---
def jacobian_loss_function(nn_model, z, use_jacobian_loss=False, loss_mode="l1", device="cpu"):
    """
    Compute Jacobian-based regularization loss.

    Parameters
    ----------
    nn_model : torch.nn.Module
        Model with `decoder_params`.
    z : torch.Tensor
        Latent variables (batch_size, latent_dim).
    use_jacobian_loss : bool, default=False
        If False, returns zero loss.
    loss_mode : {"l1", "l2"}
    device : str, default="cpu"

    Returns
    -------
    torch.Tensor (scalar loss)
    """
    if not use_jacobian_loss:
        return torch.tensor(0.0, device=device)

    # Compute Jacobian
    jacobian = torch.vmap(jacfwd(nn_model.decoder_params))(z)[0]

    # Loss modes
    if loss_mode == "l1":
        loss = jacobian.abs()
    elif loss_mode == "l2":
        loss = jacobian.square().sum(0).sqrt().sum(1)
    else:
        raise ValueError(f"Unknown loss_mode: {loss_mode}")

    return loss.sum() / jacobian.shape[0]


# --- Early stopping ---
class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0, mode="min", verbose=True):
        """
        Early stopping utility to stop training when a monitored metric
        has stopped improving.

        Args:
            patience (int): Number of epochs to wait after no improvement.
            min_delta (float): Minimum change to qualify as improvement.
            mode (str): "min" (default, e.g. loss) or "max" (e.g. accuracy).
            verbose (bool): Whether to print updates.
        """
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.best_value = None
        self.best_epoch = None
        self.counter = 0
        self.should_stop = False

        # comparator depending on mode
        if mode == "min":
            self.is_better = lambda current, best: current < (best - self.min_delta)
            self.init_best = np.inf
        else:  # mode == "max"
            self.is_better = lambda current, best: current > (best + self.min_delta)
            self.init_best = -np.inf

    def step(self, current_value, epoch=None):
        """
        Check if training should stop.

        Args:
            current_value (float): Current metric value (loss/accuracy/etc).
            epoch (int or None): Current epoch number.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if self.best_value is None:
            self.best_value = self.init_best

        if self.is_better(current_value, self.best_value):
            self.best_value = current_value
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"[EarlyStopper] Reset at epoch {epoch} | New best: {current_value:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopper] No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# --- Loss plotting ---
def get_scaling(data_name, epoch_threshold=0):
    """Return scaling factors for different losses depending on dataset/epoch."""
    default_scaling = {
        "overall_loss": 1.015 if epoch_threshold > 0 else 1.025,
        "elbo_loss": 0.999 if epoch_threshold > 0 else 1.025,
        "jac_est_loss": 0.995 if epoch_threshold > 0 else 0.95,
        "rec_loss": 0.9999 if epoch_threshold > 0 else 1.015,
        "kl_loss": 1.25 if epoch_threshold > 0 else 2.5,
        "bkl_loss": 1.25 if epoch_threshold > 0 else 2.5,
    }
    if data_name == "PDX":
        return {
            "overall_loss": 1.025 if epoch_threshold == 0 else 1.025,
            "elbo_loss": 1.025 if epoch_threshold == 0 else 0.995,
            "jac_est_loss": 0.95 if epoch_threshold == 0 else 0.995,
            "rec_loss": 1.015 if epoch_threshold == 0 else 0.998,
            "kl_loss": 2.5 if epoch_threshold == 0 else 1.25,
            "bkl_loss": 2.5 if epoch_threshold == 0 else 1.25,
        }
    return default_scaling


def prepare_loss_df(training_loss, validation_loss, test_loss=None, decimals=1):
    """Return tidy DataFrames for training/validation/test losses."""
    loss_dict = {
        "train": {k: np.round(v[-1], decimals) for k, v in training_loss.items()},
        "valid": {k: np.round(v[-1], decimals) for k, v in validation_loss.items()},
    }
    if test_loss is not None:
        loss_dict["test"] = {k: np.round(v[-1], decimals) for k, v in test_loss.items()}

    loss_df = (
        pd.DataFrame(loss_dict)
        .reset_index()
        .melt(id_vars="index", var_name="loss_type", value_name="loss")
    )

    df_train = pd.DataFrame(training_loss).assign(loss_type="train", epoch=lambda x: x.index)
    df_valid = pd.DataFrame(validation_loss).assign(loss_type="valid", epoch=lambda x: x.index)
    df_concat = pd.concat([df_train, df_valid], ignore_index=True)

    return df_concat, loss_df


def plt_single_loss(df_concat, loss_df, loss_select, loss_title, epoch_threshold, ax, factor_increase=1.0,
                    test_loss=None, decimals=1):
    """Plot a single loss curve."""
    df_plot = (
        df_concat.query("epoch > @epoch_threshold")
        .merge(loss_df.query("index == @loss_select"), on="loss_type", how="left")
        .assign(loss_type=lambda x: x.loss_type + ": " + x.loss.astype(str))
    )

    graph = sns.lineplot(data=df_plot, x="epoch", y=loss_select, hue="loss_type", ax=ax)

    if test_loss is not None and loss_select in test_loss:
        test_val = test_loss[loss_select][0]
        ax.axhline(test_val, color="r", linestyle="-.", alpha=0.5)
        ann_val = loss_df.query("index == @loss_select and loss_type == 'test'")["loss"].values
        if len(ann_val):
            ax.annotate(
                f"Test: {np.round(float(ann_val), decimals)}",
                xy=(max(50, df_concat["epoch"].max() - 35), ann_val[0] * factor_increase),
                va="center", ha="center",
                bbox=dict(boxstyle="round", fc="none", alpha=0.5, ec="gray"),
            )

    graph.get_legend().set_title("Type")
    graph.set(xlabel="epoch", ylabel="Loss", title=loss_title)
    return graph


def plt_training_loss(epoch_threshold, model_dir=None, training_loss=None, validation_loss=None, test_loss=None,
                      model_name=None, data_name=None, plot_mode="train", decimals=1):
    """Plot all training/validation/test losses in a grid."""
    df_concat, loss_df = prepare_loss_df(training_loss, validation_loss, test_loss, decimals)

    if epoch_threshold >= df_concat["epoch"].max() or df_concat["epoch"].max() < 2:
        return None

    scaling = get_scaling(data_name, epoch_threshold)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 16))
    loss_plots = [
        ("overall_loss", "Overall Loss"),
        ("elbo_loss", "ELBO Loss"),
        ("jac_est_loss", "Jacobian Loss"),
        ("rec_loss", "Reconstruction Error"),
        ("kl_loss", "KL Loss"),
        ("bkl_loss", "β·KL Loss"),
    ]
    for ax, (loss, title) in zip(axes.flat, loss_plots):
        plt_single_loss(df_concat, loss_df, loss, title,
                        epoch_threshold, ax,
                        factor_increase=scaling.get(loss, 1.0),
                        test_loss=test_loss, decimals=decimals)

    save_path = None
    if plot_mode == "train" and model_dir:
        save_path = model_dir / f"training/loss{epoch_threshold}.png"
    elif plot_mode == "analysis" and model_name:
        save_path = model_dir / f"full-loss{epoch_threshold}.png"

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    plt.close(fig)


# --- Reconstruction plotting ---
def _plot_vae_reconstructions_basic(
        model,
        data_loader,
        n_samples=16,
        save_dir="reconstructions",
        device="cpu",
        cmap="viridis",
        normalize=False,
        epoch=0,
        pad_value=0.0,
        num_rows=None,
        model_decoder="gaussian",
):
    """
    Plot VAE reconstructions (original vs reconstructed side-by-side) with color map and subtitles.
    One combined panel per sample. Used for gaussian, poisson, and nb/zinb when decoder params are not plotted.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for data in data_loader:
            decoder_params, _, _, _ = model.forward(*[items.to(device) for items in data])
            break

    x = data[0][:n_samples]
    recon = decoder_params[0][:n_samples]

    if model_decoder in ["poisson", "nb", "zinb"]:
        x = torch.log1p(x)
        recon = torch.log1p(recon)

    x = ensure_image_shape(x, num_rows=num_rows, pad_value=pad_value)
    recon = ensure_image_shape(recon, num_rows=num_rows, pad_value=pad_value)

    n_cols = math.ceil(math.sqrt(n_samples))
    n_rows = math.ceil(n_samples / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 6, n_rows * 6),
        gridspec_kw={"wspace": 0.35, "hspace": 0.35},
    )
    axes = axes.flatten()

    for i in range(n_samples):
        ax = axes[i]
        ax.axis("off")

        img_orig = x[i].cpu().squeeze().numpy()
        img_recon = recon[i].cpu().squeeze().numpy()

        if normalize:
            img_orig = (img_orig - img_orig.min()) / (img_orig.max() - img_orig.min() + 1e-8)
            img_recon = (img_recon - img_recon.min()) / (img_recon.max() - img_recon.min() + 1e-8)

        pad = 5
        combined = combine_side_by_side_matplotlib(img_orig, img_recon, pad=pad)

        im = ax.imshow(combined, cmap=cmap)
        ax.set_title(f"Sample {i+1}", fontsize=12)

        h, w = combined.shape
        half = w // 2
        ax.text(half / 2, h + 3, "Original", color="black", ha="center", va="top", fontsize=10)
        ax.text(half + half / 2, h + 3, "Reconstructed", color="black", ha="center", va="top", fontsize=10)

        cbar = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.08, shrink=0.8)
        cbar.ax.tick_params(labelsize=10)

    for j in range(n_samples, len(axes)):
        axes[j].axis("off")

    plt.tight_layout(pad=1.2)
    save_path = os.path.join(save_dir, f"vae_reconstructions_{str(epoch).zfill(4)}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"✅ Saved reconstruction grid to: {save_path}")


def _plot_vae_reconstructions_with_decoder_params(
        model,
        data_loader,
        n_samples=16,
        save_dir="reconstructions",
        device="cpu",
        cmap="viridis",
        normalize=False,
        epoch=0,
        pad_value=0.0,
        num_rows=None,
        model_decoder="zinb",
):
    """
    Plot VAE reconstructions for nb/zinb with one panel per quantity: Original, Reconstructed,
    Variance (log1p), and for zinb Zero-inflation. Each sample is a row; each column has its own scale.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for data in data_loader:
            decoder_params, _, _, _ = model.forward(*[items.to(device) for items in data])
            break

    x = data[0][:n_samples]
    recon = decoder_params[0][:n_samples]
    x = torch.log1p(x)
    recon = torch.log1p(recon)

    x = ensure_image_shape(x, num_rows=num_rows, pad_value=pad_value)
    recon = ensure_image_shape(recon, num_rows=num_rows, pad_value=pad_value)

    variance = decoder_params[1][:n_samples]
    variance = ensure_image_shape(
        torch.log1p(variance), num_rows=num_rows, pad_value=pad_value
    )
    n_plot_cols = 4 if model_decoder == "zinb" else 3
    if model_decoder == "zinb":
        zi = decoder_params[2][:n_samples]
        zi = ensure_image_shape(zi, num_rows=num_rows, pad_value=pad_value)

    fig, axes = plt.subplots(
        n_samples, n_plot_cols,
        figsize=(n_plot_cols * 5, n_samples * 4.5),
        squeeze=False,
        gridspec_kw={"wspace": 0.2, "hspace": 0.25},
    )
    col_titles = ["Original", "Reconstructed", "Variance (log1p)", "Zero-inflation"]
    if model_decoder == "nb":
        col_titles = col_titles[:3]

    def col_min_max(tensor_4d):
        arr = tensor_4d.cpu().numpy()
        return float(np.nanmin(arr)), float(np.nanmax(arr))

    vmin0, vmax0 = col_min_max(x)
    vmin1, vmax1 = col_min_max(recon)
    vmin2, vmax2 = col_min_max(variance)
    if model_decoder == "zinb":
        vmin3, vmax3 = 0.0, 1.0
    vmins = [vmin0, vmin1, vmin2] + ([vmin3] if model_decoder == "zinb" else [])
    vmaxs = [vmax0, vmax1, vmax2] + ([vmax3] if model_decoder == "zinb" else [])

    for i in range(n_samples):
        for j in range(n_plot_cols):
            ax = axes[i, j]
            ax.axis("off")
            if j == 0:
                img = x[i].cpu().squeeze().numpy()
            elif j == 1:
                img = recon[i].cpu().squeeze().numpy()
            elif j == 2:
                img = variance[i].cpu().squeeze().numpy()
            else:
                img = zi[i].cpu().squeeze().numpy()

            if normalize and j < 2:
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            im = ax.imshow(img, cmap=cmap, vmin=vmins[j], vmax=vmaxs[j])
            ax.set_title(f"Sample {i+1} – {col_titles[j]}", fontsize=9)
            cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.06, shrink=0.85)
            cbar.ax.tick_params(labelsize=8)

    plt.tight_layout(pad=0.8)
    save_path = os.path.join(save_dir, f"vae_reconstructions_{str(epoch).zfill(4)}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"✅ Saved reconstruction grid to: {save_path}")


def plot_vae_reconstructions(
        model,
        data_loader,
        n_samples=16,
        save_dir="reconstructions",
        device="cpu",
        cmap="viridis",
        normalize=False,
        epoch=0,
        pad_value=0.0,
        num_rows=None,
        model_decoder="gaussian",
        include_decoder_params=True,
):
    """
    Plot VAE reconstructions. Routes to a basic (original | reconstructed) plot or to a
    multi-panel plot with decoder params (variance, zero-inflation) for nb/zinb when
    include_decoder_params is True.
    """
    if include_decoder_params and model_decoder in ("nb", "zinb"):
        _plot_vae_reconstructions_with_decoder_params(
            model=model,
            data_loader=data_loader,
            n_samples=n_samples,
            save_dir=save_dir,
            device=device,
            cmap=cmap,
            normalize=normalize,
            epoch=epoch,
            pad_value=pad_value,
            num_rows=num_rows,
            model_decoder=model_decoder,
        )
    else:
        _plot_vae_reconstructions_basic(
            model=model,
            data_loader=data_loader,
            n_samples=n_samples,
            save_dir=save_dir,
            device=device,
            cmap=cmap,
            normalize=normalize,
            epoch=epoch,
            pad_value=pad_value,
            num_rows=num_rows,
            model_decoder=model_decoder,
        )


def unflatten_to_rows(
    tensor: torch.Tensor,
    num_rows: int,
    pad_value: float = 0.0,
) -> torch.Tensor:
    if tensor.dim() != 2:
        raise ValueError(f"Expected [B, D] tensor, got {tensor.shape}")

    if num_rows <= 0:
        raise ValueError("num_rows must be positive")

    B, D = tensor.shape
    row_size = math.ceil(D / num_rows)
    total = num_rows * row_size
    pad_len = int(total - D)

    if pad_len > 0:
        tensor = F.pad(
            tensor,
            (0, pad_len),      # tuple of INTS
            mode="constant",
            value=pad_value,   # float padding value
        )

    return tensor.view(B, 1, num_rows, row_size)


def ensure_image_shape(tensor, num_rows=None, pad_value=0.0):
    """Convert flattened tensors into square image tensors."""
    if tensor.dim() == 4:
        return tensor
    if tensor.dim() == 3 and tensor.size(1) in [1, 3]:
        return tensor
    if tensor.dim() == 2 and num_rows is None:
        B, D = tensor.shape
        side = int(math.sqrt(D))
        if side * side != D:
            new_side = math.ceil(math.sqrt(D))
            new_D = new_side * new_side
            pad_len = new_D - D
            tensor = F.pad(tensor, (pad_value, pad_len), mode="constant", value=0)
            side = new_side
        tensor = tensor.view(B, 1, side, side)
        return tensor
    elif tensor.dim() == 2 and num_rows is not None:
        return unflatten_to_rows(tensor, num_rows, pad_value)
    raise ValueError(f"Unexpected tensor shape {tensor.shape} for image conversion.")


def combine_side_by_side_matplotlib(img1, img2, pad=5):
    """Combine two NumPy image arrays side by side with a small gap."""
    h = max(img1.shape[0], img2.shape[0])
    pad_block = torch.ones((h, pad)) * img1.max()  # bright divider
    combined = torch.cat([
        torch.tensor(img1),
        pad_block,
        torch.tensor(img2)
    ], dim=1)
    return combined.numpy()
