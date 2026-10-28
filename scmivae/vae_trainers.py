from typing import Optional, Callable, Tuple, List, Any
import time
import numpy as np
import torch
import torch.nn as nn

from dataclasses import dataclass
from .utils_trainer import load_ckp, save_ckp, write_status, jacobian_loss_function, plt_training_loss, save_loss, EarlyStopper, plot_vae_reconstructions

@dataclass
class LossBatch:
    """Container for tracking losses in a batch or epoch.
    Loss fields are per-sample averages (from _run_epoch); n_batches is metadata only."""
    beta_loss: float
    overall_loss: float
    elbo_loss: float
    rec_loss: float
    kl_loss: float
    jac_est_loss: float
    n_batches: int


class ModelTrainerBase:
    """
    Base trainer class for PyTorch models.

    Handles:
      - Saving/loading checkpoints
      - Tracking train/valid/test losses
      - Printing/logging utilities
      - Plotting and saving loss curves
    """

    LOSS_KEYS = ["overall_loss", "elbo_loss", "jac_est_loss", "rec_loss", "kl_loss", "bkl_loss"]

    def __init__(
        self,
        model, optimizer,
        load_model=False, resume_from_latest=False,
        early_stopping_start=25, epoch_checkpoint_freq=5,
        checkpoint_path_latest=None, checkpoint_path_best=None,
        device=None, verbose=2,
        model_dir: Optional[str] = None,
        *args, **kwargs
    ):
        self.model = model
        self.model_dir = model_dir
        self.optimizer = optimizer
        self.load_model = load_model
        self.resume_from_latest = resume_from_latest

        # Loss dicts
        self.train_loss = self._init_loss_dict()
        self.valid_loss = self._init_loss_dict()
        self.test_loss = self._init_loss_dict()

        self.epoch_start = 0
        self.valid_loss_best = np.inf
        self.epoch_best = None

        # Saving/loading params
        self.verbose = verbose
        self.early_stopping_start = early_stopping_start
        self.epoch_checkpoint_freq = epoch_checkpoint_freq
        self.checkpoint_path_latest = checkpoint_path_latest
        self.checkpoint_path_best = checkpoint_path_best

        # Device handling
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Load model if available
        if self.load_model:
            self._load_checkpoint()

    # -----------------
    # Utilities
    # -----------------
    def _init_loss_dict(self):
        return {k: [] for k in self.LOSS_KEYS}

    def _log(self, msg, level=2):
        if self.verbose >= level:
            print(msg)

    def _load_checkpoint(self):
        path = self.checkpoint_path_best if self.resume_from_latest else self.checkpoint_path_latest
        assert path, "Checkpoint path is missing"
        mode = "best" if self.resume_from_latest else "latest"
        self._log(f"\nLoading existing ({mode}) model ...\n")

        self.model, self.optimizer, self.epoch_start, self.train_loss, self.valid_loss = load_ckp(path, self.model, self.optimizer)

    # -----------------
    # Checkpoint saving
    # -----------------
    def _save_checkpoint(self, checkpoint, is_best, epoch_idx):
        msg = "BEST VALIDATION" if is_best else "checkpoint"
        self._log(f"\nSaving {msg} at epoch {epoch_idx}\n")
        save_ckp(
            checkpoint,
            checkpoint_path=self.checkpoint_path_latest,
            best_model_path=self.checkpoint_path_best,
            is_best=is_best
        )

    def save_model_ckp(self, epoch, valid_loss):
        epoch_idx = self.epoch_start + epoch + 1

        checkpoint = {
            "epoch": epoch_idx,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_loss": self.train_loss,
            "valid_loss": self.valid_loss,
        }

        # Regular checkpoint saving
        if epoch % self.epoch_checkpoint_freq == 0:
            self._save_checkpoint(checkpoint, is_best=False, epoch_idx=epoch_idx)

        # Save best model
        if epoch_idx > self.early_stopping_start and valid_loss < self.valid_loss_best:
            self.valid_loss_best = valid_loss
            self.epoch_best = epoch_idx
            self._save_checkpoint(checkpoint, is_best=True, epoch_idx=epoch_idx)

    # -----------------
    # Loss tracking
    # -----------------
    def _append_loss_dict(self, loss_dict, loss: LossBatch):
        # LossBatch values are already per-sample averages from _run_epoch (no n_batches scaling)
        loss_dict["overall_loss"].append(loss.overall_loss)
        loss_dict["elbo_loss"].append(loss.elbo_loss)
        loss_dict["rec_loss"].append(loss.rec_loss)
        loss_dict["kl_loss"].append(loss.kl_loss)
        loss_dict["bkl_loss"].append(loss.beta_loss * loss.kl_loss)
        loss_dict["jac_est_loss"].append(loss.jac_est_loss)

    # -----------------
    # Printing
    # -----------------
    def _format_losses(self, loss: LossBatch, n_digits=3):
        # LossBatch values are already per-sample averages
        return (
            f"Avg Loss: {loss.overall_loss:.{n_digits}f}\t"
            f"Rec Loss: {loss.rec_loss:.{n_digits}f}\t"
            f"bKL Loss: {loss.kl_loss:.{n_digits}f}\t"
            f"Jac Loss: {loss.jac_est_loss:.{n_digits}f}\t"
            f"ELBO Loss: {loss.elbo_loss:.{n_digits}f}"
        )

    def print_epoch(self, epoch, epochs):
        print(f"Epoch {self.epoch_start + epoch + 1} / {self.epoch_start + epochs} complete!")

    def print_loss(self, loss: LossBatch, n_digits=3, loss_mode="Training"):
        print(f"\t{loss_mode}:\t{self._format_losses(loss, n_digits)}")

    # -----------------
    # Plotting & saving
    # -----------------
    def plt_training_loss(
        self,
        epoch_threshold,
        model_dir=None,
        training_loss=None,
        validation_loss=None,
        test_loss=None,
        model_name=None,
        data_name=None,
        plot_mode="train",
        decimals=None,
    ):
        if decimals is None:
            decimals = 4 if getattr(self.model, "decoder", None) in ["nb", "zinb"] else 1

        training_loss = training_loss or self.train_loss
        validation_loss = validation_loss or self.valid_loss
        model_dir = model_dir or self.model_dir
        if plot_mode in ["test", "analysis"]:
            if self.test_loss['overall_loss']:
                test_loss = test_loss or self.test_loss
        data_name = data_name or "PDX"

        plt_training_loss(
            epoch_threshold=epoch_threshold,
            model_dir=model_dir,
            training_loss=training_loss,
            validation_loss=validation_loss,
            test_loss=test_loss,
            model_name=model_name,
            data_name=data_name,
            plot_mode=plot_mode,
            decimals=decimals,
        )

    def save_loss(self, str_save, decimals):
        save_loss(self.train_loss, self.valid_loss, self.test_loss, str_save=str_save, decimals=decimals)

    def _collect_batches(self, loader, fn, device):
        out = []
        with torch.no_grad():
            for data in loader:
                out.append(fn(*[items.to(device) for items in data]))
        return out


class ModelTrainerIVAE(ModelTrainerBase):
    """
    Trainer for iVAE models.

    Major features:
    - modular train/valid epoch separation
    - robust NaN checks and optional gradient clipping
    - early stopping integration (uses EarlyStopper.step)
    - optional periodic plotting (delegates to base plt_training_loss and plt_training_rec)
    - infer/test/decode helpers
    """
    DEFAULT_PLOT_INTERVALS = [0, 100, 500, 1000]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # -------------------------
    # Public API
    # -------------------------
    def train_ivae(
        self,
        epochs: int,
        device: Optional[torch.device] = None,
        train_loader: torch.utils.data.DataLoader = None,
        valid_loader: Optional[torch.utils.data.DataLoader] = None,
        beta_loss: float = 1.0,
        jacobian_loss: bool = False, # --- jacobian loss
        jacobian_lambda: float = 10.0,
        jacobian_loss_mode: str = "l1",
        max_norm: Optional[float] = None, # --- gradient clipping
        patience: int = 50, # --- early stopping
        min_delta: float = 0.0,
        rec_loader: Optional[torch.utils.data.DataLoader] = None, # ---reconstruction plotting
        rec_samples: int = 16,
        rec_num_rows: Optional[int] = None,
        rec_cmap: str = "viridis",
        plot_intervals: Optional[List[int]] = None,
        callback: Optional[Callable[[int, dict], Any]] = None,
        **kwargs,
    ) -> dict:
        """
        Train the iVAE model.

        Returns a dictionary with summary information (best_epoch, best_valid_loss, n_epochs_trained).
        """
        device = device or self.device
        self.model.to(device)
        plot_intervals = plot_intervals or self.DEFAULT_PLOT_INTERVALS

        has_validation = valid_loader is not None
        if not has_validation:
            self._log("No validation provided: model will only save at regular intervals.", level=1)

        n_train_batches = len(train_loader)
        n_valid_batches = len(valid_loader) if has_validation else 1
        early_stopper = EarlyStopper(patience=patience, min_delta=min_delta, mode="min", verbose=(self.verbose >= 2))

        self._log(f"Start training iVAE for {epochs} epochs on device {device}.", level=1)
        start_time = time.time()

        for epoch in range(epochs):
            epoch_idx = self.epoch_start + epoch + 1
            if self.verbose >= 3:
                t0 = time.time()

            train_loss, nan_batches = self._run_and_wrap_epoch(
                train_loader, device, train=True, n_batches=n_train_batches,
                jacobian_loss=jacobian_loss, jacobian_lambda=jacobian_lambda,
                jacobian_loss_mode=jacobian_loss_mode, beta_loss=beta_loss,
                max_norm=max_norm,
            )

            valid_loss = self._run_validation_epoch(
                valid_loader=valid_loader,
                has_validation=has_validation,
                n_valid_batches=n_valid_batches,
                device=device,
                jacobian_loss=jacobian_loss,
                jacobian_lambda=jacobian_lambda,
                jacobian_loss_mode=jacobian_loss_mode,
                beta_loss=beta_loss,
            )

            self._log_epoch_summary(
                epoch=epoch,
                epochs=epochs,
                train_loss=train_loss,
                valid_loss=valid_loss,
                has_validation=has_validation,
            )
            self._append_loss_dict(self.train_loss, train_loss)
            self._append_loss_dict(self.valid_loss, valid_loss)

            if self._maybe_abort_on_nan(train_loss=train_loss, epoch_idx=epoch_idx):
                break

            self._handle_epoch_artifacts(
                epoch=epoch,
                epoch_idx=epoch_idx,
                valid_loss=valid_loss,
                plot_intervals=plot_intervals,
                rec_loader=rec_loader,
                rec_samples=rec_samples,
                rec_num_rows=rec_num_rows,
                rec_cmap=rec_cmap,
                device=device,
            )

            if self.verbose >= 3:
                t1 = time.time()
                epoch_time = t1 - t0
                remaining = (epochs - epoch - 1) * epoch_time
                self._log(f"Epoch time: {epoch_time:.2f}s | Estimated remaining: {remaining/60:.2f} min", level=3)

            should_stop = early_stopper.step(valid_loss.elbo_loss, epoch=epoch_idx) if has_validation else False
            if should_stop:
                self._log(f"Early stopping triggered at epoch {epoch_idx}", level=1)
                break

            self._maybe_call_callback(
                callback=callback,
                epoch_idx=epoch_idx,
                train_loss=train_loss,
                valid_loss=valid_loss,
                has_validation=has_validation,
                nan_batches=nan_batches,
            )

        total_time = time.time() - start_time
        self._log(f"Training finished in {total_time/60:.2f} minutes", level=1)
        write_status(self.model_dir, status="OK", msg=f"Training finished. Best val loss: {self.valid_loss_best:.4f}")
        self.plt_training_loss(epoch_threshold=0, model_dir=self.model_dir)

        return {
            "best_epoch": self.epoch_best,
            "best_valid_loss": self.valid_loss_best,
            "trained_epochs": epoch + 1
        }

    def test_ivae(self,
        test_loader,
        device=None,
        jacobian_loss=False,
        jacobian_lambda=10,
        jacobian_loss_mode='l1',
        beta_loss=1.0
    ):

        device = device or self.device
        n_test_batches = len(test_loader)

        test_loss, _ = self._run_and_wrap_epoch(
            test_loader, device, train=False, n_batches=n_test_batches,
            jacobian_loss=jacobian_loss, jacobian_lambda=jacobian_lambda,
            jacobian_loss_mode=jacobian_loss_mode, beta_loss=beta_loss,
        )

        self._append_loss_dict(self.test_loss, test_loss)
        self.print_loss(test_loss, loss_mode="Test")

    def infer_ivae(self, infer_loader, device=None):
        device = device or self.device
        self.model.to(device)
        self.model.eval()

        # outputs of form [embedding/z/prior][embedding 1 or 2 / prior 1 or 2][mean / variance]
        outputs = self._collect_batches(
            infer_loader,
            lambda *x: self.model.forward(*x)[1:4],
            device,
        )

        # enc: embedding_list of type [(mean, variance), (u_mean, u_variance) for each batch]
        enc = [
            (
                torch.cat([o[0][i][0] for o in outputs], dim=0),
                torch.cat([o[0][i][1] for o in outputs], dim=0)
            ) for i in range(len(outputs[0][0]))
        ]
        z  = torch.cat([o[1] for o in outputs], dim=0)
        prior = [
            (
                torch.cat([o[2][i][0] for o in outputs], dim=0),
                torch.cat([o[2][i][1] for o in outputs], dim=0)
            ) for i in range(len(outputs[0][0]))
        ]
        # enc_means = torch.cat([o[0][0][0] for o in outputs], dim=0)
        # enc_vars  = torch.cat([o[0][0][1] for o in outputs], dim=0)
        # z  = torch.cat([o[1] for o in outputs], dim=0)
        # prior_means = torch.cat([o[2][0][0] for o in outputs], dim=0)
        # prior_vars  = torch.cat([o[2][0][1] for o in outputs], dim=0)

        # return (enc_means, enc_vars), z, (prior_means, prior_vars)
        return enc, z, prior

    def decode_ivae(self, decode_loader, device=None):
        device = device or self.device
        self.model.to(device)
        self.model.eval()

        outputs = self._collect_batches(
            decode_loader,
            lambda *x: self.model.forward(*x)[0],
            device,
        )

        means = torch.cat([o[0] for o in outputs], dim=0)

        if self.model.decoder_var_mode == "dec_nn":
            vars_ = torch.cat([o[1] for o in outputs], dim=0)
        else:
            vars_ = self.model.decoder_var

        return means, vars_

    def decode_from_latent(self, decode_loader, new_latents, device=None):
        device = device or self.device
        self.model.to(device)
        self.model.eval()
        if new_latents.dim() not in (2, 3):
            raise ValueError(
                "new_latents must have shape [batch, latent] or [samples, batch, latent]"
            )

        has_sample_dim = new_latents.dim() == 3
        n_samples = new_latents.shape[0] if has_sample_dim else None

        outputs = []
        with torch.no_grad():

            curr_idx = 0
            for data in decode_loader:
                batch_len = data[0].shape[0]

                _, u, aux_info, library_size = self.model.return_vars(*[items.to(device) for items in data])

                if has_sample_dim:
                    # Flatten sample and batch dimensions for the decoder, then restore them below.
                    z_cat = new_latents[:, curr_idx:curr_idx + batch_len].to(device)
                    z_cat = z_cat.reshape(n_samples * batch_len, z_cat.shape[-1])
                    u = [self._repeat_batch_tensor(item, has_sample_dim, n_samples) if item is not None else None for item in u]
                    aux_info = self._repeat_batch_tensor(aux_info, has_sample_dim, n_samples)
                    library_size = self._repeat_batch_tensor(library_size, has_sample_dim, n_samples)
                else:
                    z_cat = new_latents[curr_idx:curr_idx + batch_len].to(device)

                curr_idx += batch_len
                decoder_params = self.model.decoder_params(z_cat, *u, aux_info, library_size)

                if has_sample_dim:
                    if isinstance(decoder_params, tuple):
                        decoder_params = tuple(
                            self._reshape_decoder_param(param, has_sample_dim, n_samples, batch_len)
                            for param in decoder_params
                        )
                    else:
                        decoder_params = [
                            self._reshape_decoder_param(param, has_sample_dim, n_samples, batch_len)
                            for param in decoder_params
                        ]

                outputs.append(decoder_params)

        cat_dim = 1 if has_sample_dim else 0
        means = torch.cat([o[0] for o in outputs], dim=cat_dim)

        if self.model.decoder_var_mode == "dec_nn":
            vars_ = torch.cat([o[1] for o in outputs], dim=cat_dim)
        else:
            vars_ = self.model.decoder_var

        return means, vars_

    # -------------------------
    # Epoch execution core
    # -------------------------
    def _run_and_wrap_epoch(
        self, loader, device, *, train: bool, n_batches: int,
        jacobian_loss: bool, jacobian_lambda: float, jacobian_loss_mode: str,
        beta_loss: float, max_norm: Optional[float] = None
    ) -> Tuple[LossBatch, int]:
        metrics, nan_batches = self._run_epoch(
            loader, device,
            jacobian_loss=jacobian_loss,
            jacobian_lambda=jacobian_lambda,
            jacobian_loss_mode=jacobian_loss_mode,
            beta_loss=beta_loss,
            max_norm=max_norm if train else None,
            train=train,
        )
        lossbatch = self._to_lossbatch(
            metrics, beta_loss, max(1, n_batches - (nan_batches if train else 0))
        )
        return lossbatch, nan_batches

    def _run_epoch(
        self,
        loader, device,
        jacobian_loss: bool, jacobian_lambda: float, jacobian_loss_mode: str,
        beta_loss: float, max_norm: Optional[float] = None, train: bool = True,
    ) -> Tuple[dict, int]:
        """
        Run one epoch (training, validation, or test).

        Returns:
            metrics dict (averaged per sample, not per batch!)
            and number of NaN batches skipped (0 if not training)
        """
        overall_loss = elbo_loss = rec_loss = kl_loss = bkl_loss = jac_est_loss = 0.0
        nan_batches = 0
        total_samples = 0

        if train:
            self.model.train()
            context = torch.enable_grad()
        else:
            self.model.eval()
            context = torch.no_grad()

        # actual training
        with context:
            for batch in loader:
                batch_size = batch[0].shape[0]  # assumes first tensor has batch dim
                if train:
                    self.optimizer.zero_grad()

                try:
                    items = [it.to(device) for it in batch]
                    rec_batch_loss, kl_batch_loss, z = self.model.elbo(*items)
                except Exception as e:
                    if train:
                        self._log(f"Forward pass failed: {e}. Skipping batch.", level=1)
                        nan_batches += 1
                        continue
                    else:
                        raise RuntimeError("Error in validation or test batch") from e

                # compute jacobian loss and beta KL loss if applicable
                jac_est_batch_loss = jacobian_lambda * jacobian_loss_function(
                    self.model, z, use_jacobian_loss=jacobian_loss, loss_mode=jacobian_loss_mode, device=device
                )
                beta_kl_batch_loss = beta_loss * kl_batch_loss
                loss = rec_batch_loss + beta_kl_batch_loss + jac_est_batch_loss

                if train and (torch.isnan(loss) or not torch.isfinite(loss)):
                    self._log("Detected NaN/inf loss in batch -> skipping", level=1)
                    nan_batches += 1
                    continue

                # accumulate weighted by batch size
                overall_loss += loss.item() * batch_size
                elbo_loss += (rec_batch_loss + kl_batch_loss).item() * batch_size
                rec_loss += rec_batch_loss.item() * batch_size
                kl_loss += kl_batch_loss.item() * batch_size
                bkl_loss += beta_kl_batch_loss.item() * batch_size
                jac_est_loss += jac_est_batch_loss.item() * batch_size

                total_samples += batch_size

                if train:
                    loss.backward()
                    if max_norm is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                    self.optimizer.step()

        # avoid division by zero
        denom = max(1, total_samples)
        metrics = {
            "overall": overall_loss / denom,
            "elbo": elbo_loss / denom,
            "rec": rec_loss / denom,
            "kl": kl_loss / denom,
            "bkl": bkl_loss / denom,
            "jac_est_loss": jac_est_loss / denom,
        }

        return metrics, nan_batches if train else 0

    def _to_lossbatch(self, metrics: dict, beta_loss: float, n_batches: int) -> LossBatch:
        return LossBatch(
            beta_loss=beta_loss,
            overall_loss=metrics["overall"],
            elbo_loss=metrics["elbo"],
            rec_loss=metrics["rec"],
            kl_loss=metrics["kl"],
            jac_est_loss=metrics["jac_est_loss"],
            n_batches=n_batches,
        )

    # -------------------------
    # Training loop helpers
    # -------------------------
    def _run_validation_epoch(
        self,
        valid_loader,
        has_validation: bool,
        n_valid_batches: int,
        device,
        jacobian_loss: bool,
        jacobian_lambda: float,
        jacobian_loss_mode: str,
        beta_loss: float,
    ) -> LossBatch:
        if has_validation:
            valid_loss, _ = self._run_and_wrap_epoch(
                valid_loader, device, train=False, n_batches=n_valid_batches,
                jacobian_loss=jacobian_loss, jacobian_lambda=jacobian_lambda,
                jacobian_loss_mode=jacobian_loss_mode, beta_loss=beta_loss,
            )
            return valid_loss
        return LossBatch(beta_loss, 0, 0, 0, 0, 0, n_batches=1)

    def _log_epoch_summary(
        self,
        epoch: int,
        epochs: int,
        train_loss: LossBatch,
        valid_loss: LossBatch,
        has_validation: bool,
    ) -> None:
        self.print_epoch(epoch, epochs)
        self.print_loss(train_loss, loss_mode="Training")
        if has_validation:
            self.print_loss(valid_loss, loss_mode="Validation")

    def _maybe_abort_on_nan(self, train_loss: LossBatch, epoch_idx: int) -> bool:
        if np.isnan(train_loss.overall_loss):
            msg = f"NAN detected in epoch {epoch_idx}; aborting training."
            fail_path = write_status(self.model_dir, status="FAIL", msg=msg, exit_code=1)
            self._log(f"{msg} Log written to {fail_path}", level=1)
            return True
        return False

    def _handle_epoch_artifacts(
        self,
        epoch: int,
        epoch_idx: int,
        valid_loss: LossBatch,
        plot_intervals: List[int],
        rec_loader,
        rec_samples: int,
        rec_num_rows: Optional[int],
        rec_cmap: str,
        device,
    ) -> None:
        if epoch % self.epoch_checkpoint_freq != 0:
            return

        for threshold in plot_intervals:
            try:
                self.plt_training_loss(epoch_threshold=threshold, model_dir=self.model_dir)
            except Exception as e:
                self._log(
                    f"Plotting training loss failed at epoch {epoch_idx} for threshold {threshold}: {e}",
                    level=1,
                )

        if rec_loader is not None:
            try:
                plot_vae_reconstructions(
                    self.model,
                    rec_loader,
                    rec_samples,
                    num_rows=rec_num_rows,
                    device=device,
                    save_dir=self.model_dir / "reconstructions",
                    cmap=rec_cmap,
                    epoch=epoch,
                    model_decoder=self.model.decoder,
                )
            except Exception as e:
                self._log(f"Plotting reconstructions failed at epoch {epoch_idx}: {e}", level=1)

        self.save_model_ckp(epoch=epoch, valid_loss=valid_loss.elbo_loss)

    def _maybe_call_callback(
        self,
        callback: Optional[Callable[[int, dict], Any]],
        epoch_idx: int,
        train_loss: LossBatch,
        valid_loss: LossBatch,
        has_validation: bool,
        nan_batches: int,
    ) -> None:
        if callback is None:
            return
        try:
            callback(epoch_idx, {
                "train": train_loss,
                "valid": valid_loss if has_validation else {},
                "nan_batches": nan_batches
            })
        except Exception:
            self._log("User callback raised an exception; continuing training.", level=1)

    # -------------------------
    # Latent decoding helpers
    # -------------------------
    @staticmethod
    def _repeat_batch_tensor(tensor, has_sample_dim: bool, n_samples: Optional[int]):
        if tensor is None or not has_sample_dim:
            return tensor
        return tensor.unsqueeze(0).expand(n_samples, *tensor.shape).reshape(
            n_samples * tensor.shape[0], *tensor.shape[1:]
        )

    @staticmethod
    def _reshape_decoder_param(param, has_sample_dim: bool, n_samples: Optional[int], batch_len: int):
        if not has_sample_dim or not torch.is_tensor(param):
            return param
        return param.reshape(n_samples, batch_len, *param.shape[1:])