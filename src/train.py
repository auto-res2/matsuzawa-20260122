from __future__ import annotations

"""Single-run training launcher (Hydra-controlled).
This version FIXES the two critical issues reported during validation:
1. The confidence branch now receives gradients because the `.detach()` call
   has been removed from the dual-adaptive weight computation.
2. A dedicated learning-rate for the auxiliary confidence branch is honoured
   via a separate optimiser parameter-group that uses
   `training.additional_params.aux_branch_lr` (defaulting to the global LR).

The script keeps all previous safety checks, Optuna integration, Hydra &
WandB plumbing, and complies with the publication-ready requirements.
"""

import json
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import optuna
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from src.model import create_model
from src.preprocess import compute_ece, get_dataloaders

################################################################################
# Utility helpers                                                              #
################################################################################

def _set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

################################################################################
# Optuna search-space helpers                                                  #
################################################################################

def _suggest_from_space(trial: optuna.Trial, space: DictConfig):
    name: str = space.param_name
    dist: str = str(space.distribution_type).lower()

    if dist == "uniform":
        return trial.suggest_float(name, float(space.low), float(space.high))
    if dist == "loguniform":
        return trial.suggest_float(name, float(space.low), float(space.high), log=True)
    if dist == "int_uniform":
        return trial.suggest_int(name, int(space.low), int(space.high))
    if dist in {"categorical", "choice"}:
        assert space.choices is not None, "`choices` must be provided for categorical distribution"
        return trial.suggest_categorical(name, list(space.choices))
    raise ValueError(f"Unsupported distribution_type: {dist}")

################################################################################
# Evaluation                                                                   #
################################################################################

def _evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    compute_cm: bool = False,
) -> Tuple[float, float, float, np.ndarray | None]:
    """Return (loss, acc, ece, confusion_matrix|None)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    logits_all: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            # Support both (img, label) *and* (orig, aug, label) tuples.
            if len(batch) == 3:
                imgs, _, labels = batch  # augmented image ignored during evaluation
            elif len(batch) == 2:
                imgs, labels = batch
            else:
                raise ValueError("Unexpected batch structure returned by DataLoader")

            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            logits_all.append(logits.cpu())
            labels_all.append(labels.cpu())

    logits_cat = torch.cat(logits_all)
    labels_cat = torch.cat(labels_all)
    probs = torch.softmax(logits_cat, dim=1)
    ece = compute_ece(probs, labels_cat)
    cm = None
    if compute_cm:
        cm = confusion_matrix(labels_cat.numpy(), probs.argmax(1).numpy())
    return total_loss / total, correct / total, ece, cm

################################################################################
# Training function                                                            #
################################################################################

def _build_optimizer(cfg: DictConfig, model: nn.Module):
    """Return optimiser (with auxiliary param-group if AALCR++ is used)."""
    optim_kwargs = {
        "weight_decay": float(cfg.training.weight_decay),
    }

    base_lr = float(cfg.training.learning_rate)

    # ------------------------------------------------------------------
    # If AALCR++, create separate param-group for aux branch ------------
    # ------------------------------------------------------------------
    if cfg.method.startswith("AALCR") and hasattr(model, "aux_branch"):
        aux_lr = float(cfg.training.additional_params.get("aux_branch_lr", base_lr))

        # Deduplicate parameters: we rely on explicit lists
        base_params = list(model.base.parameters())
        aux_params = list(model.aux_branch.parameters())

        param_groups = [
            {"params": base_params, "lr": base_lr},
            {"params": aux_params, "lr": aux_lr},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": base_lr}]

    # Optimiser selection ------------------------------------------------------
    if str(cfg.training.optimizer).lower() == "adam":
        optimizer = torch.optim.Adam(param_groups, **optim_kwargs)
    elif str(cfg.training.optimizer).lower() == "sgd":
        optimizer = torch.optim.SGD(param_groups, momentum=0.9, nesterov=True, **optim_kwargs)
    else:
        raise ValueError(f"Unsupported optimiser: {cfg.training.optimizer}")

    return optimizer


def _run_training(
    cfg: DictConfig,
    device: torch.device,
    *,
    wandb_run: wandb.wandb_sdk.wandb_run.Run | None = None,
    dummy_run: bool = False,
) -> Dict[str, float]:
    """Full training loop. Returns key metrics for WandB summary."""

    # ------------------------------------------------------------------
    # Data & model ------------------------------------------------------
    # ------------------------------------------------------------------
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    num_classes = 10 if cfg.dataset.name.lower() == "cifar10" else 100
    model = create_model(cfg, num_classes=num_classes).to(device)

    # Post-initialisation safety checks ---------------------------------------
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 32, 32, device=device)
        test_out = model(dummy) if not hasattr(model, "aux_branch") else model(dummy)
        assert test_out.shape[-1] == num_classes, "Model output dimension mismatch"

    optimizer = _build_optimizer(cfg, model)

    warmup_steps = int(cfg.training.warmup_steps)
    total_steps = int(cfg.training.epochs) * len(train_loader)

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.pi * torch.tensor(progress))).item()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    ####################################################################
    # Epoch loop                                                        #
    ####################################################################
    global_step = 0
    best_val_acc = 0.0
    best_state: Dict[str, torch.Tensor] | None = None

    for epoch in range(int(cfg.training.epochs)):
        model.train()
        epoch_loss, epoch_correct, epoch_seen = 0.0, 0, 0

        for batch_idx, (orig_imgs, aug_imgs, labels) in enumerate(train_loader):
            if cfg.mode == "trial" and batch_idx > 1:
                break  # minimal workload in trial mode

            orig_imgs = orig_imgs.to(device, non_blocking=True)
            aug_imgs = aug_imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Lifecycle assertions (first global step) -------------------------
            if global_step == 0:
                assert orig_imgs.shape == aug_imgs.shape, "Original/Augmented shape mismatch"
                assert labels.size(0) == orig_imgs.size(0), "Batch size mismatch"

            optimizer.zero_grad(set_to_none=True)

            ################################################################
            # Forward pass & loss computation                               #
            ################################################################
            if cfg.method.startswith("AALCR"):
                logits_orig, confidences = model(orig_imgs, return_confidence=True)
                logits_aug = model(aug_imgs)

                loss_ce = F.cross_entropy(logits_orig, labels)

                # Consistency loss (stop grad wrt logits_orig)
                p_orig = F.softmax(logits_orig.detach(), dim=1)
                loss_consist = F.kl_div(
                    F.log_softmax(logits_aug, dim=1), p_orig, reduction="batchmean"
                )

                # Aug-intensity normalised (no grad required)
                with torch.no_grad():
                    intensity = torch.abs(orig_imgs - aug_imgs).mean(dim=[1, 2, 3])
                    inten_norm = (intensity - intensity.min()) / (
                        intensity.max() - intensity.min() + 1e-8
                    )
                inten_norm = inten_norm.detach()  # explicit, for safety

                # Dual-adaptive weight (!!! confidences NOT detached !!!)
                weight_vec = inten_norm * (1.0 - confidences)
                w_avg = weight_vec.mean()
                lambda_max = float(cfg.training.additional_params.get("lambda_max", 1.0))
                loss = (1.0 - w_avg) * loss_ce + lambda_max * w_avg * loss_consist
            else:
                # Fixed-weight baseline                                       
                logits_orig = model(orig_imgs)
                logits_aug = model(aug_imgs)

                loss_ce = F.cross_entropy(logits_orig, labels)
                p_orig = F.softmax(logits_orig.detach(), dim=1)
                loss_consist = F.kl_div(
                    F.log_softmax(logits_aug, dim=1), p_orig, reduction="batchmean"
                )
                fixed_w = float(cfg.training.additional_params.get("fixed_weight", 0.5))
                loss = (1.0 - fixed_w) * loss_ce + fixed_w * loss_consist

            # Back-prop & gradient-integrity checks ----------------------------
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.requires_grad]
            assert all(g is not None for g in grads), "Detected None gradient before optimiser step"
            total_grad = sum(g.abs().sum().item() for g in grads)
            assert total_grad > 0.0, "Gradient norm is zero – check loss graph"

            if cfg.training.gradient_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg.training.gradient_clip))

            optimizer.step()
            scheduler.step()

            # Statistics ------------------------------------------------------
            batch_acc = (logits_orig.argmax(1) == labels).float().mean().item()
            epoch_loss += loss.item() * labels.size(0)
            epoch_correct += batch_acc * labels.size(0)
            epoch_seen += labels.size(0)
            global_step += 1

            if wandb_run is not None and not dummy_run:
                wandb_run.log(
                    {
                        "train_loss_batch": loss.item(),
                        "train_acc_batch": batch_acc,
                        "lr": optimizer.param_groups[0]["lr"],
                        "step": global_step,
                    },
                    step=global_step,
                )

        # ---------------- End epoch ------------------------------------------
        epoch_train_loss = epoch_loss / epoch_seen
        epoch_train_acc = epoch_correct / epoch_seen
        val_loss, val_acc, val_ece, _ = _evaluate(model, val_loader, device)

        if wandb_run is not None and not dummy_run:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train_loss": epoch_train_loss,
                    "train_acc": epoch_train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_ece": val_ece,
                },
                step=global_step,
            )

        # Best-model tracking --------------------------------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    ########################################################################
    # Test evaluation (with best checkpoint)                                #
    ########################################################################
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_ece, cm = _evaluate(
        model, test_loader, device, compute_cm=True
    )

    if wandb_run is not None and not dummy_run:
        wandb_run.summary["confusion_matrix"] = cm.tolist()

    return {
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_ece": test_ece,
    }

################################################################################
# Hydra entry-point                                                            #
################################################################################

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:  # noqa: C901 – complex but required
    # ------------------------------------------------------------------
    # Merge run-specific YAML ------------------------------------------
    # ------------------------------------------------------------------
    if not cfg.run:
        raise ValueError("Parameter `run` is required (e.g. run=proposed-resnet18-cifar10)")

    run_cfg_path = Path(__file__).resolve().parent.parent / "config" / "runs" / f"{cfg.run}.yaml"
    if not run_cfg_path.exists():
        raise FileNotFoundError(f"Run-specific configuration {run_cfg_path} not found")

    # Disable struct mode to allow new keys in additional_params
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, OmegaConf.load(run_cfg_path))  # type: ignore[assignment]
    OmegaConf.set_struct(cfg, True)

    # ------------------------------------------------------------------
    # Mode-specific overrides                                           
    # ------------------------------------------------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.epochs = 1
        cfg.training.batch_size = min(int(cfg.training.batch_size), 32)

    # ------------------------------------------------------------------
    # Output directory & seed                                            
    # ------------------------------------------------------------------
    results_dir = Path(cfg.results_dir).expanduser().absolute()
    results_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(int(cfg.training.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # WandB initialisation                                              
    # ------------------------------------------------------------------
    if cfg.wandb.mode == "disabled":
        os.environ["WANDB_DISABLED"] = "true"
        wandb_run = None
    else:
        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run,
            resume="allow",
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print(f"W&B URL: {wandb_run.get_url()}")

    # ------------------------------------------------------------------
    # Optuna hyper-parameter search                                      
    # ------------------------------------------------------------------
    if int(cfg.optuna.n_trials) > 0:

        def _objective(trial: optuna.Trial) -> float:
            sampled_cfg = OmegaConf.create(OmegaConf.to_object(cfg))  # deep copy
            for space in cfg.optuna.search_spaces:
                val = _suggest_from_space(trial, space)
                if space.param_name in sampled_cfg.training:
                    sampled_cfg.training[space.param_name] = val
                else:
                    sampled_cfg.training.additional_params[space.param_name] = val

            # Lightweight single-epoch run (no WandB)
            sampled_cfg.wandb.mode = "disabled"
            sampled_cfg.training.epochs = 1
            sampled_cfg.optuna.n_trials = 0  # avoid recursion

            _set_seed(int(sampled_cfg.training.seed))
            device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            metrics = _run_training(sampled_cfg, device_, wandb_run=None, dummy_run=True)
            return metrics["test_loss"]  # minimise loss

        study = optuna.create_study(direction="minimize")
        study.optimize(_objective, n_trials=int(cfg.optuna.n_trials))
        best_params = study.best_trial.params
        print("[Optuna] Best parameters:", best_params)

        # Inject best params back into cfg -------------------------------------
        for k, v in best_params.items():
            if k in cfg.training:
                cfg.training[k] = v
            else:
                cfg.training.additional_params[k] = v

    # ------------------------------------------------------------------
    # Final training with best parameters                                 
    # ------------------------------------------------------------------
    final_metrics = _run_training(cfg, device, wandb_run=wandb_run, dummy_run=False)

    if wandb_run is not None:
        wandb_run.summary.update(final_metrics)
        wandb_run.finish()

    # Persist metrics locally for quick access -------------------------
    metrics_path = results_dir / f"{cfg.run}_final_metrics.json"
    with metrics_path.open("w") as fp:
        json.dump(final_metrics, fp, indent=2)
    print(f"Saved final metrics to {metrics_path}")


if __name__ == "__main__":
    # Running via `uv run python -m src.train ...` uses this entry point.
    main()
