"""
XRayEarth — train.py
Seeing through disaster with satellite vision.

Responsibilities:
    - Full training loop with AMP (FP16) mixed precision
    - Gradient scaling to prevent FP16 underflow
    - WandB logging (losses, metrics, GPU stats)
    - Checkpoint saving (latest + best)
    - Early stopping
    - Config-driven experiment runs
    - Smoke test mode (2 batches, no WandB)
    - Epoch-level validation with macro F1

CLI:
    python src/train.py --config configs/v1.yaml
    python src/train.py --config configs/v1.yaml --smoke-test
    python src/train.py --config configs/v1.yaml --resume outputs/checkpoints/v1_latest.pth
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from rich.table import Table

# Local imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_config,
    set_seed,
    get_device,
    setup_logging,
    save_checkpoint,
    load_checkpoint,
    init_wandb,
    print_config,
    print_banner,
    ensure_dirs,
    format_metrics,
    get_gpu_memory_mb,
    console,
)
from dataset  import build_dataloader, XBDDataset, compute_class_distribution
from model    import build_model, freeze_backbone, unfreeze_backbone
from loss     import XRayEarthLoss, build_classification_loss
from eval     import evaluate


# ═══════════════════════════════════════════════════════════
#  1. ARGUMENT PARSER
# ═══════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XRayEarth — Disaster Damage Assessment Training"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to experiment config (e.g. configs/v1.yaml)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run 2 batches only — no WandB, quick sanity check"
    )
    parser.add_argument(
        "--max-batches", type=int, default=2,
        help="Max batches per epoch in smoke test mode"
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable WandB logging"
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════
#  2. OPTIMIZER & SCHEDULER
# ═══════════════════════════════════════════════════════════

def build_optimizer(
    model: nn.Module,
    cfg,
) -> torch.optim.Optimizer:
    """
    Build AdamW optimizer.

    Uses different LR for backbone vs heads:
        - Backbone: lr * 0.1  (fine-tuning, lower rate)
        - Heads:    lr        (full rate)

    Args:
        model: XRayEarthModel
        cfg:   Experiment config

    Returns:
        Configured optimizer
    """
    lr           = cfg.training.learning_rate
    weight_decay = cfg.training.weight_decay

    # Separate backbone and head parameters
    backbone_params = []
    head_params     = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {"params": head_params,     "lr": lr},
        {"params": backbone_params, "lr": lr * 0.1},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=weight_decay,
    )

    console.log(
        f"[green]✓[/green] Optimizer: AdamW "
        f"(lr={lr}, backbone_lr={lr*0.1:.2e}, "
        f"weight_decay={weight_decay})"
    )
    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build OneCycleLR scheduler.

    OneCycleLR warms up LR then anneals — works well
    with small batch sizes on GPU-constrained setups.

    Args:
        optimizer:       Configured optimizer
        cfg:             Experiment config
        steps_per_epoch: Number of batches per epoch

    Returns:
        Configured LR scheduler
    """
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr         = cfg.training.learning_rate,
        epochs         = cfg.training.epochs,
        steps_per_epoch= steps_per_epoch,
        pct_start      = 0.1,    # 10% warmup
        anneal_strategy= "cos",
        div_factor     = 25,     # start_lr = max_lr / 25
        final_div_factor=1e4,    # end_lr = start_lr / 10000
    )

    console.log(
        f"[green]✓[/green] Scheduler: OneCycleLR "
        f"(epochs={cfg.training.epochs}, "
        f"steps_per_epoch={steps_per_epoch})"
    )
    return scheduler


# ═══════════════════════════════════════════════════════════
#  3. TRAINING STEP
# ═══════════════════════════════════════════════════════════

def train_one_step(
    model:     nn.Module,
    pre_imgs:  list,
    post_imgs: list,
    targets:   list,
    optimizer: torch.optim.Optimizer,
    scaler:    GradScaler,
    loss_wrapper: "XRayEarthLoss",
    device:    torch.device,
    cfg,
) -> Dict[str, float]:
    """
    Single training step with AMP.

    Args:
        model:        XRayEarthModel
        pre_imgs:     List of pre-disaster tensors
        post_imgs:    List of post-disaster tensors
        targets:      List of target dicts
        optimizer:    AdamW optimizer
        scaler:       GradScaler for FP16
        loss_wrapper: XRayEarthLoss
        device:       CUDA/CPU device
        cfg:          Experiment config

    Returns:
        Dict of scalar loss values for logging
    """
    # Move to device
    pre_imgs  = [img.to(device)  for img  in pre_imgs]
    post_imgs = [img.to(device)  for img  in post_imgs]
    targets   = [
        {k: v.to(device) for k, v in t.items()}
        for t in targets
    ]

    optimizer.zero_grad()

    # ── Mixed Precision Forward Pass ──────────────────────
    with autocast():
        loss_dict = model(pre_imgs, post_imgs, targets)

    # ── Compute total weighted loss ───────────────────────
    total_loss, scalar_losses = loss_wrapper.compute_total_loss(loss_dict)

    # ── Scaled Backward Pass ──────────────────────────────
    scaler.scale(total_loss).backward()

    # ── Gradient Clipping (prevent exploding gradients) ───
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=cfg.training.clip_grad_norm,
    )

    # ── Optimizer Step ────────────────────────────────────
    scaler.step(optimizer)
    scaler.update()

    return scalar_losses


# ═══════════════════════════════════════════════════════════
#  4. TRAINING EPOCH
# ═══════════════════════════════════════════════════════════

def train_one_epoch(
    model:        nn.Module,
    dataloader:   DataLoader,
    optimizer:    torch.optim.Optimizer,
    scheduler,
    scaler:       GradScaler,
    loss_wrapper: "XRayEarthLoss",
    device:       torch.device,
    cfg,
    epoch:        int,
    smoke_test:   bool = False,
    max_batches:  int  = 2,
) -> Dict[str, float]:
    """
    Run one full training epoch.

    Args:
        model:        XRayEarthModel in train mode
        dataloader:   Training DataLoader
        optimizer:    Optimizer
        scheduler:    LR scheduler
        scaler:       AMP GradScaler
        loss_wrapper: XRayEarthLoss
        device:       Compute device
        cfg:          Experiment config
        epoch:        Current epoch number
        smoke_test:   If True, stop after max_batches
        max_batches:  Max batches in smoke test mode

    Returns:
        Dict of averaged loss values for the epoch
    """
    model.train()

    epoch_losses: Dict[str, float] = {}
    num_batches = 0

    pbar = tqdm(
        dataloader,
        desc    = f"Epoch {epoch:03d} [train]",
        leave   = False,
        dynamic_ncols = True,
    )

    for batch_idx, (pre_imgs, post_imgs, targets, _) in enumerate(pbar):

        # Smoke test: stop early
        if smoke_test and batch_idx >= max_batches:
            break

        # Skip batches with no annotations
        if all(len(t["boxes"]) == 0 for t in targets):
            continue

        step_losses = train_one_step(
            model, pre_imgs, post_imgs, targets,
            optimizer, scaler, loss_wrapper, device, cfg,
        )

        # Step scheduler
        scheduler.step()

        # Accumulate losses
        for k, v in step_losses.items():
            epoch_losses[k] = epoch_losses.get(k, 0.0) + v

        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{step_losses.get('loss_total', 0):.4f}",
            "lr":   f"{scheduler.get_last_lr()[0]:.2e}",
            "gpu":  f"{get_gpu_memory_mb():.0f}MB",
        })

    # Average losses over epoch
    if num_batches > 0:
        epoch_losses = {k: v / num_batches for k, v in epoch_losses.items()}

    epoch_losses["lr"] = scheduler.get_last_lr()[0]

    return epoch_losses


# ═══════════════════════════════════════════════════════════
#  5. MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════

def train(args: argparse.Namespace) -> None:
    """
    Main training function.

    Full workflow:
        1. Load config + set seed
        2. Build dataset + dataloader
        3. Build model
        4. Build optimizer + scheduler + scaler
        5. Resume from checkpoint (optional)
        6. Init WandB
        7. Training loop:
            a. Train one epoch
            b. Validate
            c. Log to WandB
            d. Save checkpoint
            e. Early stopping check
        8. Final summary
    """

    # ── Setup ─────────────────────────────────────────────
    print_banner()
    cfg    = load_config(args.config)
    device = get_device()

    set_seed(cfg.project.seed)
    ensure_dirs(cfg)

    log_dir = os.path.join(cfg.paths.output_dir, "logs")
    logger  = setup_logging(log_dir, cfg.project.version)

    console.rule(f"[bold blue]XRayEarth — {cfg.project.version}[/bold blue]")
    print_config(cfg)

    # ── Dataset & DataLoader ──────────────────────────────
    console.log("[bold]Building datasets...[/bold]")

    train_loader = build_dataloader(cfg, split="train", epoch=0)
    val_loader   = build_dataloader(cfg, split="val",   epoch=0)

    # Compute class distribution for loss weighting
    console.log("[bold]Computing class distribution...[/bold]")
    train_dataset   = train_loader.dataset
    class_counts    = compute_class_distribution(train_dataset)

    console.log(f"  Class distribution: {class_counts}")

    # ── Model ─────────────────────────────────────────────
    console.log("[bold]Building model...[/bold]")
    model = build_model(cfg).to(device)

    # ── Loss ──────────────────────────────────────────────
    loss_wrapper = XRayEarthLoss(cfg, class_counts)

    # ── Optimizer & Scheduler ─────────────────────────────
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler    = GradScaler()  # AMP gradient scaler

    # ── Resume from Checkpoint ────────────────────────────
    start_epoch  = 0
    best_metric  = 0.0
    patience_ctr = 0

    if args.resume:
        ckpt_info   = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = ckpt_info["epoch"] + 1
        best_metric = ckpt_info["metrics"].get("macro_f1", 0.0)
        console.log(
            f"[green]✓[/green] Resuming from epoch {start_epoch} "
            f"(best macro_f1={best_metric:.4f})"
        )

    # ── WandB Init ────────────────────────────────────────
    no_wandb = args.no_wandb or args.smoke_test
    init_wandb(cfg, smoke_test=no_wandb)

    if not no_wandb:
        wandb.watch(model, log="gradients", log_freq=100)

    # ── Training Loop ─────────────────────────────────────
    console.rule("[bold green]Training Started[/bold green]")

    for epoch in range(start_epoch, cfg.training.epochs):

        epoch_start = time.time()

        # Update dataset epoch (for hybrid cache)
        train_loader.dataset.set_epoch(epoch)

        # ── V5: Freeze backbone ───────────────────────────
        if cfg.model.get("freeze_backbone", False):
            freeze_backbone(model)

        # ── V6: Unfreeze at epoch 0 ───────────────────────
        # (freeze_backbone=False means full fine-tuning)

        # ── Train ─────────────────────────────────────────
        train_losses = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            scaler, loss_wrapper, device, cfg,
            epoch       = epoch,
            smoke_test  = args.smoke_test,
            max_batches = args.max_batches,
        )

        # ── Validate ──────────────────────────────────────
        val_metrics = evaluate(
            model      = model,
            dataloader = val_loader,
            device     = device,
            cfg        = cfg,
            epoch      = epoch,
            smoke_test = args.smoke_test,
        )

        epoch_time = time.time() - epoch_start

        # ── Logging ───────────────────────────────────────
        log_dict = {}

        # Train losses
        for k, v in train_losses.items():
            log_dict[f"train/{k}"] = v

        # Val metrics
        for k, v in val_metrics.items():
            log_dict[f"val/{k}"] = v

        # System stats
        log_dict["system/gpu_memory_mb"]   = get_gpu_memory_mb()
        log_dict["system/epoch_time_sec"]  = epoch_time
        log_dict["epoch"]                  = epoch

        if not no_wandb:
            wandb.log(log_dict, step=epoch)

        # ── Console Summary ───────────────────────────────
        macro_f1 = val_metrics.get("macro_f1", 0.0)

        console.log(
            f"[bold]Epoch {epoch:03d}[/bold] | "
            f"loss={train_losses.get('loss_total', 0):.4f} | "
            f"macro_f1={macro_f1:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        # ── Checkpoint ────────────────────────────────────
        is_best = macro_f1 > best_metric

        if is_best:
            best_metric  = macro_f1
            patience_ctr = 0
        else:
            patience_ctr += 1

        if (
            epoch % cfg.training.save_every == 0
            or is_best
            or args.smoke_test
        ):
            save_checkpoint(
                model     = model,
                optimizer = optimizer,
                epoch     = epoch,
                metrics   = val_metrics,
                cfg       = cfg,
                is_best   = is_best,
            )

        # ── Early Stopping ────────────────────────────────
        patience = cfg.training.early_stop_patience
        if patience_ctr >= patience and not args.smoke_test:
            console.log(
                f"[yellow]⚠[/yellow] Early stopping at epoch {epoch} "
                f"(no improvement for {patience} epochs)"
            )
            break

        # ── Smoke Test: stop after first epoch ────────────
        if args.smoke_test:
            console.log("[green]✅ Smoke test passed![/green]")
            break

    # ── Final Summary ─────────────────────────────────────
    console.rule("[bold green]Training Complete[/bold green]")

    summary_table = Table(title="XRayEarth Training Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value",  style="green")

    summary_table.add_row("Version",      cfg.project.version)
    summary_table.add_row("Best macro_f1", f"{best_metric:.4f}")
    summary_table.add_row("Total epochs", str(epoch + 1))
    summary_table.add_row("Loss type",    cfg.loss.type)
    summary_table.add_row("Siamese",      str(cfg.model.siamese))

    console.print(summary_table)

    if not no_wandb:
        wandb.summary["best_macro_f1"] = best_metric
        wandb.summary["total_epochs"]  = epoch + 1
        wandb.finish()

    console.log(
        f"[bold green]🌍 XRayEarth {cfg.project.version} "
        f"training complete! Best macro_f1={best_metric:.4f}[/bold green]"
    )


# ═══════════════════════════════════════════════════════════
#  6. ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()
    train(args)