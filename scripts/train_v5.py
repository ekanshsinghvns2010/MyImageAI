
import argparse
import copy
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from src.model_v5 import MyImageAIModelV5

CLASSES = ["cat", "dog", "wild"]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoints", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--timesteps", type=int, default=300)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)


def make_diffusion(timesteps, device):
    # Linear beta schedule for the initial V5 baseline.
    betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def save_checkpoint(path, model, ema, optimizer, scaler,
                    epoch, step, losses, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save({
        "version": 5,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "step": step,
        "losses": losses,
        "config": {
            "image_size": 128,
            "num_classes": 3,
            "timesteps": args.timesteps,
            "base_channels": 48,
            "class_names": CLASSES,
        },
    }, temp)
    os.replace(temp, path)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp_dtype = torch.float16

    root = Path(args.data)
    ckpt_dir = Path(args.checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest = ckpt_dir / "latest.pt"

    # Required layout: data/cat, data/dog, data/wild.
    for name in CLASSES:
        if not (root / name).is_dir():
            raise FileNotFoundError(
                f"Missing class folder: {root / name}. "
                "Do not train until approved images are prepared."
            )

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])

    dataset = datasets.ImageFolder(root=str(root), transform=transform)
    expected = {name: i for i, name in enumerate(CLASSES)}
    if dataset.class_to_idx != expected:
        raise ValueError(
            f"Class folder order mismatch: {dataset.class_to_idx}; "
            f"expected {expected}"
        )
    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=use_amp,
        drop_last=True,
        worker_init_fn=seed_worker,
        persistent_workers=args.workers > 0,
    )

    model = MyImageAIModelV5().to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    betas, alphas, alpha_bars = make_diffusion(args.timesteps, device)

    start_epoch, global_step = 0, 0
    losses = []

    if args.resume and latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        if state["config"]["timesteps"] != args.timesteps:
            raise ValueError("Checkpoint timestep setting differs.")
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"])
        global_step = int(state["step"])
        losses = state.get("losses", [])
        print(f"Resuming at epoch {start_epoch + 1}, step {global_step}")

    print("Device:", device)
    print("Images:", len(dataset))
    print("Classes:", dataset.class_to_idx)
    print("Parameters:", sum(p.numel() for p in model.parameters()))

    model.train()
    for epoch in range(start_epoch, args.epochs):
        epoch_loss, batches = 0.0, 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            b = images.shape[0]

            # Classifier-free guidance training: occasionally drop labels.
            dropped = torch.rand(b, device=device) < 0.1
            labels = labels.clone()
            labels[dropped] = model.null_class

            t = torch.randint(0, args.timesteps, (b,), device=device)
            noise = torch.randn_like(images)
            ab = alpha_bars[t].view(b, 1, 1, 1)
            noisy = ab.sqrt() * images + (1.0 - ab).sqrt() * noise

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                predicted_noise = model(noisy, t, labels)
                loss = F.mse_loss(predicted_noise.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                decay = min(0.999, (global_step + 1) / (global_step + 10))
                for ep, p in zip(ema.parameters(), model.parameters()):
                    ep.lerp_(p, 1.0 - decay)
                for eb, mb in zip(ema.buffers(), model.buffers()):
                    eb.copy_(mb)

            global_step += 1
            batches += 1
            epoch_loss += loss.item()

            if global_step % args.save_every == 0:
                save_checkpoint(
                    latest, model, ema, optimizer, scaler,
                    epoch, global_step, losses, args
                )
                print(
                    f"Saved step {global_step}; "
                    f"epoch {epoch + 1}; loss {loss.item():.5f}"
                )

        average = epoch_loss / max(batches, 1)
        losses.append(average)
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={average:.6f}")

        # Save completed epoch as epoch+1 so resume begins correctly.
        save_checkpoint(
            latest, model, ema, optimizer, scaler,
            epoch + 1, global_step, losses, args
        )
        torch.save(ema.state_dict(), ckpt_dir / "ema_weights.pt")

    print("Training finished. Latest checkpoint:", latest)


if __name__ == "__main__":
    main()
