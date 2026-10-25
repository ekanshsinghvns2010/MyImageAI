"""Train MyImageAI V4 on class-labelled CIFAR-10."""
import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model_v4 import MyImageAIModelV4
from src.diffusion import Diffusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--class-dropout", type=float, default=0.10)
    parser.add_argument("--checkpoint-dir",
        default="/content/drive/MyDrive/MyImageAI/v4_checkpoints")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.timesteps < 2:
        parser.error("epochs/batch-size must be positive; timesteps >= 2")
    if args.learning_rate <= 0 or not 0 <= args.class_dropout <= 1:
        parser.error("Invalid learning rate or class dropout")

    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "architecture": "class_conditional_residual_attention_unet_v4",
        "in_channels": 3, "base_channels": 64, "time_dim": 256,
        "num_classes": 10, "image_size": 32, "dataset": "CIFAR10",
        "class_dropout": args.class_dropout, "dropout": 0.05,
    }
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "latest.pt"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    dataset = datasets.CIFAR10(
        root="/content/MyImageAI-data", train=True,
        download=True, transform=transform
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=device.type == "cuda"
    )
    model = MyImageAIModelV4(
        in_channels=3, base_channels=64, time_dim=256,
        num_classes=10, dropout=0.05
    ).to(device)
    diffusion = Diffusion(timesteps=args.timesteps, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )

    start_epoch = 0
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        if checkpoint.get("model_config") != config:
            raise ValueError("Checkpoint config mismatch; use a new V4 directory.")
        if checkpoint.get("timesteps") != args.timesteps:
            raise ValueError("Timestep mismatch; use original timestep count.")
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        print("Resuming from epoch", start_epoch)

    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("Checkpoint:", ckpt_path)
    if start_epoch >= args.epochs:
        print("Target epochs already reached.")
        return

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels in progress:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, dtype=torch.long)
            batch = images.shape[0]
            t = torch.randint(0, diffusion.timesteps, (batch,), device=device)
            noisy, noise = diffusion.add_noise(images, t)
            labels_in = labels.clone()
            labels_in[torch.rand(batch, device=device) < args.class_dropout] = (
                MyImageAIModelV4.NULL_CLASS
            )
            pred = model(noisy, t, labels_in)
            loss = F.mse_loss(pred, noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg = total_loss / len(loader)
        print(f"Epoch {epoch}: average loss = {avg:.5f}")
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg, "timesteps": args.timesteps,
            "model_config": config,
        }, ckpt_path)
        print("Checkpoint saved:", ckpt_path)
    print("V4 training finished.")


if __name__ == "__main__":
    main()
