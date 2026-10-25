
"""Train MyImageAI v2 on CIFAR-10."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_v2 import MyImageAIModelV2
from src.diffusion import Diffusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--checkpoint-dir",
        default="/content/drive/MyDrive/MyImageAI/v2_checkpoints",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Start fresh and replace the v2 checkpoint.",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.timesteps < 1:
        parser.error("Epochs, batch size and timesteps must be positive.")
    if args.learning_rate <= 0:
        parser.error("Learning rate must be positive.")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("MyImageAI v2 training")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: GPU unavailable; training will be slow.")

    checkpoint_dir = Path(args.checkpoint_dir)
    sample_dir = checkpoint_dir / "samples"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "latest.pt"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = datasets.CIFAR10(
        root="/content/MyImageAI-data",
        train=True,
        download=True,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    config = {
        "in_channels": 3,
        "base_channels": 32,
        "time_dim": 128,
        "image_size": 32,
        "dataset": "CIFAR10",
        "architecture": "residual_unet_v2",
    }

    model = MyImageAIModelV2(
        in_channels=3,
        base_channels=32,
        time_dim=128,
    ).to(device)

    diffusion = Diffusion(
        timesteps=args.timesteps,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    start_epoch = 0

    if checkpoint_path.exists() and not args.restart:
        print("Resuming existing v2 checkpoint.")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )

        if checkpoint.get("model_config") != config:
            raise ValueError(
                "The saved v2 model configuration differs. "
                "Use another checkpoint directory or --restart."
            )

        if checkpoint.get("timesteps") != args.timesteps:
            raise ValueError(
                "Timestep count differs from checkpoint. "
                "Use the original timestep count."
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        start_epoch = int(checkpoint.get("epoch", 0))
        print("Last completed epoch:", start_epoch)

    print("Training images:", len(dataset))
    print("Trainable parameters:", sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ))
    print("Target total epochs:", args.epochs)
    print("Checkpoint:", checkpoint_path)

    if start_epoch >= args.epochs:
        print("Target epoch count already reached.")
        print("Increase --epochs to continue training.")
        return

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(
            loader, desc=f"Epoch {epoch}/{args.epochs}"
        )

        for images, _ in progress:
            images = images.to(
                device, non_blocking=(device.type == "cuda")
            )
            t = torch.randint(
                0, diffusion.timesteps,
                (images.shape[0],), device=device
            )

            noisy_images, noise = diffusion.add_noise(images, t)
            predicted_noise = model(noisy_images, t)
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        average_loss = total_loss / len(loader)
        print(f"Epoch {epoch}: average loss = {average_loss:.5f}")

        # Save a complete checkpoint after every epoch.
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": average_loss,
            "timesteps": args.timesteps,
            "model_config": config,
        }, checkpoint_path)

        print("Checkpoint saved:", checkpoint_path)

        model.eval()
        with torch.no_grad():
            generated = diffusion.sample(
                model, image_size=32, batch_size=16, channels=3
            )

        sample_path = sample_dir / f"epoch_{epoch:03d}.png"
        save_image(
            (generated + 1).clamp(0, 2) / 2,
            sample_path,
            nrow=4,
        )
        print("Sample saved:", sample_path)

    print("MyImageAI v2 training finished.")


if __name__ == "__main__":
    main()
