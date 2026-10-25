
"""Train MyImageAI v3 on class-labelled CIFAR-10."""

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

from src.model_v3 import MyImageAIModelV3
from src.diffusion import Diffusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--class-dropout", type=float, default=0.10)
    parser.add_argument(
        "--checkpoint-dir",
        default="/content/drive/MyDrive/MyImageAI/v3_checkpoints",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.timesteps < 2:
        parser.error("Epochs and batch size must be positive; timesteps >= 2.")
    if args.learning_rate <= 0:
        parser.error("Learning rate must be positive.")
    if not 0.0 <= args.class_dropout <= 1.0:
        parser.error("class-dropout must be between 0 and 1.")

    torch.manual_seed(42)
    random.seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("MyImageAI v3 training")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: GPU unavailable; training will be slow.")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "latest.pt"

    config = {
        "architecture": "class_conditional_residual_unet_v3",
        "in_channels": 3,
        "base_channels": 32,
        "time_dim": 128,
        "num_classes": 10,
        "image_size": 32,
        "dataset": "CIFAR10",
        "class_dropout": args.class_dropout,
    }

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ),
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
        drop_last=False,
    )

    model = MyImageAIModelV3(
        in_channels=3,
        base_channels=32,
        time_dim=128,
        num_classes=10,
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

    if checkpoint_path.exists():
        print("Resuming existing v3 checkpoint.")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )

        if checkpoint.get("model_config") != config:
            raise ValueError(
                "v3 model configuration differs from saved checkpoint. "
                "Use a new --checkpoint-dir; do not overwrite old work."
            )

        if checkpoint.get("timesteps") != args.timesteps:
            raise ValueError(
                "Timestep count differs from saved checkpoint. "
                "Use the original timestep count."
            )

        model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = int(checkpoint.get("epoch", 0))
        print("Last completed epoch:", start_epoch)

    print("Training images:", len(dataset))
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("Target total epochs:", args.epochs)
    print("Checkpoint:", checkpoint_path)

    if start_epoch >= args.epochs:
        print("Target epoch count already reached.")
        print("Use a higher --epochs value to continue.")
        return

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")

        for images, labels in progress:
            images = images.to(
                device, non_blocking=(device.type == "cuda")
            )
            labels = labels.to(device, dtype=torch.long)

            batch_size = images.shape[0]
            timesteps = torch.randint(
                0,
                diffusion.timesteps,
                (batch_size,),
                device=device,
            )

            noisy_images, noise = diffusion.add_noise(images, timesteps)

            # Randomly remove class information for classifier-free guidance.
            drop_mask = (
                torch.rand(batch_size, device=device) < args.class_dropout
            )
            training_labels = labels.clone()
            training_labels[drop_mask] = MyImageAIModelV3.NULL_CLASS

            predicted_noise = model(
                noisy_images, timesteps, training_labels
            )
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        average_loss = total_loss / len(loader)
        print(f"Epoch {epoch}: average loss = {average_loss:.5f}")

        # Save only to the dedicated v3 directory.
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": average_loss,
            "timesteps": args.timesteps,
            "model_config": config,
        }, checkpoint_path)

        print("Checkpoint saved:", checkpoint_path)

    print("MyImageAI v3 training finished.")


if __name__ == "__main__":
    main()
