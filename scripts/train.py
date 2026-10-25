
"""Train MyImageAI's first diffusion model on MNIST."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

# Allow imports from the repository's src directory.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import MyImageAIModel
from src.diffusion import Diffusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--checkpoint-dir",
        default="/content/MyImageAI-checkpoints",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("MyImageAI training starting")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    checkpoint_dir = Path(args.checkpoint_dir)
    sample_dir = checkpoint_dir / "samples"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Scale grayscale pixels to [-1, 1].
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = datasets.MNIST(
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

    model = MyImageAIModel().to(device)
    diffusion = Diffusion(
        timesteps=args.timesteps,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    print("Training images:", len(dataset))
    print("Trainable parameters:", sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")

        for images, _ in progress:
            images = images.to(
                device,
                non_blocking=(device.type == "cuda"),
            )

            batch_size = images.shape[0]
            timesteps = torch.randint(
                0,
                diffusion.timesteps,
                (batch_size,),
                device=device,
            )

            noisy_images, noise = diffusion.add_noise(images, timesteps)
            predicted_noise = model(noisy_images, timesteps)
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        average_loss = total_loss / len(loader)
        print(f"Epoch {epoch}: average loss = {average_loss:.5f}")

        # Save a resumable training checkpoint outside the Git repo.
        checkpoint_path = checkpoint_dir / "latest.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": average_loss,
            "timesteps": diffusion.timesteps,
            "model_config": {
                "in_channels": 1,
                "base_channels": 64,
                "time_dim": 128,
            },
        }, checkpoint_path)

        # Generate a sample grid after each epoch.
        generated = diffusion.sample(
            model,
            image_size=28,
            batch_size=16,
            channels=1,
        )

        save_image(
            (generated + 1) / 2,
            sample_dir / f"epoch_{epoch:03d}.png",
            nrow=4,
        )

        print("Checkpoint saved:", checkpoint_path)
        print("Sample saved:", sample_dir / f"epoch_{epoch:03d}.png")

    print("Training finished!")


if __name__ == "__main__":
    main()
