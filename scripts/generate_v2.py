
"""Generate RGB images using MyImageAI v2."""

import argparse
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_v2 import MyImageAIModelV2
from src.diffusion import Diffusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="/content/drive/MyDrive/MyImageAI/v2_checkpoints/latest.pt",
    )
    parser.add_argument(
        "--output",
        default="/content/drive/MyDrive/MyImageAI/v2_generated.png",
    )
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be positive")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        parser.error(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    config = checkpoint.get("model_config", {})

    if config.get("architecture") != "residual_unet_v2":
        parser.error("This is not a MyImageAI v2 checkpoint.")

    model = MyImageAIModelV2(
        in_channels=config["in_channels"],
        base_channels=config["base_channels"],
        time_dim=config["time_dim"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    diffusion = Diffusion(
        timesteps=checkpoint["timesteps"],
        device=device,
    )

    print("MyImageAI v2")
    print("Device:", device)
    print("Images:", args.count)
    print("Resolution: 32 x 32 RGB")
    print("Trainable parameters:", sum(
        p.numel() for p in model.parameters()
    ))

    with torch.no_grad():
        images = diffusion.sample(
            model,
            image_size=32,
            batch_size=args.count,
            channels=3,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(
        (images + 1).clamp(0, 2) / 2,
        output_path,
        nrow=4,
    )

    print("Saved:", output_path)


if __name__ == "__main__":
    main()
