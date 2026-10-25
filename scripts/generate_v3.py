
"""Generate class-conditional images with MyImageAI v3."""

import argparse
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_v3 import MyImageAIModelV3
from src.diffusion import Diffusion


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@torch.no_grad()
def sample_conditioned(
    model,
    diffusion,
    labels,
    image_size=32,
    guidance_scale=3.0,
):
    """DDPM sampling with classifier-free guidance."""
    device = diffusion.device
    batch_size = labels.shape[0]

    images = torch.randn(
        batch_size, 3, image_size, image_size, device=device
    )

    null_labels = torch.full(
        (batch_size,),
        MyImageAIModelV3.NULL_CLASS,
        device=device,
        dtype=torch.long,
    )

    for step in reversed(range(diffusion.timesteps)):
        timesteps = torch.full(
            (batch_size,),
            step,
            device=device,
            dtype=torch.long,
        )

        # One model call for conditional and unconditional predictions.
        combined_images = torch.cat([images, images], dim=0)
        combined_times = torch.cat([timesteps, timesteps], dim=0)
        combined_labels = torch.cat([null_labels, labels], dim=0)

        predicted = model(
            combined_images,
            combined_times,
            combined_labels,
        )

        predicted_unconditional, predicted_conditional = predicted.chunk(2)

        predicted_noise = (
            predicted_unconditional
            + guidance_scale
            * (predicted_conditional - predicted_unconditional)
        )

        alpha = diffusion.alphas[step]
        alpha_bar = diffusion.alpha_bars[step]
        beta = diffusion.betas[step]

        images = (
            images
            - beta / torch.sqrt(1.0 - alpha_bar) * predicted_noise
        ) / torch.sqrt(alpha)

        if step > 0:
            images = images + torch.sqrt(beta) * torch.randn_like(images)

    return images.clamp(-1, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="/content/drive/MyDrive/MyImageAI/v3_checkpoints/latest.pt",
    )
    parser.add_argument(
        "--output",
        default="/content/drive/MyDrive/MyImageAI/v3_generated.png",
    )
    parser.add_argument(
        "--class-name",
        choices=CLASS_NAMES,
        default="cat",
    )
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be positive")
    if args.guidance < 0:
        parser.error("--guidance must be non-negative")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        parser.error(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    config = checkpoint.get("model_config", {})
    if config.get("architecture") != "class_conditional_residual_unet_v3":
        parser.error("Checkpoint is not a compatible MyImageAI v3 checkpoint.")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = MyImageAIModelV3(
        in_channels=config["in_channels"],
        base_channels=config["base_channels"],
        time_dim=config["time_dim"],
        num_classes=config["num_classes"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    diffusion = Diffusion(
        timesteps=checkpoint["timesteps"],
        device=device,
    )

    class_id = CLASS_NAMES.index(args.class_name)
    labels = torch.full(
        (args.count,), class_id, device=device, dtype=torch.long
    )

    print("MyImageAI v3")
    print("Device:", device)
    print("Class:", args.class_name, f"(label {class_id})")
    print("Images:", args.count)
    print("Guidance:", args.guidance)
    print("Seed:", args.seed)

    images = sample_conditioned(
        model,
        diffusion,
        labels,
        image_size=config["image_size"],
        guidance_scale=args.guidance,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_image(
        (images + 1).clamp(0, 2) / 2,
        output_path,
        nrow=min(4, args.count),
    )

    print("Saved:", output_path)


if __name__ == "__main__":
    main()
