"""Generate images using a trained MyImageAI checkpoint."""

import argparse
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

ROOT = Path(file).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import MyImageAIModel
from src.diffusion import Diffusion

def main():
parser = argparse.ArgumentParser(
description="Generate grayscale images with MyImageAI."
)
parser.add_argument(
"--checkpoint",
default="/content/MyImageAI-checkpoints/latest.pt",
)
parser.add_argument("--output", default="/content/MyImageAI-generated.png")
parser.add_argument("--count", type=int, default=16)
parser.add_argument("--timesteps", type=int, default=None)
args = parser.parse_args()

if args.count < 1:
    parser.error("--count must be at least 1")

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
model = MyImageAIModel(
    in_channels=config.get("in_channels", 1),
    base_channels=config.get("base_channels", 64),
    time_dim=config.get("time_dim", 128),
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

steps = args.timesteps or checkpoint.get("timesteps", 300)
diffusion = Diffusion(timesteps=steps, device=device)

with torch.no_grad():
    images = diffusion.sample(
        model,
        image_size=28,
        batch_size=args.count,
        channels=1,
    )

output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)
save_image((images + 1) / 2, str(output_path), nrow=4)

print(f"Generated {args.count} images.")
print(f"Device: {device}")
print(f"Output: {output_path}")

if name == "main":
main()