
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

CLASSES = ("cat", "dog", "wild")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--check-images", action="store_true")
    parser.add_argument(
        "--commercial-rights-verified",
        action="store_true",
        help="Use only after independently verifying the dataset's terms."
    )
    args = parser.parse_args()

    root = Path(args.data)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.commercial_rights_verified:
        raise SystemExit(
            "STOP: Commercial rights have not been confirmed. "
            "Review dataset and image-level licenses before proceeding."
        )

    records = []
    errors = []

    for class_name in CLASSES:
        folder = root / class_name
        if not folder.is_dir():
            errors.append(f"Missing class folder: {folder}")
            continue

        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue

            record = {
                "path": str(path.resolve()),
                "class": class_name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "valid_image": None,
            }

            if args.check_images:
                try:
                    with Image.open(path) as image:
                        image.verify()
                    record["valid_image"] = True
                except Exception as exc:
                    record["valid_image"] = False
                    errors.append(f"Invalid image: {path}: {exc}")

            records.append(record)

    output = {
        "dataset_root": str(root.resolve()),
        "classes": list(CLASSES),
        "commercial_rights_verified_by_user": True,
        "file_count": len(records),
        "records": records,
        "errors": errors,
    }
    manifest_path.write_text(json.dumps(output, indent=2))

    print("Manifest:", manifest_path)
    print("Images found:", len(records))
    print("Errors:", len(errors))
    if errors:
        print("\nFirst errors:")
        for error in errors[:10]:
            print("-", error)


if __name__ == "__main__":
    main()
