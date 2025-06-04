from pathlib import Path
from typing import Callable
import typer
import cv2

from crop_person.logging_utils import get_logger

log = get_logger()


def validate_input_dir(path: Path) -> Path:
    if not path.exists():
        typer.echo(f"❌ Source directory does not exist: {path}")
        raise typer.Exit(code=1)
    if not path.is_dir():
        typer.echo(f"❌ Source path is not a directory: {path}")
        raise typer.Exit(code=1)
    return path.resolve()


def validate_output_dir(path: Path) -> Path:
    if not path.exists():
        typer.echo(f"⚠️ Destination folder does not exist. Creating: {path}")
        path.mkdir(parents=True, exist_ok=True)
    elif not path.is_dir():
        typer.echo(f"❌ Destination path is not a directory: {path}")
        raise typer.Exit(code=1)
    return path.resolve()


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def load_images_from_directory(source_dir: Path) -> list:
    """
    Loads images from folder into: list of (image_path, img)
    """
    image_paths = [p for p in source_dir.iterdir() if p.is_file() and is_image_file(p)]
    images = []

    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        if img is not None:
            images.append((image_path, img))
            log.info("Loaded image", path=str(image_path))
        else:
            log.error("Failed to load image", path=str(image_path))

    return images


def save_images(
    images: list,
    destination_dir: Path,
    name_func: Callable,
) -> None:
    """
    Generic image saver.

    images: list of (image_path, img, *metadata)
    name_func: function(image_path, img, metadata_tuple) -> filename (str)

    Example name_func:
    lambda image_path, img, meta: f"{image_path.stem}_person.jpg"
    """
    log.info("Saving images", count=len(images), destination_dir=str(destination_dir))

    for image_path, img, *meta in images:
        filename = name_func(image_path, img, tuple(meta))
        output_path = destination_dir / filename

        cv2.imwrite(str(output_path), img)

        log.debug("Saved image", path=str(output_path), metadata=meta)

    log.info("Finished saving images", total_saved=len(images))
