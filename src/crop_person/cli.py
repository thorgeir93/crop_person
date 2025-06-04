from pathlib import Path
import typer
import shutil

from ultralytics import YOLO

from crop_person.logging_utils import get_logger
from crop_person.processes.blur import filter_sharp_images_from_images
from crop_person.utils import (
    load_images_from_directory,
    save_images,
    validate_input_dir,
    validate_output_dir,
)
from crop_person.processes.face import filter_images_with_faces
from crop_person.processes.crop import (
    get_cropped_persons_from_directory,
)
from crop_person.utils import (
    validate_input_dir,
    validate_output_dir,
)
from crop_person.workflow import run_workflow

app = typer.Typer(
    help="Crop person(s) from images and filter sharp images.",
    pretty_exceptions_enable=False,
)


@app.command("run-workflow")
def run_workflow_command(
    source_dir: Path = typer.Argument(
        ..., help="Path to source directory with images."
    ),
    workflow_dir: Path = typer.Argument(..., help="Path to workflow output directory."),
    model_path: str = typer.Option(
        "yolov8n.pt",
        "--model",
        "-m",
        help="Path to YOLO model to use for person detection (default: yolov8n.pt).",
    ),
    min_confidence: float = typer.Option(
        0.9,
        "--min-confidence",
        "-c",
        help="Minimum confidence threshold for person detection (default: 0.9). Only persons detected with confidence >= this value will be cropped.",
    ),
    blur_threshold: float = typer.Option(
        100.0,
        "--blur-threshold",
        "-t",
        help="Blur threshold for sharp image filtering (default: 100.0). Images with blur score below this will be considered blurred and excluded.",
    ),
    clean: bool = typer.Option(
        False,
        "--clean",
        help="If set, intermediate folders (cropped, sharp) will be deleted after workflow completes.",
    ),
    save: bool = typer.Option(
        True,
        "--save",
        help="If set (default), saves final images with faces to workflow_dir/faces. If False, runs in-memory only.",
    ),
) -> None:
    run_workflow(
        source_dir=source_dir,
        workflow_dir=workflow_dir,
        model_path=model_path,
        min_confidence=min_confidence,
        blur_threshold=blur_threshold,
        clean=clean,
        save=save,
    )


@app.command("crop-person")
def crop_person_command(
    source_dir: Path = typer.Argument(
        ..., help="Path to source directory with images."
    ),
    output_dir: Path = typer.Argument(
        ..., help="Path to output directory for cropped persons."
    ),
    model_path: str = typer.Option(
        "yolov8n.pt", "--model", "-m", help="Path to YOLO model."
    ),
    min_confidence: float = typer.Option(
        0.9, "--min-confidence", "-c", help="Minimum confidence threshold."
    ),
) -> None:
    source_dir = validate_input_dir(source_dir)
    output_dir = validate_output_dir(output_dir)

    model = YOLO(model_path)
    cropped_persons = get_cropped_persons_from_directory(
        source_dir, model, min_confidence
    )

    save_images(
        cropped_persons,
        output_dir,
        name_func=lambda image_path,
        img,
        meta: f"{image_path.stem}_person_{meta[0] * 100:.0f}_blur_{meta[1]:.0f}.jpg",
    )


def filter_sharp_command(
    input_dir: Path = typer.Argument(..., help="Path to input directory with images."),
    output_dir: Path = typer.Argument(
        ..., help="Path to output directory for sharp images."
    ),
    blur_threshold: float = typer.Option(
        100.0, "--blur-threshold", "-t", help="Blur threshold."
    ),
) -> None:
    """
    Filter sharp images from input_dir → output_dir.
    Accepts any images.
    """
    input_dir = validate_input_dir(input_dir)
    output_dir = validate_output_dir(output_dir)

    images = load_images_from_directory(input_dir)

    sharp_images = filter_sharp_images_from_images(images, blur_threshold)

    save_images(
        sharp_images,
        output_dir,
        name_func=lambda image_path, img, meta: f"{image_path.name}",
    )


@app.command("filter-faces")
def filter_faces_command(
    source_dir: Path = typer.Argument(..., help="Path to directory with images."),
    output_dir: Path = typer.Argument(
        ..., help="Path to output directory for images with faces."
    ),
) -> None:
    source_dir = validate_input_dir(source_dir)
    output_dir = validate_output_dir(output_dir)

    images = load_images_from_directory(source_dir)

    images_with_faces = filter_images_with_faces(images)

    save_images(
        images_with_faces,
        output_dir,
        name_func=lambda image_path, img, meta: f"{image_path.name}",
    )
