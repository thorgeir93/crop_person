from pathlib import Path
import shutil
from ultralytics import YOLO
from crop_person.processes.crop import get_cropped_persons_from_directory
from crop_person.processes.face import filter_images_with_faces
from crop_person.processes.blur import filter_sharp_images_from_images
from crop_person.utils import (
    save_images,
    split_upper_lower,
    validate_input_dir,
    validate_output_dir,
)
from crop_person.logging_utils import get_logger

log = get_logger()


def run_workflow(
    source_dir: Path,
    workflow_dir: Path,
    model_path: str,
    min_confidence: float,
    blur_threshold: float,
    clean: bool,
    save: bool,
) -> list:
    """
    Run the full workflow:
    - Step 1: Crop persons
    - Step 2: Filter sharp images
    - Step 3: Filter images with faces

    Returns: list of (image_path, img, faces_count) of final images with faces.
    If `save` is True, saves them to workflow_dir/faces.
    If `clean` is True, deletes intermediate folders (cropped, sharp) if they exist.
    """

    # Validate paths
    source_dir = validate_input_dir(source_dir)
    workflow_dir = validate_output_dir(workflow_dir)

    # Define subdirectories (will use faces_dir if save=True)
    cropped_dir = workflow_dir / "cropped"
    sharp_dir = workflow_dir / "sharp"
    final_dir = workflow_dir / "final"

    # Always create faces_dir (if saving) so it's ready
    final_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Starting workflow", source_dir=str(source_dir), workflow_dir=str(workflow_dir)
    )

    # Load YOLO model once
    log.info("Loading YOLO model", model_path=model_path)
    model = YOLO(model_path)

    # STEP 1: Crop persons
    log.info("STEP 1: Crop persons")
    cropped_persons = get_cropped_persons_from_directory(
        source_dir, model, min_confidence
    )

    # STEP 2: Filter sharp images
    log.info("STEP 2: Filter sharp images")
    sharp_images = filter_sharp_images_from_images(cropped_persons, blur_threshold)

    # STEP 3: Filter images with faces
    log.info("STEP 3: Filter images with faces")
    images_with_faces = filter_images_with_faces(sharp_images)

    log.info("STEP 4: Split images into upper and lower body")
    upper_images, lower_images = split_upper_lower(images_with_faces)

    # Save final images if requested
    if save:
        log.info("Saving final images (upper body)")
        upper_body_dir = final_dir / "upper"
        upper_body_dir.mkdir(parents=True, exist_ok=True)
        save_images(
            upper_images,
            upper_body_dir,
            name_func=lambda image_path, img, meta: f"{image_path.name}",
        )

        log.info("Saving final images (lower body)")
        lower_body_dir = final_dir / "lower"
        lower_body_dir.mkdir(parents=True, exist_ok=True)
        save_images(
            lower_images,
            lower_body_dir,
            name_func=lambda image_path, img, meta: f"{image_path.name}",
        )

    # TODO: think about saving as well, we do not need to clean if we do not save I guess.
    # Optional cleanup
    if clean:
        log.info("Cleaning intermediate folders if they exist")
        for folder in [cropped_dir, sharp_dir]:
            if folder.exists():
                try:
                    shutil.rmtree(folder)
                    log.info("Deleted folder", folder=str(folder))
                except Exception as e:
                    log.error("Error deleting folder", folder=str(folder), error=str(e))

    return images_with_faces
