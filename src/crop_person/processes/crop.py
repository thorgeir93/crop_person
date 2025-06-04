from tabnanny import verbose
from typing import Any
import cv2
from pathlib import Path
from ultralytics import YOLO

from crop_person.utils import is_image_file
from crop_person.processes.blur import is_blurred, blur_score
from crop_person.logging_utils import get_logger

log = get_logger()


def get_cropped_persons(img: Any, model: YOLO, min_confidence: float) -> list:
    results = model(img, verbose=False)
    cropped_persons = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            confidence = float(box.conf[0])
            if label == "person" and confidence >= min_confidence:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped = img[y1:y2, x1:x2]
                cropped_persons.append((cropped, confidence))

    return cropped_persons


def crop_persons_from_image(
    image_path: Path, model: YOLO, min_confidence: float
) -> list:
    """
    Returns: list of (image_path, cropped_person_img, confidence, blur_score, is_blurred)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        log.error("Failed to load image", path=str(image_path))
        return []

    cropped_persons_raw = get_cropped_persons(img, model, min_confidence)
    cropped_persons = []

    for cropped, confidence in cropped_persons_raw:
        score = blur_score(cropped)
        blurred = is_blurred(cropped, threshold=100.0)

        cropped_persons.append((image_path, cropped, confidence, score, blurred))

    return cropped_persons


def get_cropped_persons_from_directory(
    source_dir: Path, model: YOLO, min_confidence: float
) -> list:
    """
    Returns: list of (image_path, cropped_person_img, confidence, blur_score, is_blurred)
    """
    image_paths = [p for p in source_dir.iterdir() if p.is_file() and is_image_file(p)]
    if not image_paths:
        log.warning(
            "No image files found in source directory", source_dir=str(source_dir)
        )
        return []

    log.info(
        "Found images to process for person cropping",
        count=len(image_paths),
        source_dir=str(source_dir),
    )

    all_cropped_persons = []

    for image_path in image_paths:
        cropped_persons = crop_persons_from_image(image_path, model, min_confidence)
        all_cropped_persons.extend(cropped_persons)

    log.info("Total cropped persons", total=len(all_cropped_persons))
    return all_cropped_persons
