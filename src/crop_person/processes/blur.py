from typing import Any
import cv2
import numpy as np

from crop_person.logging_utils import get_logger

log = get_logger()


def is_blurred(img: Any, threshold: float = 100.0) -> bool:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


def blur_score(img: Any) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def filter_sharp_images_from_images(images: list, blur_threshold: float) -> list:
    """
    Given: list of (image_path, img)
    Returns: list of (image_path, img, blur_score, is_blurred)
    """
    sharp_images = []

    # TODO: consider to move the blue functionallity to here instead of in crop
    #       then we can skip _,_,_.
    for image_path, img, _, _, _ in images:
        score = blur_score(img)
        blurred = is_blurred(img, threshold=blur_threshold)

        if not blurred:
            sharp_images.append((image_path, img, score, blurred))
            log.debug("Image is sharp", path=str(image_path), blur_score=f"{score:.2f}")
        else:
            log.debug(
                "Image is blurred", path=str(image_path), blur_score=f"{score:.2f}"
            )

    log.info("Filtered sharp images", total=len(sharp_images))
    return sharp_images
