import cv2
from crop_person.logging_utils import get_logger

log = get_logger()

# Load face detector once
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
)


def filter_images_with_faces(images: list) -> list:
    """
    Given: list of (image_path, img, *meta) OR (image_path, img)
    Returns: list of (image_path, img, faces_count)
    """
    images_with_faces = []

    for image in images:
        if len(image) >= 2:
            image_path, img = image[0], image[1]
        else:
            log.error("Invalid image tuple", image=image)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        faces_count = len(faces)

        if faces_count > 0:
            images_with_faces.append((image_path, img, faces_count))
            log.debug("Image has faces", path=str(image_path), faces_count=faces_count)
        else:
            pass
            log.debug("Image has no faces", path=str(image_path))

    log.info("Filtered images with faces", total=len(images_with_faces))
    return images_with_faces
