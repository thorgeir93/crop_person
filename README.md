# Crop Person from an Image

## Background
This project was created to make photography post-processing faster and easier. Editing photos manually takes a lot of time and effort. By training a model to score images based on running style, we can speed up this process. This project is one part of that bigger goal.

## About
This repository helps create clean data for training a model. It focuses on cropping the person from an image so the main part of the image can be used for training. You can also split the cropped person into two parts—upper body and lower body (legs and feet)—to improve training results.

## Installation

Install [`justfile`](https://github.com/casey/just)

Then install the project:
```
uv sync
```

Then use the `just` command recently installed:

```
just run-workflow --help
```

## How to Run the Workflow

### Simple Workflow
Run the following command:
```justfile
just run /path/to/media/images /path/to/media
```

This command will create a `/path/to/media/final` directory containing cropped persons from the `/path/to/media/images`. The workflow includes these steps:

1. Crop person from images
    * uses a deep learning CNN (YOLOv8/YOLOv9 object detection model).
2. Filter out blurred images, keeping only sharp images.
    * Uses no model — pure image processing (Laplacian variance from OpenCV).
3. Filter out images without faces, retaining only those that contain faces.
    * Uses a classical pre-trained computer vision model (Haar Cascade face detector from OpenCV).

# TODO
- [ ] Add --overwrite flag if the user wants to overwrite the final directory.
- [ ] Use async if possible.
- [ ] Split the cropped person into two sections: upper body and lower body for better model training.
- [ ] Suppress logging messages, too noisy.

# More Detailed Workflow (Manual)

Example: accept a directory and store cropped image to a different directory:
```
uv run python -m src.crop_person.__main__ /home/thorgeir/training_data/sport/event/running/good_style/original /home/thorgeir/training_data/sport/event/running/good_style/cropped
```

The filter sharp images:
```
uv run python -m src.crop_person.__main__ filter-sharp /home/thorgeir/training_data/sport/event/running/good_style/cropped /home/thorgeir/training_data/sport/event/running/good_style/cropped_sharp
```

Then filter only images containing faces:
```
uv run python -m src.crop_person.__main__ filter-faces /home/thorgeir/training_data/sport/event/running/good_style/cropped_sharp /home/thorgeir/training_data/sport/event/running/good_style/cropped_sharp_only_with_faces
```
