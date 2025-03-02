# selfiemulticlasssegmentation

A simple computer vision project that performs multiclass segmentation using mediapipe on selfie images and videos. This project can identify and segment different elements in selfie photos such as hair, skin, clothing, background, and accessories.

## Features

- Multiclass semantic segmentation of selfie images
- Support for multiple segmentation classes (skin, hair, clothing, background, etc.)

## Installation

```bash
# Install Rye if you don't have it already
Visit https://rye-up.com/ and install for your environment

# Clone and navigate to the project
cd selfiemulticlasssegmentation

# Sync dependencies and create virtual environment
rye sync

# Activate the environment if you need
.venv\Scripts\activate
```

## Download the Model

Before running the application, you need to download the required model file:

### Direct Download Link

[Download selfie_multiclass_256x256.tflite](https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite)

Save the tflite file directly into src/selfiesegmentation/assets/models dir.

## Usage

### Command Line Interface

```bash
cd src/selfiemulticlasssegmentation

# For processing a single image
rye run python multiclass_segmentation_image.py --image path/to/image.jpg

# For processing a video file
rye run python multiclass_segmentation_video.py --video path/to/video.mp4 

# For real-time segmentation using webcam
rye run python multiclass_segmentation_webcam.py
```
