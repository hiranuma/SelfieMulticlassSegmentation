# Segmentation Models

This directory must contain the TFLite models used for multiclass segmentation.

## Required Model

- `selfie_multiclass_256x256.tflite` - Multiclass segmentation model for selfies

## Download the Model

Before running the application, you need to download the required model file:

### Direct Download Link
[Download selfie_multiclass_256x256.tflite](https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite)

Save this file directly into this directory.

### Programmatic Download
You can also download the model programmatically using Python:

```python
import os
import urllib.request

# Define model URL and directory
model_url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
model_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(model_dir, "selfie_multiclass_256x256.tflite")

# Download the model if it doesn't exist
if not os.path.exists(model_path):
    print(f"Downloading model to {model_path}")
    urllib.request.urlretrieve(model_url, model_path)
    print("Model downloaded successfully")
else:
    print("Model already exists")
```

## MediaPipe Version Compatibility

**Important:** Current MediaPipe Python packages have limited or no support for custom segmentation models:

- Standard MediaPipe Python package (via pip): Does not support custom models for segmentation
- MediaPipe 0.9.0+ or custom builds: May support the `image_segmentation` module

The application will automatically detect support and use one of these approaches:
1. If `image_segmentation` module is available: Use custom model
2. If not supported: Fall back to the default binary segmentation model

## Model Placement

Please place the model file directly in this directory. The application specifically looks for:
```
/assets/models/selfie_multiclass_256x256.tflite
```

## For Developers

To fully utilize multiclass segmentation with custom models, you may need to:

1. Build MediaPipe from source with custom segmentation support
2. Use a development version of MediaPipe that includes the `image_segmentation` module
3. Implement your own TFLite model inference pipeline if MediaPipe doesn't support your needs

The current code provides a graceful fallback to standard segmentation when custom models aren't supported.
