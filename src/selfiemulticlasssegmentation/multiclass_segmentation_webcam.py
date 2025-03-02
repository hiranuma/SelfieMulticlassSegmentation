import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import numpy as np
import time

segmentation_result_list = []

def segmentation_callback(result, output_image, timestamp_ms):
  segmentation_result_list.append(result)

options = vision.ImageSegmenterOptions(
  base_options=BaseOptions(model_asset_path="./assets/models/selfie_multiclass_256x256.tflite"),
  output_category_mask=True,
  running_mode=vision.RunningMode.LIVE_STREAM,
  result_callback=segmentation_callback
)
segmenter = vision.ImageSegmenter.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:
  ret, frame = cap.read()
  if not ret:
    break

  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  frame_rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
  
  segmenter.segment_async(frame_rgb, time.time_ns() // 1_000_000)

  if len(segmentation_result_list) > 0:
    segmentation_result = segmentation_result_list[0]

    category_mask = segmentation_result.category_mask

    category_mask_np = category_mask.numpy_view()

    category_mask_bgr = cv2.cvtColor(category_mask_np, cv2.COLOR_GRAY2BGR)

    category_mask_bgr[np.where(category_mask_np == 0)] = (255, 0, 0) # Blue
    category_mask_bgr[np.where(category_mask_np == 1)] = (0, 255, 0) # Green
    category_mask_bgr[np.where(category_mask_np == 2)] = (0, 0, 255) # Red
    category_mask_bgr[np.where(category_mask_np == 3)] = (255, 255, 0) # Yellow
    category_mask_bgr[np.where(category_mask_np == 4)] = (0, 255, 255) # Cyan
    category_mask_bgr[np.where(category_mask_np == 5)] = (255, 0, 255) # Magenta

    cv2.imshow("category_mask_bgr", category_mask_bgr)

  segmentation_result_list.clear()
  cv2.imshow("Frame", frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()