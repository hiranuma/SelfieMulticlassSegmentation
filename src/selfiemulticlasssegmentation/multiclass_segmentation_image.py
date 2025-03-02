import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Process image for multiclass segmentation.')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image file')
    return parser.parse_args()

options = vision.ImageSegmenterOptions(
  base_options=BaseOptions(model_asset_path="./assets/models/selfie_multiclass_256x256.tflite"),
  output_category_mask=True,
  running_mode=vision.RunningMode.IMAGE
)
segmenter = vision.ImageSegmenter.create_from_options(options)

args = parse_args()
path = args.image

image = cv2.imread(path)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

segmentation_result = segmenter.segment(image_rgb)
print("segmentation_result:", segmentation_result)

category_mask = segmentation_result.category_mask
confidence_masks = segmentation_result.confidence_masks

confidence_masks_np_bg = confidence_masks[0].numpy_view()
confidence_masks_np_hair = confidence_masks[1].numpy_view()
confidence_masks_np_body_skin = confidence_masks[2].numpy_view()
confidence_masks_np_face_skin = confidence_masks[3].numpy_view()
confidence_masks_np_clothes = confidence_masks[4].numpy_view()
confidence_masks_np_others = confidence_masks[5].numpy_view()

print(confidence_masks_np_bg.dtype)
print(confidence_masks_np_hair.dtype)
print(confidence_masks_np_body_skin.dtype)
print(confidence_masks_np_face_skin.dtype)
print(confidence_masks_np_clothes.dtype)
print(confidence_masks_np_others.dtype)

category_mask_np = category_mask.numpy_view()
print(category_mask_np.dtype, category_mask_np.shape)
print(np.unique(category_mask_np))

print(category_mask_np.shape)
category_mask_bgr = cv2.cvtColor(category_mask_np, cv2.COLOR_GRAY2BGR)
print(category_mask_bgr.shape)

category_mask_bgr[np.where(category_mask_np == 0)] = (255, 0, 0) # Blue
category_mask_bgr[np.where(category_mask_np == 1)] = (0, 255, 0) # Green
category_mask_bgr[np.where(category_mask_np == 2)] = (0, 0, 255) # Red
category_mask_bgr[np.where(category_mask_np == 3)] = (255, 255, 0) # Yellow
category_mask_bgr[np.where(category_mask_np == 4)] = (0, 255, 255) # Cyan
category_mask_bgr[np.where(category_mask_np == 5)] = (255, 0, 255) # Magenta

cv2.imshow("Original", image)
# cv2.imshow("confidence_masks_np_bg", confidence_masks_np_bg)
# cv2.imshow("confidence_masks_np_hair", confidence_masks_np_hair)
# cv2.imshow("confidence_masks_np_body_skin", confidence_masks_np_body_skin)
# cv2.imshow("confidence_masks_np_face_skin", confidence_masks_np_face_skin)
# cv2.imshow("confidence_masks_np_clothes", confidence_masks_np_clothes)
# cv2.imshow("confidence_masks_np_others", confidence_masks_np_others)
cv2.imshow("category_mask_bgr", category_mask_bgr)
# cv2.imshow("cateogry_mask_np", cateogry_mask_np)

cv2.waitKey(0)
cv2.destroyAllWindows()