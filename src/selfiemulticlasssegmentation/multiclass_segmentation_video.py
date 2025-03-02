import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import numpy as np
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Process video for multiclass segmentation.')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file')
    return parser.parse_args()

options = vision.ImageSegmenterOptions(
  base_options=BaseOptions(model_asset_path="./assets/models/selfie_multiclass_256x256.tflite"),
  output_category_mask=True,
  running_mode=vision.RunningMode.VIDEO
)
segmenter = vision.ImageSegmenter.create_from_options(options)

def main():
    args = parse_args()
    video_path = args.video
    
    # Validate that the file exists
    if not os.path.isfile(video_path):
        print(f"Error: Video file '{video_path}' does not exist.")
        return
    
    cap = cv2.VideoCapture(video_path)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return

    for frame_index in range(int(frame_count)):
        ret, frame = cap.read()
        if ret == False:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        frame_timestamp_ms = int(1000 * frame_index / fps)

        segmentation_result = segmenter.segment_for_video(frame_rgb, frame_timestamp_ms)
        print("segmentation_result:", segmentation_result)

        category_mask = segmentation_result.category_mask

        category_mask_np = category_mask.numpy_view()

        category_mask_bgr = cv2.cvtColor(category_mask_np, cv2.COLOR_GRAY2BGR)

        category_mask_bgr[np.where(category_mask_np == 0)] = (255, 0, 0) # Blue
        category_mask_bgr[np.where(category_mask_np == 1)] = (0, 255, 0) # Green
        category_mask_bgr[np.where(category_mask_np == 2)] = (0, 0, 255) # Red
        category_mask_bgr[np.where(category_mask_np == 3)] = (255, 255, 0) # Yellow
        category_mask_bgr[np.where(category_mask_np == 4)] = (0, 255, 255) # Cyan
        category_mask_bgr[np.where(category_mask_np == 5)] = (255, 0, 255) # Magenta

        cv2.imshow("Frame", frame)
        cv2.imshow("category_mask_bgr", category_mask_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()