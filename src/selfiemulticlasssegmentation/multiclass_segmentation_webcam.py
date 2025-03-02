import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import numpy as np
import time
import os

# Check if running in headless environment
HEADLESS = 'DISPLAY' not in os.environ or not os.environ['DISPLAY']
if HEADLESS:
    print("Running in headless mode - visualization disabled")

# Check if GPU is available
def check_gpu_available():
    try:
        # Try to get CUDA device count from OpenCV
        cuda_available = hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0
        return cuda_available
    except:
        return False

# Global variables
segmentation_result_list = []
last_frame_time = time.time()
frame_count = 0
fps = 0
processed_frames = 0
total_processing_time = 0

# Reusable buffers
rgb_buffer = None
mask_buffer = None

def segmentation_callback(result, output_image, timestamp_ms):
    segmentation_result_list.append(result)

# Check if GPU is available
use_gpu = check_gpu_available()
print(f"GPU available: {use_gpu}")

# Configure delegate based on GPU availability
delegate = None
if use_gpu:
    try:
        # Try to use GPU delegate if available
        delegate = BaseOptions.Delegate.GPU
    except:
        delegate = None

# Configure options with GPU delegate if available
options = vision.ImageSegmenterOptions(
    base_options=BaseOptions(
        model_asset_path="./assets/models/selfie_multiclass_256x256.tflite",
        delegate=delegate
    ),
    output_category_mask=True,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=segmentation_callback
)

segmenter = vision.ImageSegmenter.create_from_options(options)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam. Testing with a static image instead.")
    # Create a test image for headless environments
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw some shapes for testing
    cv2.rectangle(test_frame, (100, 100), (300, 300), (0, 255, 0), -1)
    cv2.circle(test_frame, (450, 200), 100, (0, 0, 255), -1)
    use_test_image = True
else:
    use_test_image = False
    # Set lower resolution to reduce processing load
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Target FPS for processing (adjust based on your hardware capabilities)
target_fps = 15
frame_interval = 1.0 / target_fps

# Process a limited number of frames for testing
max_frames = 100
frame_count = 0

while frame_count < max_frames:
    current_time = time.time()
    elapsed_time = current_time - last_frame_time
    
    # Skip frames to maintain target FPS
    if elapsed_time < frame_interval:
        # Small sleep to reduce CPU usage
        time.sleep(0.001)
        continue
    
    # Capture frame or use test image
    if use_test_image:
        frame = test_frame.copy()
        ret = True
    else:
        ret, frame = cap.read()
    
    if not ret:
        break
    
    # Update timing variables
    last_frame_time = current_time
    frame_count += 1
    
    # Calculate FPS every second
    if frame_count % target_fps == 0:
        fps = 1.0 / (elapsed_time)
        print(f"Processing FPS: {fps:.1f}")
    
    # Resize frame to reduce processing load (adjust resolution as needed)
    frame = cv2.resize(frame, (320, 240))
    
    # Start timing the processing
    process_start_time = time.time()
    
    # Convert to RGB for MediaPipe (reuse the same memory)
    if rgb_buffer is None or rgb_buffer.shape != frame.shape:
        rgb_buffer = np.empty(frame.shape, dtype=frame.dtype)
    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, rgb_buffer)
    
    # Create MediaPipe image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_buffer)
    
    # Process frame asynchronously
    segmenter.segment_async(mp_image, time.time_ns() // 1_000_000)
    
    # Process results if available
    if len(segmentation_result_list) > 0:
        segmentation_result = segmentation_result_list[0]
        category_mask = segmentation_result.category_mask
        
        # Get numpy view of the mask (no copy)
        category_mask_np = category_mask.numpy_view()
        
        # Create output visualization
        if mask_buffer is None or mask_buffer.shape[:2] != category_mask_np.shape:
            mask_buffer = np.empty((category_mask_np.shape[0], category_mask_np.shape[1], 3), dtype=np.uint8)
        
        # Apply colors directly without intermediate conversion
        mask_buffer.fill(0)
        mask_buffer[category_mask_np == 0] = [255, 0, 0]    # Blue
        mask_buffer[category_mask_np == 1] = [0, 255, 0]    # Green
        mask_buffer[category_mask_np == 2] = [0, 0, 255]    # Red
        mask_buffer[category_mask_np == 3] = [255, 255, 0]  # Yellow
        mask_buffer[category_mask_np == 4] = [0, 255, 255]  # Cyan
        mask_buffer[category_mask_np == 5] = [255, 0, 255]  # Magenta
        
        # End timing the processing
        process_end_time = time.time()
        process_time = process_end_time - process_start_time
        total_processing_time += process_time
        processed_frames += 1
        
        if not HEADLESS:
            # Resize mask to display size if needed
            display_mask = cv2.resize(mask_buffer, (640, 480))
            
            # Display the mask
            cv2.imshow("Segmentation", display_mask)
            
            # Display FPS on the original frame
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the original frame
            display_frame = cv2.resize(frame, (640, 480))
            cv2.imshow("Frame", display_frame)
        
    # Clear results for next frame
    segmentation_result_list.clear()
    
    # Check for exit key if not in headless mode
    if not HEADLESS and cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate and print average processing time
if processed_frames > 0:
    avg_processing_time = total_processing_time / processed_frames
    print(f"\nPerformance Statistics:")
    print(f"Total frames processed: {processed_frames}")
    print(f"Average processing time per frame: {avg_processing_time*1000:.2f} ms")
    print(f"Effective FPS: {1.0/avg_processing_time:.2f}")

# Release resources
if not use_test_image:
    cap.release()
if not HEADLESS:
    cv2.destroyAllWindows()

print("Processing completed successfully.")
