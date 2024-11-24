from torch.backends.mkl import verbose
from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model (use a segmentation variant like YOLOv8-seg if available)
model = YOLO("yolo11n-seg.pt")

# Function to process the image and generate bounding boxes with white regions inside
def generate_bounding_boxes_with_white_region(img_path):
    # Perform object detection with YOLO
    results = model.predict(img_path, verbose=False)

    # Read the original image
    img = cv2.imread(img_path)
    height, width, _ = img.shape

    # Create a blank black image of the same size as the original
    black_img = np.zeros((height, width), dtype=np.uint8)

    # Iterate over the detected bounding boxes
    for result in results:
        # Loop through detections, for each detection, get the bounding box coordinates
        boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)  # Convert to integers

            # Fill the region inside the bounding box with white (255)
            black_img[y1:y2, x1:x2] = 255

    # return black_img

    # Save the mask image with white bounding boxes
    # save_path = img_path.replace(".jpg", "_bounding_box_mask.png").replace(".png", "_bounding_box_mask.png")
    # cv2.imwrite(save_path, black_img)
    # print(f"Saved bounding box mask at: {save_path}")
    return black_img

# Test on an image
# img = generate_bounding_boxes_with_white_region('bus.jpg')
