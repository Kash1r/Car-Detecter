import cv2
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms import functional as F
from PIL import Image

# Check if CUDA (GPU) is available and set PyTorch to use it if it is
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pre-trained Faster R-CNN model with a MobileNet v3 Large backbone
model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
# Move the model to the GPU if available
model = model.to(device)
# Set the model to evaluation mode
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Define a function to get predictions from the model
def get_prediction(img, threshold):
    # Define a transform to convert the image to a PyTorch tensor
    transform = F.to_tensor
    # Apply the transform
    img = transform(img)
    # Move the image data to the GPU if available
    img = img.to(device)
    # Get predictions from the model
    pred = model([img])
    # Get the bounding boxes from the predictions
    pred_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in
                  list(pred[0]['boxes'].detach().cpu().numpy())]
    # Get the class labels from the predictions
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    # Get the scores from the predictions
    pred_score = list(pred[0]['scores'].cpu().detach().numpy())
    # Find the last score above the threshold
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    # Only keep the bounding boxes and class labels for the scores above the threshold
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    # Return the bounding boxes and class labels
    return pred_boxes, pred_class


# Open the video file
cap = cv2.VideoCapture(r"C:\Users\Kashir\Desktop\Car_Video.mp4")

# Process each frame in the video
while (cap.isOpened()):
    # Read the next frame
    ret, frame = cap.read()

    if ret:
        # Convert the frame to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the frame to a PIL Image
        img = Image.fromarray(img)
        # Get predictions for the frame
        boxes, pred_class = get_prediction(img, threshold=0.5)

        # Draw a rectangle around each car in the frame
        for i, box in enumerate(boxes):
            if pred_class[i] == 'car':
                cv2.rectangle(frame, box[0], box[1], (0, 255, 0), 2)

        # Count the number of cars in the frame
        car_count = sum(x == 'car' for x in pred_class)
        # Draw the car count on the frame
        cv2.putText(frame, f'Car count: {car_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
        # Display the frame
        cv2.imshow('Video', frame)

        # If the 'q' key is pressed, break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # If there are no more frames, break the loop
        break

# Release the video file
cap.release()
# Close all OpenCV windows
cv2.destroyAllWindows()