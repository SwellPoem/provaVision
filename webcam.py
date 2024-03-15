import cv2
import torch
from torchvision import transforms
from PIL import Image
import torch
from yolo_hand_detection_master.yolo import YOLO  # Make sure this import is correct
from cnn import HandGestureCNN  # Adjust this import based on your directory structure

# Instantiate the YOLO detector
yolo_config = 'yolo_hand_detection_master/models/cross-hands.cfg'
yolo_weights = 'yolo_hand_detection_master/models/cross-hands.weights'
yolo_labels = ['Hand']  # Assuming you have one class for 'Hand'
yolo_size = 416  # The size parameter must match what was used during YOLO training
yolo_detector = YOLO(yolo_config, yolo_weights, yolo_labels, size=yolo_size)

# Load your HandGestureCNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hand_gesture_cnn = HandGestureCNN(nc=3, ndf=64, num_classes=7).to(device)
hand_gesture_cnn.load_state_dict(torch.load('cnn_model3.pth'))
hand_gesture_cnn.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def preprocess_for_hand_gesture_cnn(cropped_hand):
    """
    Preprocess the cropped hand image for HandGestureCNN inference.

    Parameters:
    cropped_hand (numpy.ndarray): The cropped hand region from the frame.

    Returns:
    torch.Tensor: The preprocessed image tensor.
    """
    # Convert the cropped hand numpy array to a PIL Image
    cropped_hand_pil = Image.fromarray(cropped_hand)
    # Apply the transformations
    preprocessed_hand = transform(cropped_hand_pil)
    # Add an extra batch dimension since pytorch expects batches of images
    preprocessed_hand = preprocessed_hand.unsqueeze(0).to(device)

    return preprocessed_hand

gesture_classes = ['dislike', 'L', 'like', 'ok', 'palm', 'peace', 'rock']

# Initialize webcam
cap = cv2.VideoCapture(0)

# Main loop to process the webcam feed
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference to detect hands
    width, height, inference_time, yolo_results = yolo_detector.inference(frame)

    # Display FPS
    cv2.putText(frame, f'{round(1/inference_time,2)} FPS', (15,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

    # Sort detections by confidence
    yolo_results.sort(key=lambda x: x[2], reverse=True)  # Assuming higher confidence first

    # Process each detection for gesture classification
    for detection in yolo_results:
        id, name, confidence, x, y, w, h = detection
        
        # Crop the detected hand
        cropped_hand = frame[y:y+h, x:x+w]
        
        # Preprocess the cropped hand for HandGestureCNN
        preprocessed_hand = preprocess_for_hand_gesture_cnn(cropped_hand)
        
        # Run HandGestureCNN inference to classify the gesture
        gesture = "Unknown"  # Default if no prediction exceeds threshold
        with torch.no_grad():
            prediction = hand_gesture_cnn(preprocessed_hand)
            max_value, predicted_idx = torch.max(prediction, 1)
            if max_value.item() > 0.3:  # Use your confidence threshold
                gesture = gesture_classes[predicted_idx.item()]

        # Draw bounding box and label with gesture
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        text = f"{name} ({round(confidence, 2)}): {gesture}"
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Show the modified frame
    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# import cv2
# import torch
# from torchvision import transforms
# from PIL import Image
# from cnn import HandGestureCNN
# import matplotlib.pyplot as plt

# def preprocess_frame(frame, transform):
#     """Convert BGR OpenCV frame to the desired format and apply transforms"""
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame_pil = Image.fromarray(frame_rgb)
#     frame_transformed = transform(frame_pil)
#     return frame_transformed.unsqueeze(0)  # Add batch dimension

# # Hyperparameters
# nc = 3  # Number of channels in the training images
# ndf = 64  # Size of feature maps
# num_classes = 7  # Adjust to match your model

# # Transformation pipeline
# transform = transforms.Compose([
#     transforms.Resize(64),
#     transforms.CenterCrop(64),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])

# # Initialize your CNN model
# model = HandGestureCNN(nc, ndf, num_classes)
# model.load_state_dict(torch.load('cnn_model3.pth'))
# model.eval()

# # gesture_classes = ['call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one', 'palm', 'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted', 'three', 'three2', 'two_up', 'two_up_inverted']
# gesture_classes = ['dislike', 'L', 'like', 'ok', 'palm', 'peace', 'rock']
# # gesture_classes = ['fist', 'fist_moved', 'index', 'l', 'ok', 'palm', 'thumb']

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# # while True:
# #     ret, frame = cap.read()
# #     if not ret:
# #         break

# #     frame_transformed = preprocess_frame(frame, transform)

# #     with torch.no_grad():
# #         prediction = model(frame_transformed)
# #         _, predicted_idx = torch.max(prediction, 1)
# #         gesture = gesture_classes[predicted_idx.item()]

# #     cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
# #     cv2.imshow('Webcam Feed', frame)

# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_transformed = preprocess_frame(frame, transform)

#     with torch.no_grad():
#         prediction = model(frame_transformed)
#         max_value, predicted_idx = torch.max(prediction, 1)
#         print(f"Max value: {max_value.item()}")
#         if max_value.item() > 0.3:  # Replace 'threshold' with your chosen value
#             gesture = gesture_classes[predicted_idx.item()]
#         else:
#             gesture = ""

#     cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#     cv2.imshow('Webcam Feed', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
