import cv2
import torch
from torchvision import transforms
from PIL import Image
from cnn import HandGestureCNN
import matplotlib.pyplot as plt

def preprocess_frame(frame, transform):
    """Convert BGR OpenCV frame to the desired format and apply transforms"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    frame_transformed = transform(frame_pil)
    return frame_transformed.unsqueeze(0)  # Add batch dimension

# Hyperparameters
nc = 3  # Number of channels in the training images
ndf = 64  # Size of feature maps
num_classes = 18  # Adjust to match your model

# Transformation pipeline
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Initialize your CNN model
model = HandGestureCNN(nc, ndf, num_classes)
model.load_state_dict(torch.load('cnn_model2.pth'))
model.eval()

gesture_classes = ['call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one', 'palm', 'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted', 'three', 'three2', 'two_up', 'two_up_inverted']
# gesture_classes = ['fist', 'fist_moved', 'index', 'l', 'ok', 'palm', 'thumb']

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_transformed = preprocess_frame(frame, transform)

    with torch.no_grad():
        prediction = model(frame_transformed)
        _, predicted_idx = torch.max(prediction, 1)
        gesture = gesture_classes[predicted_idx.item()]

    cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


#OLD CODE FOR GAN

# import cv2
# import torch
# from torchvision import transforms
# from PIL import Image
# from models import Discriminator
# import matplotlib.pyplot as plt

# def preprocess_frame(frame, transform):
#     """Convert BGR OpenCV frame to the desired format and apply transforms"""
#     # Convert BGR (OpenCV) to RGB (PIL)
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame_pil = Image.fromarray(frame_rgb)
    
#     # Apply the transformations
#     frame_transformed = transform(frame_pil)
    
#     return frame_transformed.unsqueeze(0)  # Add batch dimension

# # Hyperparameters
# nc = 3  # Number of channels in the training images
# ndf = 64  # Size of feature maps in the discriminator

# # Transformation pipeline as specified in dataset.py
# transform = transforms.Compose([
#     transforms.Resize(64),
#     transforms.CenterCrop(64),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])

# # Initialize your GAN model
# # Define the model architecture
# model = Discriminator(nc, ndf)
# # Load the model weights
# model.load_state_dict(torch.load('discriminator3.pth'))
# # Set the model to evaluation mode
# model.eval()

# gesture_classes = ['call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one', 'palm', 'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted', 'three', 'three2', 'two_up', 'two_up_inverted']

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Pre-process the captured frame using the specified transformations
#     frame_transformed = preprocess_frame(frame, transform)

#     with torch.no_grad():
#         # Reshape the input tensor to the expected shape
#         frame_transformed = frame_transformed.view(1, 3, 64, 64)

#         prediction = model(frame_transformed)  # No need to unsqueeze
#         print(f"Prediction shape: {prediction.shape}")  # Expected to be [num_classes]

#         # Ensure prediction is in the correct format if necessary
#         if len(prediction.shape) == 0:  # If prediction is a scalar
#             prediction = prediction.view(1)  # Convert it to a 1D tensor

#         # Get the index of the highest probability
#         _, predicted_idx = torch.max(prediction, -1)  # Ensure finding the max across the correct dimension

#         # Convert this index into a gesture
#         gesture = gesture_classes[predicted_idx.item()]  # Use .item() to get a Python scalar from a tensor



#     # Display the resulting frame with prediction
#     cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (225, 0, 0), 2, cv2.LINE_AA)
#     cv2.imshow('Webcam Feed', frame)

#     # Break the loop with 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
