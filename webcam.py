import cv2
import torch
from torchvision import transforms
from PIL import Image
from models import Discriminator

def preprocess_frame(frame, transform):
    """Convert BGR OpenCV frame to the desired format and apply transforms"""
    # Convert BGR (OpenCV) to RGB (PIL)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    
    # Apply the transformations
    frame_transformed = transform(frame_pil)
    
    return frame_transformed.unsqueeze(0)  # Add batch dimension

# Hyperparameters
nc = 3  # Number of channels in the training images
ndf = 64  # Size of feature maps in the discriminator

# Transformation pipeline as specified in dataset.py
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Initialize your GAN model
# Define the model architecture
model = Discriminator(nc, ndf, num_classes=18)
# Load the model weights
model.load_state_dict(torch.load('path_to_your_model_weights.pth'))
# Set the model to evaluation mode
model.eval()

gesture_classes = ['call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one', 'palm', 'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted', 'three', 'three2', 'two_up', 'two_up_inverted']

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Pre-process the captured frame using the specified transformations
    frame_transformed = preprocess_frame(frame, transform)

    # Make a prediction with your model
    # with torch.no_grad():
    #     prediction = model(frame_transformed)
    #     # Translate 'prediction' to a human-readable gesture
    #     # Define your gesture classes
    #     gesture_classes = ['call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one', 'palm', 'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted', 'three', 'three2', 'two_up', 'two_up_inverted']

    #     # Ensure prediction is in the correct format if necessary
    #     prediction = prediction.squeeze()  # Remove batch dimension if present

    #     # Get the index of the highest probability
    #     _, predicted_idx = torch.max(prediction, -1)  # Ensure finding the max across the correct dimension

    #     # Convert this index into a gesture
    #     gesture = gesture_classes[predicted_idx.item()]  # Use .item() to get a Python scalar from a tensor

    # with torch.no_grad():
    #     prediction = model(frame_transformed)
    #     # prediction = prediction.squeeze()  # Assuming this is the logits

    #     # Double-checking the shape of the prediction tensor
    #     print(f"Prediction shape: {prediction.shape}")  # Expected to be [num_classes]

    #     # Apply softmax to convert logits to probabilities
    #     probabilities = torch.softmax(prediction, dim=0)
    #     print(f"Probabilities: {probabilities}")  # Checking the softmax output

    #     # Get the highest probability and its corresponding index
    #     max_prob, predicted_idx = torch.max(probabilities, 0)
    #     print(f"Max probability: {max_prob.item()}, Predicted index: {predicted_idx.item()}")  # Verifying the max probability and index

    #     # Define a confidence threshold
    #     confidence_threshold = 0.8

    #     if max_prob.item() >= confidence_threshold:
    #         print(f"Confident Prediction -> Gesture: {gesture_classes[predicted_idx.item()]}, Confidence: {max_prob.item()}")
    #         gesture = gesture_classes[predicted_idx.item()]
    #     else:
    #         print("Low Confidence Prediction -> Gesture: Uncertain")
    #         gesture = 'uncertain'

    with torch.no_grad():
        # Reshape the input tensor to the expected shape
        frame_transformed = frame_transformed.view(1, 3, 64, 64)

        prediction = model(frame_transformed)  # No need to unsqueeze
        print(f"Prediction shape: {prediction.shape}")  # Expected to be [num_classes]

        # Ensure prediction is in the correct format if necessary
        if len(prediction.shape) == 0:  # If prediction is a scalar
            prediction = prediction.view(1)  # Convert it to a 1D tensor

        # Get the index of the highest probability
        _, predicted_idx = torch.max(prediction, -1)  # Ensure finding the max across the correct dimension

        # Convert this index into a gesture
        gesture = gesture_classes[predicted_idx.item()]  # Use .item() to get a Python scalar from a tensor



    # Display the resulting frame with prediction
    cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (225, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Webcam Feed', frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
