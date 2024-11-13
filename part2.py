import cv2
import numpy as np
from typing import List, Optional, Tuple
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt



def pil_to_cv2(pil_image):
    """Convert PIL Image to CV2 format"""
    # Convert PIL image to RGB if it's in RGBA
    if pil_image.mode == 'RGBA':
        pil_image = pil_image.convert('RGB')
    # Convert to numpy array
    numpy_image = np.array(pil_image)
    # Convert RGB to BGR
    cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return cv2_image

def cv2_to_pil(cv2_image):
    """Convert CV2 image to PIL format"""
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_image)
    return pil_image

def extract_eyes_from_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)
    left_eye_img = None
    right_eye_img = None

    # Define eye landmark indices
    LEFT_EYE_INDICES = list(mp.solutions.face_mesh.FACEMESH_LEFT_EYE)
    RIGHT_EYE_INDICES = list(mp.solutions.face_mesh.FACEMESH_RIGHT_EYE)

    # Loop through the detected faces to visualize
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Get image dimensions
        image_height, image_width = annotated_image.shape[:2]

        # Dictionary to store eye coordinates and images
        eye_images = {'left': None, 'right': None}

        # Extract both eyes
        for eye_type, eye_indices in [('left', LEFT_EYE_INDICES), ('right', RIGHT_EYE_INDICES)]:
            # Get eye landmarks
            x_coords = []
            y_coords = []
            
            # Flatten the eye indices (they come as pairs of connections)
            flat_indices = [idx for pair in eye_indices for idx in pair]
            unique_indices = list(set(flat_indices))
            
            # Collect all x and y coordinates for the eye
            for landmark_idx in unique_indices:
                landmark = face_landmarks[landmark_idx]
                x_coords.append(landmark.x * image_width)
                y_coords.append(landmark.y * image_height)
            
            # Calculate bounding box with padding
            padding = 20  # Adjust this value to change the size of the rectangle
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Add padding
            x_min = max(0, x_min - padding) - 30
            x_max = min(image_width, x_max + padding) + 30
            y_min = max(0, y_min - padding) -  30
            y_max = min(image_height, y_max + padding) + 30
            
            # Convert to integers
            x_min, x_max = int(x_min), int(x_max)
            y_min, y_max = int(y_min), int(y_max)
            
            # Extract eye region
            eye_img = rgb_image[y_min:y_max, x_min:x_max]
            
            # Store the eye image
            if eye_type == 'left':
                left_eye_img = eye_img
            else:
                right_eye_img = eye_img

            # Draw rectangle on annotated image (optional, for visualization)
            cv2.rectangle(annotated_image, 
                        (x_min, y_min), 
                        (x_max, y_max), 
                        (0, 255, 0),  # Green color (BGR)
                        2)  # Thickness

    return annotated_image, left_eye_img, right_eye_img

def get_output(imageid):
    base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    image = mp.Image.create_from_file(imageid)
    detection_result = detector.detect(image)
    annotated_image,l,r = extract_eyes_from_image(image.numpy_view(), detection_result)
    return annotated_image,l,r