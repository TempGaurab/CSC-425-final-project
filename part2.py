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

    # Define landmark indices
    LEFT_EYE_INDICES = list(mp.solutions.face_mesh.FACEMESH_LEFT_EYE)
    RIGHT_EYE_INDICES = list(mp.solutions.face_mesh.FACEMESH_RIGHT_EYE)
    LEFT_EYEBROW_INDICES = list(mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW)
    RIGHT_EYEBROW_INDICES = list(mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW)

    # Loop through the detected faces
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Get image dimensions
        image_height, image_width = annotated_image.shape[:2]

        # Extract both eye regions
        for side, eye_indices, brow_indices in [
            ('left', LEFT_EYE_INDICES, LEFT_EYEBROW_INDICES),
            ('right', RIGHT_EYE_INDICES, RIGHT_EYEBROW_INDICES)
        ]:
            # Get eye and eyebrow landmarks
            x_coords = []
            y_coords = []
            
            # Combine and flatten eye and eyebrow indices
            all_indices = eye_indices + brow_indices
            flat_indices = [idx for pair in all_indices for idx in pair]
            unique_indices = list(set(flat_indices))
            
            # Collect all x and y coordinates for the eye and eyebrow
            for landmark_idx in unique_indices:
                landmark = face_landmarks[landmark_idx]
                x_coords.append(landmark.x * image_width)
                y_coords.append(landmark.y * image_height)
            
            # Calculate bounding box with padding
            padding_horizontal = 20  # Horizontal padding
            padding_vertical = 15    # Vertical padding
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Add padding
            x_min = max(0, x_min - padding_horizontal)
            x_max = min(image_width, x_max + padding_horizontal)
            y_min = max(0, y_min - padding_vertical)
            y_max = min(image_height, y_max + padding_vertical)
            
            # Convert to integers
            x_min, x_max = int(x_min), int(x_max)
            y_min, y_max = int(y_min), int(y_max)
            
            # Extract eye-eyebrow region
            eye_region = rgb_image[y_min:y_max, x_min:x_max]
            
            # Store the eye region image
            if side == 'left':
                left_eye_img = eye_region
            else:
                right_eye_img = eye_region

            # Draw rectangle on annotated image
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