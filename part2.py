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
    EYE_INDICES = [mp.solutions.face_mesh.FACEMESH_LEFT_EYE, mp.solutions.face_mesh.FACEMESH_RIGHT_EYE]
    BROW_INDICES = [mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW, mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW]

    for idx, face_landmarks in enumerate(face_landmarks_list):
        image_height, image_width = annotated_image.shape[:2]

        for side, eye_indices, brow_indices in zip(['left', 'right'], EYE_INDICES, BROW_INDICES):
            eye_coords = np.array([face_landmarks[i] for i in eye_indices]).T
            brow_coords = np.array([face_landmarks[i] for i in brow_indices]).T

            eye_center_y = (eye_coords[1].max() + eye_coords[1].min()) / 2
            eye_height = eye_coords[1].max() - eye_coords[1].min()
            brow_top = brow_coords[1].min()

            x_min = min(eye_coords[0].min(), brow_coords[0].min())
            x_max = max(eye_coords[0].max(), brow_coords[0].max())
            y_min = brow_top
            y_max = eye_center_y + eye_height / 2 / 0.7

            x_min = max(0, x_min - 15)
            x_max = min(image_width, x_max + 15)
            y_min = max(0, y_min - 15)
            y_max = min(image_height, y_max + 10)

            x_min, x_max, y_min, y_max = map(int, [x_min, x_max, y_min, y_max])
            eye_region = rgb_image[y_min:y_max, x_min:x_max]

            if side == 'left':
                left_eye_img = eye_region
            else:
                right_eye_img = eye_region

            cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

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