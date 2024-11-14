#This model run first just to detect a person! If a person (driver) is detected, i.e "True", then it will send a signal to turn on the second model.
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
import tempfile
import os

def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  MARGIN = 10  # pixels
  ROW_SIZE = 10  # pixels
  FONT_SIZE = 1
  FONT_THICKNESS = 1
  TEXT_COLOR = (255, 0, 0)  # red
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image

def main(image):
    """
    Modified main function to handle both file paths and numpy arrays
    Args:
        image: Can be either a file path (str) or numpy array
    """
    if isinstance(image, np.ndarray):
        # If image is numpy array, save it temporarily and get the path
        image_path = process_image_for_detection(image)
    else:
        image_path = image
    
    try:
        title = detectperson(image_path)
        # Clean up temporary file if it was created
        if isinstance(image, np.ndarray):
            os.remove(image_path)
        return title == "person"
    except Exception as e:
        # Clean up temporary file if it was created
        if isinstance(image, np.ndarray):
            os.remove(image_path)
        raise e
    
def process_image_for_detection(image_array):
    """Convert numpy array to MediaPipe image format using a temporary file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        # Save the numpy array as an image file
        cv2.imwrite(tmp_file.name, cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        return tmp_file.name


def detectperson(imageid):
    base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                         score_threshold=0.5)
    detector = vision.ObjectDetector.create_from_options(options)

    # Load the input image
    image = mp.Image.create_from_file(imageid)
    # Detect objects in the input image
    detection_result = detector.detect(image)
    
    # Check if any detections were made
    if not detection_result.detections:
        return "no_person"
    
    title = detection_result.detections[0].categories[0].category_name
    return title