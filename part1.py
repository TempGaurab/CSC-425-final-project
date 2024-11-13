#This model run first just to detect a person! If a person (driver) is detected, i.e "True", then it will send a signal to turn on the second model.
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt


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

def main(image_path):
    title = detectperson(image_path)
    if title == "person":
        return True
    return False

def detectperson(imageid):
    base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                        score_threshold=0.5)
    detector = vision.ObjectDetector.create_from_options(options)

    # STEP 3: Load the input image.
    image = mp.Image.create_from_file(imageid)
    # STEP 4: Detect objects in the input image.
    detection_result = detector.detect(image)
    image_copy = np.copy(image.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    title = detection_result.detections[0].categories[0].category_name
    return title
