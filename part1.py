#This model run first just to detect a person! If a person (driver) is detected, i.e "True", then it will send a signal to turn on the second model.
from ultralytics import YOLO
def is_person_detected(model, image_path):
    # Perform inference on the provided image
    results = model(image_path)
    
    # Access the first result (assuming a single image)
    first_result = results[0]
    
    # Access the boxes attribute
    detected_objects = first_result.boxes
    
    # Check if any people (class index 0) were detected
    person_detected = detected_objects.shape[0] > 0  # True if any boxes are detected
    
    if person_detected:
        scores = detected_objects.conf  # Confidence scores for all detections
        max_score = scores.max().item()  # Get the highest score
    else:
        max_score = 0  # No person detected, score is 0
    
    return person_detected, max_score

def main(image_path):
    model = YOLO('yolo8-trained.pt')  # Load your trained model
    person_detected = is_person_detected(model, image_path)
    return person_detected

