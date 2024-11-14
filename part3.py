import numpy as np  
from PIL import Image
from tensorflow.keras.models import load_model

def get_model():
    best_model = load_model('bestModel.keras')
    return best_model

def preprocess_image(image_path):
    # Load and preprocess the image
    image = image_path.resize((64, 64)).convert('L')  # Convert to grayscale
    image = image.convert('L')
    image_array = np.array(image)  # Convert to numpy array
    image_array = image_array.reshape(1, 64, 64, 1)  # Reshape for model input
    image_array = image_array.astype('float32') / 255.0  # Normalize the image
    return image_array

def output(prediction):
    if prediction > 0.5:
        return 'Open'
    else:
        return "Closed"
    
def main3(image_path):
    test_image = preprocess_image(image_path)
    prediction = get_model().predict(test_image)
    result = output(prediction)
    return result,prediction