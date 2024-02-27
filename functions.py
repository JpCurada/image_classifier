import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow_hub as hub

class Functions:

    @staticmethod
    def process_image(img_file_buffer, image_size):
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img_resized = cv2.resize(img_gray, image_size)
        img_rgb = np.expand_dims(img_resized, axis=-1)  # Keep the grayscale image
        img_rgb = np.repeat(img_rgb, 3, axis=-1)  # Repeat the grayscale channel to create RGB channels
        img_normalized = img_rgb / 255.0  

        return cv2_img, img_normalized

    
    @staticmethod
    def load_model(model_choice):
        model_dict = {
            "EfficientNet" : "models\efficientnet_model.h5", 
            "Resnet50" : "models\model_resnet50_v2.h5", 
            "MobileNet V2" : "models\mobile_net_model.h5", 
            "Inception V3" : "models\inceptionv3_model.h5"}
        
        custom_objects = {'KerasLayer': hub.KerasLayer}

        model = load_model(model_dict[model_choice], custom_objects=custom_objects)
        return model

    @staticmethod
    def preprocess_and_predict_image(img_file_buffer, model, image_size=(224, 224)):
        """
        Preprocess an image from a file buffer, make predictions using the provided model,
        and return the predicted label and confidence level.

        Parameters:
        - img_file_buffer: File buffer containing the image.
        - model: The trained model for making predictions.
        - label_dict (dict): A dictionary mapping class indices to labels.
        - image_size (tuple): Target size for resizing the images (default is (224, 224)).

        Returns:
        - predicted_label: The predicted label for the input image.
        - confidence_level: The confidence level of the predicted label.
        """

        label_dict = {0: 'paisley', 
                      1: 'plain', 
                      2: 'chequered', 
                      3: 'polka-dotted', 
                      4: 'striped', 
                      5: 'zigzagged', 
                      6: 'animal-print'}

        # Read image from file buffer
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # Preprocess the image
        img_gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, image_size)
        img_rgb = np.expand_dims(img_resized, axis=-1)
        img_rgb = np.repeat(img_rgb, 3, axis=-1)
        img_normalized = img_rgb / 255.0  # Normalize pixel values to range [0, 1]

        # Expand dimensions to match model input shape
        x_test_processed = np.expand_dims(img_normalized, axis=0)

        # Make predictions
        predictions = model.predict(x_test_processed)
        predicted_index = np.argmax(predictions, axis=1)[0]
        predicted_label = label_dict[predicted_index]
        confidence_level = np.max(predictions)

        return predicted_label, confidence_level

    