from flask import Flask,render_template,request,jsonify
import logging
import tensorflow as tf
import cv2
from keras.preprocessing import image
import os
import numpy as np

app = Flask(__name__)
# Set upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Load the saved model
#with open('hdbscan_model.pkl', 'rb') as model_file:

model = tf.keras.models.load_model('image_classification_model')
logger.info('Model loaded successfully')
IMG_SIZE = (28,28)
def preprocess_image(img_path):
    """Preprocess image for model prediction."""
    img = image.load_img(img_path, target_size=IMG_SIZE)  # Resize image
    img_array = image.img_to_array(img)  # Convert to array

    # Convert to grayscale (if the image has 3 channels)
    if img_array.shape[-1] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)  # Convert to grayscale

    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension to make it (28, 28, 1)
    img_array = img_array / 255.0  # Normalize
    return img_array

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_image():
    logger.info('Received a prediction request')
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No images part in the request'}), 400

        files = request.files.getlist('files')
        predictions = []

        for file in files:
            if file.filename == '':
                return jsonify({'error': 'One or more files have no filename'}), 400

            # Save file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Preprocess and predict
            img_array = preprocess_image(filepath)
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]  # Get class index
            predicted_class_name = class_names[predicted_class]
            confidence = float(np.max(prediction))

            # Store results
            predictions.append({'filename': file.filename, 'predicted_class': predicted_class_name,'confidence':confidence})
        print(predictions)
        return jsonify(predictions), 200

    except Exception as e:
        logger.error('Error during prediction: %s', str(e))
        return 'Error during prediction', 500


if __name__ == '__main__':
    logger.info('Starting the Flask app')
    app.run(debug=True, host="0.0.0.0", port=5000)
    logger.info('Flask app is running')
