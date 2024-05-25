from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import base64
import io



# Load the pre-trained model
model = load_model('mango_classification_model.h5')

# Define a function to preprocess the image
def preprocess_image(image):
    # Resize the image to 224x224 pixels
    image = image.resize((224, 224))

    # Convert the PIL image to a NumPy array
    image_array = img_to_array(image)

    # Scale the pixel values from [0, 255] to [-1, 1]
    image_array = image_array / 255.0
    image_array = image_array * 2.0 - 1.0

    # Add a batch dimension to the array
    image_array = np.expand_dims(image_array, axis=0)

    # Return the preprocessed image
    return image_array

# Define a dictionary mapping class indices to class names
class_names = {
    0: 'Anwar Ratool',
    1: 'Chaunsa(Black)',
    2: 'Chaunsa(Summer Bahisht)',
    3: 'Chaunsa(White)',
    4: 'Dosehri',
    5: 'Fajri',
    6: 'Langra',
    7: 'Sindhri',
}

# Create a Flask application
app = Flask(__name__)


# Define a route to the home page
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('home.html')


# Define a route to the index page
@app.route('/predict')
def index():
    return render_template('index.html')


# Define a route to the predict page
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file from the form
    file = request.files.get('image')

    # If the image is not present in the form data, check if it is present in the request JSON
    if file is None:
        try:
            img_data = request.json['image']
            img_data = base64.b64decode(img_data)
            image = Image.open(io.BytesIO(img_data))
        except (KeyError, TypeError, OSError):
            # Return an error response
            return jsonify({'error': 'No image found in request'})
    else:
        # Open the image file using PIL
        image = Image.open(file)

    # Preprocess the image
    image_array = preprocess_image(image)

    # Use the pre-trained model to make predictions
    predictions = model.predict(image_array)

    # Get the predicted class label
    class_label = np.argmax(predictions, axis=1)[0]

    # Get the corresponding class name from the dictionary
    class_name = class_names[class_label]

    # Save the image to a file
    with open('static/images/predicted_image.jpg', 'wb') as f:
        image.save(f, format='JPEG')

    # Convert the image to a base64 string and return it as part of the JSON response
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return render_template('result.html', class_name=class_name)

if __name__ == '__main__':
    app.run()
