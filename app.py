from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os

# ===============================
# Config
# ===============================
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
MAX_FILE_SIZE = 10 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===============================
# Class Mapping
# ===============================
category_dict = {
    0: 'Bacterial leaf blight',
    1: 'Brown spot',
    2: 'Leaf smut',
    3: 'healthy'
}

# ===============================
# Load TFLite Model
# ===============================
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

model_path = "dis_model_with_aug (2).tflite"
if not os.path.exists(model_path):
    model_path = "dis_model_with_aug (2).tflite"

interpreter = load_tflite_model(model_path)
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
num_classes = int(output_details["shape"][-1])

class_labels = {
    index: category_dict.get(index, f"Class {index}")
    for index in range(num_classes)
}

IMG_SIZE = 128 

# Warm up the interpreter
interpreter.set_tensor(input_details["index"], np.zeros(input_details["shape"], dtype=np.float32))
interpreter.invoke()

# ===============================
# Flask App Setup
# ===============================
app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


# ===============================
# Core Preprocessing & Prediction Functions
# ===============================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_bytes(image_bytes):
    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image file provided")

    # FIX APPLIED HERE: Explicitly forcing Linear Interpolation to match Colab defaults
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    return img

def predict_from_array(img_tensor):
    # Add this right before interpreter.set_tensor(...)
    print("Total Image Array Sum:", np.sum(img_tensor))

    interpreter.set_tensor(input_details["index"], img_tensor)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details["index"])[0] 
    
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds)) # Left as decimal so frontend can * 100

    return {
        "class_id": class_id,
        "disease": class_labels[class_id],
        "confidence": confidence, 
        "all_probabilities": {
            class_labels[i]: float(preds[i]) for i in range(num_classes)
        }
    }
 

# ===============================
# API Routes
# ===============================
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model": "tensorflow-lite-rice-disease",
        "classes": class_labels
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        image_bytes = file.read()
        processed_tensor = preprocess_image_bytes(image_bytes)
        result = predict_from_array(processed_tensor)

        return jsonify({
            "success": True,
            "prediction": result
        })

    except Exception as e:
        app.logger.exception("Prediction endpoint failure triggered")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict_url', methods=['POST'])
def predict_url():
    try:
        data = request.get_json()
        url = data.get("image_url")
        if not url:
            return jsonify({"error": "Missing image_url field"}), 400

        import urllib.request
        with urllib.request.urlopen(url) as response:
            image_bytes = response.read()

        processed_tensor = preprocess_image_bytes(image_bytes)
        result = predict_from_array(processed_tensor)

        return jsonify({
            "success": True,
            "prediction": result
        })

    except Exception as e:
        app.logger.exception("URL prediction endpoint failure triggered")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html')

@app.route('/api/docs')
def docs():
    return jsonify({
        "app": "Rice Disease API",
        "framework": "TensorFlow Lite Runtime",
        "input_size": f"{IMG_SIZE}x{IMG_SIZE}",
        "channels": "BGR",
        "classes": class_labels
    })

# ===============================
# Execution Entry Point
# ===============================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)