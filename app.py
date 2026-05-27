from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import json
import os
import tempfile
import zipfile
from werkzeug.utils import secure_filename

# ===============================
# Config
# ===============================
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
IMG_SIZE = 128
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
# Load Model
# ===============================
def patch_legacy_keras_config(config):
    if isinstance(config, dict):
        config.pop("dtype", None)
        config.pop("quantization_config", None)

        class_name = config.get("class_name")
        inner_config = config.get("config")

        if isinstance(inner_config, dict):
            inner_config.pop("dtype", None)
            inner_config.pop("quantization_config", None)
            inner_config.pop("optional", None)

            if class_name == "InputLayer" and "batch_shape" in inner_config and "batch_input_shape" not in inner_config:
                inner_config["batch_input_shape"] = inner_config.pop("batch_shape")

            for value in inner_config.values():
                patch_legacy_keras_config(value)

        for key, value in list(config.items()):
            if key != "config":
                patch_legacy_keras_config(value)
    elif isinstance(config, list):
        for item in config:
            patch_legacy_keras_config(item)


def load_model_compat(model_path):
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except (TypeError, ValueError):
        if not model_path.endswith(".keras"):
            raise

        with tempfile.TemporaryDirectory() as temp_dir:
            patched_path = os.path.join(temp_dir, os.path.basename(model_path))

            with zipfile.ZipFile(model_path, "r") as source_archive, zipfile.ZipFile(patched_path, "w") as target_archive:
                for info in source_archive.infolist():
                    file_data = source_archive.read(info.filename)

                    if info.filename == "config.json":
                        config = json.loads(file_data.decode("utf-8"))
                        patch_legacy_keras_config(config)
                        file_data = json.dumps(config).encode("utf-8")

                    target_archive.writestr(info, file_data)

            return tf.keras.models.load_model(patched_path, compile=False)


model = load_model_compat("new_rice_disease_model.keras")
num_classes = int(model.output_shape[-1])
class_labels = {
    index: category_dict.get(index, f"Class {index}")
    for index in range(num_classes)
}

# ===============================
# Flask App
# ===============================
app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


# ===============================
# Helpers
# ===============================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Invalid image")

    # IMPORTANT: ensure same format as training
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # normalize
    img = img.astype('float32') / 255.0

    # add batch dimension
    img = np.expand_dims(img, axis=0)

    return img


def predict_image(image_path):
    img = preprocess_image(image_path)

    preds = model.predict(img)
    class_id = np.argmax(preds)
    confidence = float(np.max(preds))

    return {
        "class_id": int(class_id),
        "disease": class_labels[int(class_id)],
        "confidence": round(confidence, 4),
        "all_probabilities": {
            class_labels[i]: float(preds[0][i]) for i in range(num_classes)
        }
    }


# ===============================
# ROUTES (UNCHANGED STRUCTURE)
# ===============================

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model": "tensorflow-rice-disease",
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

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        result = predict_image(filepath)

        os.remove(filepath)

        return jsonify({
            "success": True,
            "prediction": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict_url', methods=['POST'])
def predict_url():
    try:
        data = request.get_json()
        url = data.get("image_url")

        import urllib.request
        filepath = os.path.join(UPLOAD_FOLDER, "temp.jpg")
        urllib.request.urlretrieve(url, filepath)

        result = predict_image(filepath)

        os.remove(filepath)

        return jsonify({
            "success": True,
            "prediction": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/')
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html')


@app.route('/api/docs')
def docs():
    return jsonify({
        "app": "Rice Disease API",
        "framework": "TensorFlow",
        "input_size": IMG_SIZE,
        "classes": class_labels
    })


# ===============================
# RUN
# ===============================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)