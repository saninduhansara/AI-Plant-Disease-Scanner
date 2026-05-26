from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from PIL import Image
import os
import tempfile
import urllib.request
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import time
import logging

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

try:
    import tensorflow as tf
    # Allow TensorFlow to decide thread parallelism by default. If you need to
    # limit threads for your environment, set TF_NUM_THREADS environment var.
    try:
        num_threads = int(os.environ.get('TF_NUM_THREADS', '0'))
    except ValueError:
        num_threads = 0
    if num_threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(num_threads)
        tf.config.threading.set_inter_op_parallelism_threads(num_threads)
except Exception:
    tf = None


# ===============================
# Configuration
# ===============================
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = os.environ.get('MODEL_PATH', 'rice_disease_model.keras')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024
IMAGE_SIZE = (128, 128)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ===============================
# Class Mapping
# ===============================
CLASS_MAPPING = {
    0: 'brown_spot',
    1: 'bacterial_leaf_blight',
    2: 'healthy',
    3: 'leaf_blast',
    4: 'narrow_brown_spot',
    5: 'leaf_scald'
}


# ===============================
# Initialize Flask App
# ===============================
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


# ===============================
# Load Model
# ===============================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"TensorFlow model file not found: {MODEL_PATH}. "
        "Place your .keras model in the project root or set MODEL_PATH."
    )

model = load_model(MODEL_PATH, compile=False)

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

# Warm-up model once to reduce first-prediction latency
try:
    logging.info('Warming up TensorFlow model...')
    dummy = np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)
    # Create a compiled TF function for faster repeated inference when TF is present
    try:
        if 'tf' in globals() and tf is not None:
            @tf.function(experimental_relax_shapes=True)
            def _tf_predict(x):
                return model(x, training=False)
            _ = _tf_predict(tf.convert_to_tensor(dummy, dtype=tf.float32)).numpy()
        else:
            _ = model.predict(dummy, verbose=0)
    except Exception:
        # Fallback to model.predict if tf.function fails for some reason
        _ = model.predict(dummy, verbose=0)
    logging.info('Model warm-up complete')
except Exception as exc:
    logging.warning(f'Model warm-up failed: {exc}')

# If TF is available, expose the compiled predict function for inference path
_tf_predict = None
if 'tf' in globals() and tf is not None:
    try:
        @tf.function(experimental_relax_shapes=True)
        def _tf_predict(x):
            return model(x, training=False)
    except Exception:
        _tf_predict = None


# ===============================
# Helper Functions
# ===============================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_temp_file(file_name, prefix):
    _, extension = os.path.splitext(secure_filename(file_name))
    extension = extension.lower() if extension.lower() in {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'} else '.jpg'
    fd, filepath = tempfile.mkstemp(prefix=prefix, suffix=extension, dir=UPLOAD_FOLDER)
    os.close(fd)
    return filepath


def preprocess_image(image_path, target_size=IMAGE_SIZE):
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        return np.expand_dims(image_array, axis=0)
    except Exception as exc:
        raise ValueError(f'Image preprocessing error: {exc}') from exc


def predict_disease(image_path):
    # measure preprocessing and prediction time for debugging
    start_total = time.time()
    image_array = preprocess_image(image_path)
    start_pred = time.time()
    # Use compiled TF function when available (faster repeated inference)
    if _tf_predict is not None:
        try:
            import tensorflow as _tf
            input_tensor = _tf.convert_to_tensor(image_array, dtype=_tf.float32)
            predictions = _tf_predict(input_tensor).numpy()
        except Exception:
            predictions = model.predict(image_array, verbose=0)
    else:
        predictions = model.predict(image_array, verbose=0)
    pred_time = time.time() - start_pred
    total_time = time.time() - start_total

    probabilities = np.asarray(predictions[0], dtype=np.float32)

    class_id = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))
    disease_name = CLASS_MAPPING.get(class_id, str(class_id))

    all_probabilities = {
        CLASS_MAPPING[i]: round(float(probabilities[i]), 4)
        for i in range(len(CLASS_MAPPING))
    }

    logging.info(f'Prediction: class={class_id} ({disease_name}), confidence={confidence:.4f}, pred_time={pred_time:.3f}s, total_time={total_time:.3f}s')

    return {
        'class_id': class_id,
        'disease': disease_name,
        'confidence': round(confidence, 4),
        'all_probabilities': all_probabilities,
        'timings': {
            'prediction_seconds': round(pred_time, 4),
            'total_seconds': round(total_time, 4),
        }
    }


def cleanup_file(filepath):
    try:
        os.remove(filepath)
    except OSError:
        pass


# ===============================
# API Routes
# ===============================
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model': 'rice_leaf_disease',
        'backend': 'tensorflow',
        'classes': CLASS_MAPPING,
        'model_path': MODEL_PATH,
    }), 200


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({
                'error': f'File type not allowed. Allowed: {", ".join(sorted(ALLOWED_EXTENSIONS))}'
            }), 400

        filepath = save_temp_file(file.filename, 'upload_')
        file.save(filepath)

        try:
            result = predict_disease(filepath)
        finally:
            cleanup_file(filepath)

        return jsonify({
            'success': True,
            'prediction': result,
        }), 200

    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'error': f'Server error: {exc}'}), 500


@app.route('/api/predict_url', methods=['POST'])
def predict_url():
    try:
        data = request.get_json(silent=True)

        if not data or 'image_url' not in data:
            return jsonify({'error': 'No image_url provided'}), 400

        image_url = data['image_url']
        filepath = save_temp_file('image.jpg', 'url_')

        try:
            urllib.request.urlretrieve(image_url, filepath)
            result = predict_disease(filepath)
        finally:
            cleanup_file(filepath)

        return jsonify({
            'success': True,
            'prediction': result,
        }), 200

    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'error': f'Server error: {exc}'}), 500


@app.route('/', methods=['GET'])
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html')


@app.route('/api/docs', methods=['GET'])
def docs():
    return jsonify({
        'app': 'Rice Leaf Disease Detection API',
        'version': '2.0',
        'backend': 'tensorflow',
        'model_path': MODEL_PATH,
        'image_size': IMAGE_SIZE,
        'endpoints': {
            '/api/health': {
                'method': 'GET',
                'description': 'Health check'
            },
            '/api/predict': {
                'method': 'POST',
                'description': 'Predict disease from uploaded image',
                'params': 'multipart/form-data with file field'
            },
            '/api/predict_url': {
                'method': 'POST',
                'description': 'Predict disease from image URL',
                'params': 'JSON with image_url field'
            }
        },
        'classes': CLASS_MAPPING,
        'example_response': {
            'success': True,
            'prediction': {
                'class_id': 2,
                'disease': 'healthy',
                'confidence': 0.9876,
                'all_probabilities': {
                    'brown_spot': 0.0001,
                    'bacterial_leaf_blight': 0.0002,
                    'healthy': 0.9876,
                    'leaf_blast': 0.0051,
                    'narrow_brown_spot': 0.0068,
                    'leaf_scald': 0.0002
                }
            }
        }
    }), 200


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': f'File too large. Max size: {MAX_FILE_SIZE} bytes'}), 413


# ===============================
# Run Flask App
# ===============================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print('\n' + '=' * 60)
    print('Rice Leaf Disease Detection API')
    print('=' * 60)
    print(f'Model: {MODEL_PATH}')
    print('Backend: TensorFlow/Keras')
    print(f'Upload Folder: {UPLOAD_FOLDER}')
    print('=' * 60)
    print(f'Starting server on http://localhost:{port}')
    print(f'Web UI: http://localhost:{port}/')
    print(f'API Documentation: http://localhost:{port}/api/docs')
    print('=' * 60 + '\n')

    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
