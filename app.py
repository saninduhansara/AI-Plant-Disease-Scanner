from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename

# ===============================
# Configuration
# ===============================
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ===============================
# Class Mapping
# ===============================
CLASS_MAPPING = {
    0: 'bacterial_leaf_blight',
    1: 'brown_spot',
    2: 'healthy',
    3: 'leaf_blast',
    4: 'leaf_scald',
    5: 'narrow_brown_spot'
}

# ===============================
# Model Architecture
# ===============================
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# ===============================
# Initialize Flask App
# ===============================
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# ===============================
# Load Model
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = CNNModel(num_classes=6).to(device)
model_path = 'best_rice_model.pth'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define transformation
transforms_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# ===============================
# Helper Functions
# ===============================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(50, 50)):
    """Load and preprocess image"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(img).astype(np.float32)
        
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
        
        img_array = img_array.transpose(2, 0, 1)  # (C, H, W)
        img_tensor = transforms_val(torch.from_numpy(img_array))
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        
        return img_tensor
    except Exception as e:
        raise ValueError(f"Image preprocessing error: {str(e)}")

def predict_disease(image_path):
    """Make prediction on image"""
    img_tensor = preprocess_image(image_path).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, class_id = torch.max(probabilities, 1)
    
    class_id = class_id.item()
    confidence = confidence.item()
    disease_name = CLASS_MAPPING[class_id]
    
    all_probs = {CLASS_MAPPING[i]: round(float(probabilities[0][i]), 4) 
                for i in range(len(CLASS_MAPPING))}
    
    return {
        'class_id': class_id,
        'disease': disease_name,
        'confidence': round(confidence, 4),
        'all_probabilities': all_probs
    }

# ===============================
# API Routes
# ===============================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model': 'rice_leaf_disease',
        'device': str(device),
        'classes': CLASS_MAPPING
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict disease from uploaded image
    
    Expected: multipart/form-data with 'file' field
    Returns: JSON with prediction results
    """
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result = predict_disease(filepath)
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify({
            'success': True,
            'prediction': result
        }), 200
        
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/predict_url', methods=['POST'])
def predict_url():
    """
    Predict disease from image URL
    
    Expected: JSON with 'image_url' field
    Returns: JSON with prediction results
    """
    try:
        data = request.get_json()
        
        if not data or 'image_url' not in data:
            return jsonify({'error': 'No image_url provided'}), 400
        
        image_url = data['image_url']
        
        # Download image
        import urllib.request
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_image.jpg')
        urllib.request.urlretrieve(image_url, filepath)
        
        # Make prediction
        result = predict_disease(filepath)
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify({
            'success': True,
            'prediction': result
        }), 200
        
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/', methods=['GET'])
def index():
    """Serve the web UI."""
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html')

@app.route('/api/docs', methods=['GET'])
def docs():
    """API documentation."""
    return jsonify({
        'app': 'Rice Leaf Disease Detection API',
        'version': '1.0',
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
                    'bacterial_leaf_blight': 0.0001,
                    'brown_spot': 0.0002,
                    'healthy': 0.9876,
                    'leaf_blast': 0.0051,
                    'leaf_scald': 0.0068,
                    'narrow_brown_spot': 0.0002
                }
            }
        }
    }), 200

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': f'File too large. Max size: {MAX_FILE_SIZE} bytes'}), 413

# ===============================
# Run Flask App
# ===============================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Rice Leaf Disease Detection API")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Upload Folder: {UPLOAD_FOLDER}")
    print("="*60)
    print("Starting server on http://localhost:8000")
    print("Web UI: http://localhost:8000/")
    print("API Documentation: http://localhost:8000/api/docs")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=8000)
