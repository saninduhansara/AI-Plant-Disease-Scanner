from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import os
import urllib.request

# ===============================
# Environment & Server Config
# ===============================
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp', 'gif'}
MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB
IMG_SIZE = (260, 260)             # EfficientNetV2-B0 Standard Input Resolution

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===============================
# Class Mapping & Agricultural Insights
# ===============================
category_dict = {
    0: 'Bacterial leaf blight',
    1: 'Brown spot',
    2: 'Leaf smut',
    3: 'healthy'
}

DISEASE_METADATA = {
    'Bacterial leaf blight': {
        'pathogen': 'Xanthomonas oryzae pv. oryzae (Bacterial pathogen)',
        'severity': 'High',
        'badge_color': '#e74c3c',
        'symptoms': 'Pale green to grayish-yellow water-soaked lesions expanding into wavy marginal streaks along leaf blades, leading to leaf blighting and canopy desiccation.',
        'treatment': [
            'Drain field for 3-4 days to arrest bacterial proliferation in standing water.',
            'Apply Copper Oxychloride (2.5 g/L) or Streptocycline/Validamycin as locally recommended.',
            'Balance nitrogen applications; avoid excessive top-dressing during tillering.'
        ],
        'prevention': [
            'Use resistant rice varieties (e.g., IRBB lines, PR106).',
            'Avoid clipping seedling tips during transplanting.',
            'Ensure adequate potassium fertilization to bolster cell wall resilience.'
        ]
    },
    'Brown spot': {
        'pathogen': 'Bipolaris oryzae / Cochliobolus miyabeanus (Fungal pathogen)',
        'severity': 'Moderate to High',
        'badge_color': '#d35400',
        'symptoms': 'Round to oval brown spots with prominent grayish-white necrotic centers and conspicuous yellow chlorotic halos across leaf blades and glumes.',
        'treatment': [
            'Foliar spray of Mancozeb (2 g/L) or Tricyclazole / Propiconazole at early tillering.',
            'Apply potassium and silicon fertilizers to reverse physiological predisposition.',
            'Maintain continuous thin-layer irrigation to avoid dry-induced moisture stress.'
        ],
        'prevention': [
            'Seed treatment with Carbendazim or Thiram (2 g/kg seed) before germination.',
            'Soil nutrient enrichment through organic compost and balanced NPK ratio.',
            'Clear infected crop residue and volunteer hosts post-harvest.'
        ]
    },
    'Leaf smut': {
        'pathogen': 'Entyloma oryzae (Fungal pathogen)',
        'severity': 'Moderate',
        'badge_color': '#8e44ad',
        'symptoms': 'Slightly raised, angular, punctate jet-black spots (sori) scattered across the leaf blade. Associated leaves often exhibit chlorotic yellowing and premature senescence.',
        'treatment': [
            'Apply broad-spectrum triazole fungicides (Propiconazole, Hexaconazole) if canopy coverage exceeds 15%.',
            'Avoid excessive nitrogen top-dressing which exacerbates sori erupting.',
            'Ensure proper plant spacing for improved sunlight penetration and canopy aeration.'
        ],
        'prevention': [
            'Burn or incorporate stubble immediately after harvest to terminate teliospore overwintering.',
            'Rotate paddies with non-cereal crops where feasible.',
            'Select certified disease-free seeds with clean pedigree.'
        ]
    },
    'healthy': {
        'pathogen': 'None (Optimal Plant Health)',
        'severity': 'None',
        'badge_color': '#27ae60',
        'symptoms': 'Vibrant, uniform green foliage without necrotic margin lesions, fungal halo pustules, or black sori ruptures.',
        'treatment': [
            'No chemical intervention required.',
            'Maintain customary agronomic fertilization and field scouting.'
        ],
        'prevention': [
            'Continue integrated pest management (IPM) practices.',
            'Conduct routine leaf color chart (LCC) monitoring for nitrogen calibration.'
        ]
    }
}

# ===============================
# TFLite Model Loading (Universal Runtime)
# ===============================
def load_tflite_interpreter():
    candidate_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "paddy_disease_model_quantized.tflite"),
        "paddy_disease_model_quantized.tflite",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts_paddy_disease", "paddy_disease_model_quantized.tflite"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "paddy_disease_model.tflite"),
        "paddy_disease_model.tflite"
    ]

    resolved_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            resolved_path = path
            break

    if not resolved_path:
        # Fallback to any existing .tflite file in workspace
        for f in os.listdir(os.path.dirname(os.path.abspath(__file__))):
            if f.endswith('.tflite'):
                resolved_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f)
                break

    if not resolved_path:
        raise FileNotFoundError("Could not find paddy_disease_model_quantized.tflite in workspace.")

    print(f"[*] Initializing TFLite Interpreter with model: {resolved_path}")

    try:
        import tflite_runtime.interpreter as tflite_rt
        interpreter = tflite_rt.Interpreter(model_path=resolved_path)
    except ImportError:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=resolved_path)

    interpreter.allocate_tensors()
    return interpreter, resolved_path

interpreter, active_model_path = load_tflite_interpreter()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Warmup run
warmup_tensor = np.zeros(input_details[0]['shape'], dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], warmup_tensor)
interpreter.invoke()
print(f"[*] Model warmed up successfully. Input shape: {input_details[0]['shape']}")

# ===============================
# Flask App Setup
# ===============================
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


# ===============================
# Calibrated Sori Biomarker & Multi-Patch Engine
# (Engineered directly from Paddy_Disease_Detection_ADA_Server.md)
# ===============================
def detect_sori_density(pil_img):
    """
    Biological biomarker for Leaf Smut (Entyloma oryzae).
    Measures sub-millimeter punctate jet-black sori on leaf tissue using
    morphological Black Top-Hat filtering on leaf-masked pixels.
    """
    arr_rgb = np.array(pil_img.convert('RGB'), dtype=np.float32)
    r, g, b = arr_rgb[..., 0], arr_rgb[..., 1], arr_rgb[..., 2]

    # Leaf tissue segmentation mask (green and chlorotic yellow leaves):
    is_green_leaf = (g > r - 15.0) & (g > b + 10.0) & (g > 40.0)
    is_yellow_leaf = (r > 70.0) & (g > 65.0) & (r + g > 2.1 * b) & (b < 120.0)
    is_leaf_tissue = (is_green_leaf | is_yellow_leaf) & (r + g + b < 680.0) & (r + g + b > 100.0)

    leaf_count = np.sum(is_leaf_tissue)
    if leaf_count < 1500:
        return 0.0

    # Morphological closing: MaxFilter(5) then MinFilter(5)
    gray = pil_img.convert('L')
    closed = gray.filter(ImageFilter.MaxFilter(size=5)).filter(ImageFilter.MinFilter(size=5))
    arr_gray = np.array(gray, dtype=np.float32)
    arr_closed = np.array(closed, dtype=np.float32)
    top_hat = np.maximum(0.0, arr_closed - arr_gray)

    # Genuine punctate black sori on leaf tissue
    is_sori = is_leaf_tissue & (top_hat > 28.0) & (arr_gray < 75.0)
    return float(np.sum(is_sori)) / float(leaf_count)


def extract_diagnostic_patches(pil_img):
    """
    Multi-Patch Lesion Saliency: Aspect-ratio aware blade isolation,
    lesion zoom, sori-sharpening, and chlorosis-desaturated texture view.
    """
    w, h = pil_img.size
    min_dim = min(w, h)

    # 1. Full center crop
    c_left = (w - min_dim) // 2
    c_top = (h - min_dim) // 2
    crop_center = pil_img.crop((c_left, c_top, c_left + min_dim, c_top + min_dim))

    # 2 & 3. Aspect-ratio aware blade isolation
    if h > w:
        # Portrait (isolates lower leaf blade from upper rice grains/panicles)
        low_y = min(int(h * 0.35), max(0, h - min_dim))
        crop_blade_a = pil_img.crop((c_left, low_y, c_left + min_dim, min(h, low_y + min_dim)))
        crop_blade_b = pil_img.crop((c_left, 0, c_left + min_dim, min_dim))
    else:
        # Landscape (isolates individual leaf blades without cutting through center seams)
        crop_blade_a = pil_img.crop((0, 0, min_dim, min_dim))
        crop_blade_b = pil_img.crop((w - min_dim, 0, w, min_dim))

    # 4. Central 55% lesion zoom
    focus_size = int(min_dim * 0.55)
    f_left = (w - focus_size) // 2
    f_top = (h - focus_size) // 2
    crop_zoom = pil_img.crop((f_left, f_top, f_left + focus_size, f_top + focus_size))

    # 5. Sori-enhanced unsharp mask crop
    crop_sori = crop_zoom.filter(ImageFilter.UnsharpMask(radius=1.5, percent=160, threshold=2))

    # 6. Desaturated texture view (focuses on lesion morphology vs background hue)
    crop_texture = ImageEnhance.Color(crop_zoom).enhance(0.20)

    patches = {
        'center': crop_center.resize(IMG_SIZE, Image.BICUBIC),
        'blade_primary': crop_blade_a.resize(IMG_SIZE, Image.BICUBIC),
        'blade_secondary': crop_blade_b.resize(IMG_SIZE, Image.BICUBIC),
        'zoom': crop_zoom.resize(IMG_SIZE, Image.BICUBIC),
        'sori_enhanced': crop_sori.resize(IMG_SIZE, Image.BICUBIC),
        'texture_focus': crop_texture.resize(IMG_SIZE, Image.BICUBIC)
    }
    return patches


def run_single_inference(patch_pil):
    """
    Feeds RGB patch [0, 255] float32 into TFLite model.
    Note: EfficientNetV2 includes Rescaling & Normalization internally.
    """
    arr = np.array(patch_pil, dtype=np.float32)
    inp_tensor = np.expand_dims(arr, axis=0)

    interpreter.set_tensor(input_details[0]['index'], inp_tensor)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]['index'])[0]
    return probs.astype(np.float32)


def predict_pipeline(pil_image):
    """
    Full diagnostic inference pipeline with Multi-Patch Saliency
    and Calibrated Biological Sori Biomarker Fusion.
    """
    sori_density = detect_sori_density(pil_image)
    crops_dict = extract_diagnostic_patches(pil_image)

    crop_keys = list(crops_dict.keys())
    all_probs = [run_single_inference(crops_dict[k]) for k in crop_keys]

    confidences = np.array([np.max(p) for p in all_probs])

    # Sharpened temperature scaling (T=0.18)
    exp_w = np.exp((confidences - np.max(confidences)) / 0.18)
    weights = exp_w / np.sum(exp_w)
    probs = np.sum([w * p for w, p in zip(weights, all_probs)], axis=0)

    winning_idx = int(np.argmax(weights))
    winning_crop_name = crop_keys[winning_idx]
    winning_crop_weight = float(weights[winning_idx] * 100)

    # Calibrated Biological Sori Biomarker Fusion:
    # Punctate black sori (>3.5% density on leaf tissue) are the biological hallmark of Leaf Smut.
    # Watermarks and blight streaks produce 0.0% sori_density.
    if sori_density > 0.035:
        smut_boost = min(0.60, (sori_density - 0.035) * 12.0 + 0.25)
        probs[2] += smut_boost
        probs[0] = max(0.0, probs[0] - smut_boost * 0.60)
        probs[1] = max(0.0, probs[1] - smut_boost * 0.40)
        probs = probs / np.sum(probs)

    pred_index = int(np.argmax(probs))
    pred_class = category_dict[pred_index]
    confidence_val = float(probs[pred_index])

    all_probabilities = {
        category_dict[i]: float(probs[i])
        for i in range(len(category_dict))
    }

    metadata = DISEASE_METADATA.get(pred_class, DISEASE_METADATA['healthy'])

    return {
        "class_id": pred_index,
        "disease": pred_class,
        "confidence": confidence_val,
        "confidence_formatted": f"{confidence_val * 100:.2f}%",
        "sori_density": round(float(sori_density * 100), 2),
        "winning_crop_name": winning_crop_name,
        "winning_crop_weight": f"{winning_crop_weight:.1f}%",
        "all_probabilities": all_probabilities,
        "details": metadata
    }


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ===============================
# API Endpoints
# ===============================
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model": "paddy_disease_model_quantized.tflite",
        "architecture": "EfficientNetV2-B0 + Residual Spatial Attention + Dual Pooling",
        "input_resolution": f"{IMG_SIZE[0]}x{IMG_SIZE[1]}",
        "classes": category_dict
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No image file provided in upload request"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No image file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": f"Invalid file type. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

        image_bytes = file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        prediction = predict_pipeline(pil_img)

        return jsonify({
            "success": True,
            "prediction": prediction
        })

    except Exception as e:
        app.logger.exception("Prediction failure encountered")
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict_url', methods=['POST'])
def predict_url():
    try:
        data = request.get_json()
        if not data or 'image_url' not in data:
            return jsonify({"error": "Missing 'image_url' parameter in JSON payload"}), 400

        url = data.get("image_url")
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=12) as response:
            image_bytes = response.read()

        pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        prediction = predict_pipeline(pil_img)

        return jsonify({
            "success": True,
            "prediction": prediction
        })

    except Exception as e:
        app.logger.exception("URL prediction failure encountered")
        return jsonify({"error": str(e)}), 500


@app.route('/api/docs')
def docs():
    return jsonify({
        "app": "Paddy Leaf Disease Detection API (v6)",
        "model_file": os.path.basename(active_model_path),
        "input_size": f"{IMG_SIZE[0]}x{IMG_SIZE[1]}",
        "color_space": "RGB",
        "saliency_pipeline": "Multi-Patch 6-View Ensemble + Biological Sori Top-Hat",
        "classes": category_dict
    })


@app.route('/')
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)