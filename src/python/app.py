from flask import Flask, request, jsonify, render_template, send_file
import base64
import numpy as np
from PIL import Image
import io
import os
import mimetypes
from face_matcher import FaceMatcher
from pca_processor import PCAProcessor

app = Flask(__name__)

# Global variables for model components
pca_processor = None
face_matcher = None

def initialize_model():
    """Initialize the PCA processor and face matcher with saved model"""
    global pca_processor, face_matcher
    
    try:
        # Initialize PCA processor
        pca_processor = PCAProcessor(n_components=50, image_size=(128, 128))
        
        # Try to load saved model
        model_dir = os.path.join(os.path.dirname(__file__), "models", "simpsons_faces_pca")
        
        if os.path.exists(os.path.join(model_dir, 'pca_model.pkl')):
            print("Loading saved PCA model...")
            pca_processor.load_model(model_dir)
            print("Model loaded successfully!")
        else:
            print("No saved model found. Please train the model first using train_PCA.py")
            return False
        
        # Initialize face matcher
        face_matcher = FaceMatcher(pca_processor, similarity_metric='cosine')
        
        return True
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/')
def index():
    """Serve the web interface"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if pca_processor is None or face_matcher is None:
        return jsonify({'status': 'error', 'message': 'Model not initialized'}), 500
    
    info = pca_processor.get_dataset_info()
    return jsonify({
        'status': 'healthy',
        'model_info': info
    })

@app.route('/image/<int:image_index>')
def serve_image(image_index):
    """Serve dataset images by index"""
    try:
        if pca_processor is None:
            return "Model not initialized", 500
        
        # Get image paths
        _, _, image_paths = pca_processor.get_dataset_features()
        
        if image_index < 0 or image_index >= len(image_paths):
            return "Image not found", 404
        
        image_path = image_paths[image_index]
        
        if not os.path.exists(image_path):
            return "Image file not found", 404
        
        # Determine mime type
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            mime_type = 'image/jpeg'
        
        return send_file(image_path, mimetype=mime_type)
        
    except Exception as e:
        print(f"Error serving image: {e}")
        return "Error serving image", 500

@app.route('/match', methods=['POST'])
def match_face():
    """Face matching endpoint"""
    try:
        if pca_processor is None or face_matcher is None:
            return jsonify({'error': 'Model not initialized'}), 500
        
        # Handle different input formats
        if request.is_json:
            data = request.get_json()
            if 'image_data' in data:
                # Base64 encoded image
                image_data = base64.b64decode(data['image_data'])
                image = Image.open(io.BytesIO(image_data))
            else:
                return jsonify({'error': 'No image_data in JSON'}), 400
        else:
            # Handle file upload
            if 'image' not in request.files:
                return jsonify({'error': 'No image file provided'}), 400
            
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No image file selected'}), 400
            
            image = Image.open(file.stream)
        
        # Get number of matches requested
        top_k = request.args.get('top_k', 5, type=int)
        match_type = request.args.get('match_type', 'person')  # 'all', 'person', or 'diverse'
        max_per_person = request.args.get('max_per_person', 2, type=int)
        
        # Find matches based on type
        if match_type == 'person':
            matches = face_matcher.get_person_matches(image, top_k=top_k)
        elif match_type == 'diverse':
            matches = face_matcher.get_diverse_matches(image, top_k=top_k, max_per_person=max_per_person)
        else:  # 'all'
            matches = face_matcher.match_face(image, top_k=top_k)
        
        # Add image URLs to matches
        for match in matches:
            if 'index' in match:
                match['image_url'] = f"/image/{match['index']}"
        
        return jsonify({
            'matches': matches,
            'total_matches': len(matches),
            'match_type': match_type,
            'unique_persons': len(set(match['person'] for match in matches))
        })
    
    except Exception as e:
        print(f"Error in face matching: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Face Matching Service...")
    
    if initialize_model():
        print("Model initialized successfully!")
        print("Web interface available at: http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Failed to initialize model. Please check the error messages above.")