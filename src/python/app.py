from flask import Flask, request, jsonify
import base64
import numpy as np
from PIL import Image
import io
import os
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
        model_dir = os.path.join(os.path.dirname(__file__),"models")
        
        if os.path.exists(os.path.join(model_dir, 'pca_model.pkl')):
            print("Loading saved PCA model...")
            pca_processor.load_model(model_dir)
            print("Model loaded successfully!")
        else:
            print("No saved model found. Please train the model first using train_model.py")
            return False
        
        # Initialize face matcher
        face_matcher = FaceMatcher(pca_processor, similarity_metric='cosine')
        
        return True
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return False

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
        match_type = request.args.get('match_type', 'all')  # 'all' or 'person'
        
        # Find matches
        if match_type == 'person':
            matches = face_matcher.get_person_matches(image, top_k=top_k)
        else:
            matches = face_matcher.match_face(image, top_k=top_k)
        
        return jsonify({
            'matches': matches,
            'total_matches': len(matches),
            'match_type': match_type
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
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Failed to initialize model. Please check the error messages above.")