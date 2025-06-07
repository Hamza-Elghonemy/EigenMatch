from flask import Flask, request, jsonify
import base64
import numpy as np
from PIL import Image
import io
from face_matcher import FaceMatcher
from pca_processor import PCAProcessor
from image_preprocessor import ImagePreprocessor

app = Flask(__name__)

# Initialize components
image_preprocessor = ImagePreprocessor()
pca_processor = PCAProcessor(n_components=50)
face_matcher = FaceMatcher(pca_processor, dataset=[])  # Load your dataset here

@app.route('/match', methods=['POST'])
def match_face():
    try:
        data = request.get_json()
        image_data = base64.b64decode(data['image_data'])
        
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Process the image and find matches
        matches = face_matcher.match_face(image)
        
        return jsonify({'matches': matches})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)