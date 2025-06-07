import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cv2
import pickle
import json

class PCAProcessor:
    def __init__(self, n_components=0.95, image_size=(128, 128)):
        self.n_components = n_components
        self.image_size = image_size
        self.pca = None
        self.scaler = StandardScaler()
        self.dataset = []
        self.dataset_labels = []
        self.dataset_image_paths = []  # Store image paths
        self.dataset_pca_features = None  # Store transformed dataset

    def load_dataset(self, dataset_path):
        """Load images from dataset directory"""
        self.dataset = []
        self.dataset_labels = []
        self.dataset_image_paths = []
        
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path {dataset_path} does not exist")
        
        print(f"Loading dataset from: {dataset_path}")
        # Load images from subdirectories (each subdirectory represents a person)
        print(os.listdir(dataset_path))
        person_name = os.listdir(dataset_path)[0]
        person_dir = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_dir):
            person_count = 0
            for image_file in os.listdir(person_dir):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_path = os.path.join(person_dir, image_file)
                    try:
                        image = Image.open(image_path)
                        processed_image = self.preprocess_image(image)
                        self.dataset.append(processed_image.flatten())
                        self.dataset_labels.append(person_name)
                        self.dataset_image_paths.append(image_path)
                        person_count += 1
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
            print(f"Loaded {person_count} images for {person_name}")
        
        print(f"Total loaded: {len(self.dataset)} images from dataset")
        return len(self.dataset)

    def preprocess_image(self, image):
        """Preprocess image: resize, convert to grayscale, normalize"""
        # Convert PIL Image to numpy array if needed
        if hasattr(image, 'convert'):
            # Convert to RGB if it's a PIL image
            image = image.convert('RGB')
            image_array = np.array(image)
        else:
            image_array = image
        
        # Convert to grayscale using OpenCV
        if len(image_array.shape) == 3:
            gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image_array
        
        # Resize image
        resized_image = cv2.resize(gray_image, self.image_size)
        
        # Normalize pixel values to [0, 1]
        normalized_image = resized_image.astype(np.float32) / 255.0
        
        return normalized_image

    def fit(self, dataset_path=None):
        """Fit the PCA model to the dataset"""
        if dataset_path:
            self.load_dataset(dataset_path)
        
        if not self.dataset:
            raise ValueError("No dataset loaded. Please load dataset first.")
        
        print("Fitting PCA model...")
        # Convert to numpy array
        dataset_array = np.array(self.dataset)
        
        # Standardize the dataset
        scaled_data = self.scaler.fit_transform(dataset_array)
        
        # Fit PCA
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(scaled_data)
        
        # Transform the dataset and store for future matching
        self.dataset_pca_features = self.pca.transform(scaled_data)
        
        print(f"PCA fitted with {self.pca.n_components_} components")
        print(f"Explained variance ratio: {sum(self.pca.explained_variance_ratio_):.4f}")
        
        return self.pca

    def save_model(self, model_dir="models"):
        """Save the trained PCA model and dataset features"""
        if self.pca is None:
            raise ValueError("No model to save. Train the model first.")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save PCA model and scaler
        model_data = {
            'pca': self.pca,
            'scaler': self.scaler,
            'image_size': self.image_size,
            'n_components': self.n_components
        }
        
        with open(os.path.join(model_dir, 'pca_model.pkl'), 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save dataset features, labels, and image paths
        dataset_data = {
            'dataset_pca_features': self.dataset_pca_features,
            'dataset_labels': self.dataset_labels,
            'dataset_image_paths': self.dataset_image_paths,
            'total_images': len(self.dataset_labels)
        }
        
        with open(os.path.join(model_dir, 'dataset_features.pkl'), 'wb') as f:
            pickle.dump(dataset_data, f)
        
        # Save metadata as JSON for easy inspection
        metadata = {
            'total_images': len(self.dataset_labels),
            'unique_persons': len(set(self.dataset_labels)),
            'image_size': self.image_size,
            'pca_components': self.pca.n_components_,
            'explained_variance_ratio': float(sum(self.pca.explained_variance_ratio_))
        }
        
        with open(os.path.join(model_dir, 'model_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {model_dir}/")
        return model_dir

    def load_model(self, model_dir="models"):
        """Load the trained PCA model and dataset features"""
        pca_model_path = os.path.join(model_dir, 'pca_model.pkl')
        dataset_features_path = os.path.join(model_dir, 'dataset_features.pkl')
        
        if not os.path.exists(pca_model_path) or not os.path.exists(dataset_features_path):
            raise ValueError(f"Model files not found in {model_dir}/")
        
        # Load PCA model and scaler
        with open(pca_model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.pca = model_data['pca']
        self.scaler = model_data['scaler']
        self.image_size = model_data['image_size']
        self.n_components = model_data['n_components']
        
        # Load dataset features, labels, and image paths
        with open(dataset_features_path, 'rb') as f:
            dataset_data = pickle.load(f)
        
        self.dataset_pca_features = dataset_data['dataset_pca_features']
        self.dataset_labels = dataset_data['dataset_labels']
        self.dataset_image_paths = dataset_data.get('dataset_image_paths', [])
        
        print(f"Model loaded from {model_dir}/")
        print(f"Loaded {len(self.dataset_labels)} images with {self.pca.n_components_} PCA components")
        return True

    def transform(self, image):
        """Transform the image into PCA space"""
        if self.pca is None:
            raise ValueError("PCA model not fitted. Call fit() or load_model() first.")
        
        processed_image = self.preprocess_image(image)
        # Reshape to match dataset format (flatten)
        flattened_image = processed_image.flatten().reshape(1, -1)
        
        # Standardize using the same scaler
        scaled_image = self.scaler.transform(flattened_image)
        
        return self.pca.transform(scaled_image)

    def transform_dataset(self):
        """Transform the entire dataset into PCA space"""
        if self.pca is None:
            raise ValueError("PCA model not fitted. Call fit() first.")
        
        dataset_array = np.array(self.dataset)
        scaled_data = self.scaler.transform(dataset_array)
        return self.pca.transform(scaled_data)

    def inverse_transform(self, pca_data):
        """Inverse transform PCA data back to original space"""
        if self.pca is None:
            raise ValueError("PCA model not fitted. Call fit() first.")
        
        # Inverse transform
        scaled_data = self.pca.inverse_transform(pca_data)
        original_data = self.scaler.inverse_transform(scaled_data)
        
        # Reshape back to image format
        return original_data.reshape(-1, *self.image_size)

    def get_dataset_info(self):
        """Get information about the loaded dataset"""
        return {
            'total_images': len(self.dataset_labels) if self.dataset_labels else 0,
            'unique_persons': len(set(self.dataset_labels)) if self.dataset_labels else 0,
            'image_size': self.image_size,
            'feature_dimension': len(self.dataset[0]) if self.dataset else 0,
            'pca_components': self.pca.n_components_ if self.pca else 0
        }

    def get_dataset_features(self):
        """Get the PCA features of the dataset for matching"""
        if self.dataset_pca_features is None:
            raise ValueError("Dataset features not available. Train or load model first.")
        return self.dataset_pca_features, self.dataset_labels, self.dataset_image_paths
