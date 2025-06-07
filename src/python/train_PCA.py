"""
Script to train and save the PCA model
Run this once to train the model, then use the saved model for inference
"""

from pca_processor import PCAProcessor
import os

def main():
    print("Starting PCA model training...")
    
    # Initialize PCA processor
    pca_processor = PCAProcessor(n_components=50, image_size=(128, 128))
    
    # Dataset path - update this to your actual dataset path
    dataset_path = r"C:\Users\hamza\.cache\kagglehub\datasets\ashwingupta3012\human-faces\versions\1"
    
    try:
        # Check if dataset exists
        if not os.path.exists(dataset_path):
            print(f"Dataset path does not exist: {dataset_path}")
            print("Please update the dataset_path variable in this script")
            return
        
        # Train the model
        print("Loading dataset and training PCA model...")
        pca_processor.fit(dataset_path)
        
        # Save the trained model
        model_dir = os.path.join(os.path.dirname(__file__),"models")
        saved_path = pca_processor.save_model(model_dir)
        
        # Display model info
        info = pca_processor.get_dataset_info()
        print("\nModel Training Complete!")
        print("=" * 40)
        for key, value in info.items():
            print(f"{key}: {value}")
        
        print(f"\nModel saved to: {saved_path}")
        print("You can now use the saved model for face matching!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()