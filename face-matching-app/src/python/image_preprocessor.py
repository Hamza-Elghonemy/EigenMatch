class ImagePreprocessor:
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size

    def load_image(self, image_path):
        from PIL import Image
        image = Image.open(image_path)
        return image

    def preprocess_image(self, image):
        image = image.resize(self.image_size)
        image = image.convert('RGB')
        return image