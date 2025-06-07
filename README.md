# Face Matching Application

This project implements a face matching application using both Python and Go, leveraging Principal Component Analysis (PCA) to match an input user face image with a dataset of images. The application outputs the images that are closest matches to the input face.

## Project Structure

```
face-matching-app
├── src
│   ├── python
│   │   ├── face_matcher.py       # Contains the FaceMatcher class for matching faces
│   │   ├── pca_processor.py       # Defines the PCAProcessor class for PCA operations
│   │   ├── image_preprocessor.py   # Exports the ImagePreprocessor class for image loading and preprocessing
│   │   └── utils
│   │       └── __init__.py        # Initializer for the utils package
│   ├── go
│   │   ├── main.go                # Entry point for the Go application
│   │   ├── handlers
│   │   │   └── face_handler.go     # Handles HTTP requests related to face matching
│   │   ├── models
│   │   │   └── face_model.go       # Defines the FaceModel struct for face image data
│   │   └── services
│   │       └── face_service.go     # Contains business logic for processing face matching requests
│   └── shared
│       └── config.yaml            # Configuration settings for the application
├── data
│   ├── dataset
│   │   └── .gitkeep               # Keeps the dataset directory in version control
│   └── uploads
│       └── .gitkeep               # Keeps the uploads directory in version control
├── api
│   └── openapi.yaml               # API specification using OpenAPI
├── requirements.txt               # Python dependencies required for the project
├── go.mod                         # Go module definition
├── go.sum                         # Checksums for the dependencies
├── Dockerfile                     # Instructions for building a Docker image
├── docker-compose.yml             # Defines services for running the application with Docker Compose
└── README.md                      # Documentation for the project
```

## Setup Instructions

### Prerequisites

- Python 3.x
- Go 1.x
- Docker (optional, for containerization)

### Python Setup

1. Navigate to the `src/python` directory.
2. Install the required Python packages:

   ```
   pip install -r requirements.txt
   ```

### Go Setup

1. Navigate to the `src/go` directory.
2. Initialize the Go module:

   ```
   go mod tidy
   ```

### Running the Application

1. Start the Go server:

   ```
   go run main.go
   ```

2. The server will be running and ready to handle face matching requests.

## Usage

- Send a POST request to the Go server with an image file to receive the closest matching images from the dataset.
- The application will process the image using PCA and return the results.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.