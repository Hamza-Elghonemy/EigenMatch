# Face Matching Application

This project implements a face matching application using Python for the core machine learning (PCA-based face matching) and a Go service that can act as an API gateway or a standalone service interacting with the Python backend. The application matches an input user face image with a dataset of images and outputs the closest matches. It features a web interface, a command-line tool, and a desktop GUI for interacting with the Python matching service.

## Project Structure

```
EigenMatch/
├── src/
│   ├── python/
│   │   ├── app.py                # Flask web server for the Python ML service & web UI
│   │   ├── face_matcher.py       # Contains the FaceMatcher class
│   │   ├── pca_processor.py      # Defines the PCAProcessor class for PCA operations
│   │   ├── train_PCA.py          # Script to train new PCA models
│   │   ├── cli_tool.py           # Command-line interface for face matching
│   │   ├── gui_app.py            # Desktop GUI application for face matching
│   │   ├── templates/
│   │   │   └── index.html        # HTML template for the web interface
│   │   └── models/               # Directory for storing trained PCA models
│   │       └── (model_name)/     # Subdirectory for each specific trained model
│   │           ├── pca_model.pkl
│   │           ├── dataset_features.pkl
│   │           └── model_metadata.json
│   ├── go/
│   │   ├── main.go               # Entry point for the Go application
│   │   ├── handlers/
│   │   │   └── face_handler.go   # Handles HTTP requests
│   │   └── services/
│   │       └── face_service.go   # Business logic for Go service
│   └── shared/
│       └── config.yaml           # Shared configuration (if used by Go service)
├── data/
│   ├── dataset/
│   │   └── dataset.py          # Script to download datasets (e.g., from Kaggle)
│   │   └── .gitkeep
│   └── uploads/                # For temporary user uploads (if implemented)
│       └── .gitkeep
├── requirements.txt              # Python dependencies
├── go.mod                        # Go module definition
├── go.sum                        # Go module checksums
├── Dockerfile                    # For building a combined Go app + Python environment (optional)
├── docker-compose.yml            # For running Python and Go services together
└── README.md                     # This file
```

## Features

* **PCA-based Face Matching**: Uses Principal Component Analysis for feature extraction and matching.
* **Multiple Interfaces**:
  * Web UI (via Flask in `app.py`)
  * Command-Line Tool (`cli_tool.py`)
  * Desktop GUI (`gui_app.py`)
* **Configurable Model Training**: Train PCA models on different datasets with various parameters.
* **Flexible Dataset Loading**: Designed to work with datasets where images are organized into subdirectories by class/person.
* **Docker Support**: Includes `Dockerfile` and `docker-compose.yml` for containerized deployment.
* **Go API Service**: A Go service that can interact with the Python backend.

## Setup Instructions

### Prerequisites

* Python 3.8+
* Go 1.17+
* Docker and Docker Compose (optional, for containerization)
* Kaggle API token configured (if using `data/dataset/dataset.py` with private Kaggle datasets)

### 1. Clone the Repository

```bash
git clone <repository_url>
cd EigenMatch
```

### 2. Python Setup

Navigate to the Python source directory:

```bash
cd src/python
```

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install Python dependencies:

```bash
pip install -r ../../requirements.txt # requirements.txt is in the root
```

### 3. Go Setup (Optional - if running Go service separately)

Navigate to the Go source directory:

```bash
cd src/go
```

Download Go dependencies:

```bash
go mod tidy
```

## Usage

### 1. Download a Dataset

You can use the provided script to download datasets from Kaggle.
Navigate to `data/dataset/`:

```bash
cd ../../data/dataset # from src/python or src/go
```

Run the script with the Kaggle dataset identifier:

```bash
python dataset.py "owner/dataset-slug"
# Example: python dataset.py "kostastokis/simpsons-faces"
```

This will download the dataset, typically into your `~/.cache/kagglehub/datasets/` directory. Note the path where it's downloaded.

### 2. Train a PCA Model

Navigate to the Python source directory (`src/python`).
Run the `train_PCA.py` script, providing the path to your downloaded dataset and a name for your model.

```bash
# Example:
python train_PCA.py "C:\Users\YourUser\.cache\kagglehub\datasets\kostastokis\simpsons-faces\versions\1\simpsons_dataset" "simpsons_model_v1" --n_components 0.98 --image_width 100 --image_height 100
```

* Replace the dataset path with the actual path from step 1.
* `simpsons_model_v1` is the name your model will be saved under in `src/python/models/`.
* Adjust `--n_components`, `--image_width`, `--image_height` as needed.

### 3. Running the Python Face Matching Service (Flask App)

The Python Flask application serves the web UI and the matching API.

Navigate to `src/python`:

```bash
cd src/python # If not already there
```

Set the `ACTIVE_MODEL_NAME` environment variable to the model you trained (e.g., `simpsons_model_v1` from the training step). If not set, it defaults to `simpsons_faces_pca`.

**Linux/macOS:**

```bash
export ACTIVE_MODEL_NAME="simpsons_model_v1"
python app.py
```

**Windows (PowerShell):**

```powershell
$env:ACTIVE_MODEL_NAME="simpsons_model_v1"
python app.py
```

**Windows (CMD):**

```cmd
set ACTIVE_MODEL_NAME=simpsons_model_v1
python app.py
```

The service will start, typically on `http://localhost:5000`. You can access the web UI here.

### 4. Using the Interfaces

* **Web UI**: Open `http://localhost:5000` in your browser.
* **Command-Line Tool (`cli_tool.py`)**:

    ```bash
    # Ensure you are in src/python and your venv is active
    python cli_tool.py path/to/your/image.jpg --url http://localhost:5000 --top-k 5
    # For interactive mode:
    python cli_tool.py --interactive --url http://localhost:5000
    ```

* **Desktop GUI (`gui_app.py`)**:

    ```bash
    # Ensure you are in src/python and your venv is active
    python gui_app.py
    ```

    The GUI will attempt to connect to the API at `http://localhost:5000`.

### 5. Running with Docker Compose (Python + Go)

Docker Compose can be used to run both the Python Flask service and the Go service. The Go service is configured to communicate with the Python service.

From the project root directory (`EigenMatch/`):

```bash
docker-compose up --build
```

* The Python service will be available internally to Docker at `http://python-service:5000`.
* The Go service will be exposed on `http://localhost:8080`.
* **Note**: For Docker Compose to use a specific trained model, you might need to:
    1. Ensure the model is present in `src/python/models/`.
    2. Modify `docker-compose.yml` or the Python `app.py` to correctly set/use `ACTIVE_MODEL_NAME` within the container. One way is to add it to the `environment` section of `python-service` in `docker-compose.yml`.

    Example `docker-compose.yml` modification:

    ```yaml
    services:
      python-service:
        build:
          context: ./src/python
          dockerfile: Dockerfile # Assuming Dockerfile is in src/python
        volumes:
          - ./src/python/models:/app/models # Mount models directory
          # - ./data/uploads:/app/data/uploads # If you have local uploads
          # - ./path_to_your_datasets_on_host:/app/data/dataset # If training inside container
        environment:
          - PYTHONUNBUFFERED=1
          - ACTIVE_MODEL_NAME=your_trained_model_name # Set your model here
    # ... rest of the file
    ```

### 6. Running the Go Service (Standalone)

If you want to run the Go service separately (assuming the Python service is already running and accessible):

Navigate to `src/go`:

```bash
cd src/go
```

Update `PythonServiceURL` in `main.go` if your Python service is not at `http://localhost:5000`.
Run the Go application:

```bash
go run main.go
```

The Go server will start, typically on port `8080`.

## Development Notes

* **Dataset Structure**: The `PCAProcessor` expects datasets to be structured with subdirectories for each class/person (e.g., `dataset_root/person_A/img1.jpg`, `dataset_root/person_B/img1.jpg`).
* **Model Persistence**: Trained models are saved in `src/python/models/<model_name>/`. Each model includes the PCA object, scaler, dataset features, and metadata.
* **Similarity Metric**: The `FaceMatcher` currently uses 'cosine' similarity. This can be configured.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License.
