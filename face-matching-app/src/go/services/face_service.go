package services

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	"net/http"
	"os/exec"
)

type FaceService struct {
	PythonServiceURL string
}

type MatchRequest struct {
	ImageData []byte `json:"image_data"`
}

type MatchResponse struct {
	Matches []string `json:"matches"`
}

type InputImage struct {
    ImageData string `json:"imageData"` // Base64 encoded image data
}

func (fs *FaceService) MatchFace(imageData []byte) ([]string, error) {
	requestBody, err := json.Marshal(MatchRequest{ImageData: imageData})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}

	resp, err := http.Post(fs.PythonServiceURL+"/match", "application/json", bytes.NewBuffer(requestBody))
	if err != nil {
		return nil, fmt.Errorf("failed to call Python service: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("received non-200 response: %s", resp.Status)
	}

	var matchResponse MatchResponse
	if err := json.NewDecoder(resp.Body).Decode(&matchResponse); err != nil {
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}

	return matchResponse.Matches, nil
}

func (fs *FaceService) SaveImage(imageData []byte, filename string) error {
	img, _, err := image.Decode(bytes.NewReader(imageData))
	if err != nil {
		return fmt.Errorf("failed to decode image: %v", err)
	}

	out, err := exec.Command("convert", "-", filename).StdinPipe()
	if err != nil {
		return fmt.Errorf("failed to create command: %v", err)
	}
	defer out.Close()

	if err := jpeg.Encode(out, img, nil); err != nil {
		return fmt.Errorf("failed to encode image: %v", err)
	}

	return nil
}