package models

type FaceModel struct {
    ImagePath string `json:"image_path"`
    Metadata  map[string]interface{} `json:"metadata"`
}