package handlers

import (
    "encoding/json"
    "net/http"
    "face-matching-app/src/go/services"
)

type FaceHandler struct {
    FaceService *services.FaceService
}

func (h *FaceHandler) MatchFace(w http.ResponseWriter, r *http.Request) {
    var inputImage services.InputImage
    if err := json.NewDecoder(r.Body).Decode(&inputImage); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    matchedImages, err := h.FaceService.MatchFace([]byte(inputImage.ImageData))
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(matchedImages)
}