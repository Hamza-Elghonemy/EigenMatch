package main

import (
    "log"
    "net/http"
    "github.com/gorilla/mux"
    "face-matching-app/src/go/handlers"
    "face-matching-app/src/go/services"
)

func main() {
    // Initialize the face service
    faceService := &services.FaceService{
        PythonServiceURL: "http://localhost:5000", // Update this URL as needed
    }
    
    // Initialize the handler
    faceHandler := &handlers.FaceHandler{
        FaceService: faceService,
    }
    
    r := mux.NewRouter()
    
    // Define routes
    r.HandleFunc("/match", faceHandler.MatchFace).Methods("POST")

    // Start the server
    log.Println("Starting server on :8080")
    if err := http.ListenAndServe(":8080", r); err != nil {
        log.Fatalf("Could not start server: %s\n", err)
    }
}