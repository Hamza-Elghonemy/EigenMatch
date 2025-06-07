FROM python:3.9-slim AS python-base

# Set the working directory
WORKDIR /app

# Copy Python requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python source code
COPY src/python ./src/python

# Set up the Go environment
FROM golang:1.17 AS go-base

# Set the working directory
WORKDIR /app

# Copy Go module files and download dependencies
COPY go.mod go.sum ./
RUN go mod download

# Copy the Go source code
COPY src/go ./src/go

# Build the Go application
RUN go build -o face-matching-app ./src/go/main.go

# Final stage
FROM python-base

# Copy the built Go application from the Go stage
COPY --from=go-base /app/face-matching-app /usr/local/bin/face-matching-app

# Copy the shared configuration
COPY src/shared/config.yaml ./src/shared/config.yaml

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["face-matching-app"]