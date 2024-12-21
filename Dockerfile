# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV and other dependencies
RUN apt-get update && apt-get install -y \
    libomp-dev \
    libhdf5-serial-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libcurl4-openssl-dev

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Copy only the requirements file to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Set environment variables for Flask app
ENV FLASK_ENV=production
ENV FLASK_APP=main.py

# Expose port 8080 for the Flask app
EXPOSE 8080

# Command to run the Flask application
CMD ["python", "main.py"]
