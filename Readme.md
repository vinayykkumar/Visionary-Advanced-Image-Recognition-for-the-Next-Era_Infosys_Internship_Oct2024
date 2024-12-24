Visionary: Advanced Image Recognition for the Next Era
Project Description
Visionary is an advanced image recognition project designed to classify images into four categories: Akshay Kumar, Amitabh Bachchan, Prabhas, and Vijay. The project utilizes TensorFlow's MobileNetV2 for fine-tuned image classification and Streamlit for an interactive web interface.

Features
Deep Learning: Powered by MobileNetV2 for efficient and accurate image recognition.
Streamlit Interface: User-friendly web application for predictions.
Data Augmentation: Enhanced training with robust data preprocessing techniques.
Docker Support: Containerized deployment for consistent and portable execution.
Folder Structure
bash
Copy code
Visionary/
│
├── src/
│   ├── app.py          # Main Streamlit application
│   ├── components/     # Reusable components (e.g., class_reference.py)
│   ├── models/         # Trained model files
│   ├── Processed/      # Training and validation datasets
│
├── My dataset/         # Raw dataset (organized by class)
├── Dockerfile          # Docker configuration
├── config.yaml         # Configuration file for project settings
└── README.md           # Project documentation
Requirements
To install the required dependencies, run:

bash
Copy code
pip install -r requirements.txt
Usage
Train the Model
To train the model on your dataset:

bash
Copy code
python train.py
Run the Application
To start the Streamlit application:

bash
Copy code
streamlit run src/app.py
Docker Deployment
Build and run the project using Docker:

bash
Copy code
docker build -t visionary .
docker run -p 8501:8501 visionary
Configuration
All configurable parameters (e.g., model path, dataset paths) are stored in config.yaml. Update the file as needed to match your environment.

Predictions
To classify an image, upload it through the Streamlit application or use the following script:

python
Copy code
from predict import predict_image

image_path = "path_to_your_image.jpg"
result = predict_image(image_path)
print("Predicted Class:", result)
Contributors
Shyam (Developer)
License
This project is licensed under the MIT License.Visionary:
