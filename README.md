# AI-Powered Cavity Monitoring and Cancer Detection

## Overview
This project leverages **Artificial Intelligence** to assist dental professionals in detecting **dental cavities** and **oral cancers** using images captured via an **intraoral camera**. The system processes images in real-time through a **mobile app built with Flutter**, providing instant predictions and visual highlights of affected areas.

---

## Features
- **Cavity Detection**: Detects dental cavities from intraoral images.  
- **Cancer Detection**: Identifies potential oral cancer lesions.  
- **Real-Time Analysis**: Fast predictions via mobile app or backend API.  
- **Visualization**: Highlights affected areas to aid diagnosis.  
- **Cross-Platform**: Mobile app built in **Flutter** for iOS and Android.  

---

## Technologies
- **Python** – AI model training and inference  
- **TensorFlow / Keras** – Deep learning framework  
- **TFLite** – Optimized model for mobile deployment  
- **OpenCV** – Image preprocessing and visualization  
- **FastAPI** – Backend API for predictions  
- **Flutter** – Mobile app interface  
- **NumPy, Pandas** – Data handling  
- **Matplotlib / PIL** – Visualization  

---

## System Architecture
1. **Image Capture:** High-quality images from the intraoral camera  
2. **Mobile App (Flutter):** Uploads images to backend  
3. **AI Model Processing:** CNN-based model analyzes the images  
4. **Prediction Output:** Healthy, Cavity, or Potential Cancer  
5. **Visualization:** Highlighted areas for easy diagnosis  

---

## Installation
# Clone the repository
git clone <repository-url>
cd ai-cavity-cancer-detection

# Create virtual environment (optional)
python -m venv venv
# Activate
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install Python dependencies
pip install -r requirements.txt

# Run FastAPI server
uvicorn app:app --reload
The Flutter app should be opened in Android Studio or VS Code and connected to the backend API for real-time predictions.

## Usage
1. Open the Flutter mobile app on your device.  
2. Connect the **intraoral camera** and capture images of the patient’s teeth/oral cavity.  
3. The app sends images to the FastAPI backend.  
4. Receive predictions: **Healthy**, **Cavity**, or **Potential Cancer**.  
5. Visualized results show affected regions for easy diagnosis.

---

## Dataset
- **Caries Dataset:** Images of teeth with and without cavities.  
- **Oral Cancer Dataset:** Images of oral lesions with confirmed diagnoses.  

> All patient data is anonymized to ensure privacy and ethical compliance.

---

## Model Details
- **Architecture:** Convolutional Neural Networks (CNN) for image classification  
- **Training:** Data augmentation applied for robustness  
- **Deployment:** Converted to `.tflite` for fast mobile inference  

---

## Future Enhancements
- Implement severity scoring for cavities and lesions  
- Support multi-modal input (X-rays + intraoral images) for higher accuracy  
- Expand dataset to include rare oral conditions  
- Integrate cloud storage for patient records 
