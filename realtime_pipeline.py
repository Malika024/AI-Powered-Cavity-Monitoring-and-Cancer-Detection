# =====================================================================
# Dental AI Realtime API using FastAPI + TFLite
# =====================================================================
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
LESION_MODEL_PATH = r"C:\DentalAI_API\lesion_model.tflite"
CAVITY_MODEL_PATH = r"C:\DentalAI_API\cavity_model.tflite"
CANCER_MODEL_PATH = r"C:\DentalAI_API\cancer_model.tflite"

# ---------------------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------------------
def load_tflite(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

print("🧠 Loading TFLite models...")
lesion_interpreter = load_tflite(LESION_MODEL_PATH)
cavity_interpreter = load_tflite(CAVITY_MODEL_PATH)
cancer_interpreter = load_tflite(CANCER_MODEL_PATH)
print("✅ Models loaded successfully!")

# ---------------------------------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------------------------------
def run_inference(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    h, w = input_details[0]['shape'][1:3]
    img = cv2.resize(image, (w, h))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return float(output.flatten()[0])

# ---------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------
app = FastAPI(title="Dental AI Sequential API")

# ------------------- ADD CORS MIDDLEWARE HERE -------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],       # allow GET, POST, etc.
    allow_headers=["*"],       # allow all headers
)
# ---------------------------------------------------------------------

@app.get("/")
def index():
    return {"message": "Dental AI API running successfully!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        data = await file.read()
        image_array = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Run sequential inferences
        lesion_prob = run_inference(lesion_interpreter, img)
        cavity_prob = run_inference(cavity_interpreter, img)
        cancer_prob = run_inference(cancer_interpreter, img)

        # Interpret results
        results = {
            "lesion": {
                "label": "Lesion Detected" if lesion_prob > 0.5 else "No Lesion",
                "confidence": round(lesion_prob * 100, 2)
            },
            "cavity": {
                "label": "Cavity Detected" if cavity_prob > 0.5 else "No Cavity",
                "confidence": round(cavity_prob * 100, 2)
            },
            "cancer": {
                "label": "Cancerous" if cancer_prob > 0.5 else "Non-Cancerous",
                "confidence": round(cancer_prob * 100, 2)
            }
        }

        return JSONResponse(content=results)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)