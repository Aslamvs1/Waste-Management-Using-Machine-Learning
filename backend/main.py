# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.applications.efficientnet import preprocess_input
# from PIL import Image
# import io
# import numpy as np

# app = FastAPI()

# # Allow CORS requests from your React frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],  # Update for production if needed
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Path to your trained model file
# MODEL_PATH = "models/final_model_epoch_50_20250328_190103.keras"
# try:
#     model = load_model(MODEL_PATH)
#     print("Model loaded successfully.")
# except Exception as e:
#     print("Error loading model:", e)
#     model = None

# # Define the expected input size of your model
# IMG_SIZE = (224, 224)

# def preprocess_image(image: Image.Image) -> np.ndarray:
#     # Resize the image
#     image = image.resize(IMG_SIZE)
#     # Convert image to array
#     image_array = img_to_array(image)
#     # Expand dimensions to add the batch dimension
#     image_array = np.expand_dims(image_array, axis=0)
#     # Apply EfficientNet preprocessing (matches your training pipeline)
#     image_array = preprocess_input(image_array)
#     return image_array

# def predict_waste(image: Image.Image) -> str:
#     if model is None:
#         return "Model not loaded"
#     processed_image = preprocess_image(image)
#     prediction = model.predict(processed_image)
#     # Define your six classes
#     classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
#     predicted_class = classes[np.argmax(prediction)]
#     return predicted_class

# @app.post("/classify")
# async def classify_image(file: UploadFile = File(...)):
#     # Check if the uploaded file is an image
#     if not file.content_type.startswith("image/"):
#         raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
#     try:
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Error processing image: {e}")
    
#     waste_type = predict_waste(image)
#     return {"waste_type": waste_type}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import io
import numpy as np
import os

app = FastAPI()

# Allow CORS requests from your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to your trained model file
MODEL_PATH = "models/final_model_epoch_50_20250328_190103.keras"

# Load the model with error handling
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
else:
    print("❌ Model file not found.")

# Define the expected input size of your model
IMG_SIZE = (224, 224)

# Waste information dictionary
WASTE_INFO = {
    "cardboard": {
        "recyclable": True,
        "energy_conversion": ["Composting", "Incineration with energy recovery"],
        "description": "Cardboard is 100% recyclable and biodegradable. Recycled cardboard saves 25% energy compared to new production.",
        "handling": "Flatten and remove any non-paper materials before recycling."
    },
    "glass": {
        "recyclable": True,
        "energy_conversion": ["Melting for new products"],
        "description": "Glass is infinitely recyclable without loss of quality. Recycling glass saves 30% energy compared to new production.",
        "handling": "Rinse containers and remove lids before recycling."
    },
    "metal": {
        "recyclable": True,
        "energy_conversion": ["Melting and reforming", "Aluminum can recycling saves 95% energy"],
        "description": "Metals are infinitely recyclable. Recycling aluminum saves 95% of the energy needed for primary production.",
        "handling": "Clean cans and separate ferrous/non-ferrous metals if possible."
    },
    "paper": {
        "recyclable": True,
        "energy_conversion": ["Composting", "Incineration with energy recovery"],
        "description": "Paper can be recycled 5-7 times. Recycling paper saves 60% energy compared to virgin paper production.",
        "handling": "Keep dry and free from food contamination."
    },
    "plastic": {
        "recyclable": True,
        "energy_conversion": ["Pyrolysis to fuel", "Incineration with energy recovery"],
        "description": "Only certain plastics are recyclable (look for resin codes). Plastic-to-fuel conversion can recover energy.",
        "handling": "Check local recycling guidelines for accepted types."
    },
    "trash": {
        "recyclable": False,
        "energy_conversion": ["Waste-to-energy incineration"],
        "description": "Non-recyclable waste can be processed in waste-to-energy plants to generate electricity.",
        "handling": "Dispose in proper waste bins and avoid contamination."
    }
}

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocesses an image for model prediction.
    """
    image = image.resize(IMG_SIZE)
    image_array = img_to_array(image)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

def predict_waste(image: Image.Image) -> str:
    """
    Predicts the waste type from an image.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    predicted_class = classes[np.argmax(prediction)]
    return predicted_class

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    Handles image uploads & returns waste classification results.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")
    
    # Predict waste type
    waste_type = predict_waste(image)
    
    # Validate the predicted class
    if waste_type not in WASTE_INFO:
        raise HTTPException(status_code=500, detail="Unknown waste classification.")
    
    # Return full response
    return {
        "waste_type": waste_type,
        "recyclable": WASTE_INFO[waste_type]["recyclable"],
        "energy_conversion": WASTE_INFO[waste_type]["energy_conversion"],
        "description": WASTE_INFO[waste_type]["description"],
        "handling": WASTE_INFO[waste_type]["handling"]
    }

# Run FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
