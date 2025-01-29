# digit_classifier_using_fastapi.py

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from PIL import Image
import io
import numpy as np

app = FastAPI()                                 

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Custom API for Digit Classifier",
        version="1.0.0",
        description="This is a custom API documentation",
        routes=app.routes,
    )
    # Remove unnecessary routes or details here
    app.openapi_schema = openapi_schema
    return app.openapi_schema

# Override the default OpenAPI schema
app.openapi = custom_openapi

tracking_uri = "F:\\bongoDev ML Course\\000.Exercises\\BongoDev ML Course Practise\\MLflow\\mlruns"
experiment_id = "900972381690289215"
run_id = "fcabd4d90e734dbab3c10206164de136"
model_uri = f"{tracking_uri}\\{experiment_id}\\{run_id}\\artifacts\\model\\data\\model.pth"

model = torch.load(model_uri)

def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("L")
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image_array = (image - 0.1307) / 0.3081
    image_tensor = torch.tensor(image_array).unsqueeze(0).unsqueeze(0)
    return image_tensor.float()

class PredictionResponse(BaseModel):
    prediction: int
    
@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    input_tensor = preprocess_image(image)
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.argmax(dim=1).item()
        
    return {"prediction": prediction}

"""
uvicorn digit_classifier_using_fastapi:app --reload

After that go to the local host /docs
http://127.0.0.1:8000/docs

"""
