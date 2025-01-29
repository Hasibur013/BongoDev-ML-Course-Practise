# create and deploy an ML application.using gradio

from PIL import Image
import numpy as np
import torch
import gradio as gr

"""
Input an image of shape 28 x 28 or any other shape
Output as an image of shape 1 x 1 x 28 x 28
"""

tracking_uri = "F:\\bongoDev ML Course\\000.Exercises\\BongoDev ML Course Practise\\MLflow\\mlruns"
experiment_id = "900972381690289215"
run_id = "fcabd4d90e734dbab3c10206164de136"
model_uri = f"{tracking_uri}\\{experiment_id}\\{run_id}\\artifacts\\model\\data\\model.pth"

model = torch.load(model_uri)

def preprocess_image(image):
    image = image.convert('L')
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = (image - 0.1307) / 0.3081
    image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0)
    return image_tensor.float()

# Prediction function
def predict_image(image):
    # Directly use the image passed by Gradio (already a PIL image)
    input_tensor = preprocess_image(image)  # No need to open the image again
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.argmax(dim=1).item()
    
    return prediction

if __name__ == '__main__':
    interface = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(),
        title="Hand Written Digit Recognizer App",
        description="Upload a grayscale image to predict the digits"
    )

    interface.launch(
        share=True,
    )
    
    # if not deploying the app, run the following code
    # interface.launch()