from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from PIL import Image
import requests
from io import BytesIO

app = FastAPI()

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

results_store = {}

@app.post("/classify-url/")
async def classify_image_url(image_url: str = Form(...)):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    except requests.RequestException as e:
        return HTMLResponse(content=f"Error fetching image: {str(e)}", status_code=400)
    except IOError:
        return HTMLResponse(content="The provided URL does not contain a valid image.", status_code=400)

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_label]

    return HTMLResponse(content=f"Predicted class: {predicted_class}")

@app.get("/")
async def main():
    content = """
<body>
<form action="/classify-url/" method="post">
    <input name="image_url" type="text" placeholder="Enter direct image URL here">
    <input type="submit" value="Classify">
</form>
</body>
    """
    return HTMLResponse(content=content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080)