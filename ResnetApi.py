import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
import uvicorn
from PIL import Image
from config import class_names
import pandas as pd
from fastapi.responses import JSONResponse

model = torch.load('Model/resnet18_full.pth')
model.to("cpu")
model.eval()

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    
app = FastAPI(debug=True)

@app.get("/")
def home():
    return {"text": "disease identification"}

@app.post("/predict")
async def create_upload_file(file: UploadFile):
    image = Image.open("Test_folder/Data-sample-of-a-Apple-scab-b-Cherry-powdery-mildew-c-Corn-northern-leaf-blight_Q320.jpg")
    image = expand2square(image, (256, 256, 256))
    image = image.convert('RGB')
    image.save('Test_folder/API_testing/after_out.jpg', quality=95)
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.softmax(output, dim=1)
        top_p, top_class = predicted_class.topk(37, dim = 1)

    predicted_labels = [class_names[i] for i in top_class[0]]
    probabilities = [round(float(p), 5) for p in top_p[0]]
    output_zip=zip(predicted_labels,probabilities)
    response_content = [{"class": class_name, "confidence": confidence} for class_name, confidence in output_zip]

    # You can use the `JSONResponse` class to customize the response
    return JSONResponse(content=response_content)
if __name__ == "__main__":
    uvicorn.run(app)