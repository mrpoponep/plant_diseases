import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from config import class_names
from fastapi.responses import JSONResponse
import io

model = torch.load('Model/resnet18_full_2.pth', map_location=torch.device('cuda'))
model.eval()



transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

app = FastAPI(debug=True)

@app.get("/")
def home():
    return {"text": "Disease Identification"}

@app.post("/predict")
async def create_upload_file(file: UploadFile):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image = image.convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to("cuda")
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.softmax(output, dim=1)
        top_p, top_class = predicted_class.topk(5, dim = 1)

    predicted_labels = [class_names[i] for i in top_class[0]]
    probabilities = [round(float(p), 5) for p in top_p[0]]
    output_zip = zip(predicted_labels, probabilities)
    response_content = [{"class": class_name, "confidence": confidence} for class_name, confidence in output_zip]

    return JSONResponse(content=response_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)