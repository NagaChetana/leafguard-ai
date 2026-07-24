from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from backend.predictor import predict_image

app = FastAPI(title="Plant Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Plant Disease Detection API is Running 🚀"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)

    result = predict_image(image)

    return result