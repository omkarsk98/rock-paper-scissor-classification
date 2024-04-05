from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import uvicorn
import tensorflow as tf

MODEL = tf.keras.models.load_model("../models/secondModel.keras")

app = FastAPI()

@app.get("/ping")
async def ping():
  return "Pong!"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
  image = await file.read()
  img = np.array(Image.open(BytesIO(image)))
  imgBatch = np.expand_dims(img, axis=0)
  prediction = MODEL.predict(imgBatch)
  return {"pred": prediction}


if __name__ == "__main__":
  uvicorn.run(app, host="localhost", port= 3005)