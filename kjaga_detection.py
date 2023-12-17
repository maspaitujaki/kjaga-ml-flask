import os
import time
from keras.applications.mobilenet_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import tensorflow_hub as hub
from detection_helpers import sliding_window
from detection_helpers import image_pyramid
from detection_helpers import decode_predictions
import numpy as np
import urllib.request
import cv2
import imutils
from google.cloud import storage

# Nilai konstanta
WIDTH = 600
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = (250, 250)
INPUT_SIZE = (224, 224)

# Load model yang telah dibuat
print("[INFO] Loading model...")
model_url = os.environ.get("MODEL_URL")
# model = load_model("./model/modelv1-5.h5", custom_objects={'KerasLayer':hub.KerasLayer})
MODEL = load_model(model_url, custom_objects={'KerasLayer':hub.KerasLayer})
print("[INFO] Model loaded...")

def reload_model():
  global MODEL
  print("[INFO] Loading model...")
  model_url = os.environ.get("MODEL_URL")
  MODEL = load_model(model_url, custom_objects={'KerasLayer':hub.KerasLayer})
  print("[INFO] Model loaded...")

def predict_image(name, bucket):
  global MODEL
  client = storage.Client()
  bucket = client.get_bucket(bucket)
  blob = bucket.get_blob(name)
  downloaded_bytes = blob.download_as_bytes()
  image = np.asarray(downloaded_bytes, dtype="uint8")
  original_image = cv2.imdecode(image, cv2.IMREAD_COLOR)

  # Load gambar dan mendapatkan dimensinya
  # original_image = cv2.imread("test/nasi-tempe.jpg")
  original_image = imutils.resize(original_image, width=WIDTH)
  (H, W) = original_image.shape[:2]

  # Melakukan inisialisasi image pyramid generator
  pyramid = image_pyramid(original_image, scale=PYR_SCALE, minSize=ROI_SIZE)

  # Menyimpan nilai ROI pada list
  # Menyimpan lokasi ROI pada original image
  rois = []
  locations = []

  # Melakukan looping pada setiap gambar di pyramid
  # Mengaplikasikan sliding windows
  for image in pyramid:
    scale = W / float(image.shape[1])
    for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
      x = int(x * scale)
      y = int(y * scale)
      w = int(ROI_SIZE[0] * scale)
      h = int(ROI_SIZE[1] * scale)
      roi = cv2.resize(roiOrig, INPUT_SIZE)
      roi = img_to_array(roi)
      roi = preprocess_input(roi)
      roi = roi/255.
      rois.append(roi)
      locations.append((x, y, x + w, y + h))

  rois = np.array(rois, dtype="float32")
  # Melakukan prediksi dan mapping label
  # sesuai dengan nilai prediksi
  preds = MODEL.predict(rois)
  preds = decode_predictions(preds)

  # Membuat set dan menambahkan nilai label
  # dengan nilai probabilitas > 85%
  labels = set()
  for (label, prob) in preds:
    if prob >= 0.85:
      labels.add(label)

  return labels

# labels = predict_image('image/nasi-tempe.jpg','kjaga-ml-public')
# print(labels)