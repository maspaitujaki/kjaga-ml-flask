import os
import base64
from flask import Flask, jsonify, request
from keras.models import load_model
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from dotenv import load_dotenv
import json

load_dotenv()


app = Flask(__name__)
model_url = os.environ.get("MODEL_URL")
# model = load_model("./model/model_v1.0.h5", custom_objects={'KerasLayer':hub.KerasLayer})
model = load_model(model_url, custom_objects={'KerasLayer':hub.KerasLayer})

@app.route("/")
def hello_world():
    print("")
    c = 'Hello from flask'
    return c, 200

@app.route("/reload-model")
def update_model():
  global model
  model = load_model(model_url, custom_objects={'KerasLayer':hub.KerasLayer})
  return "Ok"

@app.route("/predict", methods=['POST'])
def predict():
  # local_image_path = './test/test_ayam_goreng.jpg'

  """Receive and parse Pub/Sub messages."""
  envelope = request.get_json()
  
  if not envelope:
      msg = "no Pub/Sub message received"
      print(f"error: {msg}")
      return f"Bad Request: {msg}", 400

  if not isinstance(envelope, dict) or "message" not in envelope:
      msg = "invalid Pub/Sub message format"
      print(f"error: {msg}")
      return f"Bad Request: {msg}", 400

  pubsub_message = envelope["message"]

  predicted_class = None
  if isinstance(pubsub_message, dict) and "data" in pubsub_message:
      event_data = base64.b64decode(pubsub_message["data"]).decode("utf-8").strip()
      event_data_dict = json.loads(event_data)

      name = event_data_dict["name"]
      bucket = event_data_dict["bucket"]
      # Make predictions for the local image
      raw = tf.io.read_file("gs://"+bucket+name)
      img = tf.image.decode_image(raw, channels=3)
      img.set_shape([224,224])
      # img = image.load_img("gs://"+bucket+name, target_size=(224, 224))  # Adjust target_size as needed
      img_array = image.img_to_array(img)
      img_array = np.expand_dims(img_array, axis=0)
      img_array = preprocess_input(img_array)

      # Make predictions
      predictions = model.predict(img_array)
      # print(predictions * 100)

      predicted_class = np.argmax(predictions)

      # print(f"The predicted class is: {predicted_class}")
      # return str(predicted_class)
      
  print(f"The predicted class is: {predicted_class}")

  return ("",204)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))