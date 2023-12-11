import os
import base64
from flask import Flask, jsonify, request
from keras.models import load_model
import tensorflow_hub as hub
import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from dotenv import load_dotenv

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

@app.route("/predict", methods=['POST'])
def predict():
  # local_image_path = './test/test_ayam_goreng.jpg'

  # # Make predictions for the local image
  # img = image.load_img(local_image_path, target_size=(224, 224))  # Adjust target_size as needed
  # img_array = image.img_to_array(img)
  # img_array = np.expand_dims(img_array, axis=0)
  # img_array = preprocess_input(img_array)

  # # Make predictions
  # predictions = model.predict(img_array)
  # # print(predictions * 100)

  # predicted_class = np.argmax(predictions)

  # # print(f"The predicted class is: {predicted_class}")
  # return str(predicted_class)

  """Receive and parse Pub/Sub messages."""
  envelope = request.get_json()
  print('envelope start')
  print(envelope)
  print('envelope end')
  if not envelope:
      msg = "no Pub/Sub message received"
      print(f"error: {msg}")
      return f"Bad Request: {msg}", 400

  if not isinstance(envelope, dict) or "message" not in envelope:
      msg = "invalid Pub/Sub message format"
      print(f"error: {msg}")
      return f"Bad Request: {msg}", 400

  pubsub_message = envelope["message"]

  name = "World"
  if isinstance(pubsub_message, dict) and "data" in pubsub_message:
      name = base64.b64decode(pubsub_message["data"]).decode("utf-8").strip()

  print(f"Hello {name}!")

  return ("",204)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))