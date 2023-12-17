import os
import base64
from flask import Flask, request
from keras.models import load_model
import tensorflow_hub as hub
from keras.applications.inception_v3 import preprocess_input
from dotenv import load_dotenv
import json
import psycopg2
from kjaga_detection import predict_image, reload_model
from datetime import datetime
load_dotenv()


app = Flask(__name__)

def update_prediction_result(name: str, result: str):
  sql = """ UPDATE predictions 
            SET (predicted_at, status, result) = (%s, 'DONE', %s) where id = %s"""
  conn = None
  updated_rows = 0
  try:
      # connect to the PostgreSQL database
      conn = psycopg2.connect(database=os.environ.get("PG_DATABASE"),
                          host=os.environ.get("PG_HOST"),
                          user=os.environ.get("PG_USER"),
                          password=os.environ.get("PG_PASSWORD"),
                          port="5432")
      # create a new cursor
      cur = conn.cursor()

      ts = datetime.now()

      # execute the UPDATE  statement
      cur.execute(sql, (ts, result, name))
      # get the number of updated rows
      updated_rows = cur.rowcount
      # Commit the changes to the database
      conn.commit()
      # Close communication with the PostgreSQL database
      cur.close()
  except (Exception, psycopg2.DatabaseError) as error:
      print(error)
  finally:
      if conn is not None:
          conn.close()

@app.route("/")
def hello_world():
    print("")
    c = 'Hello from flask'
    return c, 200

@app.route("/reload-model")
def update_model():
  reload_model()

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

  if isinstance(pubsub_message, dict) and "data" in pubsub_message:
      event_data = base64.b64decode(pubsub_message["data"]).decode("utf-8").strip()
      event_data_dict = json.loads(event_data)

      name = event_data_dict["name"]
      bucket = event_data_dict["bucket"]
      predictions = list(predict_image(name,bucket))
      print(f"The predicted class is: {predictions}")
      update_prediction_result(name, str(predictions))
      
      

  return ("",204)

# @app.route("/predict-local")
# def predictLocal():
#   # local_image_path = './test/test_ayam_goreng.jpg'

#   predicted_class = None

#   raw = tf.io.read_file("test/test_ayam_goreng.jpg")
#   img = tf.image.decode_image(raw, channels=3)
#   img1 = tf.image.resize(img,[224,224])

#   img = image.load_img("test/test_ayam_goreng.jpg", target_size=(224, 224))  # Adjust target_size as needed

#   print(type(img), type(img1), type(img1) == type(img))
#   img_array = image.img_to_array(img)
#   img_array = np.expand_dims(img_array, axis=0)

#   img_array1 = image.img_to_array(img1)
#   img_array1 = np.expand_dims(img_array1, axis=0)
#   img_array_copy = np.copy(img_array1)

#   print(type(img_array), type(img_array1), type(img_array) == type(img_array1))


#   img_array = preprocess_input(img_array_copy)
#   # Make predictions
#   predictions = model.predict(img_array)
#   # print(predictions * 100)

#   predicted_class = np.argmax(predictions)

#   # print(f"The predicted class is: {predicted_class}")
#   # return str(predicted_class)
      
#   print(f"The predicted class is: {predicted_class}")
#   update_prediction_result('Aku', int(predicted_class))


#   return ("",204)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))