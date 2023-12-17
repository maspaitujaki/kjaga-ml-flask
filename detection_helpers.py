from io import StringIO
import os
import imutils
import numpy as np
import json
from google.cloud import storage

# with open("label.json", 'r') as file:
# 	data = json.load(file)
# 	id_to_label_mapping = {entry['id']: entry['label'] for entry in data}


ID_TO_LABEL_MAPPING = None

def readLabels():
	global ID_TO_LABEL_MAPPING
	client = storage.Client()
	bucket = client.get_bucket(os.environ.get("BUCKET_NAME"))
	blob = bucket.get_blob(os.environ.get("LABEL_PATH"))
	blob = blob.download_as_string()
	blob = blob.decode('utf-8')
	blob = StringIO(blob)
	data = json.load(blob)
	ID_TO_LABEL_MAPPING = {entry['id']: entry['label'] for entry in data}
readLabels()
# with open('label.json', 'w') as outfile:
#     json.dump(data, outfile)
# print(data)

def sliding_window(image, step, ws):
	# slide a window across the image
	for y in range(0, image.shape[0] - ws[1], step):
		for x in range(0, image.shape[1] - ws[0], step):
			# yield the current window
			yield (x, y, image[y:y + ws[1], x:x + ws[0]])
			

def image_pyramid(image, scale=1.5, minSize=(224, 224)):
	# yield the original image
	yield image
	# keep looping over the image pyramid
	while True:
		# compute the dimensions of the next image in the pyramid
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		# yield the next image in the pyramid
		yield image




def decode_predictions(arr):
	temp = []
	for element in arr:
		predicted_class = np.argmax(element)
		temp.append((ID_TO_LABEL_MAPPING.get(predicted_class), element[predicted_class]))
	return temp