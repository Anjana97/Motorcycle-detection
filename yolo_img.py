import numpy as np
import time
import cv2
import os

import fn

labelsPath = "yolo-coco/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weightsPath = "yolo-coco/yolov3.weights"
configPath = "yolo-coco/yolov3.cfg"

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

image = cv2.imread('images/27.jpg')
(H, W) = image.shape[:2]

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

print("[INFO] YOLO took {:.6f} seconds".format(end - start))

boxes = []
confidences = []
classIDs = []

for output in layerOutputs:
	for detection in output:
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]

		if confidence > 0.5:
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")

			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)

idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
#ls = ['person','motorbike']

if len(idxs) > 0:
	for i in idxs.flatten():
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])
		if LABELS[classIDs[i]] == 'person':
			crop_img = image[y-15:y+h, x:x+w]
			ret , pos = fn.detect_fn(crop_img)
			if ret:
				pass			
				#cv2.rectangle(image, (x-15, y-15), (x+w+15, y+h+15), (0,255,0), 2)
				#cv2.rectangle(image, (x+pos[0], y+pos[1]), (x+pos[2], y+pos[3]), (255,255,0), 2)
				#cv2.putText(image, 'helmet', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
			else:
				#cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 2)
				#cv2.putText(image, 'no_helmet', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
				cv2.putText(image, 'wear helmet during driving', (0,10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,0,255), 2)
cv2.imshow("Image", image)
cv2.waitKey(0)