import numpy as np
import time
import cv2
import os

labelsPath = "custom/obj.names"
LABELS = open(labelsPath).read().strip().split("\n")


np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


weightsPath = "custom/yolov3-obj_2400.weights"
configPath = "custom/yolov3-obj.cfg"

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

	
def detect_fn(image):
	x = 0
	y = 0
	w = 0
	h =0
	#image = cv2.imread('images/im0001.jpg')
	#image = cv2.imread(path)
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

	if len(idxs) > 0:
		for i in idxs.flatten():

			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			#color = [int(c) for c in COLORS[classIDs[i]]]
			#cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			#print(x,y,x+w,x+h)
			#text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			#cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		return True , (x,y,x+w,y+h)
	
	else:
		return False, (x,y,x+w,y+h)

	#cv2.imshow("Image", image)
	#cv2.waitKey(0)
