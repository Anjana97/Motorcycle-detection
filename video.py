import numpy as np
import time
import cv2
import os
import imutils

import fn

labelsPath = "yolo-coco/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weightsPath = "yolo-coco/yolov3.weights"
configPath = "yolo-coco/yolov3.cfg"

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


vs = cv2.VideoCapture('videos/3.avi')
writer = None
(W, H) = (None, None)

try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

while True:
	(grabbed, frame) = vs.read()

	if not grabbed:
		break

	if W is None or H is None:
		(H, W) = frame.shape[:2]

	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

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

	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)

	if len(idxs) > 0:
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			if LABELS[classIDs[i]] == 'person':
				crop_img = frame[y-15:y+h, x:x+w]
				ret , pos = fn.detect_fn(crop_img)
				#print(ret)
				if ret:
					cv2.rectangle(frame, (x-15, y-15), (x+w+15, y+h+15), (0,255,0), 2)
					cv2.putText(frame, 'helmet', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
				else:				
					cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
					cv2.putText(frame, 'no_helmet', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
					#cv2.putText(frame, 'warning no helmet', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
					cv2.putText(frame, 'wear helmet during driving', (0,10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,0,255), 2)

	if writer is None:
	
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter('output/output.avi', fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))

	writer.write(frame)

print("[INFO] cleaning up...")
writer.release()
vs.release()