import cv2
import time
import numpy as np
from app.configs import CONFIG_PATH, WEIGHTS_PATH, SCALE_FACTOR, SPATIAL_SIZE, CONFIDENCE, THRESHOLD


class DetectionAPI(object):
    def __init__(self, video_source):
        self.video = cv2.VideoCapture(video_source)
        self.net_yl = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
        self.ln = self.net_yl.getLayerNames()
        self.ln1 = [self.ln[i - 1] for i in self.net_yl.getUnconnectedOutLayers()]
        np.random.seed(42)
    
    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()

        # Apply Human Detection 
        ih, iw = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, SCALE_FACTOR, SPATIAL_SIZE, swapRB=True, crop=False)
        self.net_yl.setInput(blob)
        layerOutputs = self.net_yl.forward(self.ln1)
        boxes = []
        confidences = []
        classIDs = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if classID == 0:
                    if confidence > CONFIDENCE:
                        box = detection[0:4] * np.array([iw, ih, iw, ih])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)


        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)
        font = cv2.FONT_HERSHEY_PLAIN
        boxes1 = []


        for i in range(len(boxes)):
            if i in idxs:
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = (0, 255, 0)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)


        # Encode the image
        response, encodedImage = cv2.imencode('.jpg', image)
        return encodedImage.tobytes()
        
        
