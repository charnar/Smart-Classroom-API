import cv2
import time
import numpy as np
import datetime
from app.configs import CONFIG_PATH, WEIGHTS_PATH, SCALE_FACTOR, SPATIAL_SIZE, CONFIDENCE, THRESHOLD
import app.utills as utills
from app.seat_detection import get_seat_matrix


class DetectionAPI(object):
    def __init__(self, video_source, pers_points, room_dim):
        self.video = cv2.VideoCapture(video_source)
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width= int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.room_dim = tuple(room_dim)
        self.net_yl = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
        self.net_yl.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net_yl.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.ln = self.net_yl.getLayerNames()
        self.pers_points = pers_points
        self.ln1 = [self.ln[i - 1] for i in self.net_yl.getUnconnectedOutLayers()]
        np.random.seed(42)
    

    def __del__(self):
        self.video.release()


    def pre_process(self):
        success, image = self.video.read()

        # Initialize Perspective transform
        src = np.float32(np.array(self.pers_points[:4]))
        dst = np.float32([[0, self.height], [self.width, self.height], [self.width, 0], [0, 0]])
        self.pers_transform = cv2.getPerspectiveTransform(src, dst)

        if success:
            transformed_first_frame = utills.perspective_transform_frame(image, self.pers_points)
            self.seat_matrix = get_seat_matrix(transformed_first_frame)
            cv2.imwrite('out/reference_img.jpg', transformed_first_frame)

        # Transform seat points 
        utills.get_transformed_seat_points(self.seat_matrix, self.pers_transform)

        # The 3 points for horizontal and vertical unit length (180cm)
        pts = np.float32(np.array([self.pers_points[4:7]]))
        warped_pt = cv2.perspectiveTransform(pts, self.pers_transform)[0]

        self.distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
        self.distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
        pnts = np.array(self.pers_points[:4], np.int32)

        #Scaling height and width
        sc_w = self.distance_w / (1918-0)
        sc_h = self.distance_h / (907-79)
        #Specify bird_width
        self.bird_width = 400

        #Find suitable size of bird eye view window
        while True:
            self.bird_height = int(sc_h * self.bird_width / sc_w)
            # bird_height = int(sc_w*bird_width/sc_h)
            if self.bird_width * self.bird_height < 200000:
                self.bird_width = self.bird_width + 40
            elif self.bird_width * self.bird_height > 360000:
                self.bird_width = self.bird_width - 40
            else:
                break

        self.scale_w, self.scale_h = utills.get_scale(self.width, self.height, self.bird_width, self.bird_height)

                    
    def get_frame(self):
        success, image = self.video.read()
        frameHeight, frameWidth = image.shape[:2]

        # Person detection
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
                        box = detection[0:4] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])
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
                boxes1.append(boxes[i])


        persons = []
        for box in boxes1:
            if success:
                if box[0] >= 0 and box[1] >= 0:
                    # Hand detection implementation is not present
                    hand_boxes = []
                    persons.append((box, hand_boxes))
                    
        
         # Transform person points and hand points
        person_points = utills.get_transformed_points(boxes1, self.pers_transform, self.seat_matrix)

        # Calculate distance between transformed points
        distances_mat, bxs_mat = utills.get_distances(boxes1, person_points, self.distance_w, self.distance_h)
        risk_count = utills.get_count(distances_mat)


        image1 = np.copy(image)
        bird_image, all_person_points, classified_distances = utills.get_bird_eye_view(image, distances_mat, person_points, self.scale_w, self.scale_h, risk_count, self.seat_matrix, self.room_dim)


        final_image = utills.social_distancing_view(image1, bxs_mat, persons, risk_count)

        # Encode the image
        response, encoded_image = cv2.imencode('.jpg', final_image)
        response1, encoded_bird_image = cv2.imencode('.jpg', bird_image)

        return encoded_image, encoded_bird_image, all_person_points, classified_distances, self.seat_matrix


    def frame_skip(self):
        success, image = self.video.read()
        return success


