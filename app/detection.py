import cv2
import time


class DetectionAPI(object):
    def __init__(self, video_source):
        self.video = cv2.VideoCapture(video_source)
    
    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        response, encodedImage = cv2.imencode('.jpg', image)
        return encodedImage.tobytes()
        
        
