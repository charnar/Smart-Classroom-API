from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from app.detection import DetectionAPI
from app.configs import FRAME_TIME_DELAY
import time
import app.utills as utills
import base64

app = FastAPI()


@app.websocket("/demo/{footage_id}")
async def demo_connection(websocket: WebSocket, footage_id):
    await websocket.accept()

    # Set source path
    path = f'sample-footage/{footage_id}.mp4'
    detection = DetectionAPI(path)

    previous_time = utills.get_utc_time()

    try:
        while True:
            detection.frame_skip()
            current_time = utills.get_utc_time()
            if current_time >= previous_time + FRAME_TIME_DELAY:
                buffer, processed_flag = detection.get_frame()
                jpg_as_text = base64.b64encode(buffer)
                previous_time = current_time
                await websocket.send_text("{\"frame_base64\": "+ str(jpg_as_text) +"}")
            
    except WebSocketDisconnect:
        print("Disconnected!")

