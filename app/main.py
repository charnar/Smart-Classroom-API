from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from app.detection import DetectionAPI
from app.configs import FRAME_TIME_DELAY
import time
import app.utills as utills
import base64
import yaml
import json

app = FastAPI()


@app.websocket("/demo/{footage_id}")
async def demo_connection(websocket: WebSocket, footage_id):
    try:
        await websocket.accept()

        # Open footage config
        path_yml = f'sample-footage/{footage_id}.yml'
        with open(path_yml) as stream:
            sample_cfg = yaml.safe_load(stream)

        detection = DetectionAPI(sample_cfg['path'], sample_cfg['perspective_points'], sample_cfg['area_dimensions'])

        # Pre-processing Stage
        detection.pre_process()

        previous_time = utills.get_utc_time()
        count = 0
        # Main loop

        while True:
            if (count <= detection.fps * FRAME_TIME_DELAY):
                success = detection.frame_skip()
                if not success:
                    break

                count += 1

            current_time = utills.get_utc_time()
            if current_time >= previous_time + FRAME_TIME_DELAY:
                buffer, buffer_bird, all_person_points, classified_distances, seats = detection.get_frame()
                frame_jpg_as_text = base64.b64encode(buffer)
                bird_jpg_as_text = base64.b64encode(buffer_bird)


                payload = {
                    "timestamp": str(current_time),
                    "person_distances": {
                        "safe_pairs": classified_distances[0],
                        "low_risk_pairs": classified_distances[1],
                        "high_risk_pairs": classified_distances[2]
                    },
                    "seats": [{"row" : s.row, "column" : s.col, "status" : s.status, "physical_coords" : s.physical_coords} for s in seats],
                    "frame_base64": str(frame_jpg_as_text),
                    "bird_base64": str(bird_jpg_as_text)
                }

                previous_time = current_time
                count = 0
                await websocket.send_text(json.dumps(payload))
        
    except WebSocketDisconnect:
        print("Something went wrong...")

