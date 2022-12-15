from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from app.detection import DetectionAPI
from app.configs import FRAME_TIME_DELAY, DATABASE_URL
import app.utills as utills
import pymongo
import time
import base64
import yaml
import json
from datetime import datetime
from app.schemas import ImagePayload


app = FastAPI()
client = pymongo.MongoClient(DATABASE_URL)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory='templates')


@app.get("/")
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/uploader")
async def uploader(request: Request):
    return templates.TemplateResponse("uploader.html", {"request": request})


@app.post("/full-process")
async def full_frame_process(imagePayload: ImagePayload):
    # Read default values from yml file if nothing was given
    if (imagePayload.point1 == "" or imagePayload.point2 == "" or imagePayload.point3 == "" or imagePayload.point4 == "" or imagePayload.area_dim == ""):
        path_yml = 'app/default.yml'
        with open(path_yml) as stream:
            default_cfg = yaml.safe_load(stream)

        pers_points = default_cfg['perspective_points']
        room_dim = default_cfg['area_dimensions']
    else: 
        room_dim = utills.tuple_string_to_float_array(imagePayload.area_dim)
        pers_points = [utills.tuple_string_to_int_array(imagePayload.point1), utills.tuple_string_to_int_array(imagePayload.point2), utills.tuple_string_to_int_array(imagePayload.point3), utills.tuple_string_to_int_array(imagePayload.point4), [903, 533],[1353, 547], [926, 341], [1054, 573]]
        
    # Fetch base64 image from payload and decode for processing
    frame = utills.get_image_from_base64(imagePayload.base64)
    # Initialize DetectionAPI
    detection = DetectionAPI(None, pers_points, room_dim)
    # Apply pre-processing
    detection.pre_process(frame, True)
    # Apply DetectionAPI
    buffer, buffer_bird, all_person_points, classified_distances, seats = detection.get_frame(frame, True)
    # Encode image to base64
    frame_jpg_as_text = utills.get_base64_from_cv2(buffer)
    bird_jpg_as_text = utills.get_base64_from_cv2(buffer_bird)

    payload = {
        'frame_base64': str(frame_jpg_as_text),
        'bird_base64': str(bird_jpg_as_text)
    }

    return payload


@app.get("/demo-data/{demo_id}")
async def main_page(request: Request, demo_id):
    db = client["demo_db"]
    data = []
    cursor = db[f"{demo_id}"].find()
    for document in cursor:
        data.append({"timestamp": document['timestamp'], 
                    "persons": document['persons'],
                    "person_distances": document['person_distances'],
                    "seats": document['seats']})

    returned_data = {"demo_data": data}
    return returned_data


@app.websocket("/demo/{footage_id}")
async def demo_connection(websocket: WebSocket, footage_id):
    try:
        await websocket.accept()      
        # Connect to MongoDB
        my_database = client["demo_db"]
        my_collection = my_database[f"{footage_id}"]

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

            # Uncomment and comment above for faster run
            # success = detection.frame_skip()
            # if not success:
            #     break

            current_time = utills.get_utc_time()
            if current_time >= previous_time + FRAME_TIME_DELAY:
                # Reset seat matrix
                detection.reset_seats()

                buffer, buffer_bird, all_person_points, classified_distances, seats = detection.get_frame()
                frame_jpg_as_text = base64.b64encode(buffer)
                bird_jpg_as_text = base64.b64encode(buffer_bird)

                log_payload = {
                    "timestamp": current_time,
                    "persons": all_person_points,
                    "person_distances": {
                        "safe_pairs": classified_distances[0],
                        "low_risk_pairs": classified_distances[1],
                        "high_risk_pairs": classified_distances[2]
                    },
                    "seats": [{"row" : s.row, "column" : s.col, "status" : s.status, "physical_coords" : s.physical_coords } for s in seats]
                }

                # Insert data into MongoDB
                x = my_collection.insert_one(jsonable_encoder(log_payload))

                sent_payload = log_payload | {
                    "frame_base64": str(frame_jpg_as_text),
                    "bird_base64": str(bird_jpg_as_text)
                }

                previous_time = current_time
                count = 0
                await websocket.send_text(json.dumps(sent_payload))

    except WebSocketDisconnect:
        print("Something went wrong...")

