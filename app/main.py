from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from app.detection import DetectionAPI

app = FastAPI()

def generate(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg \r\n\r\n' + frame + b'\r\n\r\n'
               )

@app.get("/")
def test():
    return {"hello", "world"}


@app.get('/samplefootage/{footage_id}')
def video_feed(footage_id):

    path = f'sample-footage/{footage_id}.mp4'
    return StreamingResponse(generate(DetectionAPI(path)), media_type='multipart/x-mixed-replace;boundary=frame')


