from pydantic import BaseModel

class ImagePayload(BaseModel):
    base64: str
    point1: str
    point2: str
    point3: str
    point4: str
    area_dim: str