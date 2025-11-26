import base64
import os
import re
from typing import List, Optional

import cv2
import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from uniface import ArcFace, RetinaFace

app = FastAPI(title="RacePhotos Face Embedding Service")
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_methods=["*"],
  allow_headers=["*"],
)

detector = RetinaFace()
arcface = ArcFace()

FACE_SERVICE_TOKEN = os.getenv("FACE_SERVICE_TOKEN")
auth_scheme = HTTPBearer(auto_error=False)


class EmbedRequest(BaseModel):
  image_base64: str = Field(..., description="Base64 encoded image data")
  min_confidence: Optional[float] = Field(default=0.45, ge=0.0, le=1.0)


class FaceEmbedding(BaseModel):
  face_index: int
  confidence: float
  bbox: List[float]
  landmarks: List[List[float]]
  embedding: List[float]


class EmbedResponse(BaseModel):
  model: str
  image_width: int
  image_height: int
  faces: List[FaceEmbedding]


def require_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
  if FACE_SERVICE_TOKEN:
    if credentials is None or credentials.credentials != FACE_SERVICE_TOKEN:
      raise HTTPException(status_code=401, detail="Unauthorized")
  return credentials


def decode_image(image_base64: str) -> np.ndarray:
  clean_base64 = re.sub(r"^data:image\/[a-zA-Z]+;base64,", "", image_base64)
  try:
    image_bytes = base64.b64decode(clean_base64)
  except Exception as exc:
    raise HTTPException(status_code=400, detail="Invalid base64 payload") from exc

  np_array = np.frombuffer(image_bytes, dtype=np.uint8)
  image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
  if image is None:
    raise HTTPException(status_code=400, detail="Unable to decode image data")
  return image


@app.get("/health")
async def health():
  return {"status": "ok"}


@app.post("/embed", response_model=EmbedResponse)
async def embed_faces(request: EmbedRequest, _=Depends(require_token)):
  image = decode_image(request.image_base64)
  detections = detector.detect(image)

  faces: List[FaceEmbedding] = []
  for index, face in enumerate(detections):
    confidence = float(face.get("confidence", 0.0))
    if request.min_confidence and confidence < request.min_confidence:
      continue

    landmarks = face.get("landmarks")
    bbox = face.get("bbox")
    if landmarks is None or bbox is None:
      continue

    landmarks_array = np.array(landmarks, dtype=np.float32)
    embedding = arcface.get_normalized_embedding(image, landmarks_array)
    faces.append(
      FaceEmbedding(
        face_index=index + 1,
        confidence=confidence,
        bbox=[float(v) for v in bbox],
        landmarks=[[float(pt[0]), float(pt[1])] for pt in landmarks],
        embedding=embedding.tolist(),
      )
    )

  return EmbedResponse(
    model="retinaface-arcface",
    image_width=int(image.shape[1]),
    image_height=int(image.shape[0]),
    faces=faces,
  )


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
