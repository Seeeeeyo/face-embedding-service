## Face Embedding Microservice

This FastAPI service wraps the `uniface` models (RetinaFace + ArcFace) and exposes a lightweight HTTP API that Supabase Edge Functions can call to compute per-face embeddings.

### Local development

```bash
cd services/face-embedding-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Send a request:

```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"image_base64":"<BASE64_DATA>"}'
```

If `FACE_SERVICE_TOKEN` is set, include `Authorization: Bearer <token>` in the request.

### Deploying

You can deploy this service to Render, Railway, or any container host:

1. Build an image with the provided files (Dockerfile not included—Render can auto-build from repo).
2. Set `PORT`, `FACE_SERVICE_TOKEN`, and optionally `PYTHONUNBUFFERED=1`.
3. Keep instance RAM ≥ 2GB to load models quickly.

The Supabase edge function expects:

- `POST /embed` that returns `{ faces: FaceEmbedding[], image_width, image_height, model }`.
- Optional bearer token authentication.
