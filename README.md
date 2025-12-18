
# Bird Counting and Weight Estimation (CCTV)

## Dataset
Uses **test2.mp4**.

## Approach
### Detection
- YOLOv8n pretrained model (allowed by PDF)
- Bird class detection

### Tracking
- Built-in ByteTrack via `model.track(persist=True)`
- Stable IDs prevent double counting

### Counting
- Unique active track IDs per timestamp

### Weight Estimation (Proxy)
- Weight Index = bounding box area / frame area
- Smoothed over time per bird
- To convert to grams:
  - Camera calibration (cm/pixel)
  - Small labeled weight dataset

## API
- GET /health
- POST /analyze_video (multipart video upload)

## Run
```bash
pip install ultralytics fastapi uvicorn opencv-python numpy
uvicorn main:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/analyze_video -F "video=@test2.mp4"
```

## Outputs
- outputs/annotated_output.mp4
- JSON response with counts and weight index

## Notes
- FPS sampling supported
- Occlusions handled via tracking persistence
