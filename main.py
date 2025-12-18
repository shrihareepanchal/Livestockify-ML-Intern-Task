from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2, os, tempfile
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

app = FastAPI(title="Bird Counting and Weight Estimation API")

# Load pretrained YOLOv8 model (allowed)
model = YOLO("yolov8n.pt")

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/analyze_video")
async def analyze_video(video: UploadFile = File(...), fps_sample: int = 5):
    # Save uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await video.read())
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25  # fallback

    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ✅ FIX 1: Output FPS must match sampling FPS
    output_fps = fps_sample

    out_path = "outputs/annotated_output.mp4"
    os.makedirs("outputs", exist_ok=True)
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (W, H)
    )

    counts = []
    weight_index = defaultdict(list)

    frame_id = 0
    sampled_frame_id = 0
    sample_interval = max(1, int(fps // fps_sample))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Frame sampling
        if frame_id % sample_interval != 0:
            frame_id += 1
            continue

        # YOLOv8 detection + tracking
        results = model.track(frame, persist=True, classes=[14])
        active_ids = set()

        for r in results:
            if r.boxes.id is None:
                continue

            for box, tid in zip(r.boxes.xyxy, r.boxes.id):
                x1, y1, x2, y2 = map(int, box)
                tid = int(tid)
                active_ids.add(tid)

                # Weight proxy: normalized bounding box area
                area = (x2 - x1) * (y2 - y1)
                weight_index[tid].append(area / (W * H))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID {tid}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

        # ✅ FIX 2: Correct timestamp for sampled frames
        current_time = round(sampled_frame_id / output_fps, 2)

        counts.append({
            "time_sec": current_time,
            "count": len(active_ids)
        })

        cv2.putText(
            frame,
            f"Count: {len(active_ids)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        writer.write(frame)

        sampled_frame_id += 1
        frame_id += 1

    cap.release()
    writer.release()
    os.remove(video_path)

    # Aggregate weight index per bird
    weights = [
        {
            "track_id": tid,
            "weight_index": round(float(np.mean(vals)), 4)
        }
        for tid, vals in weight_index.items()
    ]

    return JSONResponse({
        "counts": counts,
        "weight_estimates": weights,
        "artifacts": {
            "annotated_video": out_path
        }
    })
