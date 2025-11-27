#!/usr/bin/env python3
"""
YOLO GPU + FP16 + skip-frame + WebSocket server
- RTSP capture -> frame queue
- YOLO inference in GPU worker (FP16)
- Send normalized boxes via WebSocket
- Display annotated frames in OpenCV
- Save snapshots async
"""

import asyncio, json, time, os, signal
from concurrent.futures import ThreadPoolExecutor
import cv2
from ultralytics import YOLO
import torch
import websockets
import numpy as np

# ---------------- CONFIG ----------------
MODEL_PATH = "best.pt"
RTSP_URL = "rtsp://172.20.64.203:8554/streams/video1"
IMG_SIZE = 640
CONF_THRESH = 0.5
SAVE_EVERY_N = 30
SAVE_QUEUE_MAX = 4
WS_HOST = "0.0.0.0"
WS_PORT = 8765
SKIP_FRAME = 2  # xử lý 1 frame / SKIP_FRAME

os.makedirs("snapshots", exist_ok=True)

# ---------------- GLOBALS ----------------
CLIENTS = set()
frame_queue = asyncio.Queue(maxsize=4)
metadata_queue = asyncio.Queue(maxsize=8)
save_queue = asyncio.Queue(maxsize=SAVE_QUEUE_MAX)
executor = ThreadPoolExecutor(max_workers=4)

# ---------------- LOAD MODEL GPU + FP16 ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = YOLO(MODEL_PATH)
model.fuse()
model.to(device)
half_precision = device.startswith("cuda")
if half_precision:
    model.model.half()  # FP16

# ---------------- YOLO INFERENCE ----------------
def yolo_infer(frame):
    h, w = frame.shape[:2]
    results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False)
    boxes = []
    for result in results:
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            boxes.append({
                "x": x1 / w, "y": y1 / h,
                "w": (x2 - x1) / w, "h": (y2 - y1) / h,
                "label": model.names[int(cls)],
                "score": float(conf)
            })
    return boxes

# def draw_boxes(frame, boxes):
#     frame_copy = frame.copy()
#     h, w = frame_copy.shape[:2]
#     for b in boxes:
#         x, y = int(b["x"] * w), int(b["y"] * h)
#         w_box, h_box = int(b["w"] * w), int(b["h"] * h)
#         color = (0, 255, 0)
#         cv2.rectangle(frame_copy, (x, y), (x + w_box, y + h_box), color, 2)
#         label = f"{b['label']} {int(b['score']*100)}%"
#         cv2.putText(frame_copy, label, (x, max(y-5,0)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame_copy

def draw_boxes(frame, boxes):
    """
    Vẽ các bounding box trên GPU.
    frame: np.ndarray HxWx3, BGR
    boxes: list dict {"x","y","w","h","label","score"}
    """
    # Upload frame lên GPU
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)

    h, w = frame.shape[:2]

    # Tạo overlay trống trên GPU
    overlay = cv2.cuda_GpuMat(h, w, cv2.CV_8UC3)
    overlay.setTo((0,0,0))  # đen

    for b in boxes:
        x1 = int(b["x"]*w)
        y1 = int(b["y"]*h)
        x2 = int((b["x"]+b["w"])*w)
        y2 = int((b["y"]+b["h"])*h)
        color = (0,255,0)
        # vẽ rectangle bằng polylines
        pts = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], np.int32)
        pts = pts.reshape((-1,1,2))
        cpu_pts = pts.copy()  # cv2.cuda.polylines chưa hỗ trợ trực tiếp
        cv2.polylines(frame, [cpu_pts], isClosed=True, color=color, thickness=2)

    # Nếu muốn, bạn có thể blend overlay với frame bằng cv2.cuda.addWeighted
    # gpu_frame = cv2.cuda.addWeighted(gpu_frame, 1.0, overlay, 1.0, 0)

    return gpu_frame.download()

# ---------------- RTSP CAPTURE ----------------
async def capture_loop():
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("Cannot open RTSP stream")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.01)
            continue
        if frame_queue.full():
            try: frame_queue.get_nowait()
            except: pass
        await frame_queue.put(frame)
        await asyncio.sleep(0)  # yield
    cap.release()

# ---------------- INFERENCE WORKER ----------------
async def inference_loop():
    frame_counter = 0
    while True:
        frame = await frame_queue.get()
        process_frame = (frame_counter % SKIP_FRAME == 0)

        if process_frame:
            boxes = await asyncio.get_running_loop().run_in_executor(executor, yolo_infer, frame)
        else:
            boxes = []  # skip processing, send empty metadata

        fh, fw = frame.shape[:2]
        frame_ts = time.time()
        boxes_with_px = []
        for b in boxes:
            x_px = int(b.get('x',0)*fw)
            y_px = int(b.get('y',0)*fh)
            w_px = int(b.get('w',0)*fw)
            h_px = int(b.get('h',0)*fh)
            b2 = b.copy()
            b2.update({'x_px': x_px, 'y_px': y_px, 'w_px': w_px, 'h_px': h_px})
            boxes_with_px.append(b2)

        metadata = {
            "boxes": boxes_with_px,
            "frame_w": fw,
            "frame_h": fh,
            "frame_index": frame_counter,
            "frame_ts": frame_ts,
            "ts": time.time()
        }
        if metadata_queue.full():
            try: metadata_queue.get_nowait()
            except: pass
        await metadata_queue.put(metadata)

        # vẽ frame cho display + snapshot
        if process_frame:
            frame_drawn = await asyncio.get_running_loop().run_in_executor(executor, draw_boxes, frame, boxes_with_px)
            cv2.imshow("YOLO Overlay", frame_drawn)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_counter += 1
        frame_queue.task_done()

# ---------------- WEBSOCKET ----------------
async def ws_handler(ws):
    CLIENTS.add(ws)
    try:
        while True:
            metadata = await metadata_queue.get()
            try:
                await ws.send(json.dumps(metadata))
            except:
                break
            finally:
                metadata_queue.task_done()
    finally:
        CLIENTS.discard(ws)

async def ws_server():
    async with websockets.serve(ws_handler, WS_HOST, WS_PORT):
        await asyncio.Future()

# ---------------- MAIN ----------------
async def main():
    tasks = [
        asyncio.create_task(capture_loop()),
        asyncio.create_task(inference_loop()),
        asyncio.create_task(ws_server()),
    ]
    await asyncio.gather(*tasks)

if __name__=="__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def shutdown(*args):
        for t in asyncio.all_tasks(loop):
            t.cancel()
        cv2.destroyAllWindows()

    import signal
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        loop.run_until_complete(main())
    except asyncio.CancelledError:
        pass
    finally:
        loop.close()
