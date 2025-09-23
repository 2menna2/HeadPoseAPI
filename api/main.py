from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
import cv2, numpy as np, joblib, io, os, tempfile
import mediapipe as mp
from math import cos, sin, radians
from datetime import datetime

app = FastAPI(title="Head Pose API")

# 🟢 تحميل الموديلات
pitch_model = joblib.load("models/Pitch_SVR_model.joblib")
roll_model  = joblib.load("models/Roll_SVR_model.joblib")
yaw_model   = joblib.load("models/Yaw_RF_model.joblib")

mp_face_mesh = mp.solutions.face_mesh

OUTPUT_DIR = "output_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_landmarks(face_landmarks):
    x_coords = np.array([lm.x for lm in face_landmarks.landmark])
    y_coords = np.array([lm.y for lm in face_landmarks.landmark])
    nose_x, nose_y = x_coords[1], y_coords[1]
    x_centered = x_coords - nose_x
    y_centered = y_coords - nose_y
    max_val = max(np.abs(x_centered).max(), np.abs(y_centered).max())
    x_norm = x_centered / max_val if max_val != 0 else x_centered
    y_norm = y_centered / max_val if max_val != 0 else y_centered
    return np.concatenate([x_norm, y_norm]).reshape(1,-1)

def draw_axis_on_image(img, pitch, yaw, roll, nose_coords, size=100):
    yaw *= -1
    R_x = np.array([[1, 0, 0],
                    [0, cos(radians(pitch)), -sin(radians(pitch))],
                    [0, sin(radians(pitch)),  cos(radians(pitch))]])
    R_y = np.array([[ cos(radians(yaw)), 0, sin(radians(yaw))],
                    [0, 1, 0],
                    [-sin(radians(yaw)), 0, cos(radians(yaw))]])
    R_z = np.array([[cos(radians(roll)), -sin(radians(roll)), 0],
                    [sin(radians(roll)),  cos(radians(roll)), 0],
                    [0, 0, 1]])
    R = R_y @ R_x @ R_z

    x_axis_end = R @ np.array([size, 0, 0])
    y_axis_end = R @ np.array([0, -size, 0])
    z_axis_end = R @ np.array([0, 0, -size])

    x0, y0 = int(nose_coords[0]), int(nose_coords[1])
    cv2.arrowedLine(img, (x0, y0),
                    (int(x0 + x_axis_end[0]), int(y0 + x_axis_end[1])),
                    (0, 0, 255), 3)
    cv2.arrowedLine(img, (x0, y0),
                    (int(x0 + y_axis_end[0]), int(y0 + y_axis_end[1])),
                    (0, 255, 0), 3)
    cv2.arrowedLine(img, (x0, y0),
                    (int(x0 + z_axis_end[0]), int(y0 + z_axis_end[1])),
                    (255, 0, 0), 2)
    return img


# ⭕️ 1) معالجة صورة
@app.post("/predict_image/")
async def predict_image(file: UploadFile = File(...)):
    data = await file.read()
    np_img = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            features = preprocess_landmarks(face)
            pitch = float(pitch_model.predict(features)[0])
            yaw   = float(yaw_model.predict(features)[0])
            roll  = float(roll_model.predict(features)[0])
            nose = face.landmark[1]
            nose_coords = (nose.x * img.shape[1], nose.y * img.shape[0])
            img = draw_axis_on_image(img, pitch, yaw, roll, nose_coords)

    _, img_encoded = cv2.imencode(".jpg", img)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")


# ⭕️ 2) معالجة فيديو وإرجاع ملف للتحميل
@app.post("/predict_video/")
async def predict_video(file: UploadFile = File(...)):
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_in.write(await file.read())
    tmp_in.close()

    cap = cv2.VideoCapture(tmp_in.name)

    filename = f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    out_path = os.path.join(OUTPUT_DIR, filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            if res.multi_face_landmarks:
                face = res.multi_face_landmarks[0]
                features = preprocess_landmarks(face)
                pitch = float(pitch_model.predict(features)[0])
                yaw   = float(yaw_model.predict(features)[0])
                roll  = float(roll_model.predict(features)[0])
                nose = face.landmark[1]
                nose_coords = (nose.x * frame.shape[1], nose.y * frame.shape[0])
                frame = draw_axis_on_image(frame, pitch, yaw, roll, nose_coords)
            out.write(frame)

    cap.release()
    out.release()
    os.remove(tmp_in.name)

    # يرجع ملف الفيديو كتحميل مباشر فى Swagger
    return FileResponse(out_path, media_type="video/mp4", filename=filename)
