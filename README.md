# HeadPoseAPI 🎯

A **FastAPI**-based REST API for **head pose estimation** (pitch, yaw, roll)  
from images and videos using **OpenCV**, **MediaPipe**, and **Scikit-learn**.

```

## 📂 Project Structure
HeadPoseAPI/
│
├─ models/ # Pre-trained models
│ ├─ Pitch_SVR_model.joblib
│ ├─ Yaw_RF_model.joblib
│ └─ Roll_SVR_model.joblib
│
├─ api/
│ └─ main.py # FastAPI application
│
├─ requirements.txt # Python dependencies
└─ Dockerfile # Docker build file

```

## 🚀 Run Locally (without Docker)

1. **Create a virtual environment (optional but recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate       # Linux/Mac
    venv\Scripts\activate          # Windows
    ```

2. **Install dependencies**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3. **Start the API**
    ```bash
    uvicorn api.main:app --host 0.0.0.0 --port 8006
    ```
    Open your browser at:
    ```
    http://localhost:8006/docs
    ```

---

## 🐳 Run with Docker

1. **Build the Docker image**
    ```bash
    docker build -t headpose-api .
    ```

2. **Run the container**
    ```bash
    docker run -d -p 8006:8006 --name headpose-api headpose-api
    ```
    > If you want to use a different external port (e.g. 8080):
    > ```bash
    > docker run -d -p 8080:8006 --name headpose-api headpose-api
    > ```
    > Then open: `http://localhost:8080/docs`

3. **Stop and remove the container**
    ```bash
    docker stop headpose-api
    docker rm headpose-api
    ```

---

## 🖼️ API Endpoints

| Method | Endpoint          | Description                        |
|------- |-------------------|--------------------------------------|
| POST   | `/predict_image/` | Upload an image and get processed image |
| POST   | `/predict_video/` | Upload a video and get processed video |

---

## ⚡ Notes
- The models (`.joblib` files) are trained using `scikit-learn`.
- Make sure to match the **scikit-learn version** used during training to avoid  
  `InconsistentVersionWarning`.

---

## 📜 License
MIT License
