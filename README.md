# Face Recognition Attendance System

A real-time face recognition attendance system built with Streamlit and YOLOv8.

## Features

- Real-time face detection and recognition
- Attendance tracking with timestamps
- Attendance records management
- CSV export functionality
- Modern and responsive UI

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the application:
```bash
streamlit run appcam_b.py
```

## Usage

1. Click "Start Webcam" to begin face detection
2. The system will automatically detect and recognize faces
3. Attendance records are saved to `detection_log.csv`
4. View attendance records in the "Attendance Records" tab
5. Export attendance data as CSV

## Requirements

- Python 3.8+
- Webcam
- CUDA-capable GPU (recommended for better performance) 