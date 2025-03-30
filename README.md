# Face Recognition Attendance System

A real-time face recognition system for automated attendance tracking using YOLOv8 and Streamlit.

## Features

- 📷 Real-time face detection and recognition
- 📊 Automatic attendance tracking
- 💾 CSV export functionality
- 🎯 High accuracy using YOLOv8 model
- 🖥️ User-friendly Streamlit interface

## Screenshots

### Live Detection Interface
![Live Detection](assets/Screenshot%20from%202025-03-30%2012-37-14.png)
*Real-time face detection and recognition in action*

### Attendance Records
![Attendance Records](assets/Screenshot%20from%202025-03-30%2012-40-19.png)
*View and export attendance records*

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/crjanb/Face-Detection-and-Attendance-System.git
cd Face-Detection-and-Attendance-System
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run appcam_b.py
```

## Requirements

- Python 3.8+
- Webcam
- CUDA-compatible GPU (recommended for better performance)

## Usage

1. Start the application and navigate to the Live Detection page
2. Click "Start Webcam" to begin face detection
3. The system will automatically record attendance when a face is recognized
4. View attendance records in the Attendance Records page
5. Download attendance data as CSV when needed

## Model Information

The system uses YOLOv8, trained specifically for face detection and recognition. The model file (`largemodel3.pt`) should be present in the root directory of the project.
