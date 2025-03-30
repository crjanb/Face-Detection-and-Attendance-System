import streamlit as st
import cv2
import torch
from ultralytics import YOLO
import csv
from datetime import datetime
import os
import pandas as pd
import numpy as np
import time

# Set Streamlit page config
st.set_page_config(
    page_title="Face Recognization Attendance System",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📜"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa !important;
        padding: 1rem !important;
    }
    
    .stButton button {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        color: white !important;
        border-radius: 25px;
        padding: 12px 30px;
        font-size: 1rem;
        transition: all 0.3s ease;
        border: none;
        width: 100%;
        margin: 0.5rem 0;
    }
    
    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(116, 79, 168, 0.3);
    }
    
    .card {
        padding: 1.5rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .detection-frame {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #ffffff);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .filter-container {
        background: #ffffff;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .webcam-controls-top {
        margin-bottom: 1rem;
    }
    
    .webcam-controls-bottom {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown("<h2 style='color: #667eea; margin-bottom: 2rem;'>Navigation</h2>", unsafe_allow_html=True)
    page = st.radio("", ["📷 Live Detection", "📊 Attendance Records"], label_visibility='hidden')

# Load model
try:
    model = YOLO("./largemodel3.pt")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# CSV setup
csv_file = "./detection_log.csv"
file_exists = os.path.isfile(csv_file)
if not file_exists:
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Label", "Date", "Timestamp"])

if page == "📷 Live Detection":
    # Main Header
    st.markdown('<div class="main-header">Face Recognization Attendance System</div>', unsafe_allow_html=True)
    
    # Main layout columns
    col1, col2 = st.columns([3, 1], gap="large")
    
    with col1:
        # Webcam feed section
        st.markdown("### Live Camera Feed")
        
        # Start Webcam button at the top
        st.markdown('<div class="webcam-controls-top">', unsafe_allow_html=True)
        start_webcam = st.button("🚀 Start Webcam", key="start_webcam")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Webcam feed placeholder
        with st.container():
            stframe = st.empty()
            st.markdown('<div class="detection-frame"></div>', unsafe_allow_html=True)
        
        # Stop Webcam button at the bottom
        st.markdown('<div class="webcam-controls-bottom">', unsafe_allow_html=True)
        stop_webcam = st.button("🛑 Stop Webcam", key="stop_webcam")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Settings panel
        with st.container():
            st.markdown("### ⚙️ Configuration")
            with st.form("settings_form"):
                confidence_threshold = st.slider(
                    "Confidence Threshold",
                    0.1, 1.0, 0.5,
                    help="Adjust the minimum confidence level for detections"
                )
                iou_threshold = st.slider(
                    "IoU Threshold",
                    0.1, 1.0, 0.5,
                    help="Adjust the Intersection over Union threshold"
                )
                st.form_submit_button("Save Settings")
        
        # Status section
        st.markdown("### 📈 System Status")
        status_placeholder = st.empty()

    # Detection logic
    recorded_labels = set()
    
    if start_webcam:
        status_placeholder.markdown("""
            <div class='card'>
                <h4 style='color: #4CAF50;'>✅ Detection Active</h4>
                <p>System is monitoring for recognized faces...</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Initialize video capture
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access the webcam.")
        else:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video frame.")
                    break

                # Run detection
                results = model.predict(frame, conf=confidence_threshold, iou=iou_threshold)
                
                # Process detections
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_index = int(box.cls[0])
                        cls_name = model.names[cls_index] if conf >= confidence_threshold else "Unknown"

                        color = (0, 255, 0) if cls_name != "Unknown" else (255, 0, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{cls_name} {conf:.2f}" if cls_name != "Unknown" else "Unknown"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                        if cls_name != "Unknown":
                            current_time = datetime.now()
                            date = current_time.strftime("%Y-%m-%d")
                            timestamp = current_time.strftime("%H:%M:%S")
                            
                            # Daily unique recording
                            daily_key = f"{cls_name}_{date}"
                            if daily_key not in recorded_labels:
                                with open(csv_file, mode='a', newline='') as file:
                                    writer = csv.writer(file)
                                    writer.writerow([cls_name, date, timestamp])
                                recorded_labels.add(daily_key)
                                status_placeholder.markdown(f"""
                                    <div class='card'>
                                        <h4 style='color: #667eea;'>✅ New Detection</h4>
                                        <p>👤 <strong>{cls_name}</strong><br>
                                        📅 {date}<br>
                                        ⏰ {timestamp}</p>
                                    </div>
                                """, unsafe_allow_html=True)

                # Display frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(rgb_frame, channels="RGB", use_column_width=True)
                
                # Add a small delay to prevent overwhelming the UI
                time.sleep(0.1)

            cap.release()

    if stop_webcam:
        status_placeholder.markdown("""
            <div class='card'>
                <h4 style='color: #ff4b4b;'>🛑 Detection Stopped</h4>
                <p>System monitoring is currently inactive</p>
            </div>
        """, unsafe_allow_html=True)

elif page == "📊 Attendance Records":
    st.markdown('<div class="main-header">Attendance Records</div>', unsafe_allow_html=True)
    
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            if not df.empty:
                # Filters section
                with st.container():
                    st.markdown("### 🔍 Filters")
                    with st.expander("Filter Options", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            selected_date = st.selectbox(
                                "Select Date",
                                ["All"] + sorted(df['Date'].unique(), reverse=True)
                            )
                        with col2:
                            selected_label = st.selectbox(
                                "Select Person",
                                ["All"] + sorted(df['Label'].unique())
                            )

                # Metrics cards
                st.markdown("### 📊 Statistics")
                cols = st.columns(3)
                with cols[0]:
                    st.markdown("""
                        <div class='metric-card'>
                            <h3 style='color: #667eea;'>📋 Total Entries</h3>
                            <p style='font-size: 2rem;'>{}</p>
                        </div>
                    """.format(len(df)), unsafe_allow_html=True)
                with cols[1]:
                    st.markdown("""
                        <div class='metric-card'>
                            <h3 style='color: #667eea;'>👥 Unique Persons</h3>
                            <p style='font-size: 2rem;'>{}</p>
                        </div>
                    """.format(df['Label'].nunique()), unsafe_allow_html=True)
                with cols[2]:
                    st.markdown("""
                        <div class='metric-card'>
                            <h3 style='color: #667eea;'>🕒 Last Update</h3>
                            <p style='font-size: 1.2rem;'>{}</p>
                        </div>
                    """.format(df['Timestamp'].max()), unsafe_allow_html=True)

                # Data display
                st.markdown("### 📋 Detailed Records")
                filtered_df = df
                if selected_date != "All":
                    filtered_df = filtered_df[filtered_df['Date'] == selected_date]
                if selected_label != "All":
                    filtered_df = filtered_df[filtered_df['Label'] == selected_label]

                st.dataframe(
                    filtered_df.style
                        .applymap(lambda x: "color: #667eea" if x in df['Label'].unique() else "")
                        .set_properties(**{'background-color': '#f8f9fa'}),
                    use_container_width=True,
                    height=400
                )
                
                # Export section
                st.markdown("### 📤 Data Export")
                export_col = st.columns(3)
                with export_col[0]:
                    st.download_button(
                        "Download CSV",
                        filtered_df.to_csv(index=False),
                        "attendance_records.csv",
                        help="Download filtered results as CSV"
                    )
                
            else:
                st.markdown("""
                    <div class='card'>
                        <h4 style='color: #667eea;'>📭 No Records Found</h4>
                        <p>No attendance records have been logged yet</p>
                    </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading records: {str(e)}")
    else:
        st.markdown("""
            <div class='card'>
                <h4 style='color: #ff4b4b;'>⚠️ No Data File</h4>
                <p>No attendance records file found</p>
            </div>
        """, unsafe_allow_html=True)