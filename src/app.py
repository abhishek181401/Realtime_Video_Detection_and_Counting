import cv2
import streamlit as st
from video_stream import get_video_stream
from vehicle_detection import detect_vehicles


st.title("Real-Time Vehicle Detection")

video_url = st.text_input("Enter your mobile stream URL:", "http://192.168.1.73:8080/video")

vehicle_count_container = st.sidebar.empty() 

if st.button("Start Detection"):
    stframe = st.empty()  
    cap = get_video_stream(video_url)

    if not cap.isOpened():
        st.error("Could not open  mobile camera stream.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read video frame.")
                break

            processed_frame, vehicle_counts = detect_vehicles(frame)

            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            vehicle_count_container.subheader("Live Vehicle Count")

            for vehicle_type, count in vehicle_counts.items():
                vehicle_count_container.write(f"{vehicle_type.capitalize()}: {count}")

            stframe.image(processed_frame, channels="RGB", use_column_width=True)
