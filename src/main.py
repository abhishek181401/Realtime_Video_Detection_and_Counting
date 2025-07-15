import cv2
from video_stream import get_video_stream
from vehicle_detection import detect_vehicles


VIDEO_URL = "http://192.168.1.73:8080/video" 

def process_video():
  
    cap = get_video_stream(VIDEO_URL)  

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        processed_frame, vehicle_counts = detect_vehicles(frame)

       
        count_text = f"Cars: {vehicle_counts['car']} | Buses: {vehicle_counts['bus']} | Trucks: {vehicle_counts['truck']} | Motorcycles: {vehicle_counts['motorcycle']}"
        cv2.putText(processed_frame, count_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

       
        cv2.imshow("Vehicle Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()
