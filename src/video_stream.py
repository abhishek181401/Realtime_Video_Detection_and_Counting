import cv2

def get_video_stream(video_url):
    """
    Returns:
       OpenCV video capture object.
    """
    cap = cv2.VideoCapture(video_url)
    
    if not cap.isOpened():
        print(f"could not open video stream from {video_url}")
    
    return cap
