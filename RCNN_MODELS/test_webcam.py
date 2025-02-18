import cv2

def list_available_cameras(max_tested=10):
    available_cameras = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:  # If the camera opens successfully
            available_cameras.append(i)
        cap.release()
    return available_cameras

# 🔹 List all available webcams
cameras = list_available_cameras()
print(f"Available webcams: {cameras}")
