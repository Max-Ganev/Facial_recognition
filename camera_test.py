import cv2
import time

print("Trying to open camera...")
start = time.time()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print(f"VideoCapture() call took {time.time() - start:.1f}s")
print("isOpened():", cap.isOpened())

if cap.isOpened():
    ret, frame = cap.read()
    print("Got a frame:", ret, frame.shape if ret else None)
else:
    print("Camera failed to open.")

cap.release()