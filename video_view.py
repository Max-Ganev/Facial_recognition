import cv2
import numpy as np
import os
from PIL import Image, ImageTk
import threading

_streaming = False
_cap = None

def start_video_stream(app_instance):
    global _streaming, _cap
    if _streaming:
        return
        
    xml_name = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(xml_name):
        import urllib.request
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{xml_name}"
        try:
            urllib.request.urlretrieve(url, xml_name)
        except Exception:
            pass

    face_cascade = cv2.CascadeClassifier(xml_name)
    if face_cascade.empty():
        app_instance.status_label.config(text="Error: Could not load CascadeClassifier.")
        return

    # Open camera in a background thread so it doesn't freeze the GUI window
    def camera_loop():
        global _streaming, _cap
        _cap = cv2.VideoCapture(0)
        
        if not _cap.isOpened():
            app_instance.root.after(0, lambda: app_instance.status_label.config(text="Error: Webcam not detected."))
            return

        _streaming = True
        app_instance.root.after(0, lambda: app_instance.status_label.config(text="Webcam Active. Scanning..."))

        while _streaming:
            ret, frame = _cap.read()
            if not ret:
                break
                
            try:
                from app import identify
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                    
                for (x, y, w, h) in faces:
                    face_roi = gray_frame[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_roi, (100, 100))
                    cv2.imwrite("test.png", face_resized)
                    
                    try:
                        matched_name = identify("test.png")
                    except Exception:
                        matched_name = "Processing..."

                    box_color = (0, 0, 255) if "Unknown" in matched_name or "Access Denied" in matched_name else (0, 255, 0)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
                    cv2.putText(
                        frame, matched_name, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2, cv2.LINE_AA
                    )
            except Exception:
                pass

            # Convert to RGB for Tkinter display
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            img = img.resize((480, 360))
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Safely update GUI elements from the main thread loop
            def update_ui(token=imgtk):
                if _streaming:
                    app_instance.video_label.imgtk = token
                    app_instance.video_label.configure(image=token)
            
            app_instance.root.after(0, update_ui)

        if _cap:
            _cap.release()

    threading.Thread(target=camera_loop, daemon=True).start()

def stop_video_stream():
    global _streaming, _cap
    _streaming = False
    if _cap:
        _cap.release()