from flask import Flask, render_template, Response, jsonify
import cv2
import torch

app = Flask(__name__)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # or yolov5m, yolov5l, yolov5x

# Global flag to control the camera
camera_on = False

def generate_frames():
    cap = cv2.VideoCapture(0)  # Initialize webcam
    while camera_on:  # Only read frames when the camera is on
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb_frame)  # Perform detection
        
        # Render results on frame
        result_img = results.render()[0]  # YOLOv5 renders results directly
        result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)  # Convert back to BGR
        
        # Encode frame to bytes for Flask response
        _, buffer = cv2.imencode('.jpg', result_bgr)
        frame_bytes = buffer.tobytes()
        
        # Yield the frame as part of a multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream video feed."""
    global camera_on
    if camera_on:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response("Camera is off", mimetype='text/plain')

@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    """Toggle the camera on/off."""
    global camera_on
    camera_on = not camera_on
    return jsonify({'camera_on': camera_on})

if __name__ == "__main__":
    app.run(debug=True)
