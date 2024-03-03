import os
from flask import Flask, request, jsonify, render_template,Response
from src.pipelines.prediction_pipeline import CustomData,PredictPipline
import cv2
import mediapipe as mp
import time



app = Flask(__name__)
cap = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
output_file = 'body_size_measurement.txt'

def measure_distance(landmark1, landmark2):
    # Calculate the Euclidean distance between two 2D points
    return ((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2) ** 0.5


def calculate_measurements(results):
    measurements = {}

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Measure distance between left shoulder and right shoulder
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        measurements['Shoulder Length'] = measure_distance(left_shoulder, right_shoulder)

        # Measure distance between left hip and right hip
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        measurements['Hip Length'] = measure_distance(left_hip, right_hip)

        # Measure distance between left shoulder and left hip
        measurements['Chest Length'] = measure_distance(left_shoulder, left_hip)

        # Measure distance between right shoulder and right hip
        measurements['Waist Length'] = measure_distance(right_shoulder, right_hip)

        # Measure distance between left shoulder and left hip
        measurements['Shoulder to Waist'] = measure_distance(left_shoulder, left_hip)

    return measurements

def generate_frames(duration=10):
    start_time = time.time()
    while time.time() - start_time < duration:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to get landmarks
        results = pose.process(rgb_frame)

        # Draw landmarks on the frame
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Calculate measurements
            measurements = calculate_measurements(results)
            
            # Display the measured distances on the frame
            for idx, (measurement, value) in enumerate(measurements.items()):
                cv2.putText(frame, f"{measurement}: {value:.2f} px", (10, 30 + idx * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            for i in enumerate(measurements.items()):
                with open(output_file, 'w') as f:
                    f.write(f"{measurement}: {value:.2f} cm")

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in the response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




picFolder = os.path.join('static', 'pics')

app.config['UPLOAD_FOLDER'] = picFolder

@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            Gender=int(request.form.get('Gender')),
            Age =int(request.form.get('Age')),
            shoulder =int(request.form.get('ShoulderWidth')),
            chest =int(request.form.get('ChestWidth ')),
            waist=int(request.form.get('Waist ')),
            hips =int(request.form.get('Hips ')),
            shoulder_to_waist =int(request.form.get('ShoulderToWaist ')),
        )


        final_data = data.get_data_as_data_frame()
        predict_pipline = PredictPipline()
        pred = predict_pipline.predict(final_data)
        result = pred

        imageList1 = os.listdir('static/vshape')
        imagelist1 = ['vshape/1.jpeg','vshape/2.png','vshape/3.jpeg','vshape/4.jpg']
        
        imageList2 = os.listdir('static/Rectangular')
        imagelist2 = ['Rectangular/1.jpg','Rectangular/2.jpg','Rectangular/3.jpg','Rectangular/4.png']

        imageList3 = os.listdir('static/Hourglass')
        imagelist3 = ['Hourglass/1.png','Hourglass/2.jpg','Hourglass/3.jpg','Hourglass/4.png']

        imageList4 = os.listdir('static/Pear')
        imagelist4 = ['Pear/1.jpg','Pear/2.png','Pear/3.png','Pear/4.jpg']

        imageList5 = os.listdir('static/Triangle')
        imagelist5 = ['Triangle/1.png','Triangle/2.png','Triangle/3.jpg','Triangle/4.png']

        if result == "V-shape":
            return render_template("Results1.html",imagelist=imagelist1)
        elif result == "Rectangular":
            return render_template("Results2.html", imagelist=imagelist2)
        elif result == "Hourglass":
            return render_template("Results3.html", imagelist=imagelist3)
        elif result == "Pear":
            return render_template("Results4.html",imagelist=imagelist4)
        elif result == "Triangle":
            return render_template("Results5.html", imagelist=imagelist5)

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

# http://127.0.0.1:5000 