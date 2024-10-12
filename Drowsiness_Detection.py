

# from flask import Flask, render_template, Response, request, jsonify
# import threading
# import cv2
# import imutils
# from scipy.spatial import distance
# from imutils import face_utils
# from pygame import mixer
# import dlib

# app = Flask(__name__)

# # Initialize mixer for alarm sound
# mixer.init()
# mixer.music.load("music.wav")

# # Initialize Dlib's face detector and landmark predictor
# detect = dlib.get_frontal_face_detector()
# predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
# (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# # Parameters
# thresh = 0.25
# frame_check = 20
# flag = 0
# detection_on = False
# lock = threading.Lock()

# cap = None

# def eye_aspect_ratio(eye):
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#     ear = (A + B) / (2.0 * C)
#     return ear

# def detect_drowsiness():
#     global cap, flag, detection_on
#     while detection_on:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = imutils.resize(frame, width=450)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         subjects = detect(gray, 0)
#         for subject in subjects:
#             shape = predict(gray, subject)
#             shape = face_utils.shape_to_np(shape)
#             leftEye = shape[lStart:lEnd]
#             rightEye = shape[rStart:rEnd]
#             leftEAR = eye_aspect_ratio(leftEye)
#             rightEAR = eye_aspect_ratio(rightEye)
#             ear = (leftEAR + rightEAR) / 2.0
#             leftEyeHull = cv2.convexHull(leftEye)
#             rightEyeHull = cv2.convexHull(rightEye)
#             print(leftEyeHull)
#             cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
#             cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
#             if ear < thresh:
#                 flag += 1
#                 if flag >= frame_check:
#                     cv2.putText(frame, "****************ALERT!****************", (10, 30),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                     cv2.putText(frame, "****************ALERT!****************", (10, 325),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                     mixer.music.play()
#             else:
#                 flag = 0
#         ret, jpeg = cv2.imencode('.jpg', frame)
#         frame = jpeg.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#     cap.release()
#     cap = None

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     global cap, detection_on
#     if detection_on and cap is not None:
#         return Response(detect_drowsiness(), mimetype='multipart/x-mixed-replace; boundary=frame')
#     else:
#         return Response("")

# @app.route('/start_detection', methods=['POST'])
# def start_detection():
#     global detection_on, cap
#     with lock:
#         if not detection_on:
#             detection_on = True
#             cap = cv2.VideoCapture(0)
#     return jsonify({"status": "Detection started"})

# @app.route('/stop_detection', methods=['POST'])
# def stop_detection():
#     global detection_on, cap
#     with lock:
#         detection_on = False
#         if cap is not None:
#             cap.release()
#             cap = None
#     return jsonify({"status": "Detection stopped"})

# @app.route('/detection_status', methods=['GET'])
# def detection_status():
#     global detection_on
#     return jsonify({"detection_on": detection_on})

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, Response, request, jsonify
import threading
import cv2
import imutils
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import dlib

app = Flask(__name__)

mixer.init()
mixer.music.load("music.wav")

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"] #imu

# Param
thresh = 0.25
frame_check = 20
flag = 0
detection_on = False
lock = threading.Lock()

cap = None

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_drowsiness():
    global cap, flag, detection_on
    while detection_on:
        ret, frame = cap.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)
        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            if ear < thresh:
                flag += 1
                if flag >= frame_check:
                    cv2.putText(frame, "*****ALERT!*****", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "*****ALERT!*****", (10, 325),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    mixer.music.play()
            else:
                flag = 0
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    cap.release()
    cap = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global cap, detection_on
    if detection_on and cap is not None:
        return Response(detect_drowsiness(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response("")

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global detection_on, cap
    with lock:
        if not detection_on:
            detection_on = True
            cap = cv2.VideoCapture(0)
    return jsonify({"status": "Detection started"})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detection_on, cap
    with lock:
        detection_on = False
        if cap is not None:
            cap.release()
            cap = None
    return jsonify({"status": "Detection stopped"})

@app.route('/detection_status', methods=['GET'])
def detection_status():
    global detection_on
    return jsonify({"detection_on": detection_on})

if __name__ == '__main__':
    app.run(debug=True)