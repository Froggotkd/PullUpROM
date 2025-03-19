from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

hasStartedPullUp = False
hasStartedCurl = False

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit(1)

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def calculate_bar(a, b):
    a = np.array(a)
    b = np.array(b)

    bar = (a[1] + b[1]) / 2

    return bar


#Vars for Pull-Up
hasGoneDown = False
counter = 0
fullyExtended = False

#Vars for Curl
counterCurl = 0
hasGoneDownCurlL = False
hasGoneDownCurlR = False
isLeftDone = False

def generate():
    global counter, hasGoneDown, fullyExtended, hasStartedPullUp, hasStartedCurl, counterCurl, isLeftDone, hasGoneDownCurlL, hasGoneDownCurlR

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Convert to RGB (MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = pose.process(rgb_frame)
        landmarks = results.pose_landmarks.landmark

        # Get coordinates
        indexR = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]
        indexL = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y] 

        wristR = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        wristL = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]    

        chin = [landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x, landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y]

        shoulderR = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        shoulderL = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

        elbowR = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        elbowL = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

        #---------------------------------PULL UP LOGIC ------------------------------------
        #--------------------------------------------------------------------------------
        if hasStartedPullUp:
            try:          
                # Calculate bar
                bar = calculate_bar(wristL, wristR)

                # Calculate angles
                angleL = calculate_angle(shoulderL, elbowL, indexL)
                angleR = calculate_angle(shoulderR, elbowR, indexR)

                # Visualize
                cv2.putText(frame, "{:.2f}".format(bar),
                            tuple(np.multiply(wristL, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(frame, "{:.2f}".format(chin[1]),
                            tuple(np.multiply(chin, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(frame, "{:.2f}".format(angleR),
                            tuple(np.multiply(elbowR, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(frame, "{:.2f}".format(angleL),
                            tuple(np.multiply(elbowL, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                if chin[1] > bar and not hasGoneDown:
                        hasGoneDown = True

                if hasGoneDown and angleL > 120 and angleR > 120:
                        fullyExtended = True

                if chin[1] < bar and hasGoneDown and fullyExtended:
                        print(f"AngleL: {angleL}, AngleR: {angleR}")
                        counter += 1
                        print(f"Rep Count: {counter}")

                        hasGoneDown = False
                        fullyExtended = False

            except:
                pass

                # Draw rectangle for rep count
            cv2.rectangle(frame, (0, 0), (225, 73), (245, 117, 16), -1)

                # Draw the bar
            image_height, image_width, _ = frame.shape  
            indexR = [
                int(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x * image_width),
                int(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y * image_height)
            ]
                
            indexL = [
                int(landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x * image_width),
                int(landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y * image_height)
            ]
            indexL = tuple(indexL)
            indexR = tuple(indexR)
            cv2.line(frame, indexL, indexR, (255, 0, 230), 9)

                # Rep data
            cv2.putText(frame, 'REPS', (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, str(counter),
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        #---------------------------------CURL LOGIC ------------------------------------
        #--------------------------------------------------------------------------------
        if hasStartedCurl:
            try:

                # Calculate angles
                angleL_Curl = calculate_angle(shoulderL, elbowL, wristL)
                angleR_Curl = calculate_angle(shoulderR, elbowR, wristR)

                # Visualize
                cv2.putText(frame, "{:.2f}".format(angleR_Curl),
                            tuple(np.multiply(elbowR, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(frame, "{:.2f}".format(angleL_Curl),
                            tuple(np.multiply(elbowL, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                
                if angleL_Curl > 150 and not isLeftDone:  
                    hasGoneDownCurlL = True  

                if angleL_Curl < 40 and hasGoneDownCurlL:  
                    counterCurl += 1  
                    hasGoneDownCurlL = False  
                    isLeftDone = True  
                    print(f"Count: {counterCurl}")  

                if angleR_Curl > 150 and isLeftDone:  
                    hasGoneDownCurlR = True  

                if angleR_Curl < 40 and hasGoneDownCurlR:  
                    counterCurl += 1  
                    hasGoneDownCurlR = False  
                    isLeftDone = False  
                    print(f"Count: {counterCurl}")  


            except:
                pass

                # Draw rectangle for rep count
            cv2.rectangle(frame, (0, 0), (225, 73), (245, 117, 16), -1)

                # Rep data
            cv2.putText(frame, 'REPS', (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, str(counterCurl),
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA) 

        # Render detections
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Encode and send the frame
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if not flag:
            print("Error: Failed to encode frame.")
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

@app.route("/")
def index():
    global hasStartedPullUp 
    hasStartedPullUp = False
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    global hasStartedPullUp, hasStartedCurl
    hasStartedPullUp = False
    hasStartedCurl = False
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                             "Pragma": "no-cache",
                             "Expires": "0"})

@app.route("/startPullUp", methods=['POST'])
def startPullUps():
    global hasStartedPullUp
    hasStartedPullUp = True
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                             "Pragma": "no-cache",
                             "Expires": "0"})

@app.route("/startCurl", methods=['POST'])
def startCurl():
    global hasStartedCurl
    hasStartedCurl = True
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                             "Pragma": "no-cache",
                             "Expires": "0"})

if __name__ == "__main__":
    try:
        app.run(debug=True)
    finally:
        cap.release()