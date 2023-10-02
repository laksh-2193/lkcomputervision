import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
def faceMesh(frame, draw=True):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh() as face_mesh:
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face_landmarks_dict = {}
        if results.multi_face_landmarks:
            for i, face_landmarks in enumerate(results.multi_face_landmarks):
                face_landmarks_list = []
                for landmark_id, landmark in enumerate(face_landmarks.landmark):
                    x, y, z = landmark.x, landmark.y, landmark.z
                    face_landmarks_list.append({"x": x, "y": y, "z": z})
                face_landmarks_dict[i] = face_landmarks_list
                if draw:
                    mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
        return {"frame": frame, "landmarks": face_landmarks_dict}
    

def trackHands(frame, draw=True):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    hand_land_mark = {}
    if results.multi_hand_landmarks:
        for idx, landmarks in enumerate(results.multi_hand_landmarks):
            landMarks = {}
            for point, landmark in enumerate(landmarks.landmark):
                x, y, z = landmark.x, landmark.y, landmark.z
                landMarks[point] = {"x": x, "y": y, "z": z}
            hand_land_mark = landMarks
            if draw:
                mp_drawing.draw_landmarks(frame, landmarks,mp_hands.HAND_CONNECTIONS)
    return {"frame": frame, "landmarks": hand_land_mark}

def detectFace(frame, draw=True):
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        faceLms = {}
        if results.detections:
            for i, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = bboxC.xmin, bboxC.ymin, bboxC.width, bboxC.height
                faceLms[i] = {"x": x, "y": y, "width": w, "height": h}
                if draw:
                    mp_drawing.draw_detection(frame, detection)
        return {"frame": frame, "landmarks": faceLms}
    
def detectPose(frame, draw=True):
    mp_pose = mp.solutions.pose
    with mp_pose.Pose() as pose:
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pose_landmarks_dict = {}
        if results.pose_landmarks:
            for landmark_id, landmark in enumerate(results.pose_landmarks.landmark):
                x, y, z = landmark.x, landmark.y, landmark.z
                pose_landmarks_dict[landmark_id] = {"x": x, "y": y, "z": z}
            if draw:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        return {"frame": frame, "landmarks": pose_landmarks_dict}
    

def detectAll(frame):
    handTracker = trackHands(frame)
    faceDetector = detectFace(frame)
    facemesh = faceMesh(frame)
    bodypose = detectPose(frame)
    landMarks = {
        "handTracking": handTracker,
        "detectFace": faceDetector,
        "faceMesh": facemesh,
        "detectPose": bodypose
    }
    return landMarks