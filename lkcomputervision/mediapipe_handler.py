import cv2
import mediapipe as mp

class MediaPipeHandler:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh

        self.hands = self.mp_hands.Hands()
        self.pose = self.mp_pose.Pose()
        self.__face_detection = None
        self.face_mesh = self.mp_face_mesh.FaceMesh()

    def trackHands(self, frame, draw=True):
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hand_land_mark = {}

        if results.multi_hand_landmarks:
            for idx, landmarks in enumerate(results.multi_hand_landmarks):
                landMarks = {}
                for point, landmark in enumerate(landmarks.landmark):
                    x, y, z = landmark.x, landmark.y, landmark.z
                    landMarks[point] = {"x": x, "y": y, "z": z}
                hand_land_mark = landMarks
                if draw:
                    self.mp_drawing.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
        return {"frame": frame, "landmarks": hand_land_mark}
    def detectFace(self, frame, draw=True, min_confidence_of_detection=0.5):

        self.__face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=min_confidence_of_detection)
        with self.mp_face_detection.FaceDetection(min_detection_confidence=min_confidence_of_detection) as face_detection:
            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            faceLms = {}

            if results.detections:
                for i, detection in enumerate(results.detections):
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = bboxC.xmin, bboxC.ymin, bboxC.width, bboxC.height
                    faceLms[i] = {"x": x, "y": y, "width": w, "height": h}
                    if draw:
                        self.mp_drawing.draw_detection(frame, detection)
            return {"frame": frame, "landmarks": faceLms}

    def faceMesh(self, frame, draw=True):
        with self.mp_face_mesh.FaceMesh() as face_mesh:
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
                        self.mp_drawing.draw_landmarks(frame, face_landmarks, self.mp_face_mesh.FACEMESH_TESSELATION)
            return {"frame": frame, "landmarks": face_landmarks_dict}

    def detectPose(self, frame, draw=True):
        with self.mp_pose.Pose() as pose:
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pose_landmarks_dict = {}

            if results.pose_landmarks:
                for landmark_id, landmark in enumerate(results.pose_landmarks.landmark):
                    x, y, z = landmark.x, landmark.y, landmark.z
                    pose_landmarks_dict[landmark_id] = {"x": x, "y": y, "z": z}
                if draw:
                    self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            return {"frame": frame, "landmarks": pose_landmarks_dict}

    def detectAll(self, frame):
        handTracker = self.trackHands(frame)
        faceDetector = self.detectFace(frame)
        facemesh = self.faceMesh(frame)
        bodypose = self.detectPose(frame)
        landMarks = {
            "handTracking": handTracker["landmarks"],
            "detectFace": faceDetector["landmarks"],
            "faceMesh": facemesh["landmarks"],
            "detectPose": bodypose["landmarks"]
        }
        return landMarks
