import cv2
from lkcomputervision.mediapipe_handler import faceMesh, trackHands, detectFace, detectPose, detectAll

# Capture video from the webcam (you can also specify a video file path)
cap = cv2.VideoCapture(0)  # 0 represents the default webcam

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    if not ret:
        break

    # Process the frame to track hands
    # result = trackHands(frame)
    # result = detectFace(frame)
    # result = detectPose(frame)
    # result = faceMesh(frame)
    result = detectAll(frame)

    # Retrieve the frame with hand landmarks drawn on it for individual functions
    # frame_with_landmarks = result["frame"]
    # print(frame_with_landmarks)
    
    # for detectAll
    for i in result.keys():
        print(f'{i}', end="")
        print(result[i]["landmarks"])
        cv2.imshow("Hand Tracking", result[i]["frame"])
        
    # Display the frame with landmarks
    # cv2.imshow("Hand Tracking", frame_with_landmarks)

    # Exit the loop when the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
