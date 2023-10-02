import cv2
from lkcomputervision import MediaPipeHandler

# Initialize the MediaPipeHandler for computer vision tasks
mp = MediaPipeHandler()

# Capture video from the default webcam (index 0) using OpenCV
cap = cv2.VideoCapture(0)

# Enter a continuous loop for real-time video processing
while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Check if the frame was read successfully; if not, break out of the loop
    if not ret:
        break

    # Process the frame to detect human body pose using MediaPipe
    result = mp.detectPose(frame)

    # Retrieve the frame with the detected pose landmarks drawn on it
    frame_with_landmarks = result["frame"]

    # Print the detected pose landmarks (for debugging or analysis)
    print(result["landmarks"])

    # Display the frame with the pose landmarks in a window named "Hand Tracking"
    cv2.imshow("Hand Tracking", frame_with_landmarks)

    # Exit the loop when the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture resources (stop capturing video)
cap.release()

# Close any OpenCV windows that were opened during execution
cv2.destroyAllWindows()

