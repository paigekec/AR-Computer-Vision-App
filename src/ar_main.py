import cv2
import numpy as np
from object_detection import detect_objects

# Function to overlay information on the detected object
def overlay_info(frame, text, position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, position, font, 1, (0, 255, 0), 2, cv2.LINE_AA)

def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Detect objects in the frame
        detections = detect_objects(frame)

        # Overlay information for each detected object
        for (label, (x, y, w, h)) in detections:
            overlay_info(frame, label, (x, y - 10))  # Overlay text above the detected object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle around the object

        # Display the frame with augmented reality information
        cv2.imshow("AR Application", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close the windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
