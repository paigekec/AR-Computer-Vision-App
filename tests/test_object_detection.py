import cv2
from src.object_detection import detect_objects

def test_object_detection():
    # Load a sample test image
    test_image = cv2.imread('data/test_image.jpg')
    
    # Perform object detection
    detections = detect_objects(test_image)

    # Check that at least one object is detected
    assert len(detections) > 0
