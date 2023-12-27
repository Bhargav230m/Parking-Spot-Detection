import cv2
from Classes.utils import ParkingSpotClassifier

def demonstration():
    rect_width, rect_height = 105, 45
    car_park_positions_path = "../data/source/positions"
    video_path = "../data/source/carPark.mp4"
    classifier = ParkingSpotClassifier(car_park_positions_path, rect_width, rect_height)
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        new_frame = classifier.implement_process(frame)
        denoted_image = classifier.classify(frame, new_frame)
        cv2.imshow("Parking Spot Detection", denoted_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    demonstration()
