import cv2
import pickle
import numpy as np


class ParkingSpotClassifier:
    def __init__(self, positions_path: str, rect_width: int = 107, rect_height: int = 48):
        self.positions = self._read_positions(positions_path)
        self.rect_height = rect_height
        self.rect_width = rect_width

    def _read_positions(self, positions_path: str) -> list:
        positions = []
        try:
            with open(positions_path, 'rb') as f:
                positions = pickle.load(f)
        except Exception as e:
            print(f"Error: {e}")
        return positions

    def classify(self, image: np.ndarray, processed_image: np.ndarray, threshold: int = 900) -> np.ndarray:
        empty_spots = 0
        for x, y in self.positions:
            col_start, col_stop = x, x + self.rect_width
            row_start, row_stop = y, y + self.rect_height

            crop = processed_image[row_start:row_stop, col_start:col_stop]
            count = cv2.countNonZero(crop)

            empty_spots, color, thick = [empty_spots + 1, (0, 255, 0), 5] if count < threshold else [empty_spots,
                                                                                                     (0, 0, 255), 2]

            start_point, stop_point = (x, y), (x + self.rect_width, y + self.rect_height)
            cv2.rectangle(image, start_point, stop_point, color, thick)

        cv2.rectangle(image, (55, 35), (260, 78), (180, 0, 180), -1)

        ratio_text = f'Spots Free: {empty_spots}/{len(self.positions)}'
        cv2.putText(image, ratio_text, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        return image

    def implement_process(self, image: np.ndarray) -> np.ndarray:
        kernel_size = np.ones((3, 5), np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 1)

        thresholded = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
        blur = cv2.medianBlur(thresholded, 5)

        dilate = cv2.dilate(blur, kernel_size, iterations=1)

        return dilate


class coordinateFinder:
    def __init__(self, rect_width: int = 107, rect_height: int = 48, positions_path: str = "../data/source/CarParkPos"):
        self.rect_width = rect_width
        self.rect_height = rect_height
        self.positions_path = positions_path
        self.positions = []

    def read_positions(self) -> list:
        try:
            with open(self.positions_path, 'rb') as f:
                self.positions = pickle.load(f)
        except Exception as e:
            print(f"Error: {e}")

        return self.positions

    def mouse_click(self, events: int, x: int, y: int, flags: int, params: int):
        if events == cv2.EVENT_LBUTTONDOWN:
            self.positions.append((x, y))

        if events == cv2.EVENT_MBUTTONDOWN:
            for index, pos in enumerate(self.positions):
                x1, y1 = pos
                is_x_in_range = x1 <= x <= x1 + self.rect_width
                is_y_in_range = y1 <= y <= y1 + self.rect_height

                if is_x_in_range and is_y_in_range:
                    self.positions.pop(index)

        with open(self.positions_path, 'wb') as f:
            pickle.dump(self.positions, f)
