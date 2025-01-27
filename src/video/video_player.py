import cv2
from ultralytics import YOLO

from sort.sort import Sort

class VideoPlayer:

    def __init__(self, video_path):
        self.video_path = video_path

        self.multi_tracker = cv2.legacy.MultiTracker.create()


    def _draw_rectangle(self, frame, x1, y1, x2, y2):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # csrt
    # def track_face

    def play(self):


        cap = cv2.VideoCapture(self.video_path)

        yolo_model = YOLO("best.pt")


        ret, frame = cap.read()

        detect_results = yolo_model(frame)

        for box in detect_results[0].boxes:


            x1, y1, x2, y2 = map(int, box.xyxy[0])

            tracker = cv2.legacy.TrackerCSRT.create()  # 중요: .create() 메서드 사용
            self._draw_rectangle(frame, x1, y1, x2, y2)

            bounding_box = (x1, y1, x2 - x1, y2 - y1)
            self.multi_tracker.add(tracker, frame, bounding_box)


        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break

            success, boxes = self.multi_tracker.update(frame)
            for box in boxes:
                x, y, w, h = map(int, box)
                self._draw_rectangle(frame, x, y, x + w, y + h)

            cv2.imshow('frame', frame)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        
