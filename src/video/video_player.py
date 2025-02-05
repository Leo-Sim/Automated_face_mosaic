import cv2
import numpy as np

from ultralytics import YOLO
from obejct_tracker import ObjectTracker


# from sort.sort import Sort

class VideoPlayer:

    def __init__(self, video_path):
        self.video_path = video_path

    def _draw_rectangle(self, frame, x1, y1, x2, y2):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # csrt
    # def track_face

    def play(self):

        cap = cv2.VideoCapture(self.video_path)
        yolo_model = YOLO("best.pt")

        frame_num = 0

        tracker = ObjectTracker()

        frame_num = 0
        while (cap.isOpened):
            ret, frame = cap.read()
            frame_num += 1

            if not ret:
                break

            frame_num += 1
            detected_list = []
            detect_results = yolo_model(frame)





            for box in detect_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                cordinates = [x1, y1, x2, y2, confidence]



                detected_list.append(cordinates)

                # self._draw_rectangle(frame, x1, y1, x2, y2)

            # Tracker
                # SORT로 객체 추적
                if len(cordinates) > 0:
                    track_bbs_ids = tracker.update(frame_num, np.array(detected_list))
                else:
                    track_bbs_ids = tracker.update(frame_num, np.empty(0, 5))



                # 프레임에 추적 결과 표시
                for track in track_bbs_ids:
                    x1, y1, x2, y2, obj_id = track.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('frame', frame)

            if cv2.waitKey(60) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


        
