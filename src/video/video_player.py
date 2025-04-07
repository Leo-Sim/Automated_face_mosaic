import cv2
import numpy as np
import time

from typing import Optional
from ultralytics import YOLO
from obejct_tracker import ObjectTracker
from src.insightface.detector import FaceManager
import matplotlib.pyplot as plt



# from src.deepface.detector import image


# from sort.sort import Sort

class VideoPlayer:

    def __init__(self, video_path):
        self.video_path = video_path
        self.face_detector: Optional[FaceManager] = None


    def set_face_detector(self, face_detector: FaceManager):
        self.face_detector = face_detector

    def _draw_rectangle(self, frame, x1, y1, x2, y2):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # csrt
    # def track_face

    def _get_video_size(self, cap: cv2.VideoCapture):
        """
        Get frame size of the video

        :param cap:
        :return: width, height
        """
        if cap.isOpened():
            ret, frame = cap.read()

            if ret:
                height, width, _ = frame.shape
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return width, height


    def play(self):

        cap = cv2.VideoCapture(self.video_path)
        yolo_model = YOLO("best_small.pt")

        width, height = self._get_video_size(cap)

        tracker = ObjectTracker(width, height)

        detected_list = []

        frame_num = 0
        while (cap.isOpened):
            ret, frame = cap.read()
            frame_num += 1


            if not ret:
                break


            if frame_num % 13 == 12:
                detect_results = yolo_model(frame)
                detected_list = []

                # for box in detect_results[0].boxes:
                    # x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # confidence = float(box.conf[0])
                    #
                    # cordinates = [x1, y1, x2, y2, confidence]

                    # cropped_frame = frame[y1:y2, x1:x2]
                detect_result = self.face_detector.detect_face(frame)

                if detect_result is not None:

                    x1, y1, x2, y2 = np.array(detect_result.bbox)


                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)



                    # self._draw_rectangle(frame, x1, y1, x2, y2)

                # Tracker
            # SORT로 객체 추적

            # print("detected_list : ", len(detected_list))

            if len(detected_list) > 0:
                track_bbs_ids = tracker.update(frame_num, np.array(detected_list))
            # else:
            #     track_bbs_ids = tracker.update(frame_num, np.empty(0, 5))

            # print("track_bbs_ids : ", len(track_bbs_ids))

            video_frame_width = frame.shape[1]  # 프레임의 너비 (640)
            video_frame_height = frame.shape[0]  # 프레임의 높이 (360)

            if len(detected_list) > 0:
                # 프레임에 추적 결과 표시
                for track in track_bbs_ids:
                    x1, y1, x2, y2, obj_id = track.astype(int)

                    # used min to prevent get width and height outside of frame.
                    x1, x2 = max(0, x1), min(video_frame_width, x2)
                    y1, y2 = max(0, y1), min(video_frame_height, y2)

                    w = x2 - x1
                    h = y2 - y1

                    # Apply mosaic in ROI
                    roi = frame[y1:y2, x1:x2]

                    intensity = 10

                    if w < 10:
                        intensity = w % 10
                    if h < 10:
                        intensity = h % 10

                    roi = cv2.resize(roi, (w // intensity, h // intensity))
                    roi = cv2.resize(roi, (w, h))

                    frame[y1:y2, x1:x2] = roi

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('frame', frame)

            if cv2.waitKey(60) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


        
