import cv2
import numpy as np
import time

from typing import Optional
from ultralytics import YOLO
from obejct_tracker import ObjectTracker
from src.insightface.detector import FaceManager
from detection_info import FaceDetectionInfo, MosaicInfo

import matplotlib.pyplot as plt

from src.video.detection_info import DetectionInfo


# from src.deepface.detector import image


# from sort.sort import Sort

class VideoPlayer:

    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path
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

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(self.output_path, fourcc, 30.0, (720, 1280))

        # These variables are for comparing id that is detected in previous frame
        # but not in current frame
        detected_id_in_prev_frame = set()
        detected_id_in_cur_frame = set()

        frame_num = 0
        while (cap.isOpened):
            ret, frame = cap.read()
            frame_num += 1

            # if frame_num == 200:
            #
            #     break


            if not ret:
                break


            if frame_num == 1 or frame_num % 10 == 9:
                detect_results = yolo_model(frame)
                detected_list = []




                print("detected_id_in_prev_frame : ", detected_id_in_prev_frame)

                for box in detect_results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])

                    mosaicInfo: MosaicInfo = MosaicInfo([x1, y1, x2, y2], confidence, True)
                    detected_list.append(mosaicInfo)

                face_detection_result: FaceDetectionInfo = self.face_detector.detect_face(frame)

                if face_detection_result is not None:

                    # find the face that has closest distance to target face so that the model does not apply mosaic to it
                    # if target face is detected and the number of faces by yolo is 1,  no need to apply mosaic.
                    if len(detected_list) == 1:
                        detected_list[0].need_mosaic = False

                    elif len(detected_list) > 1:
                        min_distance = float('inf')
                        min_detection_id = 0

                        # Find the closest face to the face detected by insightface
                        for yolo_detection in detected_list:

                            distance = DetectionInfo.get_distance_of_two_points(face_detection_result, yolo_detection)
                            if distance < min_distance:
                                min_distance = distance
                                min_detection_id = yolo_detection.id

                        # change attribute not to apply mosaic
                        for yolo_detection in detected_list:
                            if yolo_detection.id == min_detection_id:
                                yolo_detection.need_mosaic = False
                                break

            # apply mosaic
            if len(detected_list) > 0:

                yolo_detected_infos = []

                for yolo_detection in detected_list:

                    if yolo_detection.need_mosaic is False:
                        continue

                    x1, y1, x2, y2 = yolo_detection.bbox
                    confidence = yolo_detection.confidence

                    tmp = [x1, y1, x2, y2, confidence]
                    yolo_detected_infos.append(tmp)


                track_bbs_ids = tracker.update(frame_num, np.array(yolo_detected_infos))

                video_frame_width = frame.shape[1]
                video_frame_height = frame.shape[0]


                for track in track_bbs_ids:
                    x1, y1, x2, y2, obj_id = track.astype(int)

                    detected_id_in_cur_frame.clear()
                    detected_id_in_cur_frame.add(obj_id)


                    # used min to prevent get width and height outside of frame.
                    x1, x2 = max(0, x1), min(video_frame_width, x2)
                    y1, y2 = max(0, y1), min(video_frame_height, y2)

                    w = x2 - x1
                    h = y2 - y1

                    # Apply mosaic
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


                need_estimation_id = detected_id_in_prev_frame - detected_id_in_cur_frame

                # if(len(need_estimation_id) > 0):
                #     for id in need_estimation_id:
                #         e = tracker.get_estimated_next_position(id, frame_num)
                #
                #         estimated_x1 = e.x1
                #         estimated_y1 = e.y1
                #         estimated_x2 = e.x2
                #         estimated_y2 = e.y2
                #
                #
                #         cv2.rectangle(frame, (estimated_x1, estimated_y1), (estimated_x2, estimated_y2), (255, 0, 0), 2)
                #         # cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                #         #             2)


                detected_id_in_prev_frame.clear()
                detected_id_in_prev_frame.update(detected_id_in_cur_frame)

            resized_frame = cv2.resize(frame, (720, 1280))

            # ③ 저장 (VideoWriter에 프레임 추가)
            out.write(resized_frame)


            cv2.imshow('frame', resized_frame)

            if cv2.waitKey(60) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()


        
