
from sort.sort import Sort
import numpy as np


# 객체들 관리, 객체 아이디로 검색하면, 위치 크기정보
#객체 정보 업데이트 히스토리에 추가
# 특정 시간동안 갱신이 안되면 자동 삭제 등등

# 프레임도 줘서 프레임별로 정보를 저장하게 할까

class FaceTrackInfo:

    def __init__(self, track_id, frame_num, x1, y1, x2, y2):

        self._track_id = track_id
        self._x1 = x1
        self._y1 = y1
        self._x2 = x2
        self._y2 = y2
        self._frame_num = frame_num

    @property
    def track_id(self):
        return self._track_id

    @property
    def x1(self):
        return self._x1

    @property
    def y1(self):
        return self._y1

    @property
    def x2(self):
        return self._x2

    @property
    def y2(self):
        return self._y2

    @property
    def frame_num(self):
        return self._frame_num


class FaceTrackHistory(dict):

    REMOVE_THRESHOLD_FRAME = 30
    REMOVE_INTERVAL = 100

    NUM_OF_PREVIOUS_FRAMES = 3

    def __init__(self):
        pass

    def get_estimated_next_position(self, track_id, frame_num) -> FaceTrackInfo:

        if len(self[track_id]) > FaceTrackHistory.NUM_OF_PREVIOUS_FRAMES:
            history_list = self[track_id][-FaceTrackHistory.NUM_OF_PREVIOUS_FRAMES:]

            center_position_list = []
            speed_list = []

            # get center positions of previous bounding boxes
            for history in history_list:

                cx = (history.x1 + history.x2) / 2
                cy = (history.y1 + history.y2) / 2

                center_position_list.append((cx, cy))

            center_position_leng = len(center_position_list)

            # get speed at every frame
            for i in range(center_position_leng - 1):

                # get speed between consecutive 2 frames
                sx1 = center_position_list[i][0]
                sy1 = center_position_list[i][1]

                sx2 = center_position_list[i + 1][0]
                sy2 = center_position_list[i + 1][1]

                vx = (sx2 - sx1)
                vy = (sy2 - sy1)
                speed_list.append((vx, vy))

            # get average speed for x and y directions
            avg_vx = sum(vx for vx, vy in speed_list) / len(speed_list)
            avg_vy = sum(vy for vx, vy in speed_list) / len(speed_list)

            last_frame = self[track_id][-1]
            last_cx = (last_frame.x1 + last_frame.x2) / 2
            last_cy = (last_frame.y1 + last_frame.y2) / 2

            # get estimated center position
            estimated_cx = last_cx + avg_vx
            estimated_cy = last_cy + avg_vy

            # get new bounding box position
            box_width = last_frame.x2 - last_frame.x1
            box_height = last_frame.y2 - last_frame.y1

            new_x1 = int(estimated_cx - box_width / 2)
            new_x2 = int(estimated_cx + box_width / 2)
            new_y1 = int(estimated_cy - box_height / 2)
            new_y2 = int(estimated_cy + box_height / 2)

            return FaceTrackInfo(track_id, frame_num, new_x1, new_y1, new_x2, new_y2)


        return None








    def add_frame(self, track_id, frame_num, x1, y1, x2, y2) -> None:
        # Check if id is already exist
        if self.get(track_id) is None:
            track_list = [FaceTrackInfo(id, frame_num, x1, y1, x2, y2)]
            self[track_id] = track_list

        else:
            self[track_id].append(FaceTrackInfo(track_id, frame_num, x1, y1, x2, y2))

        # Remove old information
        if frame_num % FaceTrackHistory.REMOVE_INTERVAL == FaceTrackHistory.REMOVE_INTERVAL - 1:
            remove_threshold = frame_num - FaceTrackHistory.REMOVE_THRESHOLD_FRAME
            self.remove_old_tracking_info(remove_threshold)


    def get_last_detected_frame(self, track_id) -> int:
        return self[track_id][-1].frame_num


    def remove_old_tracking_info(self, remove_threshold_frame) -> None:

        for track_id in list(self.keys()):
            # return track_info object that has hight frame number than threshold.
            filtered_list = [
                face_track_info for face_track_info in self[track_id]
                if face_track_info.frame_num >= remove_threshold_frame
            ]

            # if result list is not empty, add it to dictionary
            # Otherwise, remove it from dictionary
            if filtered_list:
                self[track_id] = filtered_list
            else:
                del self[track_id]


class ObjectTracker(Sort):

    REMOVE_HISTORY_INTERVAL = 55
    REMOVE_THRESHOLD = 50



    def __init__(self, video_width, video_height):
        super().__init__()

        self.video_width = video_width
        self.video_height = video_height

        self.face_detect_history = FaceTrackHistory()


    def update(self, frame_num: int, dets=np.empty((0, 5))):
        result = super().update(dets)

        for r in result:
            track_id = r[4]

            x1 = r[0]
            y1 = r[1]
            x2 = r[2]
            y2 = r[3]

            self.face_detect_history.add_frame(track_id, frame_num, x1, y1, x2, y2)

        return result

    def get_estimated_next_position(self, track_id, frame_num) -> FaceTrackInfo:
        return self.face_detect_history.get_estimated_next_position(track_id, frame_num)

