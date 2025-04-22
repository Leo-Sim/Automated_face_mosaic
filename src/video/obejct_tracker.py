
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

    def __init__(self):
        pass

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

    # 객체 추적 저장 포맷
    history_format = {
        "id" : {
            "x1": "",
            "x2": "",
            "y1": "y1",
            "frame" : 1
        }
    }

    # 객체 아이디 추적 저장 포맷 (특정 객체를 삭제하기 위함)
    last_detected = {
        "id" : "last_frame_num",
        "id2" : "last_frame_num"
    }

    def __init__(self, video_width, video_height):
        super().__init__()

        self.video_width = video_width
        self.video_height = video_height

        self.face_detect_history = FaceTrackHistory()




    # def _remove_old_id_from_track_history(self, frame_num):
    #
    #     threshold = ObjectTracker.REMOVE_THRESHOLD
    #
    #     if frame_num % threshold == threshold - 1:
    #         del_list = []
    #         for id in self.history:
    #             last_frame = self.history[id]["frame"]
    #             if (frame_num - last_frame) > ObjectTracker.REMOVE_THRESHOLD:
    #
    #                 del_list.append(id)
    #
    #
    #         for id in del_list:
    #             print("@@@@@@@@@@@@@@@@@@@@@@@@ remove id : ", id)
    #             print("frame_num : ", frame_num)
    #             del self.history[id]


    # Override Sort update. Add function to save object history

    # --------------------------------------------------------
    # 직전 프레임에서 얼굴이 탐지되었고, 다음프레임에서 얼굴이 탐지되지 않는 경우에,
    # 직전 프레임의  box정보를 가져와서 이것이 충분히 크고 화면상의 위치 (너무 가장자리쪽이 아니라면)  이전 정보의 위치로 모자이크
    # 또 무슨방법이 있을까.... 과거 몇장의 프레임에서 탐지된 얼굴을 가져와서 크기 변화율도 적용..?
    # --------------------------------------------------------


    """
        1. frame_num을 통해서 직전 프레임에 탐지된 애들 골라내기
        2. 너무가장자리가 아닌지 위치 확인.
        3. 그 후 앞으로 나올 3~5프레임을 보정
        
    """
    def update(self, frame_num: int, dets=np.empty((0, 5))):
        result = super().update(dets)

        # self._remove_old_id_from_track_history(frame_num)

        # get id that is not in current frame but in current - 1 frame

        for r in result:
            track_id = r[4]

            x1 = r[0]
            y1 = r[1]
            x2 = r[2]
            y2 = r[3]

            self.face_detect_history.add_frame(track_id, frame_num, x1, y1, x2, y2)



        # self.face_detect_history.add_frame(frame_num, result)

        # print("last : ", self.face_detect_history.get_last_detected_frame(frame_num))

        # for row in result:
        #     id = row[4]
        #
        #     x1 = row[0]
        #     y1 = row[1]
        #     x2 = row[2]
        #     y2 = row[3]
        #
        #     """
        #         밑의 맵 구조가 이상한데...
        #         id별 탐지된 모든 프레임을 저장해야하는데,
        #         "frame": frame_num 을 저장하면, 현재프레임만 저장되는데. 즉, 키값 id에 value가 계속 현재프레임으로 오버라이딩돼...
        #     """
        #
        #     self.history[id] = {
        #         "x1" : x1,
        #         "x2" : x2,
        #         "y1" : y1,
        #         "y2" : y2,
        #         "frame" : frame_num
        #     }
        #
        #     self.last_detected[id] = {
        #         "last_detected_frame" : frame_num,
        #     }




        return result


