
from sort.sort import Sort
import numpy as np


# 객체들 관리, 객체 아이디로 검색하면, 위치 크기정보
#객체 정보 업데이트 히스토리에 추가
# 특정 시간동안 갱신이 안되면 자동 삭제 등등

# 프레임도 줘서 프레임별로 정보를 저장하게 할까



class ObjectTracker(Sort):

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

        self.history = {}

        self.remove_threshold = 250


    def _remove_old_id_from_track_history(self, frame_num):

        if frame_num % 99 == 0:
            del_list = []
            for id in self.history:
                last_frame = self.history[id]["frame"]
                if last_frame > self.remove_threshold:

                    del_list.append(id)

            for id in del_list:
                del self.history[id]


    # Override Sort update. Add function to save object history

    # --------------------------------------------------------
    # 직전 프레임에서 얼굴이 탐지되었고, 다음프레임에서 얼굴이 탐지되지 않는 경우에,
    # 직전 프레임의  box정보를 가져와서 이것이 충분히 크고 화면상의 위치 (너무 가장자리쪽이 아니라면)  이전 정보의 위치로 모자이크
    # 또 무슨방법이 있을까.... 과거 몇장의 프레임에서 탐지된 얼굴을 가져와서 크기 변화율도 적용..?
    # --------------------------------------------------------
    def update(self, frame_num: int, dets=np.empty((0, 5))):
        result = super().update(dets)

        self._remove_old_id_from_track_history(frame_num)

        for row in result:
            id = row[4]

            x1 = row[0]
            y1 = row[1]
            x2 = row[2]
            y2 = row[3]

            self.history[id] = {
                "x1" : x1,
                "x2" : x2,
                "y1" : y1,
                "y2" : y2,
                "frame" : frame_num
            }

        return result


