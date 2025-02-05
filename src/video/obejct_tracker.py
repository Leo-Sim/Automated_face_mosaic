
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

    def __init__(self):
        super().__init__()

        self.history = {}


        self.remove_threshold = 250


    def _remove_old_id_from_history(self, frame_num):

        if frame_num % 99 == 0:
            for id in self.history:
                last_frame = self.history[id]["frame"]
                if last_frame > self.remove_threshold:
                    del self.history[id]


    # Override Sort update. Add function to save object history
    def update(self, frame_num: int, dets=np.empty((0, 5))):
        result = super().update(dets)

        #
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


