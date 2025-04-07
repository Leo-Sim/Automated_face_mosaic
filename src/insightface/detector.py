import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity


class FaceManager:

    def __init__(self, target_face_path):

        self.target_face_path = target_face_path


        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0)

        self.target_face = cv2.imread(target_face_path)

        self.target_embedding = None
        self._register_face()


    def _register_face(self):

        faces = self.app.get(self.target_face)

        if faces:
            target_embedding = faces[0].embedding
            self.target_embedding = target_embedding

        else:
            print("############# Cannot find any face in the image")


    def detect_face(self, compare_face):
        video_faces = self.app.get(compare_face)

        target_vector = self.target_embedding.reshape(1, -1)

        for i, video_face in enumerate(video_faces):


            video_face_embedding = video_face.embedding


            compare_vector = video_face_embedding.reshape(1, -1)

            similarity = cosine_similarity(target_vector, compare_vector)


class DetectInfo:


    def __init__(self, is_same, bbox):
        self._is_same = is_same
        self._bbox = bbox


    @property
    def is_same(self):
        return self._is_same

    @property
    def bbox(self):
        return self._bbox


