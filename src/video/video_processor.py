import cv2

from ultralytics import YOLO
from src.config import Config
from video_player import VideoPlayer
from src.insightface.detector import FaceManager

# video_path = "../youtube/video/test1.mp4"
video_path = "../video/test1.MOV"
# video_path = "test.MP4"

# register target face


detector = FaceManager("../insightface/nani.JPG")

video = VideoPlayer(video_path)
video.set_face_detector(detector)

video.play()

