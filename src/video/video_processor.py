import cv2

from ultralytics import YOLO
from src.config import Config
from video_player import VideoPlayer
from src.insightface.detector import FaceManager


config = Config()

input_path = config.get_video_input_path()
output_path = config.get_video_output_path()


detector = FaceManager("../insightface/nani.JPG")



video = VideoPlayer(input_path)
video.set_face_detector(detector)

video.play()

