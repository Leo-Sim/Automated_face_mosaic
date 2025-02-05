import cv2

from ultralytics import YOLO
from src.config import Config
from video_player import VideoPlayer

video_path = "../youtube/video/test.mp4"
# video_path = "test.MP4"

video = VideoPlayer(video_path)
video.play()

