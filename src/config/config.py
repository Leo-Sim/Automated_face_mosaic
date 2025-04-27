import yaml
import os

class Config:
    def __init__(self, config_path=None):

        if config_path is None:
            config_path = '../../config.yaml'
            # config_path = os.path.join(os.path.dirname(__file__), "../config.yml")

        self.config_path = config_path
        self.config = self._load_yaml()

    def _load_yaml(self):
        with open(self.config_path, "r") as file:
            try:
                config = yaml.safe_load(file)
                return config
            except yaml.YAMLError as e:
                print(f"Error loading YAML file: {e}")
                return {}

    def _get_project_root_path(self):
        config_path = os.path.abspath(__file__)
        return os.path.dirname(os.path.dirname(os.path.dirname(config_path)))


    ################# YOLO Config ##########################

    def get_yolo_config(self):
        return self.config.get("yolo")

    def get_video_config(self):
        return self.config.get("video")

    def get_video_input_path(self):
        path = self._get_project_root_path() + os.sep + self.get_video_config().get("input_path")
        return path

    def get_video_output_path(self):
        path = self._get_project_root_path() + os.sep + self.get_video_config().get("output_path")
        return path

    def get_yolo_config_image_size(self):
        return self.get_yolo_config().get("image_size")

    def get_yolo_config_epoch(self):
        return self.get_yolo_config().get("epoch")

    def get_yolo_config_batch_size(self):
        return self.get_yolo_config().get("batch_size")

    def get_yolo_config_export_path(self):
        return self.get_yolo_config().get("export_path")