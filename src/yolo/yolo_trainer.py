
from src.config import Config
if __name__ == '__main__':
    from ultralytics import YOLO



    config = Config()
    export_path = config.get_yolo_config_export_path()
    epoch = config.get_yolo_config_epoch()
    batch_size = config.get_yolo_config_batch_size()
    image_size = config.get_yolo_config_image_size()

    print(export_path)
    print(epoch)
    print(batch_size)
    print(image_size)

    model = YOLO("yolo11n.pt")
    model.train(data='yolo.yaml', epochs=epoch, batch=batch_size, imgsz=image_size, project=export_path, name="yolo_train", val=False)